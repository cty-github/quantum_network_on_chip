"""Export existing MECH/Sensate configurations to one CSV / 导出架构参数总表。

Run from the repository root with the project's qunoc Python environment.
Only MECH topology/highway construction is rerun; no routing or training runs.
仅重建 MECH 拓扑与 highway，不运行路由、训练或保真度实验。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import threading
import time
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.mech_utils import get_mech_highway_proportion
from baseline.MECH.Chiplet import gen_chiplet_array, gen_highway_layout
from qunoc.element.package import Package
from qunoc.element.chiplet import NodeType

SWEEPS = {
    'package_size': 'Package size',
    'package_shape': 'Package shape',
    'geometry': 'Compute topology',
    'inter_fidelity': 'Inter fidelity',
    'chiplet_sparsity': 'Coupler sparsity',
}
# Export only parameter columns / 仅导出参数列。
COLUMN_NAMES = {
    '变化参数': 'Sweep parameter',
    '变化参数值': 'Parameter value',
    '架构': 'Architecture',
    '布局策略': 'Layout strategy',
    'Package实际尺寸': 'Package dimensions',
    'Compute topology': 'Compute topology',
    'Inter fidelity': 'Inter fidelity',
    'Coupler sparsity': 'Coupler sparsity',
    'Chiplet数量': 'Chiplet count',
    '总比特数': 'Total qubit count',
    '通信比特数': 'Communication qubit count',
    '通信比特比例': 'Communication qubit ratio',
    '计算比特数': 'Compute qubit count',
    '计算比特比例': 'Compute qubit ratio',
    '空位数': 'Empty site count',
}
FIELDS = list(COLUMN_NAMES.values())


def progress(message):
    """Print timestamped progress immediately / 即时输出带时间戳的进度。"""
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


@contextmanager
def timed_stage(label):
    """Report stage timing and 30s heartbeats / 输出阶段耗时及每30秒状态提示。"""
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        """Report elapsed time, not estimated completion / 报告已用时间，不估算完成比例。"""
        while not stopped.wait(30):
            progress(f'{label}：仍在运行，已用 {time.monotonic() - started:.1f}s')

    progress(f'{label}：开始')
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException as exc:
        stopped.set()
        worker.join()
        progress(f'{label}：失败/中断 ({type(exc).__name__})，耗时 {time.monotonic() - started:.1f}s')
        raise
    else:
        stopped.set()
        worker.join()
        progress(f'{label}：完成，耗时 {time.monotonic() - started:.1f}s')
    finally:
        stopped.set()


def packed(value):
    """Encode structured CSV cells / 将结构化内容编码为单个单元格。"""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def axis_value(sweep, size, value):
    """Normalize shape labels, retaining orientation elsewhere / 统一形状标签但保留实际方向。"""
    if sweep == 'package_shape':
        return '×'.join(map(str, sorted(size)))
    return str(value)


def size_text(size):
    """Format coordinate extents / 格式化坐标尺寸。"""
    return '×'.join(map(str, size))


def validate_qubit_counts(compute, communication, total, empty, size):
    """Validate physical counts excluding empty sites / 校验物理比特计数，分母不含空位。"""
    if total <= 0 or min(compute, communication, empty) < 0:
        raise ValueError('Invalid physical qubit counts')
    if compute + communication != total or total + empty != math.prod(size):
        raise ValueError('Qubit counts do not match topology or package dimensions')


def mech_rows(result):
    """Reuse saved cases and highway interface / 复用已存配置及 highway 接口。"""
    for sweep, label in SWEEPS.items():
        cases = {}
        for source in sorted((result / 'baseline_mech' / sweep).glob('*.json')):
            obj = json.loads(source.read_text())
            case = obj['case']
            # Deduplicate identical hardware across workloads / 按硬件配置去重工作负载。
            key = packed(case)
            cases.setdefault(key, case)
        progress(f'MECH 分组 {list(SWEEPS).index(sweep) + 1}/{len(SWEEPS)}：{label}，共 {len(cases)} 个配置')
        for index, case in enumerate(cases.values(), 1):
            started = time.monotonic()
            tag = f'MECH | {label} | {index}/{len(cases)} | {case["raw_name"]}'
            progress(f'{tag}：开始统计')
            size = case['package_size']
            kwargs = {k: case[k] for k in (
                'structure', 'chiplet_array_dim', 'chiplet_size',
                'custom_highway_qubits', 'custom_highway_edges',
                'cross_link_sparsity', 'cross_link_row_spacing', 'cross_link_col_spacing',
            ) if k in case}
            with timed_stage(f'{tag} / 1/2 get_mech_highway_proportion'):
                ratio = get_mech_highway_proportion(**kwargs)
            array, chip = case['chiplet_array_dim'], case['chiplet_size']
            with timed_stage(f'{tag} / 2/2 重建布局以统计data标记'):
                graph = gen_chiplet_array(case['structure'], *array, *chip,
                    **{k: v for k, v in kwargs.items() if k.startswith('cross_link_')})
                if 'custom_highway_qubits' in case:
                    from baseline.MECH.Chiplet import custom_highway_layout
                    custom_highway_layout(graph, case['custom_highway_qubits'], case.get('custom_highway_edges', []))
                else:
                    gen_highway_layout(graph)
            counts = Counter(d['type'] for _, d in graph.nodes(data=True))
            total = len(graph)
            validate_qubit_counts(
                counts['data'], counts['highway'], total, math.prod(size) - total, size
            )
            assert math.isclose(ratio, counts['highway'] / total)
            assert set(counts) <= {'data', 'highway'}, counts
            actual_chip = [graph.chiplet_x_size, graph.chiplet_y_size]
            actual_size = [graph.array_x_dim * actual_chip[0], graph.array_y_dim * actual_chip[1]]
            assert list(size) == actual_size, (size, actual_size)
            yield {
                '变化参数': label, '变化参数值': axis_value(sweep, size, case['axis_values'][0]),
                '架构': 'MECH', '布局策略': 'highway', 'Package实际尺寸': size_text(size),
                'Compute topology': case['structure'], 'Inter fidelity': case.get('inter_fide', 0.983),
                'Coupler sparsity': case.get('inter_sparsity', 3),
                'Chiplet数量': math.prod(array),
                '总比特数': total, '通信比特数': counts['highway'], '通信比特比例': ratio,
                '计算比特数': counts['data'], '计算比特比例': counts['data'] / total,
                '空位数': math.prod(size) - total,
            }
            progress(f'{tag}：统计完成，配置耗时 {time.monotonic() - started:.1f}s')


def sensate_rows(result):
    """Read layouts referenced by routing metadata / 读取路由元数据引用的真实布局。"""
    for sweep, label in SWEEPS.items():
        layouts = set()
        for source in sorted((result / 'routing_scalability' / sweep).glob('*/exp_result.json')):
            metadata = json.loads(source.read_text())
            raw = metadata.get('layout_package_path')
            if raw is None:
                raise ValueError(f'Missing layout_package_path: {source}')
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                raise FileNotFoundError(path)
            layouts.add(path)
        progress(f'Sensate 分组 {list(SWEEPS).index(sweep) + 1}/{len(SWEEPS)}：{label}，共 {len(layouts)} 个布局')
        for index, path in enumerate(sorted(layouts), 1):
            started = time.monotonic()
            tag = f'Sensate | {label} | {index}/{len(layouts)} | {path.parent.name}'
            with timed_stage(f'{tag} / 读取 {path.relative_to(ROOT)}'):
                package = Package.from_pickle(filepath=str(path))
            progress(f'{tag}：统计比特标记及Chiplet总数量')
            size, fab = package.package_size, package.fab_model
            value = {'package_size': size[0], 'geometry': fab.compute_geometry,
                'inter_fidelity': fab.inter_fide, 'chiplet_sparsity': fab.inter_sparsity,
                'package_shape': None}[sweep]
            counts = Counter(d['type'] for _, d in package.package_graph.nodes(data=True))
            # Count physical types, not Register roles; Empty sites are excluded.
            # 按物理类型计数而非Register角色，Empty空位不计入比例分母。
            c, t = counts[NodeType.Compute], counts[NodeType.Transfer]
            validate_qubit_counts(c, t, c + t, counts[NodeType.Empty], size)
            assert c == len(package.compute_nodes) and t == len(package.transfer_nodes)
            assert c + t + counts[NodeType.Empty] == math.prod(size)
            compute = package.compute_chiplets
            transfer = package.transfer_chiplets
            yield {
                '变化参数': label, '变化参数值': axis_value(sweep, size, value),
                '架构': 'Sensate', '布局策略': package.design_strategy,
                'Package实际尺寸': size_text(size), 'Compute topology': fab.compute_geometry,
                'Inter fidelity': fab.inter_fide, 'Coupler sparsity': fab.inter_sparsity,
                'Chiplet数量': len(compute)+len(transfer),
                '总比特数': c+t, '通信比特数': t, '通信比特比例': t/(c+t),
                '计算比特数': c, '计算比特比例': c/(c+t), '空位数': counts[NodeType.Empty],
            }
            del package, compute, transfer
            gc.collect()
            progress(f'{tag}：统计完成，配置耗时 {time.monotonic() - started:.1f}s')


def main():
    """Write one UTF-8 BOM CSV for spreadsheet readers / 输出可供表格软件读取的单个CSV。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluate/result/parameters/architecture_parameters.csv')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {args.output}')
    started = time.monotonic()
    progress(f'开始导出，目标：{args.output}')
    progress('CSV将在所有配置统计和校验完成后写入；中断不会保存部分结果。')
    result = ROOT / 'evaluate/result'
    with timed_stage('架构 1/2 MECH'):
        rows = list(mech_rows(result))
    progress(f'已收集 {len(rows)} 行')
    with timed_stage('架构 2/2 Sensate'):
        rows.extend(sensate_rows(result))
    progress(f'已收集 {len(rows)} 行，开始排序与唯一性校验')
    order = {label: i for i, label in enumerate(SWEEPS.values())}
    def sort_key(row):
        """Sort by sweep then numeric axis / 按变化参数及数值排序。"""
        value = row['变化参数值']
        try:
            axis = (0, float(value.split('×')[0]))
        except ValueError:
            axis = (1, value)
        return order[row['变化参数']], axis, row['架构'], row['布局策略']
    rows.sort(key=sort_key)
    keys = [(r['变化参数'], r['变化参数值'], r['架构'], r['布局策略']) for r in rows]
    assert len(keys) == len(set(keys)), 'Conflicting hardware cases for the same row key'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            {english: row[original] for original, english in COLUMN_NAMES.items()}
            for row in rows
        )
    progress(f'导出完成：{len(rows)} 行 → {args.output}，总耗时 {time.monotonic() - started:.1f}s')
    progress(f'各组行数：{dict(Counter((r["变化参数"], r["架构"]) for r in rows))}')


if __name__ == '__main__':
    main()
