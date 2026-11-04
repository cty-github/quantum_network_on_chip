#!/usr/bin/env python3
"""Two Hamiltonian paper experiments / 两项哈密顿量论文实验。

The numerical and matrix-free planning implementation lives in qunoc.hamiltonian.
数值演化及无矩阵规划由 qunoc.hamiltonian 提供，本脚本只负责评估和导出。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import pickle
import tempfile
from collections import Counter
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

# 支持仓库根目录与其他目录直接执行。 Support direct script invocation from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluate.plot_result_figures import (DEFAULT_FIGURE_ROOT, write_probability_plot,
                                         write_length_plot, write_optimization_plot, write_real_case_plot)

from qunoc.hamiltonian import chain_evolution as ce
from qunoc.hamiltonian.noise import NOISE_MODEL_VERSION, NoiseConfig, calibration_reference, evaluate_plan_noise

from qunoc.hamiltonian.chain_evolution import (
    DEFAULT_G_MAX_RAD_PER_NS, REFERENCE_DISTANCE, REFERENCE_FIDELITY,
    REFERENCE_SITES, REFERENCE_TIME_NS, Pair, Stages, SimulationConfig, SimulationResult,
    run_simulation, simulate_transfer,
)

__all__ = ["simulate_transfer"]


def parse_edge_pairs(text: str) -> tuple[Pair, ...]:
    """功能 / Purpose: Parse strict a:b lists / 解析严格冒号语法。

    输入 / Inputs: Comma-separated text / 逗号分隔的字符串。
    输出 / Outputs: Pair tuple; ValueError for malformed input / pair 元组；格式错误时报错。
    """
    pairs = []
    for item in text.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise ValueError("use comma-separated a:b pairs, e.g. 0:9,5:19")
        try:
            pairs.append((int(fields[0]), int(fields[1])))
        except ValueError as exc:
            raise ValueError(f"pair endpoints must be integers: {item!r}") from exc
    return tuple(pairs)


def parse_segment_stages(text: str) -> Stages:
    """功能 / Purpose: Parse sequential/parallel mirrors / 解析顺序与并行镜像。

    输入 / Inputs: Text: semicolons separate stages, commas intervals / 分号分阶段、逗号分区间。
    输出 / Outputs: Nested stage tuple; malformed text raises ValueError / 阶段元组；拒绝空项。
    """
    return tuple(parse_edge_pairs(stage) for stage in text.split(";"))


def result_to_jsonable(result: SimulationResult) -> dict[str, object]:
    """Purpose / 功能: Serialize genuine noise results / 序列化真实噪声结果。

    Inputs / 输入: Core result / 核心结果。 Outputs / 输出: Auditable JSON dictionary / 可审计字典。
    """
    payload = asdict(result)
    payload.update({"sources": sorted(result.config.sources), "targets": sorted(result.config.targets),
                    "metric": METRIC, "time_unit": "ns", "coupling_unit": "rad/ns",
                    "model_version": ce.MODEL_VERSION, "noise_model_version": result.noise["model_version"], "citation": REFERENCE_CITATION})
    payload["fidelity_fields"] = {
        "peak_fidelity, final_fidelity": "ideal coherent target occupation probability",
        "noisy_fidelities": "actual T1 + local pure-dephasing dynamics, not a distance envelope",
        "calibrated_*": "v2 compatibility aliases for actual noisy occupation probability",
        "amplitudes": "ideal coherent amplitudes only; no noisy pure-state amplitude exists",
        "peaks": "sampled maxima, not a continuous-time optimization"}
    payload["noise_scope"] = NOISE_SCOPE
    payload["sampling"] = "num_steps per constant-Hamiltonian interval; boundaries appear once"
    return payload


def noise_summary(noisy, errors, lower, upper, metadata: dict) -> dict:
    """Purpose / 功能: Summarize noise precision / 汇总噪声精度。

    Inputs / 输入: Probability/error/CI arrays and metadata / 概率、误差、区间与元数据。
    Outputs / 输出: Completion scalars and solver provenance / 完成值与求解器来源。
    """
    metadata = copy.deepcopy(metadata)
    if metadata.get("uncertainty", {}).get("trajectories", 0):
        metadata["uncertainty"]["bound_formula"] = (
            "c=ln(4*M/(1-confidence)); conditional radius=min(1,sqrt(2*sample_variance*c/n)"
            "+7*c/(3*(n-1))); scale by jump-branch weight and relaxation; "
            "M=number of adaptive checkpoints; two-sided per-time marginal bound")
    return {"noisy_final_fidelity": float(noisy[-1]), "noisy_final_standard_error": float(errors[-1]),
            "noisy_final_ci_lower": float(lower[-1]), "noisy_final_ci_upper": float(upper[-1]),
            "noise": metadata, "model_version": ce.MODEL_VERSION,
            "noise_model_version": metadata["model_version"], "noise_scope": NOISE_SCOPE,
            "status": "complete" if metadata.get("converged", False) else "not_converged"}



def write_curve_outputs(output: Path, summary: dict, times, target, source, ideal,
                        noisy, errors, lower, upper, title: str) -> None:
    """Purpose / 功能: Export ideal amplitudes and noisy probabilities / 导出理想振幅与噪声概率。

    Inputs / 输入: New directory, summary and sampled arrays / 新目录、摘要及数组。
    Outputs / 输出: JSON, CSV, PNG/PDF; never fictional noisy amplitudes / 不输出虚构噪声振幅。
    """
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CURVE_COLUMNS)
        writer.writerows((float(t), float(a.real), float(a.imag), float(b.real), float(b.imag),
                          float(f), float(n), float(e), float(lo), float(hi))
                         for t, a, b, f, n, e, lo, hi in zip(times, target, source, ideal, noisy, errors,
                                                            lower, upper, strict=True))
    # 最后发布摘要，哈希将完整工件绑定到本次计算。 Publish the summary last, binding all artifacts to this computation.
    summary["artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in CURVE_FILES if name != "summary.json"}
    write_json(output / "summary.json", summary)


def write_outputs(output_dir: Path, result: SimulationResult, times: np.ndarray,
                  target_amps: np.ndarray, source_amps: np.ndarray, fidelities: np.ndarray) -> dict:
    """Purpose / 功能: Export core results / 导出核心结果。

    Inputs / 输入: New directory and five-tuple result / 新目录与五元组结果。
    Outputs / 输出: Summary plus curve artifacts / 摘要与曲线文件。
    """
    summary = result_to_jsonable(result)
    summary.update(noise_summary(result.noisy_fidelities, result.noisy_standard_errors,
                                 result.noisy_ci_lower, result.noisy_ci_upper, result.noise))
    summary["output_dir"] = str(output_dir)
    write_curve_outputs(output_dir, summary, times, target_amps, source_amps, fidelities,
                        result.noisy_fidelities, result.noisy_standard_errors, result.noisy_ci_lower,
                        result.noisy_ci_upper, f"XY transfer · {result.route}, N = {result.config.chain_length}")
    return summary



# Paper experiments / 论文实验：原辅助模块函数原样并入，保持数值与筛选规则。
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "evaluate/result/hamiltonian_simulation"
DEFAULT_MANIFEST = ROOT / "evaluate/result/hamiltonian_revision/compiler/selection_report.json"
DEFAULT_ROUTING_ROOT = ROOT / "evaluate/result/routing_scalability"
DEFAULT_LENGTHS = tuple(range(6, 81, 2))
SINGLE_CHAIN_CACHE_VERSION = 2
CURVE_FILES = ("summary.json", "fidelity.csv")
CURVE_COLUMNS = ("time_ns", "ideal_target_amp_real", "ideal_target_amp_imag",
                 "ideal_source_amp_real", "ideal_source_amp_imag", "ideal_target_fidelity",
                 "noisy_target_fidelity", "noisy_standard_error", "noisy_ci_lower", "noisy_ci_upper")
CLASSES = ("nonconflicting", "conflicting")
DEFAULT_PROGRAMS = ("vqe", "qaoa", "bv", "s_ghz")
EQST_KEYS = ("reference_time_ns", "reference_distance", "g_max", "max_segment_steps", "beam_width")
METRIC = "target occupation probability; not labeled-state or logical process fidelity"
REFERENCE_CITATION = {
    "source": "Nature Communications 2024; Supplementary Information p18 Table 1",
    "reference": "six sites, single excitation, 125 ns, target occupation probability 0.992",
    "coherence": "Supplementary Information p3 Fig. 2: mean T1=141.5 us, spin-echo T2=19.4 us",
    "qualification": "ideal engineered coupling profile, not measured edge-by-edge reconstruction; fitted gamma_phi is effective, not measured Tphi"}
NOISE_SCOPE = ("Uniform zero-temperature T1 and local Markovian pure dephasing over the entire transfer schedule, "
               "including idle resident excitations and switched-off primitives; no reset at stage boundaries. "
               "Excludes inter-stage circuit compute/wait outside this transfer plan and unmodeled hardware errors.")


def parse_lengths(text: str) -> tuple[int, ...]:
    """Purpose / 功能: Parse unique chain lengths / 解析不重复链长。

    Inputs / 输入: Comma-separated integers / 逗号分隔整数。
    Outputs / 输出: Sorted lengths >=2; ValueError otherwise / 有序链长或校验错误。
    """
    try:
        lengths = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise ValueError("--lengths requires comma-separated integers >=2") from exc
    if not lengths or any(n < 2 for n in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError("--lengths must be unique integers >=2")
    return tuple(sorted(lengths))


def write_json(path: Path, payload: dict) -> None:
    """Purpose / 功能: Write strict JSON / 写严格 JSON。

    Inputs / 输入: Output path and JSON-ready dictionary / 输出路径与字典。
    Outputs / 输出: None; atomically replaces UTF-8 file / 无；原子替换 UTF-8 文件。
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # 同目录原子替换；同路径复用清单不会被半写截断。 Atomic replacement also protects in-place manifest reuse.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fixed_output(root: Path, name: str, protected: tuple[Path, ...] = ()) -> Path:
    """Purpose / 功能: Reuse a safe fixed output tree without deleting history / 安全复用固定目录，不删除历史。

    Inputs / 输入: Root, experiment name, read-only source trees / 根目录、实验名称、只读源目录。
    Outputs / 输出: Existing/new directory; rejects overlaps and linked destinations / 返回目录；拒绝源重叠与链接目标。
    """
    path = root.expanduser().absolute() / name
    # 拒绝输出祖先及后代的链接，防止 CSV/绘图跟随链接写入源文件。 Never follow output links into source artifacts.
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"output paths must not contain symlinks: {parent}")
    path = path.resolve()
    for source in protected:
        source = source.expanduser().resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError(f"output directory overlaps source artifacts: {source}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        for parent, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                child = Path(parent) / name
                if child.is_symlink() or (child.is_file() and child.stat().st_nlink > 1):
                    raise ValueError(f"output tree must not contain linked artifacts: {child}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_case_files(output: Path, rows: list[dict], extra: tuple[str, ...] = ()) -> list[str]:
    """Purpose / 功能: List only this invocation's artifacts / 仅列本次调用的文件。

    Inputs / 输入: Output directory, current rows, extra case filenames / 输出目录、当前行、附加案例文件名。
    Outputs / 输出: Relative paths, never unrelated retained case directories / 相对路径，不包含保留的旧案例目录。
    """
    return sorted(str((Path(row["output_dir"]) / name).relative_to(output))
                  for row in rows for name in (*CURVE_FILES, *extra)
                  if (Path(row["output_dir"]) / name).is_file())


def fresh_output(root: Path, name: str, protected: tuple[Path, ...] = ()) -> Path:
    """Purpose / 功能: Reserve a fresh experiment directory / 预留全新实验目录。

    Inputs / 输入: Output root, name, source directories / 输出根目录、名称与源目录。
    Outputs / 输出: New directory; rejects source trees / 新目录；拒绝写入源目录树。
    """
    root = root.expanduser().resolve()
    for source in protected:
        source = source.resolve()
        if root == source or root.is_relative_to(source):
            raise ValueError(f"output directory must not be inside source artifacts: {source}")
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        path = root / (name if number == 1 else f"{name}_run_{number:03d}")
        # 已有目录和符号链接均跳过；绝不清空旧结果。 Skip old paths/symlinks, never delete.
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise ValueError("too many existing experiment output directories")


def estimate_resources(dimension: int, excitations: int, intervals: int, primitives: int, num_steps: int,
                       chain_length: int | None = None, noise: NoiseConfig | None = None,
                       route: str = "pst-segments", ideal: bool = False) -> dict:
    """Purpose / 功能: Bound resident sparse matrices and sampled states / 估算驻留稀疏矩阵与采样态内存。

    Inputs / 输入: Dimension, k, schedule counts, samples/interval / 维度、激发数、调度数与每区间样本数。
    Outputs / 输出: Conservative byte estimates, not measured RSS / 保守字节估算，非实测 RSS。
    """
    samples = 1 + intervals * (num_steps - 1)
    # 每个粒子至多两个最近邻；直接边不相交。 Both schedules have row degree <= 2k.
    degree = min(dimension, 2 * excitations)
    csr_bytes = dimension * degree * 32 + (dimension + 1) * 8
    matrix_bytes = 2 * (primitives + intervals + 2) * csr_bytes
    state_bytes = 2 * num_steps * dimension * 16 + 32 * dimension * 16
    output_bytes = samples * 256  # Python lists + final NumPy arrays / 列表与最终数组同时驻留。
    basis_bytes = dimension * (160 + 80 * excitations)
    noise = noise or NoiseConfig()
    n = chain_length or dimension
    # 顶扇区密度矩阵为 D²，绝不分配 D⁴ Liouvillian；轨迹逐条流式统计。
    # Density buffers scale as D², never a dense D⁴ Liouvillian; trajectories stream.
    # --ideal disables dissipators, not solver allocations / 理想标记关闭噪声而非求解器内存。
    if intervals == 0 or excitations == n:
        backend, noise_bytes = "exact-invariant-occupation", samples * 128
    elif route == "direct-xy" and noise.solver != "master":
        backend, noise_bytes = "independent-pair density", excitations * samples * 128
    elif noise.solver == "master" or (noise.solver == "auto" and dimension <= noise.master_max_dim):
        backend, noise_bytes = "top-sector density", (24 + 2 * num_steps) * dimension ** 2 * 16
    else:
        backend = "streamed Slater trajectories"
        noise_bytes = 12 * n * excitations * 16 + (intervals + 4) * n ** 2 * 16 + samples * 128
    return {"hilbert_dimension": dimension, "interval_count": intervals, "primitive_count": primitives,
            "sample_count": samples, "matrix_bytes": matrix_bytes, "state_bytes": state_bytes,
            "output_bytes": output_bytes, "basis_bytes": basis_bytes,
            "noise_backend_estimate": backend, "noise_workspace_bytes": noise_bytes,
            "estimated_total_bytes": matrix_bytes + state_bytes + output_bytes + basis_bytes + noise_bytes,
            "estimate_scope": "conservative numerical working set including sampled density matrices; excludes imported libraries and source pickle objects"}


def resource_exclusion(resources: dict, max_dimension: int, max_intervals: int, max_state_mb: float) -> str | None:
    """Purpose / 功能: Check numerical budgets before matrix allocation / 矩阵分配前检查预算。

    Inputs / 输入: Estimate and dimension/interval/MiB caps / 估算与维度、区间、MiB 上限。
    Outputs / 输出: Exclusion reason or None / 排除原因或 None。
    """
    if resources["hilbert_dimension"] > max_dimension:
        return "dimension_limit"
    if resources["interval_count"] > max_intervals:
        return "interval_limit"
    if resources["estimated_total_bytes"] > max_state_mb * 1024 ** 2:
        return "memory_limit"
    return None


def validate_budgets(num_steps: int, max_dimension: int, max_intervals: int, max_state_mb: float) -> None:
    """Purpose / 功能: Validate shared numerical limits / 校验共同数值上限。

    Inputs / 输入: Sampling and resource limits / 采样和资源上限。
    Outputs / 输出: None; raises ValueError on invalid limits / 无；非法参数抛错。
    """
    for name, value, minimum in (("num_steps", num_steps, 2), ("max_dimension", max_dimension, 1),
                                  ("max_intervals", max_intervals, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(max_state_mb, bool) or not math.isfinite(max_state_mb) or max_state_mb <= 0:
        raise ValueError("max_state_mb must be finite and positive")



def write_case_table(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    """Purpose / 功能: Export a stable summary table / 导出固定列摘要表。

    Inputs / 输入: CSV path, rows and column names / CSV 路径、行和列名。
    Outputs / 输出: CSV, including a header when empty / CSV；空表保留表头。
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)



def single_chain_case(n: int, g_max: float | None, num_steps: int, noise: NoiseConfig, ideal: bool,
                      reference_time_ns: float, reference_fidelity: float, coupling_policy: str,
                      resources: dict, case_output: Path) -> dict:
    """Purpose / 功能: Evolve one true NN chain with shared noise / 用公共噪声演化真实最近邻链。

    Inputs / 输入: Physical parameters, sampling, budgets and directory / 物理参数、采样、预算及目录。
    Outputs / 输出: Summary plus genuine ideal/noisy artifacts / 摘要及真实理想、噪声文件。
    """
    started = time.perf_counter()
    plan = ce.build_single_chain_plan(n, g_max, noise=noise, ideal=ideal, coupling_policy=coupling_policy,
                                      reference_time_ns=reference_time_ns, reference_fidelity=reference_fidelity)
    primitive = plan.primitives[0]
    profile = ce.engineered_pst_couplings(n)
    basis, index = ce.build_fixed_excitation_basis(n, 1)
    hamiltonian, duration, actual_g, maximum = ce.build_pst_segment_stage_hamiltonian(
        basis, index, ((0, n - 1),), primitive.effective_g_rad_per_ns, g_max)
    times, target, source, fidelities = ce.simulate_schedule_time_grid(
        n, index[(0,)], index[(n - 1,)], [hamiltonian], [duration], num_steps)
    curves = evaluate_plan_noise(plan, times, noise)
    error = float(np.max(np.abs(fidelities - np.sin(actual_g * times) ** (2 * (n - 1)))))
    summary = {"experiment": "single-chain", "chain_length": n, "excitation_count": 1,
               "sources": [0], "targets": [n - 1], "metric": METRIC, "time_unit": "ns",
               "coupling_unit": "rad/ns", "g_max_rad_per_ns": g_max,
               "peak_coupling_mhz": maximum / (2 * math.pi) * 1000,
               "effective_g_rad_per_ns": actual_g, "edge_couplings_rad_per_ns": (actual_g * profile).tolist(),
               "edges": [[j, j + 1] for j in range(n - 1)], "boundary": "open; nearest-neighbor only",
               "timing_rule": ("tau=reference_time_ns*sqrt((N-1)/5); g=pi/(2*tau)" if g_max is None else
                               "g=J_max/max_j sqrt(j*(N-j)); tau=pi/(2*g)"),
               "coupling_mode": "uncapped-request" if g_max is None else "fixed-peak",
               "peak_coupling_meaning": "required coupling, not hardware maximum" if g_max is None else "hardware maximum",
               "required_peak_coupling_rad_per_ns": maximum,
               "coupling_policy_requested": coupling_policy,
               "coupling_policy_note": ("uncapped single-chain honors requested timing; policies coincide" if g_max is None else
                                        "single-chain is already cap-limited; policies coincide, not duplicated"),
               "final_time_ns": float(times[-1]), "ideal_final_fidelity": float(fidelities[-1]),
               "ideal_peak_fidelity": float(fidelities.max()), "sample_count": len(times),
               "analytic_probability": "sin(g*t)^(2*(N-1))", "max_analytic_error": error,
               "route_details": plan.to_route_details(), "simulation_config": asdict(plan.config),
               "citation": REFERENCE_CITATION, "resources": resources, "output_dir": str(case_output),
               "runtime_seconds": time.perf_counter() - started,
               **noise_summary(curves["fidelities"], curves["standard_errors"], curves["ci_lower"],
                               curves["ci_upper"], curves["metadata"])}
    if error > 1e-9 or not math.isclose(duration, plan.transfer_time_ns, rel_tol=1e-12):
        summary["status"] = "failed_analytic_check"
    write_curve_outputs(case_output, summary, times, target, source, fidelities, curves["fidelities"],
                        curves["standard_errors"], curves["ci_lower"], curves["ci_upper"],
                        f"Nearest-neighbor PST · N = {n}, one excitation")
    return summary


def single_chain_signature(plan, num_steps: int) -> dict:
    """Purpose / 功能: Fingerprint all physical and sampling controls / 标识所有物理及采样参数。

    Inputs / 输入: Validated single-chain plan and actual sample count / 已校验单链计划与实际样本数。
    Outputs / 输出: JSON-normalized, versioned configuration / JSON 规范化版本配置。
    """
    return json.loads(json.dumps({"cache_version": SINGLE_CHAIN_CACHE_VERSION,
                                 "model_version": ce.MODEL_VERSION, "noise_model_version": NOISE_MODEL_VERSION,
                                 "simulation_config": asdict(plan.config), "num_steps": num_steps}, allow_nan=False))


def load_single_chain_cache(case_output: Path, signature: dict, duration: float) -> tuple[dict | None, str]:
    """Purpose / 功能: Accept only matching, complete converged cached curves / 仅接受匹配、完整且收敛的缓存曲线。

    Inputs / 输入: Case directory, full signature and planned duration / 案例目录、完整签名及计划时长。
    Outputs / 输出: Cached row or None, plus auditable reuse/recompute reason / 缓存行或 None，及可审计原因。
    """
    try:
        missing = [name for name in CURVE_FILES if not (case_output / name).is_file()]
        if missing:
            return None, "missing_artifacts: " + ",".join(missing)
        row = json.loads((case_output / "summary.json").read_text(encoding="utf-8"))
        if row.get("status") != "complete" or not row.get("noise", {}).get("converged"):
            return None, "incomplete_or_unconverged_summary"
        if not row["noise"].get("uncertainty", {}).get("precision_met", False):
            return None, "precision_not_met"
        if row.get("cache_signature") is not None and row["cache_signature"] != signature:
            return None, "cache_signature_mismatch"
        # 旧结果无签名时比较全部保存配置；计划的 num_steps=2 与实际输出采样分开核验。
        # Legacy summaries are accepted only with full saved controls and a separately verified actual grid.
        for key in ("model_version", "noise_model_version", "simulation_config"):
            if row.get(key) != signature[key]:
                return None, f"configuration_mismatch: {key}"
        config = signature["simulation_config"]
        n, count = config["chain_length"], signature["num_steps"]
        if (row.get("experiment") != "single-chain" or row.get("chain_length") != n
                or row.get("sources") != [0] or row.get("targets") != [n - 1]
                or row.get("sample_count") != count or row.get("excitation_count") != 1
                or row.get("g_max_rad_per_ns") != config["g_max"]
                or row.get("coupling_policy_requested") != config["coupling_policy"]):
            return None, "case_or_sampling_mismatch"
        if row["noise"].get("model_version") != NOISE_MODEL_VERSION:
            return None, "noise_version_mismatch"
        if any(row["noise"].get("parameters", {}).get(key) != value for key, value in config["noise"].items()):
            return None, "noise_controls_mismatch"
        if not math.isclose(row["final_time_ns"], duration, rel_tol=1e-12, abs_tol=1e-10):
            return None, "duration_mismatch"
        if not math.isfinite(row["max_analytic_error"]) or row["max_analytic_error"] > 1e-9:
            return None, "invalid_analytic_validation"
        for name in CURVE_FILES[1:]:
            data = (case_output / name).read_bytes()
            if not data:
                return None, f"empty_artifact: {name}"
            if "artifact_sha256" in row and row["artifact_sha256"].get(name) != hashlib.sha256(data).hexdigest():
                return None, f"artifact_checksum_mismatch: {name}"
            if name.endswith(".png") and (not data.startswith(b"\x89PNG\r\n\x1a\n") or b"IEND" not in data[-16:]):
                return None, "invalid_png"
            if name.endswith(".pdf") and (not data.startswith(b"%PDF-") or not data.rstrip().endswith(b"%%EOF")):
                return None, "invalid_pdf"
        with (case_output / "fidelity.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if tuple(next(reader)) != CURVE_COLUMNS:
                return None, "csv_columns_mismatch"
            values = np.asarray([[float(value) for value in record] for record in reader], dtype=float)
        if values.shape != (count, len(CURVE_COLUMNS)) or not np.all(np.isfinite(values)):
            return None, "csv_incomplete_or_nonfinite"
        if not np.allclose(values[:, 0], np.linspace(0, duration, count), rtol=1e-12, atol=1e-10):
            return None, "csv_sampling_mismatch"
        if (np.any(values[:, 5:7] < -1e-12) or np.any(values[:, 5:7] > 1 + 1e-9)
                or np.any(values[:, 7] < 0) or np.any(values[:, 8] > values[:, 6] + 1e-12)
                or np.any(values[:, 9] < values[:, 6] - 1e-12)
                or np.any(values[:, 8] < -1e-12) or np.any(values[:, 9] > 1 + 1e-9)):
            return None, "csv_invalid_probabilities"
        for column, key in ((5, "ideal_final_fidelity"), (6, "noisy_final_fidelity"),
                            (7, "noisy_final_standard_error"), (8, "noisy_final_ci_lower"), (9, "noisy_final_ci_upper")):
            if not math.isclose(values[-1, column], row[key], rel_tol=1e-12, abs_tol=1e-12):
                return None, f"csv_summary_mismatch: {key}"
        if not np.allclose(values[:, 5], np.abs(values[:, 1] + 1j * values[:, 2]) ** 2, atol=1e-10, rtol=1e-10):
            return None, "csv_amplitude_probability_mismatch"
        row["output_dir"] = str(case_output)
        return row, "matching_complete_cache" if "cache_signature" in row else "matching_complete_legacy_cache"
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, AttributeError) as exc:
        return None, f"invalid_cache: {type(exc).__name__}: {exc}"


def run_single_chain(output_root: Path = DEFAULT_OUTPUT_ROOT, lengths: tuple[int, ...] = DEFAULT_LENGTHS,
                     g_max: float | None = ce.DEFAULT_G_MAX_RAD_PER_NS, num_steps: int = 401,
                     max_dimension: int = 5000, max_intervals: int = 100, max_state_mb: float = 512,
                     protected: tuple[Path, ...] = (), noise: NoiseConfig | None = None, ideal: bool = False,
                     reference_time_ns: float = REFERENCE_TIME_NS, reference_fidelity: float = REFERENCE_FIDELITY,
                     coupling_policy: str = "preserve-request", existing: str = "overwrite",
                     output_name: str = "single_chain", figure_root: Path = DEFAULT_FIGURE_ROOT) -> dict:
    """Purpose / 功能: Sweep fixed-peak or uncapped requested-time NN chains / 扫描固定峰值或无上限请求时序的最近邻链。

    Inputs / 输入: Lengths, physical controls, resource caps and paths / 链长、物理参数、预算及路径。
    Outputs / 输出: Fixed per-length artifacts and current-request-only summary / 固定逐链长文件及仅本次请求的汇总。
    """
    if existing not in {"overwrite", "skip"}:
        raise ValueError("existing must be overwrite or skip")
    validate_budgets(num_steps, max_dimension, max_intervals, max_state_mb)
    noise = noise or NoiseConfig()
    if g_max is not None and (isinstance(g_max, bool) or not math.isfinite(g_max) or g_max <= 0):
        raise ValueError("single-chain requires a finite positive --g-max or no coupling cap")
    if not lengths or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in lengths):
        raise ValueError("single-chain lengths must be integers >=2")
    if len(set(lengths)) != len(lengths):
        raise ValueError("single-chain lengths must be unique")
    lengths = tuple(sorted(lengths))
    estimates = {n: estimate_resources(n, 1, 1, 1, num_steps, n, noise, ideal=ideal) for n in lengths}
    for n, resources in estimates.items():
        reason = resource_exclusion(resources, max_dimension, max_intervals, max_state_mb)
        if reason:
            raise ValueError(f"single-chain N={n}: {reason}")
    plans = {n: ce.build_single_chain_plan(n, g_max, noise=noise, ideal=ideal, coupling_policy=coupling_policy,
                                          reference_time_ns=reference_time_ns, reference_fidelity=reference_fidelity)
             for n in lengths}
    output = fixed_output(output_root, output_name, (DEFAULT_ROUTING_ROOT, DEFAULT_MANIFEST, *protected))
    figure_root = fixed_output(figure_root, ".", (output_root, ROOT / "evaluate/result", DEFAULT_MANIFEST))
    write_json(output / "summary.json", {"experiment": "single-chain", "status": "running",
                                          "requested_lengths": lengths, "cases": []})
    rows = []
    for n in lengths:
        case_output = output / f"N_{n:03d}"
        signature = single_chain_signature(plans[n], num_steps)
        row, reason = (load_single_chain_cache(case_output, signature, plans[n].transfer_time_ns)
                       if existing == "skip" else (None, "overwrite_requested"))
        action = "skipped" if row is not None else "computed"
        if row is None:
            # 先使旧完成标记失效；中断后不能把混合新旧工件当作有效缓存。
            # Invalidate completion first so an interrupted overwrite cannot look like a complete cache.
            case_output.mkdir(parents=True, exist_ok=True)
            write_json(case_output / "summary.json", {"status": "running", "cache_signature": signature,
                                                       "cache": {"action": action, "reason": reason}})
            row = single_chain_case(n, g_max, num_steps, noise, ideal, reference_time_ns, reference_fidelity,
                                    coupling_policy, estimates[n], case_output)
            row["cache_signature"] = signature
            row["cache"] = {"policy": existing, "action": action, "reason": reason}
            write_json(case_output / "summary.json", row)
        else:
            row["cache"] = {"policy": existing, "action": action, "reason": reason}
        rows.append(row)
    write_case_table(output / "case_summary.csv", rows,
                     ("chain_length", "final_time_ns", "ideal_final_fidelity", "noisy_final_fidelity",
                      "noisy_final_standard_error", "noisy_final_ci_lower", "noisy_final_ci_upper",
                      "max_analytic_error", "sample_count", "runtime_seconds", "status", "output_dir"))
    write_length_plot(figure_root, rows, g_max)
    report = {"experiment": "single-chain", "status": "complete" if all(r["status"] == "complete" for r in rows) else "incomplete",
              "output_dir": str(output), "metric": METRIC, "max_numerical_chain_length": max(lengths),
              "requested_settings": {"lengths": lengths, "g_max_rad_per_ns": g_max, "num_steps": num_steps,
                                     "max_dimension": max_dimension, "max_intervals": max_intervals,
                                     "max_state_mb": max_state_mb, "noise": asdict(noise), "ideal": ideal,
                                     "reference_time_ns": reference_time_ns, "reference_fidelity": reference_fidelity,
                                     "coupling_policy": coupling_policy, "existing": existing},
              "cache_actions": [{"chain_length": row["chain_length"], **row["cache"]} for row in rows],
              "artifact_scope": "only requested lengths; other retained N directories and numbered runs are ignored",
              "cases": rows, "generated_files": current_case_files(output, rows)
                  + ["case_summary.csv", "summary.json"],
              "figure_files": [str(figure_root / f"maximum_chain_length.{ext}") for ext in ("png", "pdf", "svg")]}
    write_json(output / "summary.json", report)
    return report


def source_package_path(metadata: dict, metadata_path: Path) -> Path:
    """Purpose / 功能: Resolve explicit saved layout provenance / 解析显式保存布局来源。

    Inputs / 输入: Saved metadata and its absolute path / 保存元数据及绝对路径。
    Outputs / 输出: Existing unambiguous package path; no guessed layout / 明确存在的封装路径，不猜测布局。
    """
    candidates = []
    for section in (metadata, metadata.get("signature", {}), metadata.get("compilation_signature", {})):
        value = section.get("layout_package_path")
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                options = {p.resolve() for p in (ROOT / path, metadata_path.parent / path) if p.is_file()}
                if len(options) != 1:
                    raise ValueError(f"missing or ambiguous relative layout_package_path: {value}")
                path = options.pop()
            candidates.append(path.resolve())
    if not candidates or len(set(candidates)) != 1 or not candidates[0].is_file():
        raise ValueError("metadata must identify one existing, consistent layout_package_path")
    return candidates[0]


def source_settings(package, circuit, metadata: dict) -> dict:
    """Purpose / 功能: Honor saved per-case timing before package defaults / 优先使用保存的逐案例计时配置。

    Inputs / 输入: Loaded package/circuit and metadata / 已加载封装、电路、元数据。
    Outputs / 输出: Provenance; only copied in-memory operation configs change / 配置来源；仅修改内存配置副本。
    """
    explicit = [("eqst_config", metadata.get("eqst_config")),
                ("signature.eqst_config", metadata.get("signature", {}).get("eqst_config"))]
    explicit = [(name, value) for name, value in explicit if value is not None]
    for name, value in explicit:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
    if len(explicit) == 2:
        shared = explicit[0][1].keys() & explicit[1][1].keys()
        if any(explicit[0][1][key] != explicit[1][1][key] for key in shared):
            raise ValueError("conflicting saved eqst_config and signature.eqst_config")
    original = package.op_config.get_eqst_config()
    settings = copy.deepcopy(package.op_config)
    merged = {}
    for _, value in reversed(explicit):
        merged.update(value)
    if merged.get("ordered", True) is not True:
        raise ValueError("saved compiler EQST settings must use ordered=True")
    if merged.get("model") not in (None, "legacy", "hamiltonian"):
        raise ValueError("unknown saved EQST model")
    if merged.get("model") == "hamiltonian" and merged.get("version", "hamiltonian-net-v1") not in {"hamiltonian-net-v1", "hamiltonian-net-v2", "hamiltonian-lindblad-v2", ce.MODEL_VERSION}:
        raise ValueError("unsupported saved Hamiltonian model version")
    for key in EQST_KEYS:
        if key in merged:
            setattr(settings, f"eqst_{key}", merged[key])
    for key in ("single_time", "intra_time", "inter_time", "qst_time_constant", "measure_time", "reset_time"):
        if key in merged:
            setattr(settings, key, merged[key])
    settings.eqst_model = "hamiltonian"
    effective = settings.get_eqst_config()
    package.op_config = settings
    circuit.op_config = copy.deepcopy(settings)
    return {"settings_source": "+".join(name for name, _ in explicit) if explicit else "package.op_config defaults",
            "saved_eqst_config": merged or None, "package_eqst_config": original,
            "effective_eqst_config": effective,
            "package_default_fields": [key for key in EQST_KEYS if key not in merged],
            "package_timing_overrides": {key: merged[key] for key in EQST_KEYS if key in merged and merged[key] != original[key]}}


def resident_batches(circuit, package) -> list[dict]:
    """Purpose / 功能: Extract full resident snapshots without planning rejected batches / 提取完整驻留快照，避免规划被排除批次。

    Inputs / 输入: Saved real Circuit and layout package / 保存的真实电路与布局封装。
    Outputs / 输出: Validated Riding/die maps including stationary residents / 含静止态的已校验阶段裸片映射。
    """
    from qunoc.element.chiplet import TransferChiplet
    from qunoc.routing.circuit import Transport

    batches = []
    node_dies = {}

    def die_at(node):
        """Purpose / 功能: Cache die lookup / 缓存裸片查询。

        Inputs / 输入: Physical node / 物理节点。 Outputs / 输出: Owning die / 所属裸片。
        """
        if node not in node_dies:
            node_dies[node] = package.get_node_chiplet(node)
        return node_dies[node]

    def group(mapping):
        """Purpose / 功能: Group all residents by die / 按裸片分组所有驻留态。

        Inputs / 输入: Logical map / 逻辑映射。 Outputs / 输出: Die-local positions / 裸片局部位置。
        """
        grouped = {}
        for logical, node in mapping.items():
            die = die_at(node)
            if isinstance(die, TransferChiplet):
                if not die.has_node(node):
                    raise ValueError("resident outside its transfer die")
                grouped.setdefault(die, {})[logical] = node[die.axis] - die.start
        return grouped

    # 复用核心最终历史重放；必须耗尽生成器，使终态一致性检查先于选择。 Exhaust replay before accepting any batch.
    for stage, before, after in circuit._replay_stage_mappings():
        starts, ends = group(before), group(after)
        actions = {}
        for action in circuit.stages[stage].circuit_actions:
            if isinstance(action, Transport):
                die = die_at(action.source_physical_qubit)
                if not isinstance(die, TransferChiplet) or die_at(action.target_physical_qubit) is not die:
                    raise ValueError("Transport must stay within one TransferChiplet")
                actions.setdefault(die, []).append(action.id)
        for die in sorted(starts.keys() | ends.keys() | actions.keys(), key=lambda d: (d.axis, d.pos, d.start, d.end)):
            initial, final = starts.get(die, {}), ends.get(die, {})
            if initial.keys() != final.keys():
                raise ValueError("Riding stage changes resident logical IDs")
            ids = sorted(initial)
            batches.append({"stage_index": stage, "die": [die.axis, die.pos, die.start, die.end],
                            "logical_ids": ids, "sources": [initial[q] for q in ids], "targets": [final[q] for q in ids],
                            "chain_length": die.end - die.start + 1, "transport_action_ids": actions.get(die, [])})
    return batches


def batch_config(batch: dict, settings: dict, num_steps: int, noise: NoiseConfig | None = None,
                 ideal: bool = False, coupling_policy: str = "preserve-request",
                 reference_fidelity: float = REFERENCE_FIDELITY) -> ce.SimulationConfig:
    """Purpose / 功能: Keep saved timing and apply current common noise / 保留保存时序并应用当前公共噪声。

    Inputs / 输入: Resident batch, effective EQST settings, sampling / 驻留批次、有效配置、采样。
    Outputs / 输出: Immutable config; saved noise is explicitly re-evaluated / 不可变配置；明确重评噪声。
    """
    noise = noise or NoiseConfig()
    return ce.SimulationConfig(chain_length=batch["chain_length"],
                               pairs=tuple(zip(batch["sources"], batch["targets"], strict=True)),
                               ordered=True, ideal=ideal, num_steps=num_steps, noise=noise or NoiseConfig(),
                               coupling_policy=coupling_policy, reference_fidelity=reference_fidelity,
                               **{key: settings[key] for key in EQST_KEYS})


def parse_programs(text: str) -> tuple[str, ...] | None:
    """Purpose / 功能: Parse an explicit compiler program filter / 解析显式编译程序筛选。

    Inputs / 输入: Comma-separated names or all / 逗号分隔名称或 all。
    Outputs / 输出: Unique names, or None for all programs / 唯一名称；全部程序返回 None。
    """
    names = tuple(name.strip().lower() for name in text.split(","))
    if names == ("all",):
        return None
    if not all(names) or "all" in names or len(set(names)) != len(names):
        raise ValueError("--programs requires unique comma-separated names, or all alone")
    return names


def stratify_compiler_cases(candidates: list[dict], cases_per_class: int,
                            min_excitations: int, max_excitations: int) -> dict:
    """Purpose / 功能: Select and account for each class × full-resident k / 按类别与完整驻留数分层选择统计。

    Inputs / 输入: Eligible batches and per-stratum quota/range / 合格批次、每层配额及范围。
    Outputs / 输出: Deterministic selection, stratum and aggregate shortages / 确定性选择、分层及汇总缺额。
    """
    candidates = sorted(candidates, key=lambda r: (
        -r["chain_length"], r["resources"]["estimated_total_bytes"], r["hilbert_dimension"],
        r["resources"]["interval_count"], r["planned_time_ns"], r["case_id"]))
    selected, strata = [], []
    for kind in CLASSES:
        for k in range(min_excitations, max_excitations + 1):
            eligible = [r for r in candidates if r["class"] == kind and r["resident_count"] == k]
            unique, duplicates, seen = [], [], set()
            for record in eligible:
                # 相同链长与完整映射跨基准优先只取一次；不足时允许重复，绝不虚报缺额。
                # Prefer distinct full mappings across benchmarks; duplicates may fill a genuine quota.
                identity = (record["chain_length"], tuple(sorted(zip(record["sources"], record["targets"], strict=True))))
                (duplicates if identity in seen else unique).append(record)
                seen.add(identity)
            chosen = (unique + duplicates)[:cases_per_class]
            selected.extend(chosen)
            strata.append({"stratum": f"{kind}/k={k}", "class": kind, "resident_count": k,
                           "requested": cases_per_class, "eligible": len(eligible), "selected": len(chosen),
                           "missing": not chosen, "shortfall": max(0, cases_per_class - len(chosen))})
    eligible_counts = {kind: sum(s["eligible"] for s in strata if s["class"] == kind) for kind in CLASSES}
    selected_counts = {kind: sum(s["selected"] for s in strata if s["class"] == kind) for kind in CLASSES}
    target_per_class = cases_per_class * (max_excitations - min_excitations + 1)
    return {"eligible_cases": candidates, "selected_cases": selected, "strata": strata,
            "eligible_stratum_counts": {s["stratum"]: s["eligible"] for s in strata},
            "selected_stratum_counts": {s["stratum"]: s["selected"] for s in strata},
            "missing_strata": [s["stratum"] for s in strata if s["missing"]],
            "stratum_shortfall": {s["stratum"]: s["shortfall"] for s in strata if s["shortfall"]},
            "eligible_class_counts": eligible_counts, "selected_class_counts": selected_counts,
            "missing_classes": [kind for kind in CLASSES if not selected_counts[kind]],
            "class_shortfall": {kind: target_per_class - selected_counts[kind] for kind in CLASSES
                                if selected_counts[kind] < target_per_class},
            "selection_order": "within each (class, full resident_count): prefer distinct (chain_length, full mapping), "
                               "then fill with duplicates; descending chain_length, ascending estimated_total_bytes, "
                               "hilbert_dimension, interval_count, planned_time_ns, case_id"}


def scan_compiler(routing_root: Path, cases_per_class: int = 3, min_excitations: int = 2,
                  max_excitations: int = 4, min_moving: int = 2, max_dimension: int = 5000,
                  max_intervals: int = 100, max_state_mb: float = 512, num_steps: int = 401,
                  noise: NoiseConfig | None = None, ideal: bool = False, coupling_policy: str = "preserve-request",
                  reference_fidelity: float = REFERENCE_FIDELITY,
                  programs: tuple[str, ...] | None = DEFAULT_PROGRAMS) -> dict:
    """Purpose / 功能: Census trusted saved circuits and deterministically select small batches / 盘点可信保存电路并确定选择小批次。

    Inputs / 输入: Read-only routing tree and selection/resource limits / 只读路由目录与筛选、资源上限。
    Outputs / 输出: Complete census, exclusions, errors, eligible and selected provenance / 统计、排除、错误及候选来源。
    """
    validate_budgets(num_steps, max_dimension, max_intervals, max_state_mb)
    for key, value in (("cases_per_class", cases_per_class), ("min_excitations", min_excitations),
                       ("max_excitations", max_excitations), ("min_moving", min_moving)):
        minimum = 1 if key == "cases_per_class" else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if min_excitations > max_excitations or min_moving > max_excitations:
        raise ValueError("excitation and moving-count limits are inconsistent")
    routing_root = routing_root.expanduser().resolve()
    if not routing_root.is_dir():
        raise ValueError(f"routing root does not exist: {routing_root}")
    paths = sorted(routing_root.rglob("exp_result.json"))
    census, exclusions, class_counts = Counter(), Counter(), Counter()
    program_counts, filtered_program_counts = Counter(), Counter()
    errors, candidates, sources, excluded_batches = [], [], [], []
    exclusion_examples = Counter()
    source_census = []
    for metadata_path in paths:
        relative = metadata_path.parent.relative_to(routing_root).as_posix()
        census["metadata_files"] += 1
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            program = metadata.get("quantum_program")
            program_counts[str(program)] += 1
            # 先按元数据筛选再加载 pickle；历史程序仅在显式允许时纳入。
            # Filter metadata before loading pickles; historical programs require explicit inclusion.
            if programs is not None and program not in programs:
                census["program_filtered_cases"] += 1
                filtered_program_counts[str(program)] += 1
                source_census.append({"source_case": relative, "quantum_program": program,
                                      "excluded_before_replay": "program_filter"})
                continue
            package_path = source_package_path(metadata, metadata_path)
            circuit_path = (metadata_path.parent / "qunoc_circuit.pkl").resolve()
            # pickle 可执行代码：仅接受用户信任的项目工件。 Pickle is executable; never accept untrusted datasets.
            with package_path.open("rb") as handle:
                package = pickle.load(handle)
            with circuit_path.open("rb") as handle:
                circuit = pickle.load(handle)
            provenance = source_settings(package, circuit, metadata)
            batches = resident_batches(circuit, package)
            source = {"source_case": relative, "quantum_program": metadata.get("quantum_program"),
                      "metadata_path": str(metadata_path.resolve()), "circuit_path": str(circuit_path),
                      "layout_package_path": str(package_path), "routing_options": metadata.get("routing_options"),
                      "source_signature": metadata.get("signature"), "compilation_signature": metadata.get("compilation_signature"),
                      **provenance}
            sources.append(source)
            census["loaded_cases"] += 1
        except Exception as exc:
            errors.append({"source_case": relative, "metadata_path": str(metadata_path.resolve()),
                           "phase": "source_replay", "error": f"{type(exc).__name__}: {exc}"})
            continue
        local_exclusions, local_classes, local_eligible = Counter(), Counter(), Counter()
        for batch in batches:
            census["batches"] += 1
            pairs = list(zip(batch["sources"], batch["targets"], strict=True))
            moving = [(s, t) for s, t in pairs if s != t]
            classification = ce.request_class(pairs)
            class_counts[classification] += 1
            local_classes[classification] += 1
            case_id = f"{relative}/stage_{batch['stage_index']}/die_" + "_".join(map(str, batch["die"]))
            record = {**batch, "case_id": case_id, "class": classification,
                      "resident_count": len(pairs), "moving_count": len(moving), "source_index": len(sources) - 1}
            reason = None
            if sorted(batch["sources"]) == sorted(batch["targets"]):
                reason = "equal_occupation_label_change" if moving else "stationary_batch"
            elif not min_excitations <= len(pairs) <= max_excitations:
                reason = "excitation_count"
            elif len(moving) < min_moving:
                reason = "moving_count"
            dimension = math.comb(batch["chain_length"], len(pairs))
            record["hilbert_dimension"] = dimension
            if reason is None and dimension > max_dimension:
                reason = "dimension_limit"
            if reason is None:
                try:
                    config = batch_config(batch, provenance["effective_eqst_config"], num_steps, noise, ideal,
                                          coupling_policy, reference_fidelity)
                    plan = ce.plan_transfer(config)  # 无基底/矩阵的预检。 Matrix-free first.
                    resources = estimate_resources(dimension, len(pairs), len(plan.interval_durations), len(plan.primitives),
                                                   num_steps, batch["chain_length"], noise, plan.route, ideal)
                    record.update({"resources": resources, "simulation_config": asdict(config), "route": plan.route,
                                   "planned_time_ns": plan.transfer_time_ns,
                                   "planned_distance_edges": plan.effective_distance_edges})
                    reason = resource_exclusion(resources, max_dimension, max_intervals, max_state_mb)
                except Exception as exc:
                    reason = "planning_error"
                    errors.append({"case_id": case_id, "phase": "planning", "error": f"{type(exc).__name__}: {exc}"})
            if reason:
                exclusions[reason] += 1
                local_exclusions[reason] += 1
                # 百万级驻留批次仅保存每种原因前20个标识，完整计数保留。 Bound examples, never census counts.
                if exclusion_examples[reason] < 20:
                    excluded_batches.append({"case_id": case_id, "class": classification, "reason": reason,
                                             "resident_count": len(pairs), "moving_count": len(moving), "hilbert_dimension": dimension})
                    exclusion_examples[reason] += 1
            else:
                candidates.append(record)
                local_eligible[classification] += 1
        source_census.append({"source_case": relative, "batch_count": len(batches), "class_counts": dict(local_classes),
                              "exclusion_counts": dict(local_exclusions), "eligible_class_counts": dict(local_eligible)})
    selection = stratify_compiler_cases(candidates, cases_per_class, min_excitations, max_excitations)
    return {"experiment": "compiler", "status": "scan_complete" if not selection["stratum_shortfall"] and not errors else "incomplete",
            "routing_root": str(routing_root), "metric": METRIC, "trusted_pickle_warning": "Only trusted project pickle artifacts may be scanned",
            "requested_settings": {"cases_per_class": cases_per_class, "min_excitations": min_excitations,
                                   "max_excitations": max_excitations, "min_moving": min_moving,
                                   "max_dimension": max_dimension, "max_intervals": max_intervals,
                                   "max_state_mb": max_state_mb, "num_steps": num_steps,
                                   "programs": list(programs) if programs is not None else "all",
                                   "cases_per_class_scope": "per (class, full resident_count), NOT per class total"},
            "census": dict(census), "class_counts": {kind: class_counts[kind] for kind in CLASSES},
            **selection, "exclusion_counts": dict(exclusions),
            "program_filter": {"allowed": list(programs) if programs is not None else "all",
                               "metadata_field": "quantum_program", "observed_metadata_counts": dict(program_counts),
                               "excluded_metadata_counts": dict(filtered_program_counts)},
            "errors": errors, "excluded_batches": excluded_batches,
            "exclusion_examples_limit_per_reason": 20, "source_census": source_census, "sources": sources,
            "evidence_limit": "same-plan timing/distance agreement is consistency, not independent physical proof"}


def reuse_compiler_manifest(path: Path, settings: dict) -> dict:
    """Purpose / 功能: Re-evaluate immutable selected batches, not a new census / 重评不可变选定批次而非重新普查。

    Inputs / 输入: Existing selection JSON and current controls / 既有选择 JSON 与当前参数。
    Outputs / 输出: Explicit provenance and replanned resource-checked subset / 明确来源及重规划预算检查子集。
    """
    path = path.expanduser().resolve()
    data = path.read_bytes()
    old = json.loads(data)
    if old.get("experiment") != "compiler" or not isinstance(old.get("selected_cases"), list):
        raise ValueError("compiler manifest must contain selected_cases from a compiler selection report")
    # 完整加载后再覆盖同路径报告；原始普查只存一份，不递归复制历次报告。
    # Snapshot before any write; keep one original census/selection, not recursively nested prior reports.
    snapshot_keys = ("census", "class_counts", "source_census", "sources", "eligible_cases", "selected_cases",
                     "eligible_class_counts", "eligible_stratum_counts", "program_filter", "requested_settings",
                     "excluded_batches", "exclusion_counts", "exclusion_examples_limit_per_reason", "errors", "routing_root")
    snapshot = copy.deepcopy(old.get("source_selection_snapshot") or
                             {key: old[key] for key in snapshot_keys if key in old})
    rows, excluded = [], []
    programs = settings.get("programs", DEFAULT_PROGRAMS)
    for original in snapshot["selected_cases"]:
        record = copy.deepcopy(original)
        source = snapshot["sources"][record["source_index"]]
        pairs = list(zip(record["sources"], record["targets"], strict=True))
        moving = [(s, t) for s, t in pairs if s != t]
        # 驻留数包含静止粒子；复用不能先按类别截断，必须过滤与重规划后再分层。
        # Count idle residents too; re-filter/replan every saved selection before stratifying.
        record.update({"resident_count": len(pairs), "moving_count": len(moving),
                       "class": ce.request_class(pairs)})
        reason = None
        if programs is not None and source.get("quantum_program") not in programs:
            reason = "program_filter"
        elif not settings["min_excitations"] <= len(pairs) <= settings["max_excitations"]:
            reason = "excitation_count"
        elif len(moving) < settings["min_moving"]:
            reason = "moving_count"
        elif sorted(record["sources"]) == sorted(record["targets"]):
            reason = "equal_occupation_label_change"
        dimension = math.comb(record["chain_length"], len(pairs))
        if reason is None and dimension > settings["max_dimension"]:
            reason = "dimension_limit"
        if reason is None:
            config = batch_config(record, source["effective_eqst_config"], settings["num_steps"],
                                  settings["noise"], settings["ideal"], settings["coupling_policy"],
                                  settings["reference_fidelity"])
            plan = ce.plan_transfer(config)
            resources = estimate_resources(dimension, len(config.pairs), len(plan.interval_durations),
                                           len(plan.primitives), config.num_steps, config.chain_length,
                                           config.noise, plan.route, config.ideal)
            reason = resource_exclusion(resources, settings["max_dimension"], settings["max_intervals"], settings["max_state_mb"])
            record.update({"resources": resources, "hilbert_dimension": dimension,
                           "simulation_config": asdict(config), "route": plan.route,
                           "planned_time_ns": plan.transfer_time_ns, "planned_distance_edges": plan.effective_distance_edges})
        if reason:
            excluded.append({"case_id": record["case_id"], "class": record["class"],
                             "resident_count": len(pairs), "reason": reason})
        else:
            rows.append(record)
    selection = stratify_compiler_cases(rows, settings["cases_per_class"], settings["min_excitations"], settings["max_excitations"])
    return {"experiment": "compiler", "status": "scan_complete" if not selection["stratum_shortfall"] else "incomplete",
            "selection_mode": "immutable saved selection reuse; NO new source scan or routing compilation",
            "manifest_path": str(path), "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "census": snapshot.get("census", old.get("historical_census_not_recomputed")),
            "historical_census_not_recomputed": snapshot.get("census", old.get("historical_census_not_recomputed")),
            "source_census": snapshot.get("source_census", []), "class_counts": snapshot.get("class_counts", {}),
            "routing_root": snapshot.get("routing_root"), "sources": snapshot["sources"],
            "source_selection_snapshot": snapshot,
            **selection, "current_eligible_cases": selection["eligible_cases"],
            "eligible_cases": snapshot.get("eligible_cases", snapshot["selected_cases"]),
            "eligible_cases_scope": "preserved source census candidates; current_eligible_cases are replanned saved selections",
            "eligible_counts_scope": "only saved selected_cases after current filters and resource checks, NOT a new census",
            "program_filter": {"allowed": list(programs) if programs is not None else "all",
                               "metadata_field": "sources[].quantum_program", "scope": "saved selected_cases only"},
            "excluded_batches": excluded, "exclusion_counts": dict(Counter(r["reason"] for r in excluded)),
            "errors": [], "metric": METRIC,
            "requested_settings": {**{k: asdict(v) if isinstance(v, NoiseConfig) else v for k, v in settings.items()},
                                   "programs": list(programs) if programs is not None else "all",
                                   "cases_per_class_scope": "per (class, full resident_count), NOT per class total"}}


def run_compiler(output_root: Path = DEFAULT_OUTPUT_ROOT, routing_root: Path = DEFAULT_ROUTING_ROOT,
                 scan_only: bool = False, compiler_manifest: Path | None = None, **settings) -> dict:
    """Purpose / 功能: Re-evaluate selected saved compiler batches with common noise / 公共噪声重评保存的编译批次。

    Inputs / 输入: Fresh output, trusted sources/manifest, physical/resource controls / 输出、可信源及物理预算参数。
    Outputs / 输出: Selection provenance and noisy/ideal evidence; sources unchanged / 来源与噪声理想证据；源不变。
    """
    for key, value in {"cases_per_class": 3, "min_excitations": 2, "max_excitations": 4,
                       "min_moving": 2, "max_dimension": 5000, "max_intervals": 100,
                       "max_state_mb": 512, "num_steps": 401, "programs": DEFAULT_PROGRAMS}.items():
        settings.setdefault(key, value)
    for key in ("cases_per_class", "min_excitations", "max_excitations", "min_moving"):
        minimum = 1 if key == "cases_per_class" else 2
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if settings["min_excitations"] > settings["max_excitations"] or settings["min_moving"] > settings["max_excitations"]:
        raise ValueError("excitation and moving-count limits are inconsistent")
    settings.setdefault("noise", NoiseConfig())
    settings.setdefault("ideal", False)
    settings.setdefault("coupling_policy", "preserve-request")
    settings.setdefault("reference_fidelity", REFERENCE_FIDELITY)
    validate_budgets(**{key: settings[key] for key in ("num_steps", "max_dimension", "max_intervals", "max_state_mb")})
    report = reuse_compiler_manifest(compiler_manifest, settings) if compiler_manifest else scan_compiler(routing_root, **settings)
    protected = (routing_root, DEFAULT_ROUTING_ROOT,
                 *((Path(report["routing_root"]),) if report.get("routing_root") else ()),
                 *(Path(source[key]).parent for source in report["sources"]
                   for key in ("layout_package_path", "metadata_path", "circuit_path") if source.get(key)))
    output = fixed_output(output_root, "compiler", protected)
    report.update({"output_dir": str(output), "scan_only": scan_only, "simulated_cases": [],
                   "noise_re_evaluation": {"model_version": ce.MODEL_VERSION,
                       "noise_model_version": NOISE_MODEL_VERSION,
                       "noise": asdict(settings["noise"]), "ideal": settings["ideal"],
                       "coupling_policy": settings["coupling_policy"],
                       "parameter_sources": "saved reference_time/distance/cap/planner; current CLI noise and coupling policy",
                       "source_artifacts_modified": False,
                       "topology": "open-nearest-neighbor-xy",
                       "classification": "site-disjoint closed intervals preserving all resident targets",
                       "qualification": "strict NN replanning and noise evaluation, NOT reproduction of historical direct-edge routes/statistics"}})
    scan_status = report["status"]
    if not scan_only:
        report["status"] = "running"
    write_json(output / "selection_report.json", report)
    if not scan_only:
        for record in report["selected_cases"]:
            identifier = hashlib.sha256(record["case_id"].encode()).hexdigest()[:16]
            case_output = output / "selected_cases" / f"{record['class']}_{identifier}"
            try:
                case_output.mkdir(parents=True, exist_ok=True)
                write_json(case_output / "summary.json", {"status": "running", "case_id": record["case_id"]})
                started = time.perf_counter()
                source = report["sources"][record["source_index"]]
                config = batch_config(record, source["effective_eqst_config"], settings["num_steps"], settings["noise"],
                                      settings["ideal"], settings["coupling_policy"], settings["reference_fidelity"])
                result, times, target, initial, fidelity = ce.run_simulation(config)
                time_match = math.isclose(result.final_time, record["planned_time_ns"], rel_tol=1e-12, abs_tol=1e-9)
                distance_match = result.route_details["total_transport_distance"] == record["planned_distance_edges"]
                samples_match = len(times) == record["resources"]["sample_count"]
                numeric_ok = bool(np.all(np.isfinite(fidelity)) and np.all(fidelity >= -1e-12)
                                  and np.all(fidelity <= 1 + 1e-9) and abs(float(fidelity[-1]) - 1) < 1e-9)
                summary = {**record, "metric": METRIC, "time_unit": "ns", "coupling_unit": "rad/ns",
                           "final_time_ns": result.final_time, "ideal_final_fidelity": result.final_fidelity,
                           "ideal_peak_fidelity": result.peak_fidelity, "sample_count": result.sample_count,
                           "route_details": result.route_details, "plan_time_matches": time_match,
                           "plan_distance_matches": distance_match, "sample_count_matches": samples_match,
                           "numeric_final_occupation_pass": numeric_ok, "citation": REFERENCE_CITATION,
                           "noise_re_evaluation": report["noise_re_evaluation"],
                           "runtime_seconds": time.perf_counter() - started, "output_dir": str(case_output),
                           **noise_summary(result.noisy_fidelities, result.noisy_standard_errors,
                                           result.noisy_ci_lower, result.noisy_ci_upper, result.noise)}
                if not (time_match and distance_match and samples_match and numeric_ok):
                    summary["status"] = "failed_validation"
                write_curve_outputs(case_output, summary, times, target, initial, fidelity,
                                    result.noisy_fidelities, result.noisy_standard_errors,
                                    result.noisy_ci_lower, result.noisy_ci_upper,
                                    f"Compiler batch · {record['class']}\nN = {record['chain_length']}, "
                                    f"residents = {record['resident_count']}, moving = {record['moving_count']}")
                write_json(case_output / "compiler_source.json", {**source, **record})
                report["simulated_cases"].append(summary)
                if summary["status"] != "complete":
                    report["errors"].append({"case_id": record["case_id"], "phase": "numerical_validation", "error": summary["status"]})
            except Exception as exc:
                report["errors"].append({"case_id": record["case_id"], "phase": "simulation", "error": f"{type(exc).__name__}: {exc}"})
        report["status"] = "complete" if scan_status == "scan_complete" and not report["errors"] else "incomplete"
    # 仅扫描不能擦除已有数值表；当前报告明确不认领旧工件。
    # A selection-only rerun leaves numerical tables intact, but never claims them as current results.
    if not scan_only:
        write_case_table(output / "case_summary.csv", report["simulated_cases"],
                         ("case_id", "class", "chain_length", "resident_count", "moving_count", "hilbert_dimension",
                          "final_time_ns", "ideal_final_fidelity", "noisy_final_fidelity", "noisy_final_standard_error",
                          "noisy_final_ci_lower", "noisy_final_ci_upper", "sample_count", "runtime_seconds", "status", "output_dir"))
    report["artifact_scope"] = "current invocation only; unlisted case directories/old tables are retained, not current evidence"
    report["generated_files"] = (["selection_report.json"] + ([] if scan_only else ["case_summary.csv"])
                                  + current_case_files(output, report["simulated_cases"], ("compiler_source.json",)))
    write_json(output / "selection_report.json", report)
    return report



def run_optimization(output_root: Path, num_steps: int = 101, noise: NoiseConfig | None = None,
                     max_dimension: int = 5000, max_intervals: int = 100, max_state_mb: float = 512,
                     reference_time_ns: float = REFERENCE_TIME_NS,
                     reference_fidelity: float = REFERENCE_FIDELITY) -> dict:
    """Purpose / 功能: Audit all nine N80 hardware projections / 审计 N80 九组硬件预测。

    Inputs / 输入: Baseline noise, reference and identical precision/budgets / 基准噪声、参考及统一精度预算。
    Outputs / 输出: All points, Pareto flags and precision-qualified best, never changes defaults / 全点与精度合格最优，不改默认。
    """
    validate_budgets(num_steps, max_dimension, max_intervals, max_state_mb)
    noise = noise or NoiseConfig()
    fit = calibration_reference(noise, reference_time_ns, reference_fidelity)
    # 标定仅在基准做一次；T1 扫描绝不隐式重新拟合退相干。 Freeze baseline gamma for every scenario.
    frozen = replace(noise, gamma_phi_ns=fit["gamma_phi_ns"])
    output = fixed_output(output_root, "optimize", (DEFAULT_ROUTING_ROOT,))
    write_json(output / "summary.json", {"experiment": "optimize", "status": "running", "cases": []})
    rows = []
    baseline = None
    for t1_us in (141.5, 200.0, 300.0):
        for cap_mhz in (6.0, 9.0, 12.0):
            scenario_noise = replace(frozen, t1_ns=t1_us * 1000)
            g_max = cap_mhz * 2 * math.pi / 1000
            resources = estimate_resources(80, 1, 1, 1, num_steps, 80, scenario_noise)
            reason = resource_exclusion(resources, max_dimension, max_intervals, max_state_mb)
            if reason:
                raise ValueError(f"optimization: {reason}")
            case_output = output / f"N_080_T1_{t1_us:g}us_cap_{cap_mhz:g}MHz"
            row = single_chain_case(80, g_max, num_steps, scenario_noise, False, reference_time_ns,
                                    reference_fidelity, "cap-limited-fast", resources, case_output)
            if baseline is None:
                baseline = row
            row.update({"t1_us": t1_us, "scenario": "baseline" if t1_us == 141.5 and cap_mhz == 6 else "PROJECTED hardware",
                        "frozen_gamma_phi_ns": frozen.gamma_phi_ns, "reference_fit": fit,
                        "historical_ideal_final": 1.0,
                        "historical_distance_envelope_final": reference_fidelity ** (79 / REFERENCE_DISTANCE),
                        "historical_envelope_note": "old phenomenological probability, NOT a Lindblad result or measured value",
                        "latency_ratio_to_baseline": row["final_time_ns"] / baseline["final_time_ns"],
                        "noisy_fidelity_change_from_baseline": row["noisy_final_fidelity"] - baseline["noisy_final_fidelity"],
                        "gap_to_historical_ideal": 1.0 - row["noisy_final_fidelity"]})
            rows.append(row)
    for row in rows:
        row["pareto_latency_fidelity"] = row["status"] == "complete" and not any(
            other["status"] == "complete" and other["final_time_ns"] <= row["final_time_ns"]
            and other["noisy_final_fidelity"] >= row["noisy_final_fidelity"]
            and (other["final_time_ns"] < row["final_time_ns"] or other["noisy_final_fidelity"] > row["noisy_final_fidelity"])
            for other in rows)
        write_json(Path(row["output_dir"]) / "summary.json", row)
    equal_precision = all(row["status"] == "complete" for row in rows)
    candidate = max(rows, key=lambda row: row["noisy_final_fidelity"])
    distinguishable = all(candidate is other or candidate["noisy_final_ci_lower"] > other["noisy_final_ci_upper"]
                          for other in rows)
    best = candidate["output_dir"] if equal_precision and distinguishable else None
    write_case_table(output / "all_points.csv", rows,
                     ("chain_length", "t1_us", "peak_coupling_mhz", "scenario", "frozen_gamma_phi_ns",
                      "final_time_ns", "ideal_final_fidelity", "noisy_final_fidelity", "noisy_final_standard_error",
                      "noisy_final_ci_lower", "noisy_final_ci_upper", "historical_ideal_final",
                      "historical_distance_envelope_final", "latency_ratio_to_baseline",
                      "noisy_fidelity_change_from_baseline", "gap_to_historical_ideal",
                      "pareto_latency_fidelity", "runtime_seconds", "status", "output_dir"))
    write_optimization_plot(fixed_output(DEFAULT_FIGURE_ROOT, ".", (output_root,)), rows)
    report = {"experiment": "optimize", "status": "complete" if equal_precision else "incomplete",
              "output_dir": str(output), "metric": METRIC, "reference_fit": fit, "cases": rows,
              "noise_controls": asdict(frozen), "precision_matched_and_converged": equal_precision,
              "best_candidate_after_precision_check": best,
              "best_selection_note": "maximum completion probability only after all points converge at common tolerances and CI separates; otherwise no selected best",
              "pareto_note": "point-estimate frontier minimizes latency and maximizes probability; not a cost-aware hardware optimum",
              "coupling_policy_note": "single-chain is already cap-limited; preserve-request and cap-limited-fast coincide, so no duplicate points",
              "limitations": "All nonbaseline points are PROJECTED hardware, not experimentally supported improvements. "
                             "No coupling-dependent added errors modeled. No protocol-round reduction implemented. "
                             "Baseline defaults are unchanged; no sufficiency threshold or process-fidelity claim.",
              "citation": REFERENCE_CITATION}
    write_json(output / "summary.json", report)
    return report


def output_contract_self_test(payload: dict, controls: dict) -> dict:
    """Purpose / 功能: Regress fixed paths, cache checks and same-path manifests / 回归固定路径、缓存及同路径清单。

    Inputs / 输入: Synthetic compiler selection and current controls / 人工编译选择及当前参数。
    Outputs / 输出: Named checks; only temporary files and small/mocked evolution / 检查名称；仅临时文件及小规模或模拟演化。
    """
    from unittest.mock import patch
    from types import SimpleNamespace

    checks = {}
    module = sys.modules[__name__]
    with tempfile.TemporaryDirectory(prefix="hamiltonian_output_contract_") as directory:
        root = Path(directory)
        with patch.object(module, "write_length_plot") as plot:
            first = run_single_chain(root, (4, 6), num_steps=5)
            overwrite = run_single_chain(root, (4,), num_steps=5)
            assert first["output_dir"] == overwrite["output_dir"] == str(root / "single_chain")
            with patch.object(module, "single_chain_case", side_effect=AssertionError("cache should skip evolution")):
                skipped = run_single_chain(root, (4,), num_steps=5, existing="skip")
            assert skipped["cases"][0]["cache"]["action"] == "skipped"
            assert [row["chain_length"] for row in plot.call_args.args[1]] == [4]
            assert (root / "single_chain/N_006/fidelity.csv").is_file()
            assert not any("N_006" in name for name in skipped["generated_files"])
            with (root / "single_chain/case_summary.csv").open() as handle:
                assert [r["chain_length"] for r in csv.DictReader(handle)] == ["4"]
            case = root / "single_chain/N_004"
            saved = json.loads((case / "summary.json").read_text())
            signature = saved["cache_signature"]
            for key, value in asdict(NoiseConfig()).items():
                changed = copy.deepcopy(signature)
                changed["simulation_config"]["noise"][key] = "different"
                assert load_single_chain_cache(case, changed, saved["final_time_ns"])[0] is None, key
            for key in ("ideal", "g_max", "reference_time_ns", "reference_fidelity", "coupling_policy"):
                changed = copy.deepcopy(signature)
                changed["simulation_config"][key] = "different"
                assert load_single_chain_cache(case, changed, saved["final_time_ns"])[0] is None, key
            for key in ("num_steps", "model_version", "noise_model_version", "cache_version"):
                changed = {**signature, key: "different"}
                assert load_single_chain_cache(case, changed, saved["final_time_ns"])[0] is None, key
            legacy = {key: value for key, value in saved.items() if key not in {"cache_signature", "artifact_sha256"}}
            write_json(case / "summary.json", legacy)
            assert load_single_chain_cache(case, signature, saved["final_time_ns"])[1] == "matching_complete_legacy_cache"
            (case / "fidelity.csv").write_text("partial\n")
            repaired = run_single_chain(root, (4,), num_steps=5, existing="skip")
            assert repaired["cases"][0]["cache"]["reason"] == "csv_columns_mismatch"
            (case / "fidelity.csv").unlink()
            repaired = run_single_chain(root, (4,), num_steps=5, existing="skip")
            assert "missing_artifacts" in repaired["cases"][0]["cache"]["reason"]
            changed = run_single_chain(root, (4,), num_steps=7, ideal=True, existing="skip")
            assert changed["cases"][0]["cache"]["reason"] == "cache_signature_mismatch"
        checks["fixed_single_chain_overwrite_skip_legacy_partial_and_requested_plot_rows"] = True
        checks["cache_all_noise_physics_sampling_and_version_controls"] = True

        compiler = root / "compiler"
        compiler.mkdir()
        manifest = compiler / "selection_report.json"
        payload = copy.deepcopy(payload)
        payload.update({"census": {"batches": 12345}, "source_census": [{"batch_count": 12345}],
                        "eligible_cases": copy.deepcopy(payload["selected_cases"])})
        payload["eligible_cases"].append({**payload["selected_cases"][0], "case_id": "extra_eligible"})
        write_json(manifest, payload)
        original = manifest.read_bytes()
        scan = run_compiler(root, root / "read_only_routing", scan_only=True, compiler_manifest=manifest, **controls)
        assert scan["manifest_sha256"] == hashlib.sha256(original).hexdigest()
        assert scan["census"] == payload["census"] and scan["eligible_cases"] == payload["eligible_cases"]
        narrower = {**controls, "min_excitations": 2, "max_excitations": 2, "cases_per_class": 1}

        def fake_evolution(config):
            """Purpose / 功能: Mock compiler numerics without expensive evolution / 模拟编译数值，避免昂贵演化。

            Inputs / 输入: Replanned config / 重规划配置。 Outputs / 输出: Consistent result tuple / 一致结果元组。
            """
            plan = ce.plan_transfer(config)
            count = 1 + len(plan.interval_durations) * (config.num_steps - 1)
            times = np.linspace(0, plan.transfer_time_ns, count)
            values, errors = np.ones(count), np.zeros(count)
            result = SimpleNamespace(final_time=plan.transfer_time_ns, final_fidelity=1.0, peak_fidelity=1.0,
                                     sample_count=count, route_details={"total_transport_distance": plan.effective_distance_edges},
                                     noisy_fidelities=values, noisy_standard_errors=errors, noisy_ci_lower=values,
                                     noisy_ci_upper=values, noise={"model_version": NOISE_MODEL_VERSION, "converged": True})
            return result, times, values.astype(complex), values.astype(complex), values

        with patch.object(ce, "run_simulation", side_effect=fake_evolution):
            numeric = run_compiler(root, root / "read_only_routing", compiler_manifest=manifest, **narrower)
            assert len(numeric["simulated_cases"]) == 2
            artifacts = {name: (compiler / name).read_bytes() for name in numeric["generated_files"]
                         if name != "selection_report.json"}
            scan = run_compiler(root, root / "read_only_routing", scan_only=True, compiler_manifest=manifest, **controls)
            assert len(scan["selected_cases"]) == 18 and not scan["simulated_cases"]
            assert scan["generated_files"] == ["selection_report.json"]
            assert all((compiler / name).read_bytes() == data for name, data in artifacts.items())
            assert scan["eligible_cases"] == payload["eligible_cases"] and scan["census"] == payload["census"]
            with patch.object(ce, "run_simulation", side_effect=RuntimeError("injected failure")):
                failed = run_compiler(root, root / "read_only_routing", compiler_manifest=manifest, **narrower)
            assert failed["status"] == "incomplete" and not failed["simulated_cases"]
            assert not any("selected_cases/" in name for name in failed["generated_files"])
            with (compiler / "case_summary.csv").open() as handle:
                assert not list(csv.DictReader(handle))
        checks["compiler_same_path_preserves_census_candidates_scan_only_numerics_and_current_rows"] = True

        source = root / "source"
        source.mkdir()
        (source / "sentinel").write_text("unchanged")
        linked = root / "linked"
        linked.mkdir()
        (linked / "single_chain").symlink_to(source, target_is_directory=True)
        nested = root / "nested"
        (nested / "single_chain").mkdir(parents=True)
        (nested / "single_chain/fidelity.csv").symlink_to(source / "sentinel")
        for destination, protected in ((source, (source,)), (linked, ()), (nested, ())):
            try:
                fixed_output(destination, "single_chain", protected)
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe output was accepted")
        assert (source / "sentinel").read_text() == "unchanged"
        assert fresh_output(root, "transfer").name == "transfer"
        assert fresh_output(root, "transfer").name == "transfer_run_002"
        with patch.object(module, "single_chain_case") as case_mock, patch.object(module, "write_optimization_plot"):
            case_mock.side_effect = lambda *args: {"output_dir": str(args[-1]), "status": "complete", "final_time_ns": 1.,
                                                   "noisy_final_fidelity": .9, "noisy_final_ci_lower": .9,
                                                   "noisy_final_ci_upper": .9}
            # 只测试九点编排；不运行 N80 求解。 Test nine-point orchestration without N80 solvers.
            for t1 in (141.5, 200., 300.):
                for cap in (6., 9., 12.):
                    (root / "optimize" / f"N_080_T1_{t1:g}us_cap_{cap:g}MHz").mkdir(parents=True)
            one = run_optimization(root, num_steps=3)
            two = run_optimization(root, num_steps=3)
            assert one["output_dir"] == two["output_dir"] == str(root / "optimize")
            assert case_mock.call_count == 18
        checks["fixed_optimization_numbered_transfer_and_read_only_source_guards"] = True
    return checks


def runner_self_test() -> dict:
    """Purpose / 功能: Validate core physics and runner orchestration in this one script / 单脚本验证物理与编排。

    Inputs / 输入: None / 无。 Outputs / 输出: Checks or AssertionError / 检查结果或断言错误。
    """
    from qunoc.hamiltonian.noise import noise_self_test
    core = noise_self_test()
    checks = {}
    # 分类必须基于区间及全部居民，而非端点唯一。 Classify intervals and all residents.
    compatible = ce.plan_transfer(SimulationConfig(13, ((10, 0), (11, 12)), ordered=True))
    assert compatible.segment_stages == (((0, 10), (11, 12)),)
    assert compatible.planner == "parallel-disjoint-pst" and compatible.route == "pst-segments"
    assert len(compatible.interval_durations) == 2
    assert ce.request_class(((0, 4), (2, 6))) == "conflicting"
    assert ce.request_class(((0, 6), (1, 1))) == "conflicting"
    spectator = ce.plan_transfer(SimulationConfig(7, ((0, 6), (1, 1))))
    assert spectator.labeled_path[-1] == (6, 1)
    manual = ce.plan_transfer(SimulationConfig(5, ((0, 4),), segments=(((0, 2),), ((2, 4),))))
    assert manual.planner == "manual" and manual.labeled_path[-1] == (4,)
    checks["strict_nn_parallel_conflict_spectator_and_manual_dispatch"] = True
    noise = NoiseConfig()
    plan = ce.build_single_chain_plan(6, noise=noise)
    assert len(plan.config.pairs) == 1 and plan.primitives[0].kind == "pst-segment"
    assert math.isclose(plan.transfer_time_ns, 125.0, abs_tol=1e-10)
    curves = evaluate_plan_noise(plan, [0.0, plan.transfer_time_ns], noise)
    assert math.isclose(curves["fidelities"][-1], .992, abs_tol=1e-8)
    checks["single_chain_true_nn_reference"] = True
    zero = ce.build_single_chain_plan(6, noise=noise, ideal=True)
    assert math.isclose(evaluate_plan_noise(zero, [125.0], noise)["fidelities"][-1], 1.0, abs_tol=1e-9)
    checks["ideal_disables_only_noise"] = zero.transfer_time_ns == plan.transfer_time_ns
    settings = {"reference_time_ns": 200., "reference_distance": 5., "g_max": DEFAULT_G_MAX_RAD_PER_NS,
                "max_segment_steps": 10, "beam_width": 128}
    batch = {"chain_length": 8, "sources": [0, 3], "targets": [2, 5]}
    config = batch_config(batch, settings, 11, noise)
    assert not config.ideal and config.noise.t1_ns == noise.t1_ns and config.reference_time_ns == 200.
    checks["compiler_inherits_timing_uses_current_noise"] = True
    assert config.noise == noise and config.noise.calibration_time_ns == REFERENCE_TIME_NS
    checks["compiler_saved_timing_never_refits_common_noise"] = True
    estimated = estimate_resources(80, 1, 1, 1, 21, 80, noise)
    assert estimated["noise_workspace_bytes"] >= 80 ** 2 * 16
    assert resource_exclusion(estimated, 5000, 100, .01) == "memory_limit"
    checks["density_resource_guard"] = True
    fit = calibration_reference(noise)
    changed = calibration_reference(replace(noise, t1_ns=300000))
    assert fit["gamma_phi_ns"] == changed["gamma_phi_ns"]
    checks["t1_projection_does_not_refit_gamma"] = True
    # 两种编译策略必须实际改变调度，而非只改报告时长。 Policies change the real plan, not labels.
    slow = ce.plan_transfer(config)
    fast = ce.plan_transfer(replace(config, coupling_policy="cap-limited-fast"))
    assert fast.transfer_time_ns < slow.transfer_time_ns
    checks["compiler_coupling_policy_changes_actual_plan"] = True
    # 人工批次只用于回归编排逻辑，绝不混入真实编译来源。 Synthetic batches test orchestration only.
    synthetic = []
    for kind in CLASSES:
        for k in range(2, 5):
            for variant in range(4):
                pairs = ([(0, 1), (3, 4)] if kind == "nonconflicting" else [(0, 2), (2, 3)])
                pairs += [(site, site) for site in range(6, 6 + k - 2)]
                record = {"case_id": f"synthetic/{kind}/k{k}/{variant}", "class": kind,
                          "chain_length": 8 + variant, "sources": [s for s, _ in pairs],
                          "targets": [t for _, t in pairs], "resident_count": k, "moving_count": 2,
                          "source_index": 0, "hilbert_dimension": math.comb(8 + variant, k),
                          "resources": {"estimated_total_bytes": 1, "interval_count": 1}, "planned_time_ns": 1.0}
                synthetic.append(record)
    selection = stratify_compiler_cases(synthetic, 3, 2, 4)
    assert len(selection["selected_cases"]) == 18 and len(selection["strata"]) == 6
    assert set(selection["eligible_stratum_counts"].values()) == {4}
    assert set(selection["selected_stratum_counts"].values()) == {3}
    assert selection["selected_class_counts"] == dict.fromkeys(CLASSES, 9)
    assert not selection["stratum_shortfall"] and not selection["missing_strata"]
    assert selection == stratify_compiler_cases(list(reversed(synthetic)), 3, 2, 4)
    duplicate = {**selection["selected_cases"][0], "case_id": "aaa_duplicate"}
    diverse = stratify_compiler_cases(synthetic + [duplicate], 3, 2, 4)
    assert len({r["chain_length"] for r in diverse["selected_cases"][:3]}) == 3
    checks["compiler_six_strata_deterministic_diverse_selection"] = True
    from tempfile import TemporaryDirectory
    controls = {"cases_per_class": 3, "min_excitations": 2, "max_excitations": 4, "min_moving": 2,
                "max_dimension": 5000, "max_intervals": 100, "max_state_mb": 512, "num_steps": 11,
                "noise": noise, "ideal": False, "coupling_policy": "preserve-request",
                "reference_fidelity": REFERENCE_FIDELITY, "programs": DEFAULT_PROGRAMS}
    with TemporaryDirectory(prefix="hamiltonian_runner_self_test_") as directory:
        manifest = Path(directory) / "synthetic_selection.json"
        payload = {"experiment": "compiler", "sources": [{"quantum_program": "vqe", "effective_eqst_config": settings}],
                   "selected_cases": selection["selected_cases"]}
        write_json(manifest, payload)
        original_bytes = manifest.read_bytes()
        reused = reuse_compiler_manifest(manifest, controls)
        assert reused["status"] == "scan_complete" and len(reused["selected_cases"]) == 18
        assert reused["selected_stratum_counts"] == selection["selected_stratum_counts"]
        assert all(r["resident_count"] == len(r["sources"]) and r["moving_count"] == 2 for r in reused["selected_cases"])
        assert all(r["resources"]["sample_count"] == 1 + r["resources"]["interval_count"] * 10 for r in reused["selected_cases"])
        assert manifest.read_bytes() == original_bytes
        # 旧类别不能直接复用：端点不重合的重叠区间仍冲突。 Reclassify stale endpoint-only labels.
        stale = copy.deepcopy(payload)
        stale["selected_cases"] = [{**stale["selected_cases"][0], "class": "nonconflicting",
                                    "sources": [0, 2], "targets": [4, 6], "chain_length": 7}]
        stale_path = Path(directory) / "stale_intervals.json"
        write_json(stale_path, stale)
        corrected = reuse_compiler_manifest(stale_path, controls)
        assert corrected["selected_cases"][0]["class"] == "conflicting"
        assert corrected["selected_cases"][0]["route"] == "pst-segments"
        narrowed = reuse_compiler_manifest(manifest, {**controls, "min_excitations": 3})
        assert len(narrowed["selected_cases"]) == 12 and len(narrowed["strata"]) == 4
        budget = reuse_compiler_manifest(manifest, {**controls, "max_dimension": 1})
        assert budget["status"] == "incomplete" and len(budget["missing_strata"]) == 6
        assert budget["exclusion_counts"] == {"dimension_limit": 18}
        filtered = reuse_compiler_manifest(manifest, {**controls, "programs": ("qaoa",)})
        assert filtered["exclusion_counts"] == {"program_filter": 18} and not filtered["selected_cases"]
        payload["selected_cases"] = [r for r in payload["selected_cases"]
                                     if not (r["class"] == "conflicting" and r["resident_count"] == 4)]
        write_json(manifest, payload)
        shortage = reuse_compiler_manifest(manifest, controls)
        assert shortage["status"] == "incomplete" and shortage["missing_strata"] == ["conflicting/k=4"]
        assert shortage["stratum_shortfall"] == {"conflicting/k=4": 3}
        assert shortage["class_shortfall"] == {"conflicting": 3} and not shortage["missing_classes"]
    checks["compiler_manifest_six_strata_filters_and_resource_rechecks"] = True
    assert parse_programs("vqe,qaoa,bv,s_ghz") == DEFAULT_PROGRAMS and parse_programs("all") is None
    defaults = build_arg_parser().parse_args([])
    assert defaults.experiment is None and defaults.num_steps is None
    assert defaults.lengths == tuple(range(6, 81, 2))
    checks["unified_default_both_and_38_even_lengths"] = True
    checks.update(output_contract_self_test({"experiment": "compiler",
                                             "sources": [{"quantum_program": "vqe", "effective_eqst_config": settings}],
                                             "selected_cases": selection["selected_cases"]}, controls))
    return {"status": "complete", "core_physics": core, "runner_checks": checks}


def target_position_scan(sources, targets, chain_length):
    """Fix each target prefix by mirrors / 用镜像逐位固定目标前缀。

    Every mirror updates ALL resident labels, including nominally stationary ones.
    每次镜像更新全部居民标签，包括原本静止的居民。
    """
    if (len(sources) != len(targets) or len(set(sources)) != len(sources)
            or len(set(targets)) != len(targets)
            or any(not 0 <= x < chain_length for x in (*sources, *targets))):
        raise ValueError("invalid resident mapping")
    current, stages = tuple(sources), []
    desired = {position: label for label, position in enumerate(targets)}
    for p in range(chain_length):
        if p in desired:
            q = current[desired[p]]
        elif p in current:
            q = next(q for q in range(p + 1, chain_length) if q not in current)
        else:
            q = p
        if q != p:
            assert q > p, "fixed prefix must never be disturbed"
            stage = ((p, q),)
            current = ce.apply_segment_stage_labeled(current, stage)
            stages.append(stage)
        assert all((current.index(x) if x in current else None) == desired.get(x)
                   for x in range(p + 1))
    assert current == tuple(targets)
    return tuple(stages)


def select_cases(manifest):
    """Select before simulation using only structure / 仅根据结构预选，禁止按结果筛选。"""
    snapshot = manifest.get("source_selection_snapshot", manifest)
    candidates = snapshot["eligible_cases"] if "eligible_cases" in snapshot else snapshot["selected_cases"]
    chosen = []
    for cls in CLASSES:
        for k in (2, 3, 4):
            pool = [r for r in candidates if len(r["sources"]) == k
                    and ce.request_class(tuple(zip(r["sources"], r["targets"]))) == cls
                    and sorted(r["sources"]) != sorted(r["targets"])
                    and math.comb(r["chain_length"], k) <= 4000]
            # Explicit requested exemplar first; otherwise smallest N, stable source ID.
            # 优先用户指定实例，否则最小链长及稳定来源编号。
            pool.sort(key=lambda r: (not (cls == "nonconflicting" and k == 2
                                         and r["sources"] == [10, 11] and r["targets"] == [0, 12]),
                                     r["chain_length"], r["case_id"]))
            if not pool:
                raise ValueError(f"no bounded saved case for {cls}, k={k}")
            row = copy.deepcopy(pool[0])
            row["historical_class"] = row["class"]
            row["class"] = cls
            row["source"] = snapshot["sources"][row["source_index"]]
            chosen.append(row)
    return chosen


def case_config(row, num_steps):
    """Keep core physics; explicitly opt into manual scan / 保留核心物理，显式使用手动扫描。"""
    noise = NoiseConfig(calibration_time_ns=125., calibration_fidelity=.992,
                        atol=.001, rtol=.005, trajectories=32768, min_trajectories=256, seed=0)
    settings = {**row["source"]["effective_eqst_config"],
                "reference_time_ns": 62.5, "reference_distance": 5., "g_max": None}
    config = batch_config(row, settings, num_steps, noise)
    if row["class"] == "conflicting":
        config = replace(config, segments=target_position_scan(config.sources, config.targets,
                                                               config.chain_length))
    return config


def run_real_case_study(output_root: Path = DEFAULT_OUTPUT_ROOT, manifest: Path = DEFAULT_MANIFEST,
                        num_steps: int = 101, select_only: bool = False,
                        figure_root: Path = DEFAULT_FIGURE_ROOT) -> dict:
    """Replay six structure-selected saved cases / 重放六个按结构预选的保存案例。"""
    from types import SimpleNamespace
    args = SimpleNamespace(output=output_root / "real_case_study", manifest=manifest,
                           num_steps=num_steps, select_only=select_only)
    if args.num_steps < 2:
        raise ValueError("num-steps must be >=2")
    raw = args.manifest.read_bytes()
    rows = select_cases(json.loads(raw))
    args.output = fixed_output(args.output.parent, args.output.name, (args.manifest, DEFAULT_ROUTING_ROOT))
    figure_root = fixed_output(figure_root, ".", (output_root, args.manifest, ROOT / "evaluate/result"))
    for index, row in enumerate(rows):
        config = case_config(row, args.num_steps)
        plan = ce.plan_transfer(config)
        # Compare same schedule/timing separately from global planner changes.
        # 分开比较相同调度的时序变化与全局规划器差异。
        default_timing = ce.plan_transfer(replace(config, reference_time_ns=ce.REFERENCE_TIME_NS,
                                                g_max=ce.DEFAULT_G_MAX_RAD_PER_NS))
        default_core = ce.plan_transfer(replace(config, segments=None,
                                              reference_time_ns=ce.REFERENCE_TIME_NS,
                                              g_max=ce.DEFAULT_G_MAX_RAD_PER_NS))
        row.update(directory=f"case_{index+1}", status="selected", simulation_config=asdict(config),
                   experiment_planner="target-position-scan" if config.segments is not None else "parallel-disjoint-pst",
                   plan=plan.to_route_details(), requested_time_ns=plan.transfer_time_ns,
                   same_schedule_default_timing_ns=default_timing.transfer_time_ns,
                   unchanged_core_default_time_ns=default_core.transfer_time_ns,
                   unchanged_core_default_plan=default_core.to_route_details())
    report = {"manifest_path": str(args.manifest.resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "selection_rule": "one per current class and k=2,3,4; known 10:0,11:12 preferred; otherwise minimum N then case_id; C(N,k)<=4000; no outcome selection",
              "planner_note": "Experiment-level target-position scan uses core manual segments; global greedy/beam/fallback planner is unchanged.",
              "calibration": calibration_reference(case_config(rows[0], args.num_steps).noise),
              "metric": METRIC, "selected_cases": rows,
              "figure_files": [str(figure_root / f"real_case_study.{ext}") for ext in ("png", "pdf", "svg")]}
    report.update(status="scan_complete" if args.select_only else "selected", output_dir=str(args.output))
    write_json(args.output / "selection.json", report)
    if args.select_only:
        return report
    report.update(status="running", output_dir=str(args.output))
    write_json(args.output / "summary.json", report)
    for row in rows:
        directory = args.output / row["directory"]
        directory.mkdir(exist_ok=True)
        (directory / "fidelity.csv").unlink(missing_ok=True)
        write_json(directory / "summary.json", {"status": "running"})
        try:
            result, times, target, source, ideal = ce.run_simulation(case_config(row, args.num_steps))
            summary = result_to_jsonable(result)
            summary.update(noise_summary(result.noisy_fidelities, result.noisy_standard_errors,
                                               result.noisy_ci_lower, result.noisy_ci_upper, result.noise))
            with (directory / "fidelity.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(CURVE_COLUMNS)
                writer.writerows(zip(times, target.real, target.imag, source.real, source.imag, ideal,
                                     result.noisy_fidelities, result.noisy_standard_errors,
                                     result.noisy_ci_lower, result.noisy_ci_upper))
            row.update(status=summary["status"], noisy_final=summary["noisy_final_fidelity"],
                       ideal_final=result.final_fidelity, solver=result.noise["solver"],
                       uncertainty=result.noise["uncertainty"],
                       trajectories=result.noise["uncertainty"]["trajectories"],
                       ci_lower=summary["noisy_final_ci_lower"], ci_upper=summary["noisy_final_ci_upper"])
        except Exception as error:
            summary = {"status": "failed", "error": repr(error)}
            row.update(summary)
        summary["source_case"] = row
        write_json(directory / "summary.json", summary)
        print(row["case_id"], row["status"], row.get("noisy_final"), flush=True)
        write_json(args.output / "summary.json", report)
    report["status"] = "complete" if all(r["status"] == "complete" for r in rows) else "incomplete"
    write_json(args.output / "summary.json", report)
    fields = ("case_id", "class", "chain_length", "sources", "targets", "experiment_planner",
              "requested_time_ns", "same_schedule_default_timing_ns", "unchanged_core_default_time_ns",
              "status", "noisy_final", "ideal_final", "solver", "trajectories", "ci_lower", "ci_upper")
    write_case_table(args.output / "case_summary.csv", rows, fields)
    write_real_case_plot(args.output, rows, figure_root)
    report["generated_files"] = ["selection.json", "summary.json", "case_summary.csv"] + [
        str(path.relative_to(args.output)) for row in rows
        for name in CURVE_FILES if (path := args.output / row["directory"] / name).is_file()]
    write_json(args.output / "summary.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Two paper experiments, both by default / 两项论文实验，默认同时运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("maximum-chain-length", "real-case-study", "single-chain"),
                        help="omit to run both; single-chain is a compatibility alias")
    parser.add_argument("--output-root", "--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="read-only saved selection manifest")
    parser.add_argument("--lengths", type=parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument("--num-steps", type=int, help="override sampling: chain defaults 401, real cases 101")
    parser.add_argument("--select-only", action="store_true", help="real-case-study only: save selection without evolution")
    parser.add_argument("--self-test", action="store_true")
    # Accepted legacy protocol spellings cannot change the paper protocol / 旧参数拼写不能修改论文协议。
    parser.add_argument("--no-coupling-cap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reference-time-ns", type=float, default=62.5, help=argparse.SUPPRESS)
    parser.add_argument("--reference-distance", type=float, default=5., help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute selected experiments with frozen physical controls / 使用冻结物理参数运行所选实验。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(runner_self_test(), indent=2, allow_nan=False))
        return 0
    if args.reference_time_ns != 62.5 or args.reference_distance != 5.:
        parser.error("paper timing is fixed at 62.5 ns / 5 edges; noise calibration remains 125 ns")
    if args.num_steps is not None and args.num_steps < 2:
        parser.error("num-steps must be >=2")
    if args.select_only and args.experiment != "real-case-study":
        parser.error("--select-only requires --experiment real-case-study")
    reports = {}
    try:
        if args.experiment in (None, "maximum-chain-length", "single-chain"):
            reports["maximum_chain_length"] = run_single_chain(
                args.output_root, args.lengths, None, num_steps=args.num_steps or 401,
                reference_time_ns=62.5, noise=NoiseConfig(calibration_time_ns=125.),
                output_name="maximum_chain_length", figure_root=args.figure_root)
        if args.experiment in (None, "real-case-study"):
            reports["real_case_study"] = run_real_case_study(
                args.output_root, args.manifest, args.num_steps or 101, args.select_only, args.figure_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({name: {key: report[key] for key in ("status", "output_dir")}
                      for name, report in reports.items()}, indent=2))
    return int(any(report["status"] not in ("complete", "scan_complete") for report in reports.values()))


if __name__ == "__main__":
    raise SystemExit(main())
