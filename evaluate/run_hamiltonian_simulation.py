#!/usr/bin/env python3
"""Standalone XY/PST diagnostics / 独立 XY/PST 命令行、绘图和文件输出。

The numerical and matrix-free planning implementation lives in qunoc.hamiltonian.
数值演化及无矩阵规划由 qunoc.hamiltonian 提供，本脚本只负责评估和导出。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

# 支持仓库根目录与其他目录直接执行。 Support direct script invocation from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qunoc.hamiltonian.chain_evolution import (
    DEFAULT_G_MAX_RAD_PER_NS, REFERENCE_DISTANCE, REFERENCE_FIDELITY,
    REFERENCE_SITES, REFERENCE_TIME_NS, Pair, Stages, SimulationConfig, SimulationResult,
    calibrated_target_fidelities, distance_exposure, distance_survival,
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
    """功能 / Purpose: Prepare serializable report / 准备可序列化报告。

    输入 / Inputs: SimulationResult / 模拟结果。
    输出 / Outputs: Dictionary with explicit occupancy-only metric / 带占据指标说明的字典。
    """
    payload = asdict(result)
    payload["sources"] = sorted(result.config.sources)
    payload["targets"] = sorted(result.config.targets)
    payload["metric"] = "target occupation probability; not labeled or process fidelity"
    payload["fidelity_fields"] = {
        "peak_time, peak_fidelity, final_fidelity": "ideal coherent occupation probability",
        "calibrated_peak_time, calibrated_peak_fidelity, calibrated_final_fidelity":
            "ideal probability multiplied by the phenomenological distance-error envelope",
        "peaks": "separate sampled maxima, not continuous-time optimized peaks",
        "amplitudes": "normalized ideal amplitudes (conditional successful-transport branch by convention); "
                      "squared modulus equals ideal, NOT calibrated fidelity",
    }
    payload["time_unit"] = "ns"
    payload["coupling_unit"] = "rad/ns (hbar = 1); J/(2*pi) in GHz"
    payload["calibration"] = {
        "reference_source": "user-provided N=6, single excitation, 125 ns, F=0.992 anchor; not a literal paper quote",
        "reference_sites": REFERENCE_SITES,
        "reference_excitation_count": 1,
        "reference_time_ns": result.config.reference_time_ns,
        "reference_distance_edges": result.config.reference_distance,
        "reference_fidelity": result.config.reference_fidelity,
        "requested_time_formula": "tau(d)=reference_time_ns*sqrt(d/reference_distance); d=abs(right-left)",
        "coupling_assumption": "direct J=pi/(2*tau); PST J_j=g*sqrt(j*(L-j)), g=pi/(2*tau); "
                               "edge cap takes priority over requested sqrt-distance timing and slows each primitive separately",
        "derived_default_edge_cap_rad_per_ns": DEFAULT_G_MAX_RAD_PER_NS,
        "default_cap_source": "six-site reference profile peak 3*g_ref, equivalent to J/(2*pi)=6 MHz; "
                              "modeling bound, not asserted hardware maximum",
        "distance_cost": "direct: abs(target-source); PST: sum(abs(left+right-2*p)) over occupations p "
                         "inside that segment at the logical stage start",
        "distance_exposure": "each primitive's cost grows linearly over its actual duration and stops at completion; "
                             "no additional idle-time penalty",
        "attenuation_per_edge": result.config.attenuation_per_edge,
        "attenuation_disabled": result.config.attenuation_per_edge == 0,
        "calibrated_fidelity_formula": "exp(-alpha*L(t))*F_ideal(t)",
        "interpretation": "phenomenological distance-error design envelope, not uniquely inferred from one datum, "
                          "not local amplitude damping or a Lindblad solution, not measured device T1, "
                          "not an experimental reproduction or process-fidelity guarantee",
    }
    payload["sampling"] = "num_steps points per constant-Hamiltonian interval; shutoff/stage boundaries appear once"
    return payload


def write_fidelity_plot(output_path: Path, result: SimulationResult, times: np.ndarray, fidelities: np.ndarray) -> None:
    """功能 / Purpose: Compare ideal and distance-calibrated curves / 比较理想与距离校准曲线。

    输入 / Inputs: PNG path, result, ns times and ideal probabilities / 图片路径、摘要、纳秒时间及理想概率。
    输出 / Outputs: None; writes fixed-light PNG with legend and closes figure / 无返回；写带图例的浅色 PNG 并关闭。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    loss_color, ideal_color, grid, axis = "#2a78d6", "#eb6834", "#e1e0d9", "#c3c2b7"
    exposure = distance_exposure(times, result.route_details["primitives"])
    calibrated = calibrated_target_fidelities(result.config, exposure, fidelities)
    with plt.rc_context({"text.color": ink, "axes.labelcolor": secondary,
                         "xtick.color": secondary, "ytick.color": secondary,
                         "font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True, facecolor=surface)
        ax.set_facecolor(surface)
        ax.plot(times, calibrated, color=loss_color, linewidth=2.0,
                solid_capstyle="round", label="Distance-calibrated model")
        ax.plot(times, fidelities, color=ideal_color, linewidth=2.0, linestyle="--",
                dash_capstyle="round", label="Ideal coherent evolution")
        for value, color, marker, label, offset in (
                (result.final_fidelity, ideal_color, "s", "Ideal", 22),
                (result.calibrated_final_fidelity, loss_color, "o", "Calibrated", -30)):
            ax.plot([result.final_time], [value], marker=marker, markersize=8,
                    color=color, markeredgecolor=surface, markeredgewidth=2)
            ax.annotate(f"{label}: {value:.6f}", xy=(result.final_time, value),
                        xytext=(-14 if label == "Ideal" else -140, offset),
                        textcoords="offset points", ha="right", va="center",
                        color=ink, arrowprops={"arrowstyle": "-", "color": secondary, "lw": 0.7},
                        bbox={"facecolor": surface, "edgecolor": "none", "pad": 1.5})
        for boundary in result.route_details["stage_end_times"][:-1]:
            ax.axvline(boundary, color=axis, linewidth=0.8)
        ax.legend(loc="upper left", frameon=False)
        ax.text(0.03, 0.78, f"Completion: {result.final_time:.3f} ns\n"
                f"Stages: {result.route_details['stage_count']} | Excitations: {len(result.config.pairs)}",
                transform=ax.transAxes, va="top", color=secondary)
        sources, targets = sorted(result.config.sources), sorted(result.config.targets)
        ax.set_title(f"Target occupation probability — {result.route}\n"
                     f"N = {result.config.chain_length}: {sources} → {targets}", loc="left", pad=12)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Target occupation fidelity")
        ax.set_ylim(-0.025, 1.13)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlim(0, result.final_time * 1.045 if result.final_time else 1.0)
        ax.grid(axis="y", color=grid, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(axis)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


def write_outputs(
    output_dir: Path, result: SimulationResult, times: np.ndarray,
    target_amps: np.ndarray, source_amps: np.ndarray, fidelities: np.ndarray,
) -> dict[str, object]:
    """功能 / Purpose: Write three reproducible artifacts / 写出三份可复现结果。

    输入 / Inputs: Directory, result, ns times and conditional ideal amplitudes/probabilities / 目录、摘要、时间和理想数组。
    输出 / Outputs: Summary and JSON/CSV/PNG; CSV separates ideal amplitudes and calibrated probability / 摘要及区分两种概率的文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result_to_jsonable(result)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    exposure = distance_exposure(times, result.route_details["primitives"])
    survival = distance_survival(result.config, exposure)
    calibrated = calibrated_target_fidelities(result.config, exposure, fidelities)
    with (output_dir / "fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ns", "conditional_ideal_target_amp_real", "conditional_ideal_target_amp_imag",
                         "conditional_ideal_source_amp_real", "conditional_ideal_source_amp_imag",
                         "ideal_target_fidelity", "distance_exposure_edges", "distance_survival_factor",
                         "calibrated_target_fidelity"])
        writer.writerows((float(t), float(ta.real), float(ta.imag), float(sa.real), float(sa.imag),
                          float(f), float(distance), float(s), float(c))
                         for t, ta, sa, f, distance, s, c in zip(times, target_amps, source_amps, fidelities,
                                                               exposure, survival, calibrated, strict=True))
    write_fidelity_plot(output_dir / "fidelity.png", result, times, fidelities)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """功能 / Purpose: Define the single automatic-routing CLI / 定义唯一自动路由入口。

    输入 / Inputs: None / 无。
    输出 / Outputs: ArgumentParser with required N and pairs / 要求链长与 pairs 的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--chain-length", type=int, required=True, help="number of sites, indexed 0..N-1")
    parser.add_argument("--pairs", type=parse_edge_pairs, required=True, help="source:target pairs, e.g. 0:9,9:10,5:19")
    cap = parser.add_mutually_exclusive_group()
    cap.add_argument("--g-max", type=float, default=DEFAULT_G_MAX_RAD_PER_NS,
                     help="positive edge cap in rad/ns (default: 3*pi/250, J/(2*pi)=6 MHz); overrides requested timing")
    cap.add_argument("--no-coupling-cap", action="store_const", const=None, dest="g_max",
                     help="remove the edge cap and use requested sqrt-distance times")
    parser.add_argument("--reference-time-ns", type=float, default=REFERENCE_TIME_NS,
                        help="positive transfer time at reference distance in ns (default: 125)")
    parser.add_argument("--reference-distance", type=float, default=REFERENCE_DISTANCE,
                        help="positive reference distance in edges (default: 5)")
    parser.add_argument("--reference-fidelity", type=float, default=REFERENCE_FIDELITY,
                        help="reference single-excitation probability in (0,1] (default: 0.992); 1 disables attenuation")
    parser.add_argument("--ideal", action="store_true", help="disable distance attenuation; keep physical timing and any cap")
    parser.add_argument("--segments", type=parse_segment_stages,
                        help="manual PST stages for conflicting pairs, e.g. '5:19;0:9,10:15'")
    parser.add_argument("--max-segment-steps", type=int, default=10, help="automatic planner stage budget (default: 10)")
    parser.add_argument("--beam-width", type=int, default=128, help="heuristic beam width (default: 128)")
    parser.add_argument("--num-steps", type=int, default=401,
                        help="samples PER constant-Hamiltonian interval including endpoints, >=2 (default: 401); "
                             "I intervals produce 1+I*(num_steps-1) points, not integration steps")
    parser.add_argument("--output-dir", type=Path,
                        help="write fidelity.png, fidelity.csv, summary.json here; overwrite same names")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """功能 / Purpose: Validate, simulate and optionally export / 校验、模拟并按需输出。

    输入 / Inputs: Optional argv; None reads process arguments / 可选参数列表，缺省读取命令行。
    输出 / Outputs: Exit code 0; prints JSON, errors exit via argparse / 返回零并打印摘要，错误时退出。
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = SimulationConfig(chain_length=args.chain_length, pairs=args.pairs,
                                  g_max=args.g_max, num_steps=args.num_steps,
                                  segments=args.segments, max_segment_steps=args.max_segment_steps,
                                  beam_width=args.beam_width, reference_fidelity=args.reference_fidelity,
                                  reference_time_ns=args.reference_time_ns, reference_distance=args.reference_distance,
                                  ideal=args.ideal)
        result, times, target_amps, source_amps, fidelities = run_simulation(config)
        summary = result_to_jsonable(result)
        if args.output_dir is not None:
            summary = write_outputs(args.output_dir, result, times, target_amps, source_amps, fidelities)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
