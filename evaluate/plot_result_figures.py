#!/usr/bin/env python3
"""Plot evaluation figures from saved result files.

The script is intentionally read-only with respect to experiment results: it only
loads CSV/JSON files that already exist under ``evaluate/result`` (plus optional
baseline/ablation files) and leaves missing results as NaN so the figure layout is
still visible while unfinished experiments are blank.

Examples
--------
    python evaluate/plot_result_figures.py
    python evaluate/plot_result_figures.py --formats pdf svg png
    python evaluate/plot_result_figures.py --ablation-quantum-program vqe
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# ----------------------------- configuration -----------------------------

PACKAGE_SIZES = [40, 60, 80, 100, 120]
LAYOUT_PACKAGE_SIZES = list(range(40, 161, 20))
QUANTUM_PROGRAMS = ["vqe", "qaoa", "bv", "s_ghz"]
LAYOUT_STRATEGIES = ["template", "heuristic", "greedy", "rl"]
PRIMARY_LAYOUT_STRATEGY = "rl"
ROUTING_STRATEGIES = ["mech", "mech_eqst", "qunoc_wo_opt", "qunoc"]

SCALABILITY_SPECS = {
    "package_shape": {
        "csv": "package_shape/output.csv",
        "axis": "PackageXLength",
        "order": [500, 250, 200, 125, 100],
        "xlabel": "Package shape (x × 10000/x)",
        "tick_labels": ["500×20", "250×40", "200×50", "125×80", "100×100"],
        "title": "Package shape",
    },
    "geometry": {
        "csv": "geometry/output.csv",
        "axis": "PackageGeometry",
        "order": ["square", "hexagon", "heavy_square", "heavy_hexagon"],
        "xlabel": "Compute-die topology",
        "tick_labels": ["Square", "Hexagon", "Heavy\nSquare", "Heavy\nHexagon"],
        "title": "Compute-die topology",
    },
    "inter_fidelity": {
        "csv": "inter_fidelity/output.csv",
        "axis": "InterFidelity",
        "order": [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99],
        "xlabel": "Inter-die coupler fidelity",
        "tick_labels": ["0.960", "0.965", "0.970", "0.975", "0.980", "0.985", "0.990"],
        "title": "Inter-die fidelity",
    },
    "chiplet_sparsity": {
        "csv": "chiplet_sparsity/output.csv",
        "axis": "ChipletSparsity",
        "order": [1, 3, 5, 7],
        "xlabel": "Inter-die coupler sparsity",
        "tick_labels": ["1", "3", "5", "7"],
        "title": "Coupler sparsity",
    },
}

# 布局扫描独立于路由配置。 / Keep layout sweeps independent of routing grids.
LAYOUT_SCALABILITY_SPECS = {
    "package_size": {
        "csv": "package_size/output.csv",
        "axis": "PackageSize",
        "order": LAYOUT_PACKAGE_SIZES,
        "xlabel": "Package size",
        "tick_labels": [f"{size}×{size}" for size in LAYOUT_PACKAGE_SIZES],
        "title": "Package size",
    },
    **{name: dict(spec) for name, spec in SCALABILITY_SPECS.items()},
}
LAYOUT_SCALABILITY_SPECS["package_shape"].update(
    order=[500, 250, 200, 125, 100],
    tick_labels=["500×20", "250×40", "200×50", "125×80", "100×100"],
    xlabel="Package shape",
)


# Palette used throughout plot_results.ipynb.
COLORS = {
    "dark_green": "#264B49",
    "teal": "#4C7F79",
    "sage": "#85BAA5",
    "light_gray": "#C0C2A6",
    "khaki": "#A79B73",
    "ink": "#000000",
    "secondary": "#525252",
    "muted": "#898781",
    "grid": "#E1E1E1",
    "axis": "#B8B8B8",
    "surface": "#FFFFFF",
}

SERIES_COLOR = {
    "template": COLORS["teal"],
    "heuristic": COLORS["dark_green"],
    "greedy": COLORS["sage"],
    "rl": COLORS["khaki"],
    "mech": COLORS["sage"],
    "mech_eqst": COLORS["khaki"],
    "qunoc_wo_opt": COLORS["light_gray"],
    "qunoc": COLORS["dark_green"],
    "vqe": COLORS["dark_green"],
    "qaoa": COLORS["teal"],
    "bv": COLORS["sage"],
    "s_ghz": COLORS["khaki"],
}

SERIES_MARKER = {
    "template": "o",
    "heuristic": "s",
    "greedy": "^",
    "rl": "D",
    "mech": "o",
    "mech_eqst": "s",
    "qunoc_wo_opt": "^",
    "qunoc": "D",
    "vqe": "o",
    "qaoa": "s",
    "bv": "^",
    "s_ghz": "D",
}

DISPLAY_NAME = {
    "template": "Template",
    "heuristic": "Heuristic Design",
    "greedy": "Greedy Design",
    "rl": "RL-based Design",
    "mech": "MECH",
    "mech_eqst": "MECH with EQST",
    "qunoc_wo_opt": "QuNoC w/o opt.",
    "qunoc": "QuNoC",
    "vqe": "VQE",
    "qaoa": "QAOA",
    "bv": "BV",
    "s_ghz": "GHZ",
}


# ----------------------------- basic helpers -----------------------------

def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.facecolor": COLORS["surface"],
            "figure.facecolor": COLORS["surface"],
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.edgecolor": COLORS["ink"],
            "legend.facecolor": COLORS["surface"],
            "legend.framealpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def apply_figure_style(fig: plt.Figure, style: dict) -> None:
    """应用调用方的局部样式。 / Apply only the calling figure's local controls."""
    for ax in fig.axes:
        ax.title.set_fontsize(style["title_fontsize"])
        ax.xaxis.label.set_fontsize(style["label_fontsize"])
        ax.yaxis.label.set_fontsize(style["label_fontsize"])
        ax.tick_params(axis="both", which="both", labelsize=style["tick_fontsize"])
        if "show_titles" in style and not style["show_titles"]:
            ax.set_title("")
        if "text_fontsize" in style:
            ax.xaxis.get_offset_text().set_fontsize(style["text_fontsize"])
            ax.yaxis.get_offset_text().set_fontsize(style["text_fontsize"])
            for text in ax.texts:
                text.set_fontsize(style["text_fontsize"])
        for line in ax.lines:
            line.set_linewidth(style["line_width"])
            line.set_markersize(style["marker_size"])
        for legend in [ax.get_legend()]:
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(style["legend_fontsize"])
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(style["legend_fontsize"])


def _layout_legend(fig, handles, labels, style):
    """独立布局图图例。 / Legend for a standalone layout figure."""
    fig.legend(handles, labels, frameon=True, edgecolor=COLORS["ink"],
               facecolor=COLORS["surface"], fancybox=False,
               fontsize=style["legend_box_fontsize"], **style["legend"])


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    duplicate_program_cols = [
        c for c in df.columns if c.startswith("QuantumProgram.") or c.startswith("Algorithm.")
    ]
    if duplicate_program_cols:
        df = df.drop(columns=duplicate_program_cols)
    if "QuantumProgram" not in df.columns and "Algorithm" in df.columns:
        df = df.rename(columns={"Algorithm": "QuantumProgram"})
    elif "QuantumProgram" in df.columns and "Algorithm" in df.columns:
        df = df.drop(columns=["Algorithm"])
    return df


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def last_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.groupby(keys, as_index=False, dropna=False).tail(1).reset_index(drop=True)


def complete_grid(df: pd.DataFrame, keys: dict[str, list], value_columns: list[str]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(keys.values(), names=keys.keys()).to_frame(index=False)
    if df.empty:
        for col in value_columns:
            grid[col] = np.nan
        return grid
    return grid.merge(df, how="left", on=list(keys.keys()))


def value_at(df: pd.DataFrame, mask: pd.Series, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    subset = df.loc[mask, column]
    if subset.empty:
        return np.nan
    value = pd.to_numeric(subset, errors="coerce").dropna()
    return float(value.iloc[-1]) if not value.empty else np.nan


def style_axis(ax: Axes, grid_axis: str = "y") -> None:
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=COLORS["ink"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["ink"])
        spine.set_linewidth(1.0)


def plot_nan_hint(ax: Axes, values: Iterable[float]) -> None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size and np.all(np.isnan(arr)):
        ax.text(
            0.5,
            0.5,
            "pending",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
            fontsize=10,
        )


def plot_line(ax: Axes, x, y, key: str, label: str | None = None, dashed: bool = False, *, style: dict) -> None:
    ax.plot(
        x,
        y,
        marker=style["markers"].get(key, "o"),
        markersize=style["marker_size"],
        markeredgecolor=COLORS["ink"],
        markeredgewidth=0.45,
        linewidth=style["line_width"],
        linestyle="--" if dashed else "-",
        color=SERIES_COLOR.get(key, COLORS["dark_green"]),
        label=label or DISPLAY_NAME.get(key, key),
    )
    plot_nan_hint(ax, y)


# ----------------------------- data loaders -----------------------------

def load_layout_scalability(eval_result_dir: Path, name: str) -> pd.DataFrame:
    spec = LAYOUT_SCALABILITY_SPECS[name]
    axis = spec["axis"]
    values = ["Fidelity", "FideStd", "Score"]
    keys = {"DesignStrategy": LAYOUT_STRATEGIES, axis: spec["order"]}
    df = read_csv(eval_result_dir / "layout_scalability" / spec["csv"])
    if not df.empty:
        numeric_axis = all(isinstance(v, (int, float)) for v in spec["order"])
        df = coerce_numeric(df, ([axis] if numeric_axis else []) + values)
        # 重跑记录取最后一条，不混合历史结果。 / Use the last persisted rerun per case.
        df = last_by(df, ["DesignStrategy", axis])
        df = df[df["DesignStrategy"].isin(LAYOUT_STRATEGIES)]
    return complete_grid(df, keys, values)


def load_layout_package_size(eval_result_dir: Path) -> pd.DataFrame:
    return load_layout_scalability(eval_result_dir, "package_size")


def load_layout_package_shape(eval_result_dir: Path) -> pd.DataFrame:
    return load_layout_scalability(eval_result_dir, "package_shape")


def load_routing_package_size(eval_result_dir: Path) -> pd.DataFrame:
    path = eval_result_dir / "routing_scalability/package_size/output.csv"
    df = read_csv(path)
    values = ["GateFidelity", "CircuitDuration", "IntraGate", "InterGate", "QSTDist"]
    grid_keys = {"PackageSize": PACKAGE_SIZES, "QuantumProgram": QUANTUM_PROGRAMS}
    if df.empty:
        return complete_grid(pd.DataFrame(), grid_keys, values)
    df = coerce_numeric(
        df,
        ["PackageSize", "#DataQubit", "#LogicGate", "CircuitDuration", "GateFidelity", "IntraGate", "InterGate", "QSTDist"],
    )
    if "LayoutDesignStrategy" in df.columns:
        df = df[df["LayoutDesignStrategy"].astype(str).str.lower().eq(PRIMARY_LAYOUT_STRATEGY)]
    df = last_by(df, ["PackageSize", "QuantumProgram"])
    df = df[df["QuantumProgram"].isin(QUANTUM_PROGRAMS)]
    return complete_grid(df, grid_keys, values)


def load_main_exp(evaluate_dir: Path) -> pd.DataFrame:
    candidates = [evaluate_dir / "result" / "overall" / "main_exp.csv", evaluate_dir / "backup" / "main_exp.csv", evaluate_dir / "main_exp.csv"]
    df = pd.DataFrame()
    for path in candidates:
        df = read_csv(path)
        if not df.empty:
            break
    if df.empty:
        out = complete_grid(
            pd.DataFrame(),
            {"PackageSize": PACKAGE_SIZES, "QuantumProgram": QUANTUM_PROGRAMS},
            ["MECHGateFidelity", "QuNoCGateFidelity", "MECHCircuitDuration", "QuNoCCircuitDuration"],
        )
        return out
    df = coerce_numeric(
        df,
        ["#DataQubit", "MECHGateFidelity", "QuNoCGateFidelity", "MECHCircuitDuration", "QuNoCCircuitDuration", "Speedup"],
    )
    size_by_qubits = {100: 40, 141: 60, 181: 80, 220: 100, 257: 120}
    df["PackageSize"] = df["#DataQubit"].map(size_by_qubits)
    df = df.dropna(subset=["PackageSize"])
    df["PackageSize"] = df["PackageSize"].astype(int)
    df = last_by(df, ["PackageSize", "QuantumProgram"])
    return complete_grid(
        df,
        {"PackageSize": PACKAGE_SIZES, "QuantumProgram": sorted(df["QuantumProgram"].dropna().unique().tolist()) or QUANTUM_PROGRAMS},
        ["MECHGateFidelity", "QuNoCGateFidelity", "MECHCircuitDuration", "QuNoCCircuitDuration"],
    )


def load_scalability(eval_result_dir: Path, name: str, spec: dict) -> pd.DataFrame:
    df = read_csv(eval_result_dir / "routing_scalability" / spec["csv"])
    axis = spec["axis"]
    values = ["CircuitDuration", "GateFidelity", "IntraGate", "InterGate", "QSTDist"]
    if df.empty:
        return complete_grid(pd.DataFrame(), {axis: spec["order"], "QuantumProgram": QUANTUM_PROGRAMS}, values)
    # Numeric axes should be coerced, but categorical axes such as geometry
    # must remain strings so they merge cleanly with the expected grid.
    numeric_axis = all(isinstance(v, (int, float, np.integer, np.floating)) for v in spec["order"])
    df = coerce_numeric(df, ([axis] if numeric_axis else []) + values)
    df = last_by(df, [axis, "QuantumProgram"])
    df = df[df["QuantumProgram"].isin(QUANTUM_PROGRAMS)]
    return complete_grid(df, {axis: spec["order"], "QuantumProgram": QUANTUM_PROGRAMS}, values)


def load_mech_scalability(root: Path, name: str, spec: dict) -> pd.DataFrame:
    """Load MECH CSVs, deriving any available package-shape JSON results."""
    axis = spec["axis"]
    values = ["CircuitDuration", "GateFidelity"]
    result_dir = root / "evaluate" / "result" / "baseline_mech" / name
    path = result_dir / "output.csv"
    if not path.exists():
        result_dir = root / "result/routing_scalability/baseline_mech" / name
        path = result_dir / "output.csv"
    df = read_csv(path)

    if df.empty and name == "package_shape":
        # MECH's long package-shape runs may leave raw JSON before output.csv is
        # complete. Derive the same persisted metrics for the available cases.
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from baseline.mech_utils import get_mech_circuit_duration, get_mech_logic_gate_fide
        from qunoc.element.package import FabricateModel

        rows = []
        for x_length in spec["order"]:
            y_length = 10000 // int(x_length)
            raw_path = result_dir / f"{x_length}_{y_length}_mech.json"
            if not raw_path.exists():
                continue
            with raw_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
            chiplet_x = max(divisor for divisor in range(1, min(int(x_length), 10) + 1) if int(x_length) % divisor == 0)
            chiplet_y = max(divisor for divisor in range(1, min(y_length, 10) + 1) if y_length % divisor == 0)
            chiplet_size = (chiplet_x, chiplet_y)
            capacity = chiplet_x * chiplet_y
            fab_model = FabricateModel(
                max_capacity=capacity,
                max_length=int(0.8 * capacity),
                compute_geometry="square",
                inter_sparsity=3,
                inter_fide=0.983,
            )
            for quantum_program in QUANTUM_PROGRAMS:
                if quantum_program not in raw:
                    continue
                result = raw[quantum_program]
                rows.append(
                    {
                        axis: x_length,
                        "QuantumProgram": quantum_program,
                        "CircuitDuration": get_mech_circuit_duration(result),
                        "GateFidelity": get_mech_logic_gate_fide(result, chiplet_size, fab_model),
                    }
                )
        df = pd.DataFrame(rows)

    if df.empty:
        return complete_grid(pd.DataFrame(), {axis: spec["order"], "QuantumProgram": QUANTUM_PROGRAMS}, values)
    numeric_axis = all(isinstance(v, (int, float, np.integer, np.floating)) for v in spec["order"])
    df = coerce_numeric(df, ([axis] if numeric_axis else []) + values)
    if name == "package_shape" and axis in df.columns:
        observed = set(pd.to_numeric(df[axis], errors="coerce").dropna().astype(int).tolist())
        expected = set(int(v) for v in spec["order"])
        reversed_expected = {10000 // int(v) for v in spec["order"]}
        if len(observed & reversed_expected) > len(observed & expected):
            df[axis] = np.divide(10000, pd.to_numeric(df[axis], errors="coerce"))
            df[axis] = pd.to_numeric(df[axis], errors="coerce").round().astype("Int64")
    if "Baseline" in df.columns:
        df = df[df["Baseline"].astype(str).str.lower().eq("mech")]
    df = last_by(df, [axis, "QuantumProgram"])
    df = df[df["QuantumProgram"].isin(QUANTUM_PROGRAMS)]
    return complete_grid(df, {axis: spec["order"], "QuantumProgram": QUANTUM_PROGRAMS}, values)


def read_optional_ablation_csv(evaluate_dir: Path) -> pd.DataFrame:
    candidates = [
        evaluate_dir / "result/ablation/package_size/output.csv",
        evaluate_dir / "backup/ablation.csv",
        evaluate_dir / "ablation.csv",
        evaluate_dir / "result/routing_scalability/ablation/output.csv",
        evaluate_dir / "result/routing_scalability/ablation/package_size/output.csv",
        evaluate_dir / "result/routing_scalability/package_size_ablation/output.csv",
    ]
    for path in candidates:
        df = read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_ablation(evaluate_dir: Path, eval_result_dir: Path, quantum_program: str) -> pd.DataFrame:
    """Load routing ablation data.

    Canonical inputs are read from result/baseline_mech/package_size,
    result/routing_scalability/package_size and result/ablation/package_size.
    Historical main_exp.csv remains a fallback for old result directories.
    """

    rows: list[dict] = []
    main_df = load_main_exp(evaluate_dir)
    main_df = main_df[main_df["QuantumProgram"] == quantum_program]
    mech_df = load_mech_scalability(
        evaluate_dir.parent,
        "package_size",
        {"axis": "PackageSize", "order": PACKAGE_SIZES},
    )
    mech_df = mech_df[mech_df["QuantumProgram"] == quantum_program]
    routing_df = load_routing_package_size(eval_result_dir)
    routing_df = routing_df[routing_df["QuantumProgram"].eq(quantum_program)]

    for size in PACKAGE_SIZES:
        mech_f = value_at(mech_df, mech_df["PackageSize"].eq(size), "GateFidelity")
        mech_d = value_at(mech_df, mech_df["PackageSize"].eq(size), "CircuitDuration")
        qunoc_f = value_at(routing_df, routing_df["PackageSize"].eq(size), "GateFidelity")
        qunoc_d = value_at(routing_df, routing_df["PackageSize"].eq(size), "CircuitDuration")
        if np.isfinite(mech_d):
            mech_d = mech_d / 1_000_000
        if np.isfinite(qunoc_d):
            qunoc_d = qunoc_d / 1_000_000
        if not np.isfinite(mech_f):
            mech_f = value_at(main_df, main_df["PackageSize"].eq(size), "MECHGateFidelity")
        if not np.isfinite(mech_d):
            mech_d = value_at(main_df, main_df["PackageSize"].eq(size), "MECHCircuitDuration")
        if not np.isfinite(qunoc_f):
            qunoc_f = value_at(main_df, main_df["PackageSize"].eq(size), "QuNoCGateFidelity")
        if not np.isfinite(qunoc_d):
            qunoc_d = value_at(main_df, main_df["PackageSize"].eq(size), "QuNoCCircuitDuration")
        rows += [
            {"PackageSize": size, "Strategy": "mech", "GateFidelity": mech_f, "CircuitDurationMs": mech_d},
            {"PackageSize": size, "Strategy": "qunoc", "GateFidelity": qunoc_f, "CircuitDurationMs": qunoc_d},
            {"PackageSize": size, "Strategy": "mech_eqst", "GateFidelity": np.nan, "CircuitDurationMs": np.nan},
            {"PackageSize": size, "Strategy": "qunoc_wo_opt", "GateFidelity": np.nan, "CircuitDurationMs": np.nan},
        ]

    ablation = pd.DataFrame(rows)
    optional = read_optional_ablation_csv(evaluate_dir)
    if optional.empty:
        return ablation

    optional = optional.copy()
    # Accept several likely column names so future experiment outputs can be used
    # without rewriting this plotting script.
    rename_map = {
        "Baseline": "Strategy",
        "RoutingStrategy": "Strategy",
        "PackageLength": "PackageSize",
        "Size": "PackageSize",
        "Fidelity": "GateFidelity",
        "Duration": "CircuitDurationMs",
        "CircuitDuration": "CircuitDurationRaw",
    }
    optional = optional.rename(columns={k: v for k, v in rename_map.items() if k in optional.columns})
    if "QuantumProgram" in optional.columns:
        optional = optional[optional["QuantumProgram"].astype(str).str.lower() == quantum_program]
    if not {"PackageSize", "Strategy"}.issubset(optional.columns):
        return ablation

    optional = coerce_numeric(optional, ["PackageSize", "GateFidelity", "CircuitDurationMs", "CircuitDurationRaw"])
    if "CircuitDurationMs" not in optional.columns and "CircuitDurationRaw" in optional.columns:
        # Router results are in ns-like units in output.csv; main_exp stores ms.
        optional["CircuitDurationMs"] = optional["CircuitDurationRaw"] / 1_000_000
    strategy_alias = {
        "mech_with_eqst": "mech_eqst",
        "mech-eqst": "mech_eqst",
        "mech_eqst": "mech_eqst",
        "qunoc_wo_sensate": "qunoc_wo_opt",
        "qunoc_without_opt": "qunoc_wo_opt",
        "qunoc_wo_opt": "qunoc_wo_opt",
        "sensate_without_optimization": "qunoc_wo_opt",
    }
    optional["Strategy"] = optional["Strategy"].astype(str).str.lower().map(lambda x: strategy_alias.get(x, x))
    optional = last_by(optional, ["PackageSize", "Strategy"])

    for _, row in optional.iterrows():
        mask = ablation["PackageSize"].eq(int(row["PackageSize"])) & ablation["Strategy"].eq(row["Strategy"])
        if not mask.any():
            continue
        for col in ["GateFidelity", "CircuitDurationMs"]:
            if col in optional.columns and not pd.isna(row.get(col)):
                ablation.loc[mask, col] = float(row[col])
    return ablation


def load_mech_operation_json(root: Path, quantum_program: str) -> pd.DataFrame:
    baseline_dir = root / "evaluate" / "result" / "baseline_mech" / "package_size"
    if not baseline_dir.exists():
        baseline_dir = root / "result/routing_scalability/baseline_mech/package_size"
    rows = []
    for size in PACKAGE_SIZES:
        path = baseline_dir / f"{size}_mech_{quantum_program}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            result = data.get("result", data.get(quantum_program, {}))
        else:
            path = baseline_dir / f"{size}_mech.json"
            if not path.exists():
                rows.append({"PackageSize": size, "OnChip": np.nan, "CrossChip": np.nan, "MeasNum": np.nan})
                continue
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            result = data.get(quantum_program, {})
        rows.append(
            {
                "PackageSize": size,
                "OnChip": result.get("on_chip", np.nan),
                "CrossChip": result.get("cross_chip", np.nan),
                "MeasNum": result.get("meas_num", np.nan),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------- figure data exporters -----------------------------


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def duration_ms(value: float) -> float:
    return value / 1_000_000 if np.isfinite(value) else np.nan


def export_figure_csv(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def build_overall_results_csv(root: Path, eval_result_dir: Path) -> pd.DataFrame:
    sensate = load_routing_package_size(eval_result_dir)
    mech = load_mech_scalability(
        root,
        "package_size",
        {"axis": "PackageSize", "order": PACKAGE_SIZES},
    )
    rows = []
    for quantum_program in QUANTUM_PROGRAMS:
        sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)]
        mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)]
        for size in PACKAGE_SIZES:
            mech_duration = value_at(mech_subset, mech_subset["PackageSize"].eq(size), "CircuitDuration")
            qunoc_duration = value_at(sensate_subset, sensate_subset["PackageSize"].eq(size), "CircuitDuration")
            mech_fidelity = value_at(mech_subset, mech_subset["PackageSize"].eq(size), "GateFidelity")
            qunoc_fidelity = value_at(sensate_subset, sensate_subset["PackageSize"].eq(size), "GateFidelity")
            rows.append(
                {
                    "Figure": "routing_scalability_main",
                    "Panel": DISPLAY_NAME.get(quantum_program, quantum_program),
                    "QuantumProgram": quantum_program,
                    "PackageSize": size,
                    "PackageLabel": f"{size}x{size}",
                    "MECHCircuitDurationRaw": mech_duration,
                    "QuNoCCircuitDurationRaw": qunoc_duration,
                    "MECHCircuitDurationMs": duration_ms(mech_duration),
                    "QuNoCCircuitDurationMs": duration_ms(qunoc_duration),
                    "CircuitSpeedup_MECH_over_QuNoC": safe_ratio(mech_duration, qunoc_duration),
                    "DurationReductionPercent": (1 - safe_ratio(qunoc_duration, mech_duration)) * 100,
                    "MECHGateFidelity": mech_fidelity,
                    "QuNoCGateFidelity": qunoc_fidelity,
                    "GateFidelityRatio_QuNoC_over_MECH": safe_ratio(qunoc_fidelity, mech_fidelity),
                    "GateFidelityDelta_QuNoC_minus_MECH": qunoc_fidelity - mech_fidelity,
                }
            )
    return pd.DataFrame(rows)


def build_layout_scalability_csv(eval_result_dir: Path) -> pd.DataFrame:
    rows = []
    for name, spec in LAYOUT_SCALABILITY_SPECS.items():
        data = load_layout_scalability(eval_result_dir, name)
        axis = spec["axis"]
        labels = dict(zip(spec["order"], spec["tick_labels"]))
        for strategy in ["rl", "greedy", "heuristic", "template"]:
            subset = data[data["DesignStrategy"].eq(strategy)]
            for _, row in subset.iterrows():
                rows.append({
                    "Figure": "layout_scalability_main" if name == "package_size" else "layout_scalability_rest",
                    "Panel": "layout_score_and_fidelity" if name == "package_size" else f"{name}_layout_score",
                    "Sweep": name,
                    "AxisColumn": axis,
                    "AxisValue": row[axis],
                    "AxisLabel": labels[row[axis]].replace("\n", " "),
                    "DesignStrategy": strategy,
                    "DesignStrategyLabel": DISPLAY_NAME[strategy],
                    "PackageSize": int(row[axis]) if name == "package_size" else np.nan,
                    "PackageLabel": f"{int(row[axis])}x{int(row[axis])}" if name == "package_size" else None,
                    "Score": row.get("Score", np.nan),
                    "Fidelity": row.get("Fidelity", np.nan),
                    "FideStd": row.get("FideStd", np.nan),
                })
    return pd.DataFrame(rows)


def build_package_shape_scalability_csv(root: Path, eval_result_dir: Path) -> pd.DataFrame:
    spec = SCALABILITY_SPECS["package_shape"]
    axis = spec["axis"]
    sensate = load_scalability(eval_result_dir, "package_shape", spec)
    mech = load_mech_scalability(root, "package_shape", spec)
    rows = []
    for quantum_program in QUANTUM_PROGRAMS:
        sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)]
        mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)]
        for axis_value, tick_label in zip(spec["order"], spec["tick_labels"]):
            mech_duration = value_at(mech_subset, mech_subset[axis].eq(axis_value), "CircuitDuration")
            qunoc_duration = value_at(sensate_subset, sensate_subset[axis].eq(axis_value), "CircuitDuration")
            mech_fidelity = value_at(mech_subset, mech_subset[axis].eq(axis_value), "GateFidelity")
            qunoc_fidelity = value_at(sensate_subset, sensate_subset[axis].eq(axis_value), "GateFidelity")
            rows.append(
                {
                    "Figure": "package_shape_scalability",
                    "Panel": DISPLAY_NAME.get(quantum_program, quantum_program),
                    "QuantumProgram": quantum_program,
                    axis: axis_value,
                    "PackageShapeLabel": tick_label.replace("\n", " "),
                    "MECHCircuitDurationRaw": mech_duration,
                    "QuNoCCircuitDurationRaw": qunoc_duration,
                    "MECHCircuitDurationMs": duration_ms(mech_duration),
                    "QuNoCCircuitDurationMs": duration_ms(qunoc_duration),
                    "CircuitSpeedup_MECH_over_QuNoC": safe_ratio(mech_duration, qunoc_duration),
                    "DurationReductionPercent": (1 - safe_ratio(qunoc_duration, mech_duration)) * 100,
                    "MECHGateFidelity": mech_fidelity,
                    "QuNoCGateFidelity": qunoc_fidelity,
                    "GateFidelityRatio_QuNoC_over_MECH": safe_ratio(qunoc_fidelity, mech_fidelity),
                    "GateFidelityDelta_QuNoC_minus_MECH": qunoc_fidelity - mech_fidelity,
                }
            )
    return pd.DataFrame(rows)


def build_routing_scalability_csv(root: Path, eval_result_dir: Path) -> pd.DataFrame:
    rows = []
    for name, spec in SCALABILITY_SPECS.items():
        if name == "package_shape":
            continue
        axis = spec["axis"]
        sensate = load_scalability(eval_result_dir, name, spec)
        mech = load_mech_scalability(root, name, spec)
        for quantum_program in QUANTUM_PROGRAMS:
            sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)]
            mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)]
            for axis_value, tick_label in zip(spec["order"], spec["tick_labels"]):
                mech_duration = value_at(mech_subset, mech_subset[axis].eq(axis_value), "CircuitDuration")
                qunoc_duration = value_at(sensate_subset, sensate_subset[axis].eq(axis_value), "CircuitDuration")
                mech_fidelity = value_at(mech_subset, mech_subset[axis].eq(axis_value), "GateFidelity")
                qunoc_fidelity = value_at(sensate_subset, sensate_subset[axis].eq(axis_value), "GateFidelity")
                rows.append(
                    {
                        "Figure": "routing_scalability",
                        "Sweep": name,
                        "SweepTitle": spec["title"],
                        "AxisColumn": axis,
                        "AxisValue": axis_value,
                        "AxisLabel": tick_label.replace("\n", " "),
                        "QuantumProgram": quantum_program,
                        "QuantumProgramLabel": DISPLAY_NAME.get(quantum_program, quantum_program),
                        "MECHCircuitDurationRaw": mech_duration,
                        "QuNoCCircuitDurationRaw": qunoc_duration,
                        "CircuitSpeedup_MECH_over_QuNoC": safe_ratio(mech_duration, qunoc_duration),
                        "MECHGateFidelity": mech_fidelity,
                        "QuNoCGateFidelity": qunoc_fidelity,
                        "GateFidelityRatio_QuNoC_over_MECH": safe_ratio(qunoc_fidelity, mech_fidelity),
                        "GateFidelityDelta_QuNoC_minus_MECH": qunoc_fidelity - mech_fidelity,
                    }
                )
    return pd.DataFrame(rows)


def build_combined_scalability_csv(root: Path, eval_result_dir: Path) -> pd.DataFrame:
    package_shape = build_package_shape_scalability_csv(root, eval_result_dir).copy()
    if not package_shape.empty:
        package_shape["Figure"] = "routing_scalability_rest"
        package_shape["PanelGroup"] = "package_shape_scalability"
        package_shape["Sweep"] = "package_shape"
        package_shape["SweepTitle"] = SCALABILITY_SPECS["package_shape"]["title"]
        package_shape["RowMetric"] = "duration_and_fidelity"

    routing = build_routing_scalability_csv(root, eval_result_dir).copy()
    if not routing.empty:
        routing["Figure"] = "routing_scalability_rest"
        routing["PanelGroup"] = "routing_scalability"
        routing["RowMetric"] = "fidelity_improvement_and_circuit_speedup"

    return pd.concat([package_shape, routing], ignore_index=True, sort=False)


def build_ablation_csv(root: Path, evaluate_dir: Path, eval_result_dir: Path, quantum_program: str) -> pd.DataFrame:
    rows = []
    ablation = load_ablation(evaluate_dir, eval_result_dir, quantum_program)
    for _, row in ablation.sort_values(["PackageSize", "Strategy"]).iterrows():
        strategy = row["Strategy"]
        rows.append(
            {
                "Figure": f"ablation_{quantum_program}",
                "Panel": "routing_method_ablation",
                "QuantumProgram": quantum_program,
                "PackageSize": int(row["PackageSize"]),
                "Strategy": strategy,
                "StrategyLabel": DISPLAY_NAME.get(strategy, strategy),
                "GateFidelity": row.get("GateFidelity", np.nan),
                "CircuitDurationMs": row.get("CircuitDurationMs", np.nan),
            }
        )

    qunoc_ops = load_routing_package_size(eval_result_dir)
    qunoc_ops = qunoc_ops[qunoc_ops["QuantumProgram"].eq(quantum_program)]
    qunoc_ops = last_by(qunoc_ops, ["PackageSize"])
    mech_ops = load_mech_operation_json(root, quantum_program)
    for size in PACKAGE_SIZES:
        qunoc_subset = qunoc_ops[qunoc_ops["PackageSize"].eq(size)]
        mech_subset = mech_ops[mech_ops["PackageSize"].eq(size)]
        rows.append(
            {
                "Figure": f"ablation_{quantum_program}",
                "Panel": "hardware_operation_decomposition",
                "QuantumProgram": quantum_program,
                "PackageSize": size,
                "MECHOnChipGate": value_at(mech_subset, mech_subset["PackageSize"].eq(size), "OnChip"),
                "MECHCrossChipGate": value_at(mech_subset, mech_subset["PackageSize"].eq(size), "CrossChip"),
                "MECHMeasNum": value_at(mech_subset, mech_subset["PackageSize"].eq(size), "MeasNum"),
                "QuNoCIntraGate": value_at(qunoc_subset, qunoc_subset["PackageSize"].eq(size), "IntraGate"),
                "QuNoCInterGate": value_at(qunoc_subset, qunoc_subset["PackageSize"].eq(size), "InterGate"),
                "QuNoCQSTDist": value_at(qunoc_subset, qunoc_subset["PackageSize"].eq(size), "QSTDist"),
            }
        )
    return pd.DataFrame(rows)


def export_all_figure_csv(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, quantum_program: str) -> list[Path]:
    layout = build_layout_scalability_csv(eval_result_dir)
    main = layout["Sweep"].eq("package_size")
    return [
        export_figure_csv(layout.loc[main], out_dir, "layout_scalability_main"),
        export_figure_csv(layout.loc[~main], out_dir, "layout_scalability_rest"),
        export_figure_csv(build_overall_results_csv(root, eval_result_dir), out_dir, "routing_scalability_main"),
        export_figure_csv(build_combined_scalability_csv(root, eval_result_dir), out_dir, "routing_scalability_rest"),
        export_figure_csv(build_ablation_csv(root, evaluate_dir, eval_result_dir, quantum_program), out_dir, f"ablation_{quantum_program}"),
    ]


# ----------------------------- figure builders -----------------------------

def _plot_overall_panels(root: Path, eval_result_dir: Path, out_dir: Path, formats: list[str], style: dict) -> None:
    """构建总体路由面板。 / Draw overall routing panels using the caller's style."""
    package_sizes = [40, 60, 80, 100, 120]
    sensate = load_routing_package_size(eval_result_dir)
    mech = load_mech_scalability(
        root,
        "package_size",
        {"axis": "PackageSize", "order": package_sizes},
    )

    fig, axes = plt.subplots(2, 2, figsize=style["figsize"], sharex=False)
    x = np.arange(len(package_sizes))
    width = style["bar_width"]
    baseline_colors = {"mech": COLORS["sage"], "sensate": COLORS["dark_green"]}
    baseline_markers = style["baseline_markers"]
    baseline_labels = {"mech": "MECH", "sensate": "Sensate"}
    legend_handles = []
    legend_labels = []

    for index, (ax, quantum_program) in enumerate(zip(axes.flat, QUANTUM_PROGRAMS)):
        duration_ax = ax.twinx()
        ax.set_zorder(duration_ax.get_zorder() + 1)
        ax.patch.set_visible(False)

        sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)].copy()
        mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)].copy()
        sensate_subset = sensate_subset[sensate_subset["PackageSize"].isin(package_sizes)].copy()
        mech_subset = mech_subset[mech_subset["PackageSize"].isin(package_sizes)].copy()
        for subset in (sensate_subset, mech_subset):
            subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=package_sizes, ordered=True)
            subset.sort_values("PackageSize", inplace=True)

        mech_fidelity = pd.to_numeric(mech_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
        sensate_fidelity = pd.to_numeric(sensate_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
        mech_duration = pd.to_numeric(mech_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000
        sensate_duration = pd.to_numeric(sensate_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000

        mech_bars = duration_ax.bar(
            x - width / 2,
            mech_duration,
            width,
            color=baseline_colors["mech"],
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            alpha=0.72,
            label=baseline_labels["mech"],
            zorder=2,
        )
        sensate_bars = duration_ax.bar(
            x + width / 2,
            sensate_duration,
            width,
            color=baseline_colors["sensate"],
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            alpha=0.72,
            label=baseline_labels["sensate"],
            zorder=2,
        )
        mech_line = ax.plot(
            x,
            mech_fidelity,
            color=baseline_colors["mech"],
            marker=baseline_markers["mech"],
            markersize=9,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.5,
            linewidth=2,
            label=baseline_labels["mech"],
            zorder=3,
        )[0]
        sensate_line = ax.plot(
            x,
            sensate_fidelity,
            color=baseline_colors["sensate"],
            marker=baseline_markers["sensate"],
            markersize=9,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.5,
            linewidth=2,
            label=baseline_labels["sensate"],
            zorder=3,
        )[0]

        if index == 0:
            legend_handles = [mech_line, sensate_line, mech_bars, sensate_bars]
            legend_labels = [
                "MECH fidelity",
                "Sensate fidelity",
                "MECH duration",
                "Sensate duration",
            ]

        panel = chr(ord("a") + index)
        ax.set_title(f"({panel}) {DISPLAY_NAME[quantum_program]}")
        ax.set_xlabel("Package size")
        ax.set_ylabel("Avg. gate fidelity")
        duration_ax.set_ylabel("Circuit duration (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{size}×{size}" for size in package_sizes])
        finite_fidelity = np.concatenate([mech_fidelity[np.isfinite(mech_fidelity)], sensate_fidelity[np.isfinite(sensate_fidelity)]])
        if finite_fidelity.size:
            y_min = max(0, float(finite_fidelity.min()) - 0.08)
            y_max = min(1.0, float(finite_fidelity.max()) + 0.08)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1)
        finite_duration = np.concatenate([mech_duration[np.isfinite(mech_duration)], sensate_duration[np.isfinite(sensate_duration)]])
        if finite_duration.size:
            duration_ax.set_ylim(0, float(finite_duration.max()) * 1.18)
        else:
            duration_ax.set_ylim(0, 1)
        ax.grid(False)
        duration_ax.grid(False)
        style_axis(ax)
        duration_ax.tick_params(axis="both", colors=COLORS["ink"])
        for twin_spine in duration_ax.spines.values():
            twin_spine.set_visible(True)
            twin_spine.set_color(COLORS["ink"])
            twin_spine.set_linewidth(1.0)
        plot_nan_hint(ax, [*mech_fidelity, *sensate_fidelity, *mech_duration, *sensate_duration])

    fig.legend(
        legend_handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        columnspacing=1.5,
        handlelength=1.5,
    )
    apply_figure_style(fig, style)
    fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(fig, out_dir, "routing_scalability_main", formats)


def plot_layout_scalability_main(eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """绘制尺寸指标。 / Plot the standalone package-size metrics figure."""
    style = {
        "figsize": (12, 4),
        "title_fontsize": 16, "label_fontsize": 16, "tick_fontsize": 16,
        "legend_fontsize": 16, "legend_box_fontsize": 14,
        "text_fontsize": 16, "show_titles": False,
        "legend": dict(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.98), columnspacing=1.3, handlelength=1.5),
        "line_width": 2.0, "marker_size": 10, "markers": dict(SERIES_MARKER), "bar_width": 0.18,
        "subplot_adjust": dict(left=0.08, right=0.92, bottom=0.22, top=0.80, wspace=0.22),
    }
    package_size = load_layout_package_size(eval_result_dir)
    strategies = ["rl", "greedy", "heuristic", "template"]
    x = np.arange(len(LAYOUT_PACKAGE_SIZES))
    bar_width = style["bar_width"]

    fig, (score_ax, fidelity_ax) = plt.subplots(1, 2, figsize=style["figsize"])

    for strategy in strategies:
        subset = package_size[package_size["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=LAYOUT_PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        score = pd.to_numeric(subset["Score"], errors="coerce").to_numpy(dtype=float)
        plot_line(score_ax, x, score, strategy, style=style)

    # score_ax.set_title("(a) Layout score scalability")
    score_ax.set_xlabel("Package size")
    score_ax.set_ylabel("Package score")
    score_ax.set_xticks(x)
    score_ax.set_xticklabels([f"{size}×{size}" for size in LAYOUT_PACKAGE_SIZES])
    score_ax.set_ylim(bottom=0)
    style_axis(score_ax)

    std_ax = fidelity_ax.twinx()
    fidelity_ax.set_zorder(std_ax.get_zorder() + 1)
    fidelity_ax.patch.set_visible(False)
    fidelity_handles = []
    fidelity_labels = []
    for index, strategy in enumerate(strategies):
        subset = package_size[package_size["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=LAYOUT_PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        fidelity = pd.to_numeric(subset["Fidelity"], errors="coerce").to_numpy(dtype=float)
        fidelity_std = pd.to_numeric(subset["FideStd"], errors="coerce").to_numpy(dtype=float)
        std_ax.bar(
            x + (index - (len(strategies) - 1) / 2) * bar_width,
            fidelity_std,
            width=bar_width,
            color=SERIES_COLOR[strategy],
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            alpha=0.72,
            zorder=2,
        )
        line = fidelity_ax.plot(
            x,
            fidelity,
            marker=style["markers"][strategy],
            markersize=10,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.45,
            linewidth=2.0,
            color=SERIES_COLOR[strategy],
            label=DISPLAY_NAME[strategy],
            zorder=3,
        )[0]
        fidelity_handles.append(line)
        fidelity_labels.append(DISPLAY_NAME[strategy])
        plot_nan_hint(fidelity_ax, [*fidelity, *fidelity_std])

    # fidelity_ax.set_title("(b) Layout fidelity scalability")
    fidelity_ax.set_xlabel("Package size")
    fidelity_ax.set_ylabel("Avg. fidelity")
    std_ax.set_ylabel("Fidelity std")
    fidelity_ax.set_xticks(x)
    fidelity_ax.set_xticklabels([f"{size}×{size}" for size in LAYOUT_PACKAGE_SIZES])
    _auto_zero_ylim(fidelity_ax, package_size["Fidelity"])
    _auto_zero_ylim(std_ax, package_size["FideStd"])
    style_axis(fidelity_ax)
    std_ax.grid(False)
    std_ax.tick_params(axis="both", colors=COLORS["ink"])
    for spine in std_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["ink"])
        spine.set_linewidth(1.0)

    # 尺寸标签和图例独立设置。 / Keep size labels and legend local to this figure.
    for ax in (score_ax, fidelity_ax):
        ax.tick_params(axis="x", labelrotation=30)
    handles, labels = score_ax.get_legend_handles_labels()
    _layout_legend(fig, handles, labels, style)
    apply_figure_style(fig, style)
    fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(fig, out_dir, "layout_scalability_main", formats)


def plot_layout_scalability_rest(eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """绘制其他布局分数扫描。 / Plot the standalone layout-score sweeps."""
    style = {
        "figsize": (12, 7),
        "title_fontsize": 16, "label_fontsize": 16, "tick_fontsize": 16,
        "legend_fontsize": 16, "legend_box_fontsize": 14,
        "text_fontsize": 16, "show_titles": False,
        "legend": dict(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.98), columnspacing=1.3, handlelength=1.5),
        "line_width": 2.0, "marker_size": 10, "markers": dict(SERIES_MARKER), "bar_width": 0.18,
        "subplot_adjust": dict(left=0.08, right=0.92, bottom=0.09, top=0.88, wspace=0.22, hspace=0.45),
    }
    strategies = ["rl", "greedy", "heuristic", "template"]
    bar_width = style["bar_width"]
    sweep_fig, sweep_axes = plt.subplots(2, 2, figsize=style["figsize"])
    for ax, panel, name in zip(
        sweep_axes.flat, "abcd", ["package_shape", "geometry", "inter_fidelity", "chiplet_sparsity"]
    ):
        spec = LAYOUT_SCALABILITY_SPECS[name]
        data = load_layout_scalability(eval_result_dir, name)
        positions = np.arange(len(spec["order"]))
        for index, strategy in enumerate(strategies):
            subset = data[data["DesignStrategy"].eq(strategy)].set_index(spec["axis"]).reindex(spec["order"])
            score = subset["Score"].to_numpy(dtype=float)
            if name == "geometry":
                ax.bar(
                    positions + (index - 1.5) * bar_width, score, bar_width,
                    color=SERIES_COLOR[strategy], edgecolor=COLORS["ink"],
                    linewidth=0.5, label=DISPLAY_NAME[strategy],
                )
                plot_nan_hint(ax, score)
            else:
                plot_line(ax, positions, score, strategy, style=style)
        ax.set_title(f"({panel}) {spec['title']}: layout score")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("Package score")
        ax.set_xticks(positions)
        ax.set_xticklabels(spec["tick_labels"])
        _auto_zero_ylim(ax, data["Score"])
        style_axis(ax)

    handles, labels = sweep_axes[0, 0].get_legend_handles_labels()
    _layout_legend(sweep_fig, handles, labels, style)
    apply_figure_style(sweep_fig, style)
    sweep_fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(sweep_fig, out_dir, "layout_scalability_rest", formats)


def plot_routing_scalability_main(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """总体路由图。 / Plot MECH-versus-Sensate routing results."""
    style = {
        "figsize": (14, 8.8), "title_fontsize": 16, "label_fontsize": 16,
        "tick_fontsize": 14, "legend_fontsize": 14,
        "line_width": 2, "marker_size": 9, "bar_width": 0.30,
        "baseline_markers": {"mech": "^", "sensate": "o"},
        "subplot_adjust": dict(left=0.075, right=0.925, bottom=0.10, top=0.82, hspace=0.52, wspace=0.50),
    }
    _plot_overall_panels(root, eval_result_dir, out_dir, formats, style)


def _set_ylim_from_config(ax: Axes, limits: tuple[float, float] | None) -> None:
    if limits is not None:
        ax.set_ylim(*limits)


def _auto_zero_ylim(ax: Axes, values: Iterable[float], pad: float = 1.12) -> None:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        ax.set_ylim(0, max(float(finite.max()) * pad, 1e-12))
    else:
        ax.set_ylim(0, 1)


def _compact_tick_label(value: float, _position: int) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def plot_routing_scalability_rest(root: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """Plot package-shape and routing scalability in one 2 × 5 paper figure."""
    # 本图专用参数。 / Figure-local paper layout controls.
    style = {
        "figsize": (24, 6),
        "title_fontsize": 13, "label_fontsize": 13, "tick_fontsize": 13, "legend_fontsize": 13,
        "line_width": 2.0, "marker_size": 7.5, "markers": dict(SERIES_MARKER),
        "baseline_markers": {"mech": "^", "sensate": "o"},
        "package_shape_bar_width": 0.34, "package_shape_duration_scale": "linear",
        "routing_bar_width": 0.18,
        "subplot_adjust": dict(left=0.055, right=0.985, bottom=0.11, top=0.80, wspace=0.30, hspace=0.20),
        "package_shape_legend": dict(loc="upper center", bbox_to_anchor=(0.24, 0.92), ncol=4, columnspacing=1.2, handlelength=1.35),
        "routing_legend": dict(loc="upper center", bbox_to_anchor=(0.71, 0.92), ncol=len(QUANTUM_PROGRAMS), columnspacing=1.2, handlelength=1.35),
        # None 自动缩放；元组指定范围。 / None autoscale; tuples set manual limits.
        "y_limits": {
            "package_shape": {
                "vqe": {"duration": None, "fidelity": (0, 1.02)},
                "qaoa": {"duration": None, "fidelity": (0, 1.02)},
                "bv": {"duration": None, "fidelity": (0, 1.02)},
                "s_ghz": {"duration": None, "fidelity": (0, 1.02)},
            },
            "routing": {
                "geometry": {"fidelity_improvement": None, "circuit_speedup": None},
                "inter_fidelity": {"fidelity_improvement": (1, 3.5), "circuit_speedup": (2, 8.5)},
                "chiplet_sparsity": {"fidelity_improvement": (1, 3.5), "circuit_speedup": (2, 8.5)},
            },
        },
    }
    fig, axes = plt.subplots(2, 5, figsize=style["figsize"], sharex=False)
    panel_letters = iter("abcdefghij")

    title_fontsize = style["title_fontsize"]
    label_fontsize = style["label_fontsize"]
    tick_fontsize = style["tick_fontsize"]
    marker_size = style["marker_size"]
    line_width = style["line_width"]

    package_spec = SCALABILITY_SPECS["package_shape"]
    package_axis = package_spec["axis"]
    package_x = np.arange(len(package_spec["order"]))
    package_width = style["package_shape_bar_width"]
    baseline_colors = {"mech": COLORS["sage"], "sensate": COLORS["dark_green"]}
    baseline_markers = style["baseline_markers"]
    sensate_package = load_scalability(eval_result_dir, "package_shape", package_spec)
    mech_package = load_mech_scalability(root, "package_shape", package_spec)

    for index, quantum_program in enumerate(QUANTUM_PROGRAMS):
        row, col = divmod(index, 2)
        ax = axes[row, col]
        fidelity_ax = ax.twinx()
        sensate_subset = sensate_package[sensate_package["QuantumProgram"].eq(quantum_program)].copy()
        mech_subset = mech_package[mech_package["QuantumProgram"].eq(quantum_program)].copy()
        for subset in (sensate_subset, mech_subset):
            subset[package_axis] = pd.Categorical(subset[package_axis], categories=package_spec["order"], ordered=True)
            subset.sort_values(package_axis, inplace=True)

        mech_duration = pd.to_numeric(mech_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000
        sensate_duration = pd.to_numeric(sensate_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000
        mech_fidelity = pd.to_numeric(mech_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
        sensate_fidelity = pd.to_numeric(sensate_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)

        ax.bar(
            package_x - package_width / 2,
            mech_duration,
            package_width,
            color=baseline_colors["mech"],
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            alpha=0.82,
            zorder=2,
        )
        ax.bar(
            package_x + package_width / 2,
            sensate_duration,
            package_width,
            color=baseline_colors["sensate"],
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            alpha=0.82,
            zorder=2,
        )
        fidelity_ax.plot(
            package_x,
            mech_fidelity,
            color=baseline_colors["mech"],
            marker=baseline_markers["mech"],
            markersize=marker_size,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.5,
            linewidth=line_width,
            zorder=3,
        )
        fidelity_ax.plot(
            package_x,
            sensate_fidelity,
            color=baseline_colors["sensate"],
            marker=baseline_markers["sensate"],
            markersize=marker_size,
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.5,
            linewidth=line_width,
            zorder=3,
        )

        panel = next(panel_letters)
        ax.set_title(f"({panel}) {DISPLAY_NAME[quantum_program]}", fontsize=title_fontsize)
        ax.set_xticks(package_x)
        ax.set_xticklabels(package_spec["tick_labels"], fontsize=tick_fontsize)
        if row == 1:
            ax.set_xlabel("Package shape", fontsize=label_fontsize)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.set_ylabel("Circuit duration (ms)", fontsize=label_fontsize)
        else:
            ax.set_ylabel("")
        if col == 1:
            fidelity_ax.set_ylabel("Avg. gate fidelity", fontsize=label_fontsize)
        else:
            fidelity_ax.set_ylabel("")
        ax.set_yscale(style.get("package_shape_duration_scale", "linear"))
        ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))
        ax.yaxis.get_offset_text().set_visible(False)
        limits = style["y_limits"]["package_shape"].get(quantum_program, {})
        if limits.get("duration") is None:
            _auto_zero_ylim(ax, [*mech_duration, *sensate_duration])
        else:
            _set_ylim_from_config(ax, limits.get("duration"))
        _set_ylim_from_config(fidelity_ax, limits.get("fidelity"))
        style_axis(ax)
        fidelity_ax.grid(False)
        fidelity_ax.spines["top"].set_visible(True)
        fidelity_ax.spines["right"].set_color(COLORS["ink"])
        fidelity_ax.tick_params(axis="both", colors=COLORS["ink"], labelsize=tick_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        plot_nan_hint(ax, [*mech_duration, *sensate_duration, *mech_fidelity, *sensate_fidelity])

    routing_specs = [
        (name, SCALABILITY_SPECS[name])
        for name in ["geometry", "inter_fidelity", "chiplet_sparsity"]
    ]
    for col_offset, (name, spec) in enumerate(routing_specs, start=2):
        axis_name = spec["axis"]
        sensate = load_scalability(eval_result_dir, name, spec)
        mech = load_mech_scalability(root, name, spec)
        x = np.arange(len(spec["order"]))
        numeric_trend = all(isinstance(value, (int, float, np.integer, np.floating)) for value in spec["order"])

        for row, (metric_key, ylabel) in enumerate(
            [("fidelity_improvement", "Fidelity improvement (×)"), ("circuit_speedup", "Circuit speedup (×)")]
        ):
            ax = axes[row, col_offset]
            plotted_values: list[float] = []
            for quantum_program_index, quantum_program in enumerate(QUANTUM_PROGRAMS):
                sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)].copy()
                mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)].copy()
                for subset in (sensate_subset, mech_subset):
                    subset[axis_name] = pd.Categorical(subset[axis_name], categories=spec["order"], ordered=True)
                    subset.sort_values(axis_name, inplace=True)

                if metric_key == "circuit_speedup":
                    numerator = pd.to_numeric(mech_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float)
                    denominator = pd.to_numeric(sensate_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float)
                else:
                    numerator = pd.to_numeric(sensate_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
                    denominator = pd.to_numeric(mech_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
                ratio = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan, dtype=float),
                    where=np.isfinite(denominator) & (denominator != 0),
                )
                plotted_values.extend(ratio.tolist())

                if numeric_trend:
                    ax.plot(
                        x,
                        ratio,
                        marker=style["markers"].get(quantum_program, "o"),
                        markersize=marker_size,
                        markeredgecolor=COLORS["ink"],
                        markeredgewidth=0.45,
                        linewidth=line_width,
                        color=SERIES_COLOR[quantum_program],
                        label=DISPLAY_NAME[quantum_program],
                    )
                else:
                    offset = (quantum_program_index - (len(QUANTUM_PROGRAMS) - 1) / 2) * style["routing_bar_width"]
                    ax.bar(
                        x + offset,
                        ratio,
                        width=style["routing_bar_width"],
                        color=SERIES_COLOR[quantum_program],
                        edgecolor=COLORS["ink"],
                        linewidth=0.5,
                        label=DISPLAY_NAME[quantum_program],
                    )

            panel = next(panel_letters)
            ax.set_title(f"({panel}) {spec['title']}", fontsize=title_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.set_xticks(x)
            ax.set_xticklabels(spec["tick_labels"], fontsize=tick_fontsize)
            if row == 1:
                ax.set_xlabel(spec["xlabel"], fontsize=label_fontsize)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            y_limit = style["y_limits"]["routing"].get(name, {}).get(metric_key)
            if y_limit is None:
                _auto_zero_ylim(ax, plotted_values)
            else:
                _set_ylim_from_config(ax, y_limit)
            style_axis(ax)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            plot_nan_hint(ax, plotted_values)

    package_handles = [
        Line2D([0], [0], color=baseline_colors["mech"], marker=baseline_markers["mech"], markersize=marker_size, linewidth=line_width, markeredgecolor=COLORS["ink"], label="MECH fidelity"),
        Line2D([0], [0], color=baseline_colors["sensate"], marker=baseline_markers["sensate"], markersize=marker_size, linewidth=line_width, markeredgecolor=COLORS["ink"], label="QuNoC fidelity"),
        Patch(facecolor=baseline_colors["mech"], edgecolor=COLORS["ink"], label="MECH duration"),
        Patch(facecolor=baseline_colors["sensate"], edgecolor=COLORS["ink"], label="QuNoC duration"),
    ]
    routing_handles = [
        Line2D(
            [0],
            [0],
            color=SERIES_COLOR[quantum_program],
            marker=style["markers"][quantum_program],
            markersize=marker_size,
            linewidth=line_width,
            markeredgecolor=COLORS["ink"],
            label=DISPLAY_NAME[quantum_program],
        )
        for quantum_program in QUANTUM_PROGRAMS
    ]
    fig.legend(
        handles=package_handles,
        labels=[handle.get_label() for handle in package_handles],
        fontsize=style["legend_fontsize"],
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        **style["package_shape_legend"],
    )
    fig.legend(
        handles=routing_handles,
        labels=[handle.get_label() for handle in routing_handles],
        fontsize=style["legend_fontsize"],
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        **style["routing_legend"],
    )
    fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(fig, out_dir, "routing_scalability_rest", formats)


def plot_ablation(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str], quantum_program: str) -> None:
    # 本图专用参数。 / Figure-local ablation controls.
    style = {
        "figsize": (16, 6),
        "title_fontsize": 16, "label_fontsize": 16, "tick_fontsize": 16, "legend_fontsize": 16,
        "line_width": 2.0, "marker_size": 9, "legend_marker_size": 8,
        "markers": dict(SERIES_MARKER),
        "routing_bar_width": 0.18, "operation_bar_width": 0.13,
        "subplot_adjust": dict(left=0.075, right=0.925, bottom=0.14, top=0.70, wspace=0.3),
        "routing_legend": dict(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3), columnspacing=0.9, handlelength=1.3),
        "operation_legend": dict(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.3), columnspacing=0.55, handlelength=1.3),
        "y_scales": {"routing_duration": "log", "operation_counts": "log"},
        "y_limits": {"routing_duration": None, "routing_fidelity": (0, 1.02), "operation_counts": None},
    }
    ablation = load_ablation(evaluate_dir, eval_result_dir, quantum_program)
    qunoc_ops = load_routing_package_size(eval_result_dir)
    qunoc_ops = qunoc_ops[qunoc_ops["QuantumProgram"] == quantum_program]
    qunoc_ops = last_by(qunoc_ops, ["PackageSize"])
    mech_ops = load_mech_operation_json(root, quantum_program)

    fig, (ablation_ax, operation_ax) = plt.subplots(1, 2, figsize=style["figsize"])
    x = np.arange(len(PACKAGE_SIZES))
    title_fontsize = style["title_fontsize"]
    label_fontsize = style["label_fontsize"]
    tick_fontsize = style["tick_fontsize"]
    marker_size = style["marker_size"]
    line_width = style["line_width"]

    # (a) Match the package-shape pattern: duration bars on the left axis and
    # fidelity lines on the right axis.
    fidelity_ax = ablation_ax.twinx()
    width = style["routing_bar_width"]
    ablation_duration_handles = []
    ablation_fidelity_handles = []
    ablation_labels = []
    for index, strategy in enumerate(ROUTING_STRATEGIES):
        subset = ablation[ablation["Strategy"].eq(strategy)].sort_values("PackageSize")
        duration = pd.to_numeric(subset["CircuitDurationMs"], errors="coerce").to_numpy(dtype=float)
        fidelity = pd.to_numeric(subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
        bars = ablation_ax.bar(
            x + (index - 1.5) * width,
            duration,
            width,
            color=SERIES_COLOR[strategy],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.82,
            label=DISPLAY_NAME[strategy],
        )
        line = fidelity_ax.plot(
            x,
            fidelity,
            color=SERIES_COLOR[strategy],
            marker=style["markers"][strategy],
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=line_width,
        )[0]
        ablation_duration_handles.append(bars)
        ablation_fidelity_handles.append(line)
        ablation_labels.append(DISPLAY_NAME[strategy])

    ablation_ax.set_title("(a) Routing-method ablation", fontsize=title_fontsize)
    ablation_ax.set_xlabel("Package side length", fontsize=label_fontsize)
    ablation_ax.set_ylabel("Circuit duration (ms)", fontsize=label_fontsize)
    fidelity_ax.set_ylabel("Average logical-gate fidelity", fontsize=label_fontsize)
    ablation_ax.set_xticks(x)
    ablation_ax.set_xticklabels(PACKAGE_SIZES, fontsize=tick_fontsize)
    ablation_ax.set_yscale(style["y_scales"]["routing_duration"])
    _set_ylim_from_config(ablation_ax, style["y_limits"]["routing_duration"])
    _set_ylim_from_config(fidelity_ax, style["y_limits"]["routing_fidelity"])
    style_axis(ablation_ax)
    ablation_ax.tick_params(axis="both", labelsize=tick_fontsize)
    fidelity_ax.grid(False)
    fidelity_ax.spines["top"].set_visible(True)
    fidelity_ax.spines["right"].set_color(COLORS["ink"])
    fidelity_ax.tick_params(axis="both", colors=COLORS["ink"], labelsize=tick_fontsize)
    line_handle = Line2D(
        [0],
        [0],
        color=COLORS["ink"],
        marker="o",
        markersize=style["legend_marker_size"],
        linewidth=line_width,
        label="Line: avg. fidelity",
    )
    bar_handle = Patch(
        facecolor=COLORS["light_gray"],
        edgecolor=COLORS["ink"],
        label="Bar: circuit duration",
    )
    ablation_ax.legend(
        ablation_fidelity_handles + [line_handle, bar_handle],
        ablation_labels + ["Line: avg. fidelity", "Bar: circuit duration"],
        fontsize=style["legend_fontsize"],
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        **style["routing_legend"],
    )

    # (b) Plot all operation components as grouped bars: MECH gate counts and
    # measurements, plus QuNoC gate counts and EQST distance.
    ax = operation_ax
    width = style["operation_bar_width"]
    mech_ops = mech_ops.sort_values("PackageSize")
    qunoc_ops = qunoc_ops.sort_values("PackageSize")
    mech_intra = pd.to_numeric(mech_ops["OnChip"], errors="coerce").to_numpy(dtype=float)
    mech_inter = pd.to_numeric(mech_ops["CrossChip"], errors="coerce").to_numpy(dtype=float)
    mech_meas = pd.to_numeric(mech_ops["MeasNum"], errors="coerce").to_numpy(dtype=float)
    qunoc_intra = pd.to_numeric(qunoc_ops["IntraGate"], errors="coerce").to_numpy(dtype=float)
    qunoc_inter = pd.to_numeric(qunoc_ops["InterGate"], errors="coerce").to_numpy(dtype=float)
    qunoc_qst = pd.to_numeric(qunoc_ops["QSTDist"], errors="coerce").to_numpy(dtype=float)

    if len(qunoc_intra) != len(PACKAGE_SIZES):
        tmp = complete_grid(qunoc_ops, {"PackageSize": PACKAGE_SIZES}, ["IntraGate", "InterGate", "QSTDist"])
        qunoc_intra = pd.to_numeric(tmp["IntraGate"], errors="coerce").to_numpy(dtype=float)
        qunoc_inter = pd.to_numeric(tmp["InterGate"], errors="coerce").to_numpy(dtype=float)
        qunoc_qst = pd.to_numeric(tmp["QSTDist"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - 2.5 * width, mech_intra, width, color=COLORS["dark_green"], edgecolor=COLORS["ink"], linewidth=0.5, label="MECH: intra-die gates")
    ax.bar(x - 1.5 * width, mech_inter, width, color=COLORS["teal"], edgecolor=COLORS["ink"], linewidth=0.5, label="MECH: inter-die gates")
    ax.bar(x - 0.5 * width, mech_meas, width, color=COLORS["khaki"], edgecolor=COLORS["ink"], linewidth=0.5, label="MECH: measurements")
    ax.bar(x + 0.5 * width, qunoc_intra, width, color=COLORS["sage"], edgecolor=COLORS["ink"], linewidth=0.5, label="QuNoC: intra-die gates")
    ax.bar(x + 1.5 * width, qunoc_inter, width, color=COLORS["light_gray"], edgecolor=COLORS["ink"], linewidth=0.5, label="QuNoC: inter-die gates")
    ax.bar(x + 2.5 * width, qunoc_qst, width, color=COLORS["muted"], edgecolor=COLORS["ink"], linewidth=0.5, label="QuNoC: EQST distance")

    ax.set_title(f"(b) Hardware-operation decomposition ({DISPLAY_NAME.get(quantum_program, quantum_program.upper())})", fontsize=title_fontsize)
    ax.set_xlabel("Package side length", fontsize=label_fontsize)
    ax.set_ylabel("Operation count / EQST distance", fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(PACKAGE_SIZES, fontsize=tick_fontsize)
    ax.set_yscale(style["y_scales"]["operation_counts"])
    _set_ylim_from_config(ax, style["y_limits"]["operation_counts"])
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    plot_nan_hint(ax, [*mech_intra, *mech_inter, *mech_meas, *qunoc_intra, *qunoc_inter, *qunoc_qst])

    operation_handles, operation_labels = ax.get_legend_handles_labels()
    ax.legend(
        operation_handles,
        operation_labels,
        fontsize=style["legend_fontsize"],
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        **style["operation_legend"],
    )
    fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(fig, out_dir, f"ablation_{quantum_program}", formats)


# Hamiltonian data-only plotting / Hamiltonian 数据重绘，不触发模拟。

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "evaluate/result/hamiltonian_simulation"
DEFAULT_FIGURE_ROOT = ROOT / "evaluate/figure"

def style_hamiltonian_axis(ax) -> None:
    """Purpose / 功能: Apply light single-series chart styling / 设置浅色单系列图样式。

    Inputs / 输入: Matplotlib axis / 绘图轴。
    Outputs / 输出: None; mutates axis / 无；修改轴样式。
    """
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")


def write_probability_plot(path: Path, times, ideal, noisy, lower, upper, title: str, noise: dict) -> None:
    """Purpose / 功能: Compare ideal and dynamical noise with uncertainty / 对比理想与动力学噪声及区间。

    Inputs / 输入: File stem, aligned curves, title and solver / 文件前缀、对齐曲线、标题及求解器。
    Outputs / 输出: Light PNG/PDF; CSV is supplied by caller / 浅色图像；调用方提供 CSV。
    """
    style = {
        "figsize": (8.2, 4.9), "title_fontsize": plt.rcParams["axes.titlesize"],
        "label_fontsize": plt.rcParams["axes.labelsize"],
        "tick_fontsize": plt.rcParams["xtick.labelsize"],
        "legend_fontsize": plt.rcParams["legend.fontsize"],
        "line_width": 2, "marker_size": 8, "constrained_layout": True,
    }
    import matplotlib
    matplotlib.use("Agg")

    # Palette validated with dataviz, 2026-10-22; styles also encode identity / 已校验配色并用线型区分。
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"}):
        fig, ax = plt.subplots(figsize=style["figsize"], constrained_layout=style["constrained_layout"], facecolor="#fcfcfb")
        style_hamiltonian_axis(ax)
        parameters = noise.get("parameters", {})
        noise_disabled = parameters.get("relaxation_disabled", False) and parameters.get("pure_dephasing_disabled", False)
        ax.plot(times, noisy, color="#2a78d6", linewidth=2,
                label="Noise disabled (ideal control)" if noise_disabled else "T1 + pure-dephasing dynamics")
        ax.plot(times, ideal, color="#eb6834", linewidth=2, linestyle="--", label="Ideal coherent evolution")
        if np.any(np.asarray(upper) > np.asarray(lower)):
            confidence = noise.get("confidence", noise.get("parameters", {}).get("confidence", 0.95))
            ax.fill_between(times, lower, upper, color="#2a78d6", alpha=0.16,
                            label=f"{100 * confidence:g}% pointwise confidence interval")
        for values, color, label, offset in ((ideal, "#eb6834", "Ideal", 15),
                                              (noisy, "#2a78d6", "Noisy", -32)):
            ax.plot(times[-1], values[-1], "o", color=color, markersize=8,
                    markeredgewidth=2, markeredgecolor="#fcfcfb")
            ax.annotate(f"{label}: {values[-1]:.6f}", (times[-1], values[-1]), xytext=(-12, offset),
                        textcoords="offset points", ha="right", color="#0b0b0b",
                        bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1})
        ax.set(title=title, xlabel="Time (ns)", ylabel="Target occupation probability",
               xlim=(0, max(float(times[-1]) * 1.04, 1)), ylim=(-0.025, 1.15))
        ax.legend(loc="upper left", frameon=False)
        ax.text(0.03, 0.70, f"Completion: {times[-1]:.3f} ns\n"
                f"Solver: {noise.get('solver', 'unspecified')}\n"
                f"Completion precision: {'converged' if noise.get('converged', False) else 'NOT CONVERGED'}",
                transform=ax.transAxes, va="top", color="#52514e")
        apply_figure_style(fig, style)
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(path.with_suffix(suffix), dpi=180)
        # 图旁保存实际曲线，不修改实验输入。 / Export plotted curves beside the figure only.
        pd.DataFrame({"time_ns": times, "ideal_target_fidelity": ideal,
                      "noisy_target_fidelity": noisy, "noisy_ci_lower": lower,
                      "noisy_ci_upper": upper}).to_csv(path.with_suffix(".csv"), index=False)
        plt.close(fig)


def write_length_plot(output: Path, rows: list[dict], g_max: float | None, *,
                      formats: Iterable[str] = ("png", "pdf", "svg"), export_csv: bool = True) -> None:
    """Purpose / 功能: Compare latency and noisy completion / 比较时延及有噪完成概率。

    Inputs / 输入: Directory, case rows, peak coupling (API compatibility) / 目录、案例行及峰值耦合（接口兼容）。
    Outputs / 输出: Separate latency/probability panels, PNG/PDF/SVG / 独立时延与概率面板。
    """
    style = {
        "figsize": (14, 4), "title_fontsize": 16, "label_fontsize": 16,
        "tick_fontsize": 16, "legend_fontsize": 16,
        "line_width": 4, "marker_size": 6, "marker": "o",
        "constrained_layout": True, "spine_width": 2,
    }
    import matplotlib
    matplotlib.use("Agg")

    # Explicit paper styling / 论文指定样式；仅影响链长汇总图。
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.labelsize": 16, "axes.titlesize": 16,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 16, "text.color": "black", "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=style["figsize"], constrained_layout=style["constrained_layout"], facecolor="#fcfcfb")
        x = [r["chain_length"] for r in rows]
        for ax in axes:
            style_hamiltonian_axis(ax)
            ax.set_xlabel("Chain length")
            ax.tick_params(colors="black", labelsize=16)
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(style["spine_width"])
        axes[0].plot(x, [r["final_time_ns"] for r in rows], color="#264B49", linewidth=4,
                     marker=style["marker"], markersize=style["marker_size"], markeredgewidth=0.8, markeredgecolor="black")
        axes[0].set(ylabel="Transfer latency (ns)", ylim=(0, None))
        axes[1].plot(x, [r["noisy_final_fidelity"] for r in rows], color="#A79B73", linewidth=4,
                     marker=style["marker"], markersize=style["marker_size"], markeredgewidth=0.8, markeredgecolor="black")
        axes[1].fill_between(x, [r["noisy_final_ci_lower"] for r in rows],
                            [r["noisy_final_ci_upper"] for r in rows], color="#A79B73", alpha=0.16)
        axes[1].set(ylabel="Target occupation probability", ylim=(0.97, 1.0))
        apply_figure_style(fig, style)
        for suffix in formats:
            fig.savefig(output / f"maximum_chain_length.{suffix}", dpi=180, transparent=True)
        if export_csv:
            pd.DataFrame(rows).to_csv(output / "maximum_chain_length.csv", index=False)
        plt.close(fig)


def write_optimization_plot(output: Path, rows: list[dict]) -> None:
    """Purpose / 功能: Show projected hardware sensitivity and latency / 展示预测硬件敏感性及时延。

    Inputs / 输入: Directory and complete grid rows / 目录与全部扫描行。
    Outputs / 输出: Two-panel plot, paired with full CSV / 双面板图及配套完整 CSV。
    """
    style = {
        "figsize": (11.8, 4.8), "title_fontsize": plt.rcParams["axes.titlesize"],
        "label_fontsize": plt.rcParams["axes.labelsize"],
        "tick_fontsize": plt.rcParams["xtick.labelsize"], "legend_fontsize": 8,
        "line_width": 2, "marker_size": 8, "reference_line_width": 1,
        "constrained_layout": True,
    }
    import matplotlib
    matplotlib.use("Agg")

    t1_values = sorted({r["t1_us"] for r in rows})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#0b0b0b"}):
        fig, axes = plt.subplots(1, 2, figsize=style["figsize"], constrained_layout=style["constrained_layout"], facecolor="#fcfcfb")
        for ax in axes:
            style_hamiltonian_axis(ax)
            ax.set_xlabel("Peak J/(2π) (MHz)")
        for t1, color, marker in zip(t1_values, ("#2a78d6", "#eb6834", "#1baf7a"), ("o", "s", "^"), strict=True):
            points = sorted((r for r in rows if r["t1_us"] == t1), key=lambda r: r["peak_coupling_mhz"])
            x = [r["peak_coupling_mhz"] for r in points]
            y = [r["noisy_final_fidelity"] for r in points]
            axes[1].plot(x, y, color=color, marker=marker, markersize=8, linewidth=2,
                         markeredgewidth=2, markeredgecolor="#fcfcfb", label=f"T1 = {t1:g} μs")
            axes[1].fill_between(x, [r["noisy_final_ci_lower"] for r in points],
                                [r["noisy_final_ci_upper"] for r in points], color=color, alpha=0.12)
        baseline_t1 = [r for r in rows if r["t1_us"] == t1_values[0]]
        axes[0].plot([r["peak_coupling_mhz"] for r in baseline_t1], [r["final_time_ns"] for r in baseline_t1],
                     color="#2a78d6", marker="o", markersize=8, linewidth=2,
                     markeredgewidth=2, markeredgecolor="#fcfcfb")
        axes[0].set(ylabel="Transfer latency (ns)", ylim=(0, None))
        old = rows[0]["historical_distance_envelope_final"]
        axes[1].axhline(1, color="#52514e", linestyle="--", linewidth=1, label="Historical ideal (no noise)")
        axes[1].axhline(old, color="#898781", linestyle=":", linewidth=1,
                       label="Historical distance envelope (NOT dynamics)")
        axes[1].set(ylabel="Target occupation probability", ylim=(min(old, min(r["noisy_final_ci_lower"] for r in rows)) - .055, 1.025))
        axes[1].legend(loc="lower right", fontsize=8, frameon=False)
        fig.suptitle(f"N = {rows[0]['chain_length']} single-excitation sensitivity · PROJECTED hardware\n"
                     "Baseline-fitted dephasing held fixed; no stronger-coupling hardware-error model")
        # 参考线保留独立线宽。 / Keep reference lines independently configurable.
        apply_figure_style(fig, style)
        for line in axes[1].lines[-2:]:
            line.set_linewidth(style["reference_line_width"])
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"optimization.{suffix}", dpi=180)
        pd.DataFrame(rows).to_csv(output / "optimization.csv", index=False)
        plt.close(fig)


def write_real_case_plot(output, rows, figure_root, *,
                         formats: Iterable[str] = ("png", "pdf", "svg"), export_csv: bool = True):
    """Noisy occupation only, with uncertainty / 只画含噪占据概率及区间。"""
    style = {
        "figsize": (16, 8), "title_fontsize": 16, "label_fontsize": 16,
        "tick_fontsize": 16, "legend_fontsize": 16, "text_fontsize": 16,
        "line_width": 2, "marker_size": 3.5, "marker": "o", "title_pad": 13,
        "subplot_adjust": dict(left=.07, right=.985, bottom=.12, top=.91, hspace=.59, wspace=.23),
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # User-fixed colors; rows and explicit labels redundantly encode class.
    # 用户固定配色；行位置和文字冗余标记类别，黑描边增强浅色可读性。
    with plt.rc_context({"font.size": 16, "font.family": "DejaVu Sans",
                         "axes.titlesize": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 16, "legend.title_fontsize": 16,
                         "figure.titlesize": 16, "figure.labelsize": 16,
                         "axes.linewidth": 2, "text.color": "black", "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 3, figsize=style["figsize"], sharey=True)
        fig.subplots_adjust(**style["subplot_adjust"])
        curve_tables = []
        for index, (ax, row) in enumerate(zip(axes.flat, rows)):
            for spine in ax.spines.values():
                spine.set(color="black", linewidth=2)
            ax.tick_params(width=2, colors="black")
            color = "#264B49" if row["class"] == "nonconflicting" else "#A79B73"
            pairs = ", ".join(f"{s}→{t}" for s, t in zip(row["sources"], row["targets"]))
            ax.set_title(f"({chr(97+index)}) {row['class'].capitalize()} · k={len(row['sources'])}, N={row['chain_length']}\n{pairs}", fontsize=style["title_fontsize"], pad=style["title_pad"])
            ax.set_ylim(0, 1.01)
            ax.set_xlabel("Time (ns)")
            if index % 3 == 0:
                ax.set_ylabel("Target occupation probability")
            if row["status"] == "failed":
                ax.text(.5, .5, "SIMULATION FAILED\nSee summary.json", ha="center", transform=ax.transAxes)
                continue
            curve = np.genfromtxt(output / row["directory"] / "fidelity.csv", delimiter=",", names=True)
            times, values = curve["time_ns"], curve["noisy_target_fidelity"]
            # 长表保留案例元数据及实际绘图采样。 / Long-form plotted samples with case metadata.
            table = pd.DataFrame({"time_ns": times, "noisy_target_fidelity": values,
                                  "noisy_ci_lower": curve["noisy_ci_lower"],
                                  "noisy_ci_upper": curve["noisy_ci_upper"]})
            table.insert(0, "case_index", index)
            for key, value in row.items():
                table[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
            curve_tables.append(table)
            ax.fill_between(times, curve["noisy_ci_lower"], curve["noisy_ci_upper"], color=color, alpha=.22)
            ax.plot(times, values, color=color, linewidth=style["line_width"], marker=style["marker"], markersize=style["marker_size"],
                    markeredgecolor="black", markeredgewidth=.45, markevery=max(1, len(times)//15))
            ax.set_xlim(0, times[-1])
            status = "" if row["status"] == "complete" else " · NOT CONVERGED"
            ax.text(.03, .95, f"Final = {values[-1]:.4f}{status}", va="top", transform=ax.transAxes, fontsize=16)
        apply_figure_style(fig, style)
        for extension in formats:
            fig.savefig(figure_root / f"real_case_study.{extension}", transparent=True, dpi=200)
        curves = pd.concat(curve_tables, ignore_index=True) if curve_tables else pd.DataFrame(
            columns=["case_index", "time_ns", "noisy_target_fidelity", "noisy_ci_lower", "noisy_ci_upper"])
        if export_csv:
            curves.to_csv(figure_root / "real_case_study.csv", index=False)
        plt.close(fig)


def plot_hamiltonian_main(argv=None):
    """Replot selected saved reports / 从选定的保存报告重绘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="*", default=["pdf", "png", "svg"])
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args(argv)
    data_root, figure_root = args.data_root.resolve(), args.figure_root.resolve()
    if (figure_root == data_root or figure_root.is_relative_to(data_root)
            or data_root.is_relative_to(figure_root)
            or figure_root.is_relative_to(ROOT / "evaluate/result")):
        parser.error("figure-root must be separate from data-root and evaluate/result")
    # Reject linked artifacts before overwriting figures / 覆盖图像前拒绝链接工件。
    for path in (args.figure_root, *args.figure_root.parents):
        if path.is_symlink():
            parser.error("figure-root must not contain symlinks")
    if args.figure_root.exists():
        for path in args.figure_root.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
                parser.error("figure-root must not contain linked artifacts")
    args.figure_root.mkdir(parents=True, exist_ok=True)
    for experiment in ("maximum-chain-length", "real-case-study"):
        output = args.data_root / experiment.replace("-", "_")
        try:
            report = json.loads((output / "summary.json").read_text())
            if experiment == "maximum-chain-length":
                write_length_plot(args.figure_root, report["cases"], report["requested_settings"]["g_max_rad_per_ns"],
                                  formats=args.formats, export_csv=not args.skip_csv)
            else:
                write_real_case_plot(output, report["selected_cases"], args.figure_root,
                                     formats=args.formats, export_csv=not args.skip_csv)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
    return 0


# ----------------------------- entry point -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate evaluation figures from existing CSV/JSON results.")
    parser.add_argument("--root", type=Path, default=default_root, help="Repository root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory. Default: evaluate/figure.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Output formats, e.g. pdf svg png.")
    parser.add_argument("--ablation-quantum-program", choices=QUANTUM_PROGRAMS, default="qaoa", help="Quantum program used for ablation operation decomposition.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not export per-figure CSV data tables.")
    parser.add_argument("--only-csv", action="store_true", help="Export per-figure CSV data tables without rendering figures.")
    parser.add_argument("--data-root", type=Path, default=None, help="Hamiltonian data root.")
    parser.add_argument("--figure-root", type=Path, default=None, help="Hamiltonian figure root.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    evaluate_dir = root / "evaluate"
    eval_result_dir = evaluate_dir / "result"
    out_dir = args.output_dir.resolve() if args.output_dir else evaluate_dir / "figure"

    saved_csvs = []
    if not args.skip_csv:
        saved_csvs = export_all_figure_csv(
            root,
            evaluate_dir,
            eval_result_dir,
            out_dir,
            args.ablation_quantum_program,
        )

    if not args.only_csv:
        setup_matplotlib()
        plot_layout_scalability_main(eval_result_dir, out_dir, args.formats)
        plot_layout_scalability_rest(eval_result_dir, out_dir, args.formats)
        plot_routing_scalability_main(root, evaluate_dir, eval_result_dir, out_dir, args.formats)
        plot_routing_scalability_rest(root, eval_result_dir, out_dir, args.formats)
        plot_ablation(root, evaluate_dir, eval_result_dir, out_dir, args.formats, args.ablation_quantum_program)

    # 默认包含两项 Hamiltonian 图，并共享导出选项。 / Include Hamiltonian with shared export controls.
    hamiltonian_dir = args.figure_root.resolve() if args.figure_root else out_dir
    hamiltonian_options = [
        "--data-root", str(args.data_root or eval_result_dir / "hamiltonian_simulation"),
        "--figure-root", str(hamiltonian_dir),
        "--formats", *([] if args.only_csv else args.formats),
    ]
    if args.skip_csv:
        hamiltonian_options.append("--skip-csv")
    plot_hamiltonian_main(hamiltonian_options)
    if not args.skip_csv:
        saved_csvs.extend(hamiltonian_dir / f"{stem}.csv"
                          for stem in ("maximum_chain_length", "real_case_study"))

    if not args.only_csv:
        print(f"Saved figures to {out_dir}")
        if hamiltonian_dir != out_dir:
            print(f"Saved Hamiltonian figures to {hamiltonian_dir}")
    if saved_csvs:
        print("Saved figure CSV data:")
        for path in saved_csvs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
