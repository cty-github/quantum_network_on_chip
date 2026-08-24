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

# Central knobs for the 2 × 5 combined scalability figure.  Edit this block to
# tune the final paper layout without touching the plotting logic below.
COMBINED_SCALABILITY_STYLE = {
    "figsize": (24, 6),
    "title_fontsize": 13,
    "label_fontsize": 13,
    "tick_fontsize": 13,
    "legend_fontsize": 13,
    "line_width": 2.0,
    "marker_size": 7.5,
    "package_shape_bar_width": 0.34,
    "package_shape_duration_scale": "linear",  # Use "log" only if the duration range becomes too large.
    "routing_bar_width": 0.18,
    "subplot_adjust": {
        "left": 0.055,
        "right": 0.985,
        "bottom": 0.11,
        "top": 0.80,
        "wspace": 0.30,
        "hspace": 0.20,
    },
    "package_shape_legend": {
        "loc": "upper center",
        "bbox_to_anchor": (0.24, 0.92),
        "ncol": 4,
        "columnspacing": 1.2,
        "handlelength": 1.35,
    },
    "routing_legend": {
        "loc": "upper center",
        "bbox_to_anchor": (0.71, 0.92),
        "ncol": len(QUANTUM_PROGRAMS),
        "columnspacing": 1.2,
        "handlelength": 1.35,
    },
    # Per-subplot y-axis ranges. Use None for automatic limits, or (low, high)
    # for manual limits. Package-shape panels have separate duration/fidelity axes.
    "y_limits": {
        "package_shape": {
            "vqe": {"duration": None, "fidelity": (0, 1.02)},
            "qaoa": {"duration": None, "fidelity": (0, 1.02)},
            "bv": {"duration": None, "fidelity": (0, 1.02)},
            "s_ghz": {"duration": None, "fidelity": (0, 1.02)},
        },
        "routing": {
            "geometry": {"fidelity_improvement": None, "circuit_speedup": None},
            "inter_fidelity": {"fidelity_improvement": (2, 3), "circuit_speedup": (2.5, 5)},
            "chiplet_sparsity": {"fidelity_improvement": (2.3, 2.9), "circuit_speedup": (1, 5)},
        },
    },
}

# Central knobs for ablation_qaoa (and any other --ablation-quantum-program).
# Edit this block to tune fonts, spacing, legends, axes, and mark sizes.
ABLATION_STYLE = {
    "figsize": (16, 6),
    "title_fontsize": 16,
    "label_fontsize": 16,
    "tick_fontsize": 14,
    "legend_fontsize": 14,
    "line_width": 2.0,
    "marker_size": 9,
    "legend_marker_size": 8,
    "routing_bar_width": 0.18,
    "operation_bar_width": 0.18,
    "subplot_adjust": {
        "left": 0.075,
        "right": 0.925,
        "bottom": 0.14,
        "top": 0.70,
        "wspace": 0.3,
    },
    "routing_legend": {
        "ncol": 3,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.3),
        "columnspacing": 0.9,
        "handlelength": 1.3,
    },
    "operation_legend": {
        "ncol": 3,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.3),
        "columnspacing": 0.55,
        "handlelength": 1.3,
    },
    "y_scales": {
        "routing_duration": "log",
        "operation_counts": "log",
        "qst_distance": "log",
    },
    # Use None for automatic limits, or (low, high) for manual limits.
    "y_limits": {
        "routing_duration": None,
        "routing_fidelity": (0, 1.02),
        "operation_counts": None,
        "qst_distance": None,
    },
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


def plot_line(ax: Axes, x, y, key: str, label: str | None = None, dashed: bool = False) -> None:
    ax.plot(
        x,
        y,
        marker=SERIES_MARKER.get(key, "o"),
        markersize=10,
        markeredgecolor=COLORS["ink"],
        markeredgewidth=0.45,
        linewidth=2.0,
        linestyle="--" if dashed else "-",
        color=SERIES_COLOR.get(key, COLORS["dark_green"]),
        label=label or DISPLAY_NAME.get(key, key),
    )
    plot_nan_hint(ax, y)


# ----------------------------- data loaders -----------------------------

def load_layout_package_size(eval_result_dir: Path) -> pd.DataFrame:
    path = eval_result_dir / "layout_scalability/package_size/output.csv"
    df = read_csv(path)
    if df.empty:
        return complete_grid(
            pd.DataFrame(),
            {"DesignStrategy": LAYOUT_STRATEGIES, "PackageSize": PACKAGE_SIZES},
            ["Fidelity", "FideStd", "Score"],
        )
    df = coerce_numeric(df, ["PackageSize", "Fidelity", "FideStd", "Score"])
    df = last_by(df, ["DesignStrategy", "PackageSize"])
    df = df[df["DesignStrategy"].isin(LAYOUT_STRATEGIES)]
    return complete_grid(
        df,
        {"DesignStrategy": LAYOUT_STRATEGIES, "PackageSize": PACKAGE_SIZES},
        ["Fidelity", "FideStd", "Score"],
    )


def load_layout_package_shape(eval_result_dir: Path) -> pd.DataFrame:
    path = eval_result_dir / "layout_scalability/package_shape/output.csv"
    package_x_lengths = [1000, 500, 250, 200, 125, 100]
    df = read_csv(path)
    if df.empty:
        return complete_grid(
            pd.DataFrame(),
            {"DesignStrategy": LAYOUT_STRATEGIES, "PackageXLength": package_x_lengths},
            ["Fidelity", "FideStd", "Score"],
        )
    df = coerce_numeric(df, ["PackageXLength", "Fidelity", "FideStd", "Score"])
    df = last_by(df, ["DesignStrategy", "PackageXLength"])
    df = df[df["DesignStrategy"].isin(LAYOUT_STRATEGIES)]
    return complete_grid(
        df,
        {"DesignStrategy": LAYOUT_STRATEGIES, "PackageXLength": package_x_lengths},
        ["Fidelity", "FideStd", "Score"],
    )


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
    result_dir = root / "evaluate" / "result" / "mechbaseline" / name
    path = result_dir / "output.csv"
    if not path.exists():
        result_dir = root / "result/routing_scalability/mech_baseline" / name
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

    Canonical inputs are read from result/mechbaseline/package_size,
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
    baseline_dir = root / "evaluate" / "result" / "mechbaseline" / "package_size"
    if not baseline_dir.exists():
        baseline_dir = root / "result/routing_scalability/mech_baseline/package_size"
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
                    "Figure": "overall_results",
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
    package_size = load_layout_package_size(eval_result_dir)
    rows = []
    for strategy in ["rl", "greedy", "heuristic", "template"]:
        subset = package_size[package_size["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        for _, row in subset.iterrows():
            rows.append(
                {
                    "Figure": "layout_scalability",
                    "Panel": "layout_score_and_fidelity",
                    "DesignStrategy": strategy,
                    "DesignStrategyLabel": DISPLAY_NAME.get(strategy, strategy),
                    "PackageSize": int(row["PackageSize"]),
                    "PackageLabel": f"{int(row['PackageSize'])}x{int(row['PackageSize'])}",
                    "Score": row.get("Score", np.nan),
                    "Fidelity": row.get("Fidelity", np.nan),
                    "FideStd": row.get("FideStd", np.nan),
                }
            )
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
        package_shape["Figure"] = "combined_scalability"
        package_shape["PanelGroup"] = "package_shape_scalability"
        package_shape["Sweep"] = "package_shape"
        package_shape["SweepTitle"] = SCALABILITY_SPECS["package_shape"]["title"]
        package_shape["RowMetric"] = "duration_and_fidelity"

    routing = build_routing_scalability_csv(root, eval_result_dir).copy()
    if not routing.empty:
        routing["Figure"] = "combined_scalability"
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
    return [
        export_figure_csv(build_overall_results_csv(root, eval_result_dir), out_dir, "overall_results"),
        export_figure_csv(build_layout_scalability_csv(eval_result_dir), out_dir, "layout_scalability"),
        export_figure_csv(build_package_shape_scalability_csv(root, eval_result_dir), out_dir, "package_shape_scalability"),
        export_figure_csv(build_routing_scalability_csv(root, eval_result_dir), out_dir, "routing_scalability"),
        export_figure_csv(build_combined_scalability_csv(root, eval_result_dir), out_dir, "combined_scalability"),
        export_figure_csv(build_ablation_csv(root, evaluate_dir, eval_result_dir, quantum_program), out_dir, f"ablation_{quantum_program}"),
    ]


# ----------------------------- figure builders -----------------------------

def plot_overall(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """Plot MECH-versus-Sensate routing results for each quantum program."""
    package_sizes = [40, 60, 80, 100, 120]
    sensate = load_routing_package_size(eval_result_dir)
    mech = load_mech_scalability(
        root,
        "package_size",
        {"axis": "PackageSize", "order": package_sizes},
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharex=False)
    x = np.arange(len(package_sizes))
    width = 0.30
    baseline_colors = {"mech": COLORS["sage"], "sensate": COLORS["dark_green"]}
    baseline_markers = {"mech": "^", "sensate": "o"}
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
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.10, top=0.82, hspace=0.52, wspace=0.50)
    save_figure(fig, out_dir, "overall_results", formats)


def plot_layout_scalability(eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """Plot package-size layout scalability for score, fidelity, and fidelity std."""
    package_size = load_layout_package_size(eval_result_dir)
    strategies = ["rl", "greedy", "heuristic", "template"]
    x = np.arange(len(PACKAGE_SIZES))
    bar_width = 0.18

    fig, (score_ax, fidelity_ax) = plt.subplots(1, 2, figsize=(14, 5.2))

    for strategy in strategies:
        subset = package_size[package_size["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        score = pd.to_numeric(subset["Score"], errors="coerce").to_numpy(dtype=float)
        plot_line(score_ax, x, score, strategy)

    score_ax.set_title("(a) Layout score scalability")
    score_ax.set_xlabel("Package size")
    score_ax.set_ylabel("Package score")
    score_ax.set_xticks(x)
    score_ax.set_xticklabels([f"{size}×{size}" for size in PACKAGE_SIZES])
    score_ax.set_ylim(bottom=0)
    style_axis(score_ax)

    std_ax = fidelity_ax.twinx()
    fidelity_ax.set_zorder(std_ax.get_zorder() + 1)
    fidelity_ax.patch.set_visible(False)
    fidelity_handles = []
    fidelity_labels = []
    for index, strategy in enumerate(strategies):
        subset = package_size[package_size["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=PACKAGE_SIZES, ordered=True)
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
            marker=SERIES_MARKER[strategy],
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

    fidelity_ax.set_title("(b) Layout fidelity scalability")
    fidelity_ax.set_xlabel("Package size")
    fidelity_ax.set_ylabel("Avg. fidelity")
    std_ax.set_ylabel("Fidelity std")
    fidelity_ax.set_xticks(x)
    fidelity_ax.set_xticklabels([f"{size}×{size}" for size in PACKAGE_SIZES])
    fidelity_ax.set_ylim(0.35, 0.88)
    std_ax.set_ylim(0, 0.30)
    style_axis(fidelity_ax)
    std_ax.grid(False)
    std_ax.tick_params(axis="both", colors=COLORS["ink"])
    for spine in std_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["ink"])
        spine.set_linewidth(1.0)

    strategy_handles, strategy_labels = score_ax.get_legend_handles_labels()
    line_handle = Line2D([0], [0], color=COLORS["ink"], marker="o", markersize=8, linewidth=2, label="Line: score / avg. fidelity")
    bar_handle = Patch(facecolor=COLORS["light_gray"], edgecolor=COLORS["ink"], label="Bar: fidelity std")
    fig.legend(
        strategy_handles + [line_handle, bar_handle],
        strategy_labels + ["Line: score / avg. fidelity", "Bar: fidelity std"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        columnspacing=1.3,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.18, top=0.76, wspace=0.45)
    save_figure(fig, out_dir, "layout_scalability", formats)


def plot_package_shape_scalability(
    root: Path,
    eval_result_dir: Path,
    out_dir: Path,
    formats: list[str],
) -> None:
    """Plot one MECH-versus-Sensate package-shape panel per quantum program."""
    spec = SCALABILITY_SPECS["package_shape"]
    axis = spec["axis"]
    sensate = load_scalability(eval_result_dir, "package_shape", spec)
    mech = load_mech_scalability(root, "package_shape", spec)
    x = np.arange(len(spec["order"]))
    width = 0.34
    baseline_colors = {"mech": COLORS["sage"], "sensate": COLORS["dark_green"]}
    baseline_markers = {"mech": "^", "sensate": "o"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=True)
    for index, (ax, quantum_program) in enumerate(zip(axes.flat, QUANTUM_PROGRAMS)):
        fidelity_ax = ax.twinx()
        sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)].copy()
        mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)].copy()
        for subset in (sensate_subset, mech_subset):
            subset[axis] = pd.Categorical(subset[axis], categories=spec["order"], ordered=True)
            subset.sort_values(axis, inplace=True)

        mech_duration = pd.to_numeric(mech_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000
        sensate_duration = pd.to_numeric(sensate_subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float) / 1_000_000
        mech_fidelity = pd.to_numeric(mech_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)
        sensate_fidelity = pd.to_numeric(sensate_subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)

        mech_bars = ax.bar(
            x - width / 2,
            mech_duration,
            width,
            color=baseline_colors["mech"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.82,
            label="MECH",
        )
        sensate_bars = ax.bar(
            x + width / 2,
            sensate_duration,
            width,
            color=baseline_colors["sensate"],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.82,
            label="Sensate",
        )
        mech_line = fidelity_ax.plot(
            x,
            mech_fidelity,
            color=baseline_colors["mech"],
            marker=baseline_markers["mech"],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
        )[0]
        sensate_line = fidelity_ax.plot(
            x,
            sensate_fidelity,
            color=baseline_colors["sensate"],
            marker=baseline_markers["sensate"],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
        )[0]

        panel = chr(ord("a") + index)
        ax.set_title(f"({panel}) {DISPLAY_NAME[quantum_program]}")
        ax.set_ylabel("Circuit duration (ms)")
        fidelity_ax.set_ylabel("Average logical-gate fidelity")
        ax.set_xticks(x)
        ax.set_xticklabels(spec["tick_labels"])
        ax.set_xlabel("Package shape")
        ax.set_yscale("log")
        fidelity_ax.set_ylim(0, 1.02)
        style_axis(ax)
        fidelity_ax.grid(False)
        fidelity_ax.spines["top"].set_visible(True)
        fidelity_ax.spines["right"].set_color(COLORS["ink"])
        fidelity_ax.tick_params(axis="both", colors=COLORS["ink"])
        plot_nan_hint(ax, [*mech_duration, *sensate_duration])
        ax.legend(
            [mech_line, sensate_line, mech_bars, sensate_bars],
            ["MECH fidelity", "Sensate fidelity", "MECH duration", "Sensate duration"],
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            frameon=True,
            edgecolor=COLORS["ink"],
            facecolor=COLORS["surface"],
            fancybox=False,
            columnspacing=1.2,
        )

    fig.tight_layout(h_pad=4.0, w_pad=2.4)
    save_figure(fig, out_dir, "package_shape_scalability", formats)


def plot_scalability(root: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """Plot MECH/Sensate ratios for the remaining scalability sweeps."""
    specs = [(name, spec) for name, spec in SCALABILITY_SPECS.items() if name != "package_shape"]
    fig, axes = plt.subplots(len(specs), 2, figsize=(14, 11.5))

    for row, (name, spec) in enumerate(specs):
        axis = spec["axis"]
        sensate = load_scalability(eval_result_dir, name, spec)
        mech = load_mech_scalability(root, name, spec)
        x = np.arange(len(spec["order"]))
        numeric_trend = all(isinstance(value, (int, float, np.integer, np.floating)) for value in spec["order"])

        for col, (metric, ylabel) in enumerate(
            [("speedup", "Circuit speedup"), ("improvement", "Fidelity improvement")]
        ):
            ax = axes[row, col]
            bar_width = 0.18
            for quantum_program_index, quantum_program in enumerate(QUANTUM_PROGRAMS):
                sensate_subset = sensate[sensate["QuantumProgram"].eq(quantum_program)].copy()
                mech_subset = mech[mech["QuantumProgram"].eq(quantum_program)].copy()
                for subset in (sensate_subset, mech_subset):
                    subset[axis] = pd.Categorical(subset[axis], categories=spec["order"], ordered=True)
                    subset.sort_values(axis, inplace=True)

                if metric == "speedup":
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

                if numeric_trend:
                    plot_line(ax, x, ratio, quantum_program)
                else:
                    offset = (quantum_program_index - (len(QUANTUM_PROGRAMS) - 1) / 2) * bar_width
                    ax.bar(
                        x + offset,
                        ratio,
                        width=bar_width,
                        color=SERIES_COLOR[quantum_program],
                        edgecolor="black",
                        linewidth=0.5,
                        label=DISPLAY_NAME[quantum_program],
                    )
                    plot_nan_hint(ax, ratio)

            panel = chr(ord("a") + row * 2 + col)
            ax.set_title(f"({panel}) {spec['title']}: {ylabel.lower()}")
            ax.set_ylabel(ylabel + " (×)")
            ax.set_xlabel(spec["xlabel"])
            ax.set_xticks(x)
            ax.set_xticklabels(spec["tick_labels"])
            ax.set_ylim(bottom=0)
            style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(QUANTUM_PROGRAMS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=2.0)
    save_figure(fig, out_dir, "routing_scalability", formats)


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


def plot_combined_scalability(root: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    """Plot package-shape and routing scalability in one 2 × 5 paper figure."""
    style = COMBINED_SCALABILITY_STYLE
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
    baseline_markers = {"mech": "^", "sensate": "o"}
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
                        marker=SERIES_MARKER.get(quantum_program, "o"),
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
            marker=SERIES_MARKER[quantum_program],
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
    save_figure(fig, out_dir, "combined_scalability", formats)


def plot_ablation(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str], quantum_program: str) -> None:
    style = ABLATION_STYLE
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
            marker=SERIES_MARKER[strategy],
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

    # (b) Two-qubit-gate counts share the left log axis; EQST distance is a
    # line on the right log axis.
    ax = operation_ax
    width = style["operation_bar_width"]
    mech_intra = pd.to_numeric(mech_ops["OnChip"], errors="coerce").to_numpy(dtype=float)
    mech_inter = pd.to_numeric(mech_ops["CrossChip"], errors="coerce").to_numpy(dtype=float)
    qunoc_intra = pd.to_numeric(qunoc_ops.sort_values("PackageSize")["IntraGate"], errors="coerce").to_numpy(dtype=float)
    qunoc_inter = pd.to_numeric(qunoc_ops.sort_values("PackageSize")["InterGate"], errors="coerce").to_numpy(dtype=float)

    if len(qunoc_intra) != len(PACKAGE_SIZES):
        tmp = complete_grid(qunoc_ops, {"PackageSize": PACKAGE_SIZES}, ["IntraGate", "InterGate", "QSTDist"])
        qunoc_intra = pd.to_numeric(tmp["IntraGate"], errors="coerce").to_numpy(dtype=float)
        qunoc_inter = pd.to_numeric(tmp["InterGate"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - 1.5 * width, mech_intra, width, color=COLORS["dark_green"], edgecolor=COLORS["ink"], linewidth=0.5, label="MECH: intra-die gates")
    ax.bar(x - 0.5 * width, mech_inter, width, color=COLORS["teal"], edgecolor=COLORS["ink"], linewidth=0.5, label="MECH: inter-die gates")
    ax.bar(x + 0.5 * width, qunoc_intra, width, color=COLORS["sage"], edgecolor=COLORS["ink"], linewidth=0.5, label="QuNoC: intra-die gates")
    ax.bar(x + 1.5 * width, qunoc_inter, width, color=COLORS["light_gray"], edgecolor=COLORS["ink"], linewidth=0.5, label="QuNoC: inter-die gates")
    qst_ax = ax.twinx()
    qst = pd.to_numeric(qunoc_ops.sort_values("PackageSize")["QSTDist"], errors="coerce").to_numpy(dtype=float)
    if len(qst) != len(PACKAGE_SIZES):
        tmp = complete_grid(qunoc_ops, {"PackageSize": PACKAGE_SIZES}, ["IntraGate", "InterGate", "QSTDist"])
        qst = pd.to_numeric(tmp["QSTDist"], errors="coerce").to_numpy(dtype=float)
    qst_line = qst_ax.plot(
        x,
        qst,
        color=COLORS["khaki"],
        marker="D",
        markersize=marker_size,
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=line_width,
        label="QuNoC: EQST distance",
    )[0]

    ax.set_title(f"(b) Hardware-operation decomposition ({DISPLAY_NAME.get(quantum_program, quantum_program.upper())})", fontsize=title_fontsize)
    ax.set_xlabel("Package side length", fontsize=label_fontsize)
    ax.set_ylabel("Number of two-qubit operations", fontsize=label_fontsize)
    qst_ax.set_ylabel("Total EQST distance", fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(PACKAGE_SIZES, fontsize=tick_fontsize)
    ax.set_yscale(style["y_scales"]["operation_counts"])
    qst_ax.set_yscale(style["y_scales"]["qst_distance"])
    _set_ylim_from_config(ax, style["y_limits"]["operation_counts"])
    _set_ylim_from_config(qst_ax, style["y_limits"]["qst_distance"])
    style_axis(ax)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    qst_ax.grid(False)
    qst_ax.spines["top"].set_visible(True)
    qst_ax.spines["right"].set_color(COLORS["ink"])
    qst_ax.tick_params(axis="both", colors=COLORS["ink"], labelsize=tick_fontsize)
    plot_nan_hint(ax, [*mech_intra, *mech_inter, *qunoc_intra, *qunoc_inter, *qst])

    operation_handles, operation_labels = ax.get_legend_handles_labels()
    ax.legend(
        operation_handles + [qst_line],
        operation_labels + ["QuNoC: EQST distance"],
        fontsize=style["legend_fontsize"],
        frameon=True,
        edgecolor=COLORS["ink"],
        facecolor=COLORS["surface"],
        fancybox=False,
        **style["operation_legend"],
    )
    fig.subplots_adjust(**style["subplot_adjust"])
    save_figure(fig, out_dir, f"ablation_{quantum_program}", formats)


# ----------------------------- entry point -----------------------------

def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate evaluation figures from existing CSV/JSON results.")
    parser.add_argument("--root", type=Path, default=default_root, help="Repository root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory. Default: evaluate/figure.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Output formats, e.g. pdf svg png.")
    parser.add_argument("--ablation-quantum-program", choices=QUANTUM_PROGRAMS, default="qaoa", help="Quantum program used for ablation operation decomposition.")
    parser.add_argument("--skip-csv", action="store_true", help="Do not export per-figure CSV data tables.")
    parser.add_argument("--only-csv", action="store_true", help="Export per-figure CSV data tables without rendering figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        plot_overall(root, evaluate_dir, eval_result_dir, out_dir, args.formats)
        plot_layout_scalability(eval_result_dir, out_dir, args.formats)
        plot_package_shape_scalability(root, eval_result_dir, out_dir, args.formats)
        plot_scalability(root, eval_result_dir, out_dir, args.formats)
        plot_combined_scalability(root, eval_result_dir, out_dir, args.formats)
        plot_ablation(root, evaluate_dir, eval_result_dir, out_dir, args.formats, args.ablation_quantum_program)

    if not args.only_csv:
        print(f"Saved figures to {out_dir}")
    if saved_csvs:
        print("Saved figure CSV data:")
        for path in saved_csvs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
