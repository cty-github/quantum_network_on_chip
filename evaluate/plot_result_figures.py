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
    python evaluate/plot_result_figures.py --ablation-algorithm vqe
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
from matplotlib.patches import Patch


# ----------------------------- configuration -----------------------------

PACKAGE_SIZES = [40, 60, 80, 100, 120]
ALGORITHMS = ["vqe", "qft", "qaoa", "bv"]
LAYOUT_STRATEGIES = ["template", "heuristic", "greedy", "rl"]
PRIMARY_LAYOUT_STRATEGY = "greedy"
ROUTING_STRATEGIES = ["mech", "mech_eqst", "qunoc_wo_opt", "qunoc"]

SCALABILITY_SPECS = {
    "package_shape": {
        "csv": "package_shape/output.csv",
        "axis": "PackageXLength",
        "order": [10, 20, 40, 50, 80, 100],
        "xlabel": "Package shape (x × 10000/x)",
        "tick_labels": ["10×1000", "20×500", "40×250", "50×200", "80×125", "100×100"],
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
    "qft": COLORS["teal"],
    "qaoa": COLORS["sage"],
    "bv": COLORS["khaki"],
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
    "qft": "s",
    "qaoa": "^",
    "bv": "D",
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
    "qft": "QFT",
    "qaoa": "QAOA",
    "bv": "BV",
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
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["secondary"],
            "ytick.color": COLORS["secondary"],
            "text.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
    # Some generated CSV files have duplicate Algorithm headers. Pandas mangles
    # the second one to Algorithm.1; keep the first canonical column.
    duplicate_algorithm_cols = [c for c in df.columns if c.startswith("Algorithm.")]
    if duplicate_algorithm_cols:
        df = df.drop(columns=duplicate_algorithm_cols)
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
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(COLORS["axis"])


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


def load_routing_package_size(eval_result_dir: Path) -> pd.DataFrame:
    path = eval_result_dir / "routing_scalability/package_size/output.csv"
    df = read_csv(path)
    if df.empty:
        return complete_grid(
            pd.DataFrame(),
            {"LayoutDesignStrategy": LAYOUT_STRATEGIES, "PackageSize": PACKAGE_SIZES, "Algorithm": ALGORITHMS},
            ["GateFidelity", "CircuitDuration", "IntraGate", "InterGate", "QSTDist"],
        )
    df = coerce_numeric(
        df,
        ["PackageSize", "#DataQubit", "#LogicGate", "CircuitDuration", "GateFidelity", "IntraGate", "InterGate", "QSTDist"],
    )
    df = last_by(df, ["LayoutDesignStrategy", "PackageSize", "Algorithm"])
    df = df[df["LayoutDesignStrategy"].isin(LAYOUT_STRATEGIES)]
    df = df[df["Algorithm"].isin(ALGORITHMS)]
    return complete_grid(
        df,
        {"LayoutDesignStrategy": LAYOUT_STRATEGIES, "PackageSize": PACKAGE_SIZES, "Algorithm": ALGORITHMS},
        ["GateFidelity", "CircuitDuration", "IntraGate", "InterGate", "QSTDist"],
    )


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
            {"PackageSize": PACKAGE_SIZES, "Algorithm": ["vqe", "qft", "qaoa"]},
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
    df = last_by(df, ["PackageSize", "Algorithm"])
    return complete_grid(
        df,
        {"PackageSize": PACKAGE_SIZES, "Algorithm": sorted(df["Algorithm"].dropna().unique().tolist()) or ["vqe", "qft", "qaoa"]},
        ["MECHGateFidelity", "QuNoCGateFidelity", "MECHCircuitDuration", "QuNoCCircuitDuration"],
    )


def load_scalability(eval_result_dir: Path, name: str, spec: dict) -> pd.DataFrame:
    df = read_csv(eval_result_dir / "routing_scalability" / spec["csv"])
    axis = spec["axis"]
    values = ["CircuitDuration", "GateFidelity", "IntraGate", "InterGate", "QSTDist"]
    if df.empty:
        return complete_grid(pd.DataFrame(), {axis: spec["order"], "Algorithm": ALGORITHMS}, values)
    # Numeric axes should be coerced, but categorical axes such as geometry
    # must remain strings so they merge cleanly with the expected grid.
    numeric_axis = all(isinstance(v, (int, float, np.integer, np.floating)) for v in spec["order"])
    df = coerce_numeric(df, ([axis] if numeric_axis else []) + values)
    df = last_by(df, [axis, "Algorithm"])
    df = df[df["Algorithm"].isin(ALGORITHMS)]
    return complete_grid(df, {axis: spec["order"], "Algorithm": ALGORITHMS}, values)


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
            for algorithm in ALGORITHMS:
                if algorithm not in raw:
                    continue
                result = raw[algorithm]
                rows.append(
                    {
                        axis: x_length,
                        "Algorithm": algorithm,
                        "CircuitDuration": get_mech_circuit_duration(result),
                        "GateFidelity": get_mech_logic_gate_fide(result, chiplet_size, fab_model),
                    }
                )
        df = pd.DataFrame(rows)

    if df.empty:
        return complete_grid(pd.DataFrame(), {axis: spec["order"], "Algorithm": ALGORITHMS}, values)
    numeric_axis = all(isinstance(v, (int, float, np.integer, np.floating)) for v in spec["order"])
    df = coerce_numeric(df, ([axis] if numeric_axis else []) + values)
    if "Baseline" in df.columns:
        df = df[df["Baseline"].astype(str).str.lower().eq("mech")]
    df = last_by(df, [axis, "Algorithm"])
    df = df[df["Algorithm"].isin(ALGORITHMS)]
    return complete_grid(df, {axis: spec["order"], "Algorithm": ALGORITHMS}, values)


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


def load_ablation(evaluate_dir: Path, eval_result_dir: Path, algorithm: str) -> pd.DataFrame:
    """Load routing ablation data.

    Canonical inputs are read from result/mechbaseline/package_size,
    result/routing_scalability/package_size and result/ablation/package_size.
    Historical main_exp.csv remains a fallback for old result directories.
    """

    rows: list[dict] = []
    main_df = load_main_exp(evaluate_dir)
    main_df = main_df[main_df["Algorithm"] == algorithm]
    mech_df = load_mech_scalability(
        evaluate_dir.parent,
        "package_size",
        {"axis": "PackageSize", "order": PACKAGE_SIZES},
    )
    mech_df = mech_df[mech_df["Algorithm"] == algorithm]
    routing_df = load_routing_package_size(eval_result_dir)
    routing_df = routing_df[
        routing_df["LayoutDesignStrategy"].eq(PRIMARY_LAYOUT_STRATEGY)
        & routing_df["Algorithm"].eq(algorithm)
    ]

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
    if "Algorithm" in optional.columns:
        optional = optional[optional["Algorithm"].astype(str).str.lower() == algorithm]
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


def load_mech_operation_json(root: Path, algorithm: str) -> pd.DataFrame:
    baseline_dir = root / "evaluate" / "result" / "mechbaseline" / "package_size"
    if not baseline_dir.exists():
        baseline_dir = root / "result/routing_scalability/mech_baseline/package_size"
    rows = []
    for size in PACKAGE_SIZES:
        path = baseline_dir / f"{size}_mech.json"
        if not path.exists():
            rows.append({"PackageSize": size, "OnChip": np.nan, "CrossChip": np.nan, "MeasNum": np.nan})
            continue
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        result = data.get(algorithm, {})
        rows.append(
            {
                "PackageSize": size,
                "OnChip": result.get("on_chip", np.nan),
                "CrossChip": result.get("cross_chip", np.nan),
                "MeasNum": result.get("meas_num", np.nan),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------- figure builders -----------------------------

def plot_overall(evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str]) -> None:
    layout = load_layout_package_size(eval_result_dir)
    routing = load_routing_package_size(eval_result_dir)

    # Match the visual language used by plot_results.ipynb.
    strategies = ["rl", "greedy", "template"]
    colors = {"rl": "#A79B73", "greedy": "#85BAA5", "template": "#4C7F79"}
    markers = {"rl": "D", "greedy": "^", "template": "s"}
    labels = {"rl": "RL-based Design", "greedy": "Greedy Design", "template": "Template"}
    x = np.arange(len(PACKAGE_SIZES))
    bar_width = 0.22

    fig, (layout_ax, routing_ax) = plt.subplots(1, 2, figsize=(14, 4.8))

    # (a) Layout evaluation: average fidelity as lines, fidelity std as bars.
    layout_std_ax = layout_ax.twinx()
    layout_handles = []
    layout_labels = []
    for index, strategy in enumerate(strategies):
        subset = layout[layout["DesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        fidelity = pd.to_numeric(subset["Fidelity"], errors="coerce").to_numpy(dtype=float)
        fidelity_std = pd.to_numeric(subset["FideStd"], errors="coerce").to_numpy(dtype=float)

        line = layout_ax.plot(
            x,
            fidelity,
            color=colors[strategy],
            marker=markers[strategy],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
            label=labels[strategy],
            zorder=3,
        )[0]
        bars = layout_std_ax.bar(
            x + (index - 1) * bar_width,
            fidelity_std,
            width=bar_width,
            color=colors[strategy],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.72,
            label=f"{labels[strategy]} std",
            zorder=2,
        )
        layout_handles.append(line)
        layout_labels.append(labels[strategy])

    layout_ax.set_title("(a) Layout design results")
    layout_ax.set_xlabel("Package side length")
    layout_ax.set_ylabel("Package evaluation avg. fidelity")
    layout_std_ax.set_ylabel("Fidelity std")
    layout_ax.set_xticks(x)
    layout_ax.set_xticklabels(PACKAGE_SIZES)
    layout_ax.set_ylim(0.5, 0.82)
    layout_std_ax.set_ylim(bottom=0)
    layout_ax.grid(True, axis="y", color="#E1E1E1", linewidth=0.8)
    layout_ax.set_axisbelow(True)
    layout_std_ax.grid(False)
    layout_ax.legend(
        layout_handles,
        layout_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
    )

    # (b) QFT routing: circuit duration/depth as grouped bars and average
    # logical-gate fidelity as lines. CircuitDuration is the depth-like routing
    # metric persisted by test_routing_scalability.py.
    qft = routing[routing["Algorithm"].eq("qft")].copy()
    routing_fidelity_ax = routing_ax.twinx()
    routing_handles = []
    routing_labels = []
    for index, strategy in enumerate(strategies):
        subset = qft[qft["LayoutDesignStrategy"].eq(strategy)].copy()
        subset["PackageSize"] = pd.Categorical(subset["PackageSize"], categories=PACKAGE_SIZES, ordered=True)
        subset = subset.sort_values("PackageSize")
        circuit_depth = pd.to_numeric(subset["CircuitDuration"], errors="coerce").to_numpy(dtype=float)
        gate_fidelity = pd.to_numeric(subset["GateFidelity"], errors="coerce").to_numpy(dtype=float)

        bars = routing_ax.bar(
            x + (index - 1) * bar_width,
            circuit_depth,
            width=bar_width,
            color=colors[strategy],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.72,
            label=labels[strategy],
            zorder=2,
        )
        line = routing_fidelity_ax.plot(
            x,
            gate_fidelity,
            color=colors[strategy],
            marker=markers[strategy],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
            label=f"{labels[strategy]} fidelity",
            zorder=3,
        )[0]
        routing_handles.append(line)
        routing_labels.append(labels[strategy])

    routing_ax.set_title("(b) QFT routing results")
    routing_ax.set_xlabel("Package side length")
    routing_ax.set_ylabel("Circuit depth")
    routing_fidelity_ax.set_ylabel("Average logical-gate fidelity")
    routing_ax.set_xticks(x)
    routing_ax.set_xticklabels(PACKAGE_SIZES)
    routing_ax.set_ylim(bottom=0)
    routing_fidelity_ax.set_ylim(0.5, 0.9)
    routing_ax.grid(True, axis="y", color="#E1E1E1", linewidth=0.8)
    routing_ax.set_axisbelow(True)
    routing_fidelity_ax.grid(False)
    routing_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    routing_ax.legend(
        routing_handles,
        routing_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.5,
    )

    for ax in [layout_ax, layout_std_ax, routing_ax, routing_fidelity_ax]:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_color("#B8B8B8")
        ax.spines["left"].set_color("#B8B8B8")
        ax.spines["right"].set_color("#B8B8B8")

    fig.tight_layout(w_pad=2.2, rect=(0, 0, 1, 0.91))
    save_figure(fig, out_dir, "overall_results", formats)


def plot_package_shape_scalability(
    root: Path,
    eval_result_dir: Path,
    out_dir: Path,
    formats: list[str],
) -> None:
    """Plot one MECH-versus-Sensate package-shape panel per algorithm."""
    spec = SCALABILITY_SPECS["package_shape"]
    axis = spec["axis"]
    sensate = load_scalability(eval_result_dir, "package_shape", spec)
    mech = load_mech_scalability(root, "package_shape", spec)
    x = np.arange(len(spec["order"]))
    width = 0.34
    baseline_colors = {"mech": COLORS["sage"], "sensate": COLORS["dark_green"]}
    baseline_markers = {"mech": "^", "sensate": "o"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2), sharex=True)
    for index, (ax, algorithm) in enumerate(zip(axes.flat, ALGORITHMS)):
        fidelity_ax = ax.twinx()
        sensate_subset = sensate[sensate["Algorithm"].eq(algorithm)].copy()
        mech_subset = mech[mech["Algorithm"].eq(algorithm)].copy()
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
        fidelity_ax.plot(
            x,
            mech_fidelity,
            color=baseline_colors["mech"],
            marker=baseline_markers["mech"],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
        )
        fidelity_ax.plot(
            x,
            sensate_fidelity,
            color=baseline_colors["sensate"],
            marker=baseline_markers["sensate"],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
        )

        panel = chr(ord("a") + index)
        ax.set_title(f"({panel}) {DISPLAY_NAME[algorithm]}")
        ax.set_ylabel("Circuit duration (ms)")
        fidelity_ax.set_ylabel("Average logical-gate fidelity")
        ax.set_xticks(x)
        ax.set_xticklabels(spec["tick_labels"])
        ax.set_xlabel("Package shape")
        ax.set_yscale("log")
        fidelity_ax.set_ylim(0, 1.02)
        style_axis(ax)
        fidelity_ax.grid(False)
        fidelity_ax.spines["top"].set_visible(False)
        fidelity_ax.spines["right"].set_color(COLORS["axis"])
        plot_nan_hint(ax, [*mech_duration, *sensate_duration])
        ax.legend(
            [mech_bars, sensate_bars],
            ["MECH", "Sensate"],
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            frameon=False,
            columnspacing=1.2,
        )

    fig.tight_layout(h_pad=2.0, w_pad=2.4)
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
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                sensate_subset = sensate[sensate["Algorithm"].eq(algorithm)].copy()
                mech_subset = mech[mech["Algorithm"].eq(algorithm)].copy()
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
                    plot_line(ax, x, ratio, algorithm)
                else:
                    offset = (algorithm_index - (len(ALGORITHMS) - 1) / 2) * bar_width
                    ax.bar(
                        x + offset,
                        ratio,
                        width=bar_width,
                        color=SERIES_COLOR[algorithm],
                        edgecolor="black",
                        linewidth=0.5,
                        label=DISPLAY_NAME[algorithm],
                    )
                    plot_nan_hint(ax, ratio)

            panel = chr(ord("a") + row * 2 + col)
            ax.set_title(f"({panel}) {spec['title']}: {ylabel.lower()}")
            ax.set_ylabel(ylabel + " (×)")
            ax.set_xlabel(spec["xlabel"])
            ax.set_xticks(x)
            ax.set_xticklabels(spec["tick_labels"])
            ax.axhline(1.0, color=COLORS["axis"], linewidth=1, linestyle="--", zorder=0)
            ax.set_ylim(bottom=0)
            style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(ALGORITHMS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0, w_pad=2.0)
    save_figure(fig, out_dir, "routing_scalability", formats)


def plot_ablation(root: Path, evaluate_dir: Path, eval_result_dir: Path, out_dir: Path, formats: list[str], algorithm: str) -> None:
    ablation = load_ablation(evaluate_dir, eval_result_dir, algorithm)
    qunoc_ops = load_routing_package_size(eval_result_dir)
    qunoc_ops = qunoc_ops[(qunoc_ops["LayoutDesignStrategy"] == PRIMARY_LAYOUT_STRATEGY) & (qunoc_ops["Algorithm"] == algorithm)]
    qunoc_ops = last_by(qunoc_ops, ["PackageSize"])
    mech_ops = load_mech_operation_json(root, algorithm)

    fig, (ablation_ax, operation_ax) = plt.subplots(1, 2, figsize=(14, 6.2))
    x = np.arange(len(PACKAGE_SIZES))

    # (a) Match the package-shape pattern: duration bars on the left axis and
    # fidelity lines on the right axis.
    fidelity_ax = ablation_ax.twinx()
    width = 0.18
    ablation_handles = []
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
        fidelity_ax.plot(
            x,
            fidelity,
            color=SERIES_COLOR[strategy],
            marker=SERIES_MARKER[strategy],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=2,
        )
        ablation_handles.append(bars)
        ablation_labels.append(DISPLAY_NAME[strategy])

    ablation_ax.set_title("(a) Routing-method ablation")
    ablation_ax.set_xlabel("Package side length")
    ablation_ax.set_ylabel("Circuit duration (ms)")
    fidelity_ax.set_ylabel("Average logical-gate fidelity")
    ablation_ax.set_xticks(x)
    ablation_ax.set_xticklabels(PACKAGE_SIZES)
    ablation_ax.set_yscale("log")
    fidelity_ax.set_ylim(0, 1.02)
    style_axis(ablation_ax)
    fidelity_ax.grid(False)
    fidelity_ax.spines["top"].set_visible(False)
    fidelity_ax.spines["right"].set_color(COLORS["axis"])
    ablation_ax.legend(
        ablation_handles,
        ablation_labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        frameon=False,
        columnspacing=0.9,
    )

    # (b) Two-qubit-gate counts share the left log axis; EQST distance is a
    # line on the right log axis.
    ax = operation_ax
    width = 0.18
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
        markersize=9,
        markeredgecolor="black",
        markeredgewidth=0.5,
        linewidth=2,
        label="QuNoC: EQST distance",
    )[0]

    ax.set_title(f"(b) Hardware-operation decomposition ({DISPLAY_NAME.get(algorithm, algorithm.upper())})")
    ax.set_xlabel("Package side length")
    ax.set_ylabel("Number of two-qubit operations")
    qst_ax.set_ylabel("Total EQST distance")
    ax.set_xticks(x)
    ax.set_xticklabels(PACKAGE_SIZES)
    ax.set_yscale("log")
    qst_ax.set_yscale("log")
    style_axis(ax)
    qst_ax.grid(False)
    qst_ax.spines["top"].set_visible(False)
    qst_ax.spines["right"].set_color(COLORS["axis"])
    plot_nan_hint(ax, [*mech_intra, *mech_inter, *qunoc_intra, *qunoc_inter, *qst])

    operation_handles, operation_labels = ax.get_legend_handles_labels()
    ax.legend(
        operation_handles + [qst_line],
        operation_labels + ["QuNoC: EQST distance"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        frameon=False,
        columnspacing=0.55,
        handlelength=1.3,
    )
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.14, top=0.70, wspace=0.55)
    save_figure(fig, out_dir, f"ablation_{algorithm}", formats)


# ----------------------------- entry point -----------------------------

def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate evaluation figures from existing CSV/JSON results.")
    parser.add_argument("--root", type=Path, default=default_root, help="Repository root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory. Default: evaluate/figure.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg"], help="Output formats, e.g. pdf svg png.")
    parser.add_argument("--ablation-algorithm", choices=ALGORITHMS, default="qaoa", help="Algorithm used for ablation operation decomposition.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    evaluate_dir = root / "evaluate"
    eval_result_dir = evaluate_dir / "result"
    out_dir = args.output_dir.resolve() if args.output_dir else evaluate_dir / "figure"

    setup_matplotlib()
    plot_overall(evaluate_dir, eval_result_dir, out_dir, args.formats)
    plot_package_shape_scalability(root, eval_result_dir, out_dir, args.formats)
    plot_scalability(root, eval_result_dir, out_dir, args.formats)
    plot_ablation(root, evaluate_dir, eval_result_dir, out_dir, args.formats, args.ablation_algorithm)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
