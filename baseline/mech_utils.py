from __future__ import annotations

import math
from typing import Any

from baseline.MECH.Chiplet import (
    custom_highway_layout,
    gen_chiplet_array,
    gen_highway_layout,
    get_highway_qubits,
)
from baseline.MECH.MECHBenchmarks import MECH_experiments
from qunoc.element.package import FabricateModel, OperationConfig


CHIPLET_LENGTH = 10
DEFAULT_INTER_SPARSITY = 3
DEFAULT_INTER_FIDE = 0.983
SENSATE_MAX_CAPACITY = 100
SENSATE_MAX_LENGTH = 80

Case = dict[str, Any]

cache: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}


def get_mech_data_qubit_num(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=None,
    cross_link_sparsity=1,
):
    graph = gen_chiplet_array(
        structure,
        chiplet_array_dim[0],
        chiplet_array_dim[1],
        chiplet_size[0],
        chiplet_size[1],
        cross_link_sparsity=cross_link_sparsity,
    )
    if custom_highway_qubits is not None:
        if custom_highway_edges is None:
            custom_highway_edges = []
        custom_highway_layout(graph, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(graph)
    return len(graph.nodes) - len(get_highway_qubits(graph))


def get_mech_highway_proportion(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=None,
    cross_link_sparsity=1,
):
    graph = gen_chiplet_array(
        structure,
        chiplet_array_dim[0],
        chiplet_array_dim[1],
        chiplet_size[0],
        chiplet_size[1],
        cross_link_sparsity=cross_link_sparsity,
    )
    if custom_highway_qubits is not None:
        if custom_highway_edges is None:
            custom_highway_edges = []
        custom_highway_layout(graph, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(graph)
    return len(get_highway_qubits(graph)) / len(graph.nodes)


def get_mech_circuit_duration(
    exp_result: dict, op_config: OperationConfig = OperationConfig()
):
    return (
        exp_result["prep_exec_depth"]
        * max(
            op_config.reset_time,
            op_config.single_time,
            op_config.intra_time,
            op_config.inter_time,
        )
        + exp_result["meas_depth"] * op_config.measure_time
    )


def get_mech_eqst_circuit_duration(exp_result: dict):
    return exp_result["circuit_duration"]


def get_mech_logic_gate_fide(
    exp_result: dict,
    chiplet_size: tuple[int, int],
    fab_model: FabricateModel = FabricateModel(),
):
    logic_gate_num = exp_result["logic_gate_num"]
    average_on_chip_num = exp_result["on_chip"] / logic_gate_num
    average_cross_chip_num = exp_result["cross_chip"] / logic_gate_num
    average_eqst_dist = exp_result.get("eqst_dist", 0) / logic_gate_num
    return (
        fab_model.get_intra_fide(chiplet_size[0] * chiplet_size[1])
        ** average_on_chip_num
    ) * (fab_model.inter_fide ** average_cross_chip_num) * (
        fab_model.enhanced_fide ** average_eqst_dist
    )


op_config = OperationConfig()
duration_modify_const = (
    (
        op_config.intra_time
        + op_config.inter_time
        + (op_config.measure_time + op_config.reset_time + op_config.single_time)
    )
    / 3
) / max(
    op_config.reset_time,
    op_config.single_time,
    op_config.intra_time,
    op_config.inter_time,
    op_config.measure_time,
)


def data_qubit_num(package_size: tuple[int, int]) -> int:
    node_num = package_size[0] * package_size[1]
    return int(100 * 2**math.log(node_num / 1600, 5))


def divisor_at_most(n: int, limit: int) -> int:
    for divisor in range(min(n, limit), 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


def exact_mech_shape(
    package_size: tuple[int, int], preferred_chiplet: int = CHIPLET_LENGTH
) -> tuple[tuple[int, int], tuple[int, int]]:
    x, y = package_size
    chiplet_x = divisor_at_most(x, preferred_chiplet)
    chiplet_y = divisor_at_most(y, preferred_chiplet)
    return (x // chiplet_x, y // chiplet_y), (chiplet_x, chiplet_y)


def _available_cross_link_counts(
    geometry: str, chiplet_size: tuple[int, int]
) -> tuple[int, int]:
    chiplet_x, chiplet_y = chiplet_size
    if geometry == "square":
        return chiplet_y, chiplet_x
    if geometry == "hexagon":
        return chiplet_y, len(range((chiplet_y + 1) % 2, chiplet_x, 2))
    if geometry == "heavy_square":
        return len(range(0, chiplet_y, 2)), len(range(0, chiplet_x, 2))
    if geometry == "heavy_hexagon":
        return len(range(0, chiplet_y, 2)), len(range(2, chiplet_x, 4))
    raise ValueError(f"Unsupported MECH geometry: {geometry}")


def sensate_spacing_to_mech_links(
    chiplet_size: tuple[int, int],
    sensate_spacing: int,
    geometry: str = "square",
) -> dict[str, Any]:
    if sensate_spacing <= 0:
        raise ValueError("Sensate inter_sparsity must be positive.")

    chiplet_x, chiplet_y = chiplet_size
    requested_row_links = math.ceil(chiplet_y / sensate_spacing)
    requested_col_links = math.ceil(chiplet_x / sensate_spacing)
    available_row_links, available_col_links = _available_cross_link_counts(
        geometry, chiplet_size
    )
    mech_row_links = min(requested_row_links, available_row_links)
    mech_col_links = min(requested_col_links, available_col_links)
    mech_link_count = min(mech_row_links, mech_col_links)
    return {
        "method": "link_count_only",
        "exact_topology_match": False,
        "sensate_inter_sparsity": sensate_spacing,
        "requested_row_link_count": requested_row_links,
        "requested_col_link_count": requested_col_links,
        "available_mech_row_link_count": available_row_links,
        "available_mech_col_link_count": available_col_links,
        "mech_row_link_count": mech_row_links,
        "mech_col_link_count": mech_col_links,
        "mech_link_count": mech_link_count,
        "uses_single_count_for_both_directions": True,
        "was_capped_by_geometry": (
            mech_row_links != requested_row_links
            or mech_col_links != requested_col_links
        ),
    }


def sensate_spacing_case_fields(
    chiplet_size: tuple[int, int],
    sensate_spacing: int = DEFAULT_INTER_SPARSITY,
    geometry: str = "square",
) -> Case:
    conversion = sensate_spacing_to_mech_links(
        chiplet_size, sensate_spacing, geometry
    )
    return {
        "inter_sparsity": sensate_spacing,
        "cross_link_sparsity": conversion["mech_link_count"],
    }


def comparison_fab_model(
    geometry: str = "square",
    inter_sparsity: int = DEFAULT_INTER_SPARSITY,
    inter_fide: float = DEFAULT_INTER_FIDE,
) -> FabricateModel:
    return FabricateModel(
        max_capacity=SENSATE_MAX_CAPACITY,
        max_length=SENSATE_MAX_LENGTH,
        compute_geometry=geometry,
        inter_sparsity=inter_sparsity,
        inter_fide=inter_fide,
    )


def get_field(row: dict[str, Any], *names: str, default: Any = 0) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def aligned_logic_gate_fidelity(
    exp_result: dict[str, Any], fab_model: FabricateModel
) -> float:
    logic_gate_num = exp_result["logic_gate_num"]
    if logic_gate_num <= 0:
        raise ValueError("Cannot normalize fidelity for a circuit without logic gates.")
    average_intra_num = exp_result["on_chip"] / logic_gate_num
    average_inter_num = exp_result["cross_chip"] / logic_gate_num
    average_qst_dist = exp_result.get("eqst_dist", 0) / logic_gate_num
    return (
        fab_model.get_intra_fide(fab_model.max_length) ** average_intra_num
        * fab_model.inter_fide**average_inter_num
        * fab_model.enhanced_fide**average_qst_dist
    )


def run_mech(
    structure: str,
    package_size: tuple[int, int],
    chiplet_array_dim: tuple[int, int],
    chiplet_size: tuple[int, int],
    algorithms: list[str],
    cross_link_sparsity: int,
    data_qubits: int,
) -> dict[str, dict[str, Any]]:
    key = (
        structure,
        package_size,
        chiplet_array_dim,
        chiplet_size,
        tuple(algorithms),
        cross_link_sparsity,
        data_qubits,
    )
    if key not in cache:
        result, _ = MECH_experiments(
            structure=structure,
            chiplet_array_dim=chiplet_array_dim,
            chiplet_size=chiplet_size,
            benchmarks=algorithms,
            data_qubit_num=data_qubits,
            cross_link_sparsity=cross_link_sparsity,
        )
        cache[key] = result
    return cache[key]
