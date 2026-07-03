from ..MECH.Chiplet import *
from qunoc.element.package import FabricateModel, OperationConfig

def get_mech_data_qubit_num(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=None,
    cross_link_sparsity=1,
):
    G = gen_chiplet_array(
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
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)
    return len(G.nodes) - len(get_highway_qubits(G))

def get_mech_highway_proportion(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=None,
    cross_link_sparsity=1,
):
    G = gen_chiplet_array(
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
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)
    return len(get_highway_qubits(G)) / len(G.nodes)

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
