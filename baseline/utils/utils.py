from ..MECH.Chiplet import *
from qunoc.element.package import FabricateModel, OperationConfig

def get_mech_data_qubit_num(structure, chiplet_array_dim, chiplet_size, custom_highway_qubits=None, custom_highway_edges=[], cross_link_sparsity=1):
    G=gen_chiplet_array(structure, chiplet_array_dim[0], chiplet_array_dim[1], chiplet_size[0], chiplet_size[1], cross_link_sparsity=cross_link_sparsity)
    if custom_highway_qubits:
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)
    return len(G.nodes) - len(get_highway_qubits(G))

def get_mech_circuit_duration(exp_result: dict, op_config: OperationConfig = OperationConfig()):
    return exp_result["depth"] * max(op_config.reset_time, op_config.single_time, op_config.intra_time, op_config.inter_time, op_config.measure_time)
    return exp_result["prep_exec_depth"] * max(op_config.reset_time, op_config.single_time, op_config.intra_time, op_config.inter_time) + exp_result["meas_depth"] * op_config.measure_time

def get_mech_logic_gate_fide(exp_result: dict, chiplet_size: tuple[int, int], fab_model: FabricateModel = FabricateModel()):
    logic_gate_num = exp_result["logic_gate_num"]
    average_on_chip_num = exp_result["on_chip"] / logic_gate_num
    average_cross_chip_num = exp_result["cross_chip"] / logic_gate_num
    return (fab_model.get_intra_fide(chiplet_size[0] * chiplet_size[1]) ** average_on_chip_num) * (fab_model.inter_fide ** average_cross_chip_num)
