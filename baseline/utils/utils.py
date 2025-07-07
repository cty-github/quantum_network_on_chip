from qunoc.element.package import FabricateModel, OperationConfig

def get_mech_circuit_depth(exp_result: dict, op_config: OperationConfig = OperationConfig()):
    return exp_result['depth'] * max(op_config.reset_time, op_config.single_time, op_config.intra_time, op_config.inter_time, op_config.measure_time)

def get_mech_gate_fidelity(exp_result: dict, chiplet_size: tuple[int, int], fab_model: FabricateModel = FabricateModel()):
    gate_num = exp_result['gate_num']
    average_on_chip_num = exp_result['on_chip'] / gate_num
    average_cross_chip_num = exp_result['cross_chip'] / gate_num
    return (fab_model.get_intra_fide(chiplet_size[0] * chiplet_size[1]) ** average_on_chip_num) * (fab_model.inter_fide ** average_cross_chip_num)
