import logging
import random as stdlib_random
from collections import Counter, defaultdict
from random import sample

from qiskit.circuit.library import hamiltonian_variational_ansatz
from qiskit.quantum_info import SparsePauliOp

from .Chiplet import *
from .Circuit import *
from .HighwayOccupancy import *
from .Router import *


def VQE(data_qubit_num, num_pauli_terms=4):
    paulis = []
    pauli_options = ["I", "X", "Y", "Z"]
    for _ in range(num_pauli_terms):
        pauli_str = "".join(stdlib_random.choice(pauli_options) for _ in range(data_qubit_num))
        paulis.append(pauli_str)
    hamiltonian = SparsePauliOp(paulis)
    ansatz = hamiltonian_variational_ansatz(hamiltonian)

    program = []
    for instr, qargs, _ in ansatz.data:
        if len(qargs) == 2:
            q0 = ansatz.qubits.index(qargs[0])
            q1 = ansatz.qubits.index(qargs[1])
            control_block = ControlBlock(q0, Counter([q1]))
            program.append(control_block)
    return program


def QFT(data_qubit_num):
    program = []
    for i in range(data_qubit_num - 1):
        for j in range(i + 1, data_qubit_num):
            control_block = ControlBlock(j, Counter([i]))
            program.append(control_block)
    return program


def QAOA(
    data_qubit_num,
    percentage=0.5,
    problem_graph="random_complete_sample",
    order_strategy="as_sampled",
    seed=None,
):
    gate_num = int(data_qubit_num * (data_qubit_num - 1) / 2 * percentage)
    rng = stdlib_random.Random(seed) if seed is not None else stdlib_random

    if problem_graph == "random_complete_sample":
        all_possible_gates = [
            (i, j) for i in range(data_qubit_num) for j in range(i + 1, data_qubit_num)
        ]
        gates = list(rng.sample(all_possible_gates, k=gate_num))
    elif problem_graph in {"cycle", "ring"}:
        gates = [(i, (i + 1) % data_qubit_num) for i in range(data_qubit_num)]
    else:
        raise ValueError(f"Unknown QAOA problem graph: {problem_graph}")

    if order_strategy == "control_grouped":
        gates.sort(key=lambda gate: (gate[0], gate[1]))
    elif order_strategy != "as_sampled":
        raise ValueError(f"Unknown QAOA order strategy: {order_strategy}")

    program = []
    for control, target in gates:
        control_block = ControlBlock(control, Counter([target]))
        program.append(control_block)
    return program


def BV(data_qubit_num, percentage=0.5):
    all_possible_gates = [(i, data_qubit_num-1) for i in range(0, data_qubit_num-1)]
    gates = list(sample(all_possible_gates, k=int(len(all_possible_gates) * percentage)))
    gates.sort()

    program = []
    control_target_dict = defaultdict(list)
    for control, target in gates:
        control_target_dict[control].append(target)
    for control, target_list in control_target_dict.items():
        for i in range(len(target_list)):
            control_block = ControlBlock(control, Counter([target_list[i]]))
            program.append(control_block)
    return program


def convert_qiskit_to_control_blocks(circuit):
    """Convert Qiskit circuit to MECH ControlBlock list"""
    program = []
    for _, qargs, _ in circuit.data:
        if len(qargs) == 2:
            q0 = circuit.qubits.index(qargs[0])
            q1 = circuit.qubits.index(qargs[1])
            control_block = ControlBlock(q0, Counter([q1]))
            program.append(control_block)
    return program


def S_GHZ(data_qubit_num):
    """Generate SUPERMARQ GHZ benchmark using ControlBlock"""
    import supermarq.benchmarks

    ghz_benchmark = supermarq.benchmarks.GHZ(num_qubits=data_qubit_num)
    circuit = ghz_benchmark.qiskit_circuit()
    return convert_qiskit_to_control_blocks(circuit)


def S_HAMILTONIAN_SIMULATION(data_qubit_num, layers=2):
    """Generate SUPERMARQ Hamiltonian Simulation benchmark using ControlBlock"""
    import supermarq.benchmarks

    ham_sim = supermarq.benchmarks.HamiltonianSimulation(
        num_qubits=data_qubit_num, time_step=0.1, total_time=layers * 0.1
    )
    circuit = ham_sim.qiskit_circuit()
    return convert_qiskit_to_control_blocks(circuit)


def S_MERMIN_BELL(data_qubit_num):
    """Generate SUPERMARQ Mermin-Bell benchmark using ControlBlock"""
    import supermarq.benchmarks

    mermin = supermarq.benchmarks.MerminBell(num_qubits=data_qubit_num)
    circuit = mermin.qiskit_circuit()
    return convert_qiskit_to_control_blocks(circuit)


def FULLY_CONNECTED(data_qubit_num, layers=1):
    """Generate fully connected ansatz using ControlBlock"""
    program = []

    for _ in range(layers):
        for i in range(data_qubit_num):
            for j in range(i + 1, data_qubit_num):
                control_block = ControlBlock(i, Counter([j]))
                program.append(control_block)

    return program


def CIRCULAR(data_qubit_num, layers=1):
    """Generate circular ansatz using ControlBlock"""
    program = []

    for _ in range(layers):
        for i in range(data_qubit_num):
            next_qubit = (i + 1) % data_qubit_num
            if i != next_qubit:  # 避免自连接
                control_block = ControlBlock(i, Counter([next_qubit]))
                program.append(control_block)

    return program


def BV_HARD(data_qubit_num):
    """
    Generate hard version BV Oracle using ControlBlock.

    Oracle structure: qubit0 controls qubits 1,2,...,n;
                     qubit1 controls qubits 2,3,...,n; and so on.

    Mathematical function: f(x) = Σ_{i=1}^{n} x_i + Σ_{i<j} x_i x_j (mod 2)
    """
    program = []

    for i in range(data_qubit_num):
        for j in range(i + 1, data_qubit_num):
            control_block = ControlBlock(j, Counter([i]))
            program.append(control_block)

    return program


def MECH_experiments_data_qubit_num(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=[],
    cross_link_sparsity=1,
    cross_link_row_spacing=None,
    cross_link_col_spacing=None,
    prep_period=None,
):
    G = gen_chiplet_array(
        structure,
        chiplet_array_dim[0],
        chiplet_array_dim[1],
        chiplet_size[0],
        chiplet_size[1],
        cross_link_sparsity=cross_link_sparsity,
        cross_link_row_spacing=cross_link_row_spacing,
        cross_link_col_spacing=cross_link_col_spacing,
    )
    if prep_period is None:
        if structure in ["square", "hexagon"]:
            prep_period = 17
        if structure in ["heavy_square", "heavy_hexagon"]:
            prep_period = 17

    if custom_highway_qubits is not None:
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)

    return len(G.nodes) - len(get_highway_qubits(G))

def get_benchmark_program(benchmark_name, data_qubit_num, **kwargs):
    if benchmark_name == "vqe":
        return VQE(data_qubit_num, **kwargs)

    if benchmark_name == "qft":
        return QFT(data_qubit_num)

    if benchmark_name == "qaoa":
        return QAOA(data_qubit_num, **kwargs)

    if benchmark_name == "bv":
        return BV(data_qubit_num, **kwargs)

    if benchmark_name in {"s_ghz", "ghz"}:
        return S_GHZ(data_qubit_num)

    if benchmark_name == "s_hamiltonian_simulation":
        return S_HAMILTONIAN_SIMULATION(data_qubit_num, **kwargs)

    if benchmark_name in {"s_mermin_bell", "mermin_bell"}:
        return S_MERMIN_BELL(data_qubit_num)

    if benchmark_name in {"fully_connected", "fully_connected_ansatz"}:
        return FULLY_CONNECTED(data_qubit_num, **kwargs)

    if benchmark_name in {"circular", "circular_ansatz"}:
        return CIRCULAR(data_qubit_num, **kwargs)

    if benchmark_name == "bv_hard":
        return BV_HARD(data_qubit_num)

    raise ValueError(f"Unknown benchmark name: {benchmark_name}")

def MECH_experiments(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=[],
    benchmarks=["vqe", "qft", "qaoa", "bv"],
    data_qubit_num=None,
    cross_link_sparsity=1,
    cross_link_row_spacing=None,
    cross_link_col_spacing=None,
    prep_period=None,
    meas_period=2,
    cross_chip_gate_weight=7.4,
    meas_weight=2.2,
    benchmark_options=None,
):
    G = gen_chiplet_array(
        structure,
        chiplet_array_dim[0],
        chiplet_array_dim[1],
        chiplet_size[0],
        chiplet_size[1],
        cross_link_sparsity=cross_link_sparsity,
        cross_link_row_spacing=cross_link_row_spacing,
        cross_link_col_spacing=cross_link_col_spacing,
    )
    if prep_period is None:
        if structure in ["square", "hexagon"]:
            prep_period = 17
        if structure in ["heavy_square", "heavy_hexagon"]:
            prep_period = 17

    if custom_highway_qubits is not None:
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)

    if data_qubit_num is None:
        data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))
    # benchmark_program_dict = {
    #     "vqe": VQE(data_qubit_num),
    #     "qft": QFT(data_qubit_num),
    #     "qaoa": QAOA(data_qubit_num),
    #     "bv": BV(data_qubit_num),
    #     "s_ghz": S_GHZ(data_qubit_num),
    #     "s_hamiltonian_simulation": S_HAMILTONIAN_SIMULATION(data_qubit_num),
    #     "s_mermin_bell": S_MERMIN_BELL(data_qubit_num),
    #     "fully_connected": FULLY_CONNECTED(data_qubit_num),
    #     "circular": CIRCULAR(data_qubit_num),
    #     "bv_hard": BV_HARD(data_qubit_num),
    # }
    result = defaultdict(dict)
    circuit = dict()
    benchmark_options = benchmark_options or {}

    for b in benchmarks:
        logging.info("-" * 16 + "executing " + b + "-" * 16)
        program = get_benchmark_program(
            b, data_qubit_num, **benchmark_options.get(b, {})
        )
        initial_v2p = dict()
        data_idx, highway_idx = 0, len(G.nodes) - len(get_highway_qubits(G))
        nodes = set(G.nodes)
        while len(nodes) > 0:
            node = nodes.pop()
            if get_node_type(G, node) == "data":
                initial_v2p[data_idx] = node
                data_idx += 1
            else:
                initial_v2p[highway_idx] = node
                highway_idx += 1

        logic_gate_num = 0
        for control_block in program:
            logic_gate_num += int(sum(list(control_block.target_counter.values())))

        # if os.path.exists(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl"):
        #     with open(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl", "rb") as f:
        #         router = pickle.load(f)
        #         logging.info(f"Loaded router from ./result/baseline/mech/router_{b}_{data_qubit_num}.pkl")
        # else:
        router = Router(
            G, initial_v2p=initial_v2p, prep_period=prep_period, meas_period=meas_period
        )
        for i, control_block in enumerate(program):
            router.execute_control_multi_target_block(
                control_block, cross_chip_gate_weight, cross_chip_overhead=3
            )
        router.highway_manager.end_shuttle(
            len(router.highway_manager.shuttle_stack) - 1, router.circuit
        )
        # with open(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl", "wb") as f:
        #     pickle.dump(router, f)
        #     logging.info(f"Saved router to ./result/baseline/mech/router_{b}_{data_qubit_num}.pkl")

        prep_exec_depth = 0
        for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
            for idx in range(
                router.highway_manager.get_shuttle_prep_start_time(shuttle_idx),
                router.highway_manager.get_shuttle_exec_start_time(shuttle_idx),
            ):
                for line in range(len(router.circuit.circuit_lines)):
                    assert router.circuit.is_position_empty(line, idx), (
                        f"line {line}, idx {idx} should be empty but is occupied by {router.circuit.take_node(line, idx)}"
                    )
            source_measured_qubits_dict = (
                router.highway_manager.bridge_throughout_highway(
                    router.circuit, shuttle_idx
                )
            )
            router.highway_manager.reentangle_throughout_highway(
                router.circuit, shuttle_idx, source_measured_qubits_dict
            )
            prep_exec_depth += router.highway_manager.get_shuttle_meas_start_time(
                shuttle_idx
            ) - router.highway_manager.get_shuttle_prep_start_time(shuttle_idx)

        meas_depth = 0
        for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
            router.highway_manager.measure_throughout_highway(router.circuit, shuttle_idx)
            meas_depth += router.highway_manager.get_shuttle_end_time(
                shuttle_idx
            ) - router.highway_manager.get_shuttle_meas_start_time(shuttle_idx)

        on_chip_gate_num = 0
        cross_chip_gate_num = 0
        meas_num = 0
        for idx in range(router.circuit.depth):
            for line in range(len(router.circuit.circuit_lines)):
                if router.circuit.take_role(line, idx) == "q":
                    meas_num += 1
                if router.circuit.take_role(line, idx) in ["t", "mt"]:
                    node = router.circuit.take_node(line, idx)
                    if isinstance(node, OpNode):
                        control_line = node.control
                    if isinstance(node, MOpNode):
                        control_line = node.shared
                    control_qubit, target_qubit = (
                        router.highway_manager.idx_qubit_dict[control_line],
                        router.highway_manager.idx_qubit_dict[line],
                    )

                    if (
                        router.chip.has_edge(control_qubit, target_qubit)
                        and router.chip.edges[(control_qubit, target_qubit)]["type"]
                        == "cross_chip"
                    ):
                        cross_chip_gate_num += 1
                    else:
                        on_chip_gate_num += 1

        eff_gate_num = (
            on_chip_gate_num
            + cross_chip_gate_num * cross_chip_gate_weight
            + meas_num * meas_weight
        )
        result[b] = {
            "data_qubit_num": data_qubit_num,
            "logic_gate_num": logic_gate_num,
            "depth": router.circuit.depth,
            "eff_gate_num": eff_gate_num,
            "prep_exec_depth": prep_exec_depth,
            "meas_depth": meas_depth,
            "on_chip": on_chip_gate_num,
            "cross_chip": cross_chip_gate_num,
            "meas_num": meas_num,
            "weight": (cross_chip_gate_weight, meas_weight),
            "periods": (prep_period, meas_period),
            "shuttle_num": len(router.highway_manager.shuttle_stack),
            "sparsity": cross_link_sparsity,
            "cross_link_row_spacing": cross_link_row_spacing,
            "cross_link_col_spacing": cross_link_col_spacing,
            "cross_link_rows": list(G.cross_link_rows),
            "cross_link_cols": list(G.cross_link_cols),
        }
        circuit[b] = router.circuit

    return result, circuit


def MECH_EQST_experiments(
    structure,
    chiplet_array_dim,
    chiplet_size,
    custom_highway_qubits=None,
    custom_highway_edges=[],
    benchmarks=["vqe", "qft", "qaoa", "bv"],
    data_qubit_num=None,
    cross_link_sparsity=1,
    cross_link_row_spacing=None,
    cross_link_col_spacing=None,
    prep_period=None,
    meas_period=2,
    cross_chip_gate_weight=7.4,
    meas_weight=2.2,
    basic_time=1000,
    benchmark_options=None,
):
    G = gen_chiplet_array(
        structure,
        chiplet_array_dim[0],
        chiplet_array_dim[1],
        chiplet_size[0],
        chiplet_size[1],
        cross_link_sparsity=cross_link_sparsity,
        cross_link_row_spacing=cross_link_row_spacing,
        cross_link_col_spacing=cross_link_col_spacing,
    )
    if prep_period is None:
        if structure in ["square", "hexagon"]:
            prep_period = 17
        if structure in ["heavy_square", "heavy_hexagon"]:
            prep_period = 17

    if custom_highway_qubits is not None:
        custom_highway_layout(G, custom_highway_qubits, custom_highway_edges)
    else:
        gen_highway_layout(G)

    if data_qubit_num is None:
        data_qubit_num = len(G.nodes) - len(get_highway_qubits(G))
    benchmark_options = benchmark_options or {}
    result = defaultdict(dict)
    circuit = dict()

    for b in benchmarks:
        logging.info("-" * 16 + "executing " + b + "-" * 16)
        program = get_benchmark_program(
            b, data_qubit_num, **benchmark_options.get(b, {})
        )
        initial_v2p = dict()
        data_idx, highway_idx = 0, len(G.nodes) - len(get_highway_qubits(G))
        nodes = set(G.nodes)
        while len(nodes) > 0:
            node = nodes.pop()
            if get_node_type(G, node) == "data":
                initial_v2p[data_idx] = node
                data_idx += 1
            else:
                initial_v2p[highway_idx] = node
                highway_idx += 1

        logic_gate_num = 0
        for control_block in program:
            logic_gate_num += int(sum(list(control_block.target_counter.values())))

        # if os.path.exists(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl"):
        #     with open(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl", "rb") as f:
        #         router = pickle.load(f)
        #         logging.info(f"Loaded router from ./result/baseline/mech/router_{b}_{data_qubit_num}.pkl")
        # else:
        router = Router(
            G, initial_v2p=initial_v2p, prep_period=prep_period, meas_period=meas_period
        )
        for i, control_block in enumerate(program):
            router.execute_control_multi_target_block_eqst(
                control_block, cross_chip_gate_weight, cross_chip_overhead=3
            )
        # router.highway_manager.end_shuttle(len(router.highway_manager.shuttle_stack)-1, router.circuit)
        # with open(f"./result/baseline/mech/router_{b}_{data_qubit_num}.pkl", "wb") as f:
        #     pickle.dump(router, f)
        #     logging.info(f"Saved router to ./result/baseline/mech/router_{b}_{data_qubit_num}.pkl")

        # prep_exec_depth = 0
        # for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
        #     for idx in range(router.highway_manager.get_shuttle_prep_start_time(shuttle_idx), router.highway_manager.get_shuttle_exec_start_time(shuttle_idx)):
        #         for line in range(len(router.circuit.circuit_lines)):
        #             assert router.circuit.is_position_empty(line, idx), f"line {line}, idx {idx} should be empty but is occupied by {router.circuit.take_node(line, idx)}"
        #     source_measured_qubits_dict = router.highway_manager.bridge_throughout_highway(router.circuit, shuttle_idx)
        #     router.highway_manager.reentangle_throughout_highway(router.circuit, shuttle_idx, source_measured_qubits_dict)
        #     prep_exec_depth += router.highway_manager.get_shuttle_meas_start_time(shuttle_idx) - router.highway_manager.get_shuttle_prep_start_time(shuttle_idx)

        # meas_depth = 0
        # for shuttle_idx in range(len(router.highway_manager.shuttle_stack)):
        #     router.highway_manager.measure_throughout_highway(router.circuit, shuttle_idx)
        #     meas_depth += router.highway_manager.get_shuttle_end_time(shuttle_idx) - router.highway_manager.get_shuttle_meas_start_time(shuttle_idx)

        on_chip_gate_num = 0
        cross_chip_gate_num = 0
        meas_num = 0
        circuit_duration = 0
        for idx in range(router.circuit.depth):
            circuit_duration += max(
                basic_time, (router.circuit.eqst_depths[idx] - 1) * 25
            )
            for line in range(len(router.circuit.circuit_lines)):
                if router.circuit.take_role(line, idx) == "q":
                    meas_num += 1
                if router.circuit.take_role(line, idx) in ["t", "mt"]:
                    node = router.circuit.take_node(line, idx)
                    if isinstance(node, OpNode):
                        control_line = node.control
                    if isinstance(node, MOpNode):
                        control_line = node.shared
                    control_qubit, target_qubit = (
                        router.highway_manager.idx_qubit_dict[control_line],
                        router.highway_manager.idx_qubit_dict[line],
                    )

                    if (
                        router.chip.has_edge(control_qubit, target_qubit)
                        and router.chip.edges[(control_qubit, target_qubit)]["type"]
                        == "cross_chip"
                    ):
                        cross_chip_gate_num += 1
                    else:
                        on_chip_gate_num += 1

        eqst_dist = router.circuit.eqst_dist
        eff_gate_num = (
            on_chip_gate_num
            + cross_chip_gate_num * cross_chip_gate_weight
            + meas_num * meas_weight
        )
        result[b] = {
            "data_qubit_num": data_qubit_num,
            "logic_gate_num": logic_gate_num,
            "depth": router.circuit.depth,
            "eff_gate_num": eff_gate_num,
            # 'prep_exec_depth': prep_exec_depth,
            # 'meas_depth': meas_depth,
            "on_chip": on_chip_gate_num,
            "cross_chip": cross_chip_gate_num,
            "eqst_dist": eqst_dist,
            "meas_num": meas_num,
            "weight": (cross_chip_gate_weight, meas_weight),
            "periods": (prep_period, meas_period),
            "shuttle_num": len(router.highway_manager.shuttle_stack),
            "sparsity": cross_link_sparsity,
            "cross_link_row_spacing": cross_link_row_spacing,
            "cross_link_col_spacing": cross_link_col_spacing,
            "cross_link_rows": list(G.cross_link_rows),
            "cross_link_cols": list(G.cross_link_cols),
            "circuit_duration": circuit_duration,
        }
        circuit[b] = router.circuit

    return result, circuit
