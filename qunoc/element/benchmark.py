import random
from random import sample

import matplotlib.pyplot as plt
import networkx as nx
import qiskit
import qiskit_superstaq as qss
import supermarq
from qiskit.circuit.library import hamiltonian_variational_ansatz
from qiskit.quantum_info import SparsePauliOp


class Gate:
    """
    Represents the logic gate in the program.

    Attributes:
        id (int): Unique identifier for the gate.
    """
    _counter = 0

    def __init__(self):
        """
        Initialize a logic gate.
        """
        self.id = Gate._counter
        Gate._counter += 1

class Control(Gate):
    """
    Represents a control gate in the program.

    Attributes:
        control_qubit (int): The qubit that controls the gate.
        target_qubit (int): The qubit that is affected by the gate.
    """
    
    def __init__(self, control_qubit: int, target_qubit: int):
        """
        Initialize a control gate.

        Args:
            control_qubit (int): The qubit that controls the gate.
            target_qubit (int): The qubit that is affected by the gate.
        """
        super().__init__()
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.control_qubit}_{self.target_qubit}"
    
class Barrier(Gate):
    """
    Represents a barrier gate in the program.

    Attributes:
        qubit (int): The qubit that the barrier is applied to.
    """
    
    def __init__(self, qubit: int):
        """
        Initialize a barrier gate.

        Args:
            qubit (int): The qubit that the barrier is applied to.
        """
        super().__init__()
        self.qubit = qubit

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.qubit}"


class Program:
    """
    Represents the quantum program.

    Attributes:
        algo_name (str): Name of the algorithm.
        data_qubit_num (int): Number of data qubits.
        logic_gates (list[Gate]): List of logic gates in the program.
        gate_depend_graph (nx.DiGraph): Directed acyclic graph representing dependencies between gates.
    """

    def __init__(self, algo_name: str, data_qubit_num: int, logic_gates: list[Gate]):
        """
        Initialize a quantum program.

        Args:
            algo_name (str): Name of the algorithm.
            data_qubit_num (int): Number of data qubits.
            logic_gates (list[Gate]): List of logic gates in the program.
        """
        self.algo_name = algo_name
        self.data_qubit_num = data_qubit_num
        self.logic_gates = logic_gates

        self.get_gate_depend_graph()
        self.get_qiskit_circuit()
    
    def __repr__(self):
        return f"{self.algo_name.capitalize()}_{self.data_qubit_num}_{self.__class__.__name__}"

    def get_gate_depend_graph(self) -> nx.DiGraph:
        """
        Get the directed acyclic graph representing dependencies between gates.

        Returns:
            nx.DiGraph: The gate dependency graph.
        """
        if not hasattr(self, "gate_depend_graph"):
            recent_qubit_info = {i: {"recent_role": None, "recent_ctrls": [], "recent_other": -1} for i in range(self.data_qubit_num)}
            self.gate_depend_graph: nx.DiGraph = nx.DiGraph()
            self.gate_depend_graph.add_node(-1, gate=None)
            for gate in self.logic_gates:
                if isinstance(gate, Barrier):
                    self.gate_depend_graph.add_node(gate.id, gate=gate)
                    qubit = gate.qubit
                    if recent_qubit_info[qubit]["recent_role"] == "control":
                        for ctrl_gate_id in recent_qubit_info[qubit]["recent_ctrls"]:
                            self.gate_depend_graph.add_edge(ctrl_gate_id, gate.id)
                    else:
                        self.gate_depend_graph.add_edge(recent_qubit_info[qubit]["recent_other"], gate.id)
                    recent_qubit_info[qubit]["recent_role"] = "barrier"
                    recent_qubit_info[qubit]["recent_ctrls"] = []
                    recent_qubit_info[qubit]["recent_other"] = gate.id
                elif isinstance(gate, Control):
                    self.gate_depend_graph.add_node(gate.id, gate=gate)
                    control = gate.control_qubit
                    self.gate_depend_graph.add_edge(recent_qubit_info[control]["recent_other"], gate.id)
                    recent_qubit_info[control]["recent_role"] = "control"
                    recent_qubit_info[control]["recent_ctrls"].append(gate.id)
                    target = gate.target_qubit
                    if recent_qubit_info[target]["recent_role"] == "control":
                            for ctrl_gate_id in recent_qubit_info[target]["recent_ctrls"]:
                                self.gate_depend_graph.add_edge(ctrl_gate_id, gate.id)
                    else:
                        self.gate_depend_graph.add_edge(recent_qubit_info[target]["recent_other"], gate.id)
                    recent_qubit_info[target]["recent_role"] = "target"
                    recent_qubit_info[target]["recent_ctrls"] = []
                    recent_qubit_info[target]["recent_other"] = gate.id
                else:
                    raise ValueError("Unknown gate type")
            
            for gate in self.logic_gates:
                if isinstance(gate, Barrier):
                    pred = list(self.gate_depend_graph.predecessors(gate.id))
                    succ = list(self.gate_depend_graph.successors(gate.id))
                    for p in pred:
                        for s in succ:
                            if p != s:
                                self.gate_depend_graph.add_edge(p, s)
                    self.gate_depend_graph.remove_node(gate.id)

        assert nx.is_directed_acyclic_graph(self.gate_depend_graph), "Gate dependency graph is not a DAG."
        return self.gate_depend_graph
    
    def get_circuit_depth(self) -> int:
        """
        Get the depth of the quantum circuit represented by the program.

        Returns:
            int: The depth of the quantum circuit.
        """
        if not hasattr(self, "circuit_depth"):
            self.circuit_depth = nx.algorithms.dag_longest_path_length(self.gate_depend_graph)
        return self.circuit_depth

    def get_qiskit_circuit(self) -> qiskit.QuantumCircuit:
        """
        Convert the program to a Qiskit quantum circuit.

        Returns:
            qiskit.QuantumCircuit: The Qiskit quantum circuit representation of the program.
        """
        if not hasattr(self, "qiskit_circuit"):
            self.qiskit_circuit = qiskit.QuantumCircuit(self.data_qubit_num)
            for gate in self.logic_gates:
                if isinstance(gate, Barrier):
                    self.qiskit_circuit.barrier(gate.qubit)
                elif isinstance(gate, Control):
                    self.qiskit_circuit.cx(gate.control_qubit, gate.target_qubit)
                else:
                    raise ValueError("Unknown gate type")
        return self.qiskit_circuit

    def draw_depend(self, 
        marginrate: float = 0.2, 
        markersize: float = 5, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
    ) -> plt.Figure:
        """
        Draw the gate dependency graph to present the program.

        Returns:
            plt.Figure: A matplotlib figure representing the gate dependency graph.
        """
        pos = nx.spring_layout(self.gate_depend_graph)
        figsize = (self.get_circuit_depth(), self.get_circuit_depth())
        fig = plt.figure(figsize=figsize)
        nx.draw(
            self.gate_depend_graph, 
            pos, 
            with_labels=False, 
            arrows=True, 
            node_size=markersize, 
            width=linewidth_t,
            node_color="black", 
            edge_color="black", 
        )
        labels = {node: str(node) for node in self.gate_depend_graph.nodes()}
        label_offset = (-0.04, 0.03)
        label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(
            self.gate_depend_graph,
            label_pos,
            labels=labels,
            font_size=10,
            font_color="black"
        )
        plt.close()
        return fig
    
    def draw_circuit(self,
        marginrate: float = 0.2,
        markersize: float = 5,
        linewidth_t: float = 1,
        linewidth_b: float = 2,
    ) -> plt.Figure:
        """
        Draw the circuit representation of the program.

        Returns:
            plt.Figure: A matplotlib figure representing the circuit.
        """
        return self.qiskit_circuit.draw(output="mpl")

def vqe(data_qubit_num: int, num_pauli_terms: int = 4) -> Program:
    """
    Generate a VQE program.

    Args:
        data_qubit_num (int): Number of data qubits.
        num_pauli_terms (int): Number of pauli terms in the Hamiltonian.

    Returns:
        Program: A quantum program with VQE logic gates.
    """
    paulis = []
    pauli_options = ['I', 'X', 'Y', 'Z']
    for _ in range(num_pauli_terms):
        pauli_str = ''.join(random.choice(pauli_options) for _ in range(data_qubit_num))
        paulis.append(pauli_str)
    hamiltonian = SparsePauliOp(paulis)
    ansatz = hamiltonian_variational_ansatz(hamiltonian)

    Gate._counter = 0
    logic_gates = []
    for instr, qargs, _ in ansatz.data:
        if len(qargs) == 2:
            q0 = ansatz.qubits.index(qargs[0])
            q1 = ansatz.qubits.index(qargs[1])
            logic_gates.append(Control(q0, q1))
    return Program("vqe", data_qubit_num, logic_gates)

def qft(data_qubit_num: int) -> Program:
    """
    Generate a QFT program.

    Args:
        data_qubit_num (int): Number of data qubits.

    Returns:
        Program: A quantum program with QFT logic gates.
    """
    Gate._counter = 0
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
        control_gates = [Control(j, i) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program("qft", data_qubit_num, logic_gates)

def qaoa(data_qubit_num: int, percentage: float = 0.5, blocklayer: int = 1) -> Program:
    """
    Generate a QAOA program.

    Args:
        data_qubit_num (int): Number of data qubits.
        percentage (float): Percentage of control gates to be selected.
        blocklayer (int): Number of block layers.

    Returns:
        Program: A quantum program with QAOA logic gates.
    """
    Gate._counter = 0
    all_possible_gates = [(i, j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    logic_gates = []
    for i, gate in enumerate(control_gates):
        logic_gates.append(Control(*gate))
        if (i+1) % (data_qubit_num*blocklayer) == 0 and (i+1) != len(control_gates):
            logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
    return Program("qaoa", data_qubit_num, logic_gates)

def bv(data_qubit_num: int, percentage: float = 0.5) -> Program:
    """
    Generate a BV program.

    Args:
        data_qubit_num (int): Number of data qubits.
        percentage (float): Percentage of control gates to be selected.

    Returns:
        Program: A quantum program with BV logic gates.
    """
    Gate._counter = 0
    all_possible_gates = [(i, data_qubit_num-1) for i in range(0, data_qubit_num-1)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    control_gates.sort()
    logic_gates = []
    for gate in control_gates:
        logic_gates.append(Control(*gate))
    return Program("bv", data_qubit_num, logic_gates)

def fully_connected_ansatz(data_qubit_num: int, layers: int = 1) -> Program:
    """
    Generate a fully connected ansatz program.

    Args:
        data_qubit_num (int): Number of data qubits.
        layers (int): Number of layers.

    Returns:
        Program: A quantum program with fully connected ansatz logic gates.
    """
    Gate._counter = 0
    logic_gates = []

    for layer in range(layers):
        for i in range(data_qubit_num):
            for j in range(i+1, data_qubit_num):
                logic_gates.append(Control(i, j))

        if layer < layers - 1:
            logic_gates.extend([Barrier(k) for k in range(data_qubit_num)])

    return Program("fully_connected", data_qubit_num, logic_gates)

def circular_ansatz(data_qubit_num: int, layers: int = 1) -> Program:
    """
    Generate a circular ansatz program.

    Args:
        data_qubit_num (int): Number of data qubits.
        layers (int): Number of layers.

    Returns:
        Program: A quantum program with circular ansatz logic gates.
    """
    Gate._counter = 0
    logic_gates = []

    for layer in range(layers):
        for i in range(data_qubit_num):
            next_qubit = (i + 1) % data_qubit_num
            if i != next_qubit: 
                logic_gates.append(Control(i, next_qubit))

        if layer < layers - 1:
            logic_gates.extend([Barrier(k) for k in range(data_qubit_num)])

    return Program("circular", data_qubit_num, logic_gates)

def s_ghz(data_qubit_num: int) -> Program:
    """
    Generate a SUPERMARQ GHZ state preparation benchmark program.

    Args:
        data_qubit_num (int): Number of data qubits.

    Returns:
        Program: A quantum program with GHZ logic gates.
    """
    import supermarq.benchmarks

    Gate._counter = 0
    logic_gates = []

    # 使用 SUPERMARQ 的 GHZ 基准测试
    ghz_benchmark = supermarq.benchmarks.GHZ(num_qubits=data_qubit_num)
    circuit = ghz_benchmark.qiskit_circuit()

    # 转换 Qiskit 电路为项目的 Control 门格式
    for instr, qargs, _ in circuit.data:
        if len(qargs) == 2:  # 只处理双量子比特门
            q0 = circuit.qubits.index(qargs[0])
            q1 = circuit.qubits.index(qargs[1])
            logic_gates.append(Control(q0, q1))

    return Program("s_ghz", data_qubit_num, logic_gates)


def s_hamiltonian_simulation(data_qubit_num: int, layers: int = 2) -> Program:
    """
    Generate a SUPERMARQ Hamiltonian Simulation benchmark program.
    Simulates 1D Transverse Field Ising Model dynamics.

    Args:
        data_qubit_num (int): Number of data qubits.
        layers (int): Number of Trotter layers (affects total_time).

    Returns:
        Program: A quantum program with Hamiltonian simulation logic gates.
    """
    import supermarq.benchmarks

    Gate._counter = 0
    logic_gates = []

    ham_sim = supermarq.benchmarks.HamiltonianSimulation(
        num_qubits=data_qubit_num,
        time_step=0.1,
        total_time=layers * 0.1
    )
    circuit = ham_sim.qiskit_circuit()

    for instr, qargs, _ in circuit.data:
        if len(qargs) == 2:  
            q0 = circuit.qubits.index(qargs[0])
            q1 = circuit.qubits.index(qargs[1])
            logic_gates.append(Control(q0, q1))

    return Program("s_hamiltonian_simulation", data_qubit_num, logic_gates)


def s_mermin_bell(data_qubit_num: int) -> Program:
    """
    Generate a SUPERMARQ Mermin-Bell benchmark program.
    Tests quantum non-locality using Bell inequality violations.

    Args:
        data_qubit_num (int): Number of data qubits.

    Returns:
        Program: A quantum program with Mermin-Bell logic gates.
    """
    import supermarq.benchmarks

    Gate._counter = 0
    logic_gates = []

    # 使用 SUPERMARQ 的 MerminBell 基准测试
    mermin = supermarq.benchmarks.MerminBell(num_qubits=data_qubit_num)
    circuit = mermin.qiskit_circuit()

    # 转换 Qiskit 电路为项目的 Control 门格式
    for instr, qargs, _ in circuit.data:
        if len(qargs) == 2:  # 只处理双量子比特门
            q0 = circuit.qubits.index(qargs[0])
            q1 = circuit.qubits.index(qargs[1])
            logic_gates.append(Control(q0, q1))

    return Program("s_mermin_bell", data_qubit_num, logic_gates)


def bv_hard(data_qubit_num: int) -> Program:
    """
    Generate hard version BV Oracle with upper triangular matrix form.

    Oracle structure: qubit0 controls qubits 1,2,...,n;
                     qubit1 controls qubits 2,3,...,n; and so on.

    Mathematical function: f(x) = Σ_{i=1}^{n} x_i + Σ_{i<j} x_i x_j (mod 2)

    Args:
        data_qubit_num (int): Number of data qubits.

    Returns:
        Program: A quantum program with hard BV Oracle logic gates.
    """
    Gate._counter = 0
    logic_gates = []

    # 创建上三角矩阵形式的控制连接
    for i in range(data_qubit_num):
        for j in range(i + 1, data_qubit_num):
            logic_gates.append(Control(j, i))

    return Program("bv_hard", data_qubit_num, logic_gates)

def get_program(algorithm: str, data_qubit_num: int, **kwargs) -> Program:
    """
    Get a quantum program by name.

    Args:
        algorithm (str): Name of the quantum algorithm (e.g., "vqe", "qft", "qaoa", "bv", "s_ghz").
        data_qubit_num (int): Number of data qubits.
        **kwargs: Additional keyword arguments for specific algorithms.

    Returns:
        Program: The requested quantum program.
    """
    if algorithm == "vqe":
        return vqe(data_qubit_num)
    elif algorithm == "qft":
        return qft(data_qubit_num)
    elif algorithm == "qaoa":
        return qaoa(data_qubit_num, **kwargs)
    elif algorithm == "bv":
        return bv(data_qubit_num, **kwargs)
    elif algorithm == "fully_connected":
        return fully_connected_ansatz(data_qubit_num, **kwargs)
    elif algorithm == "circular":
        return circular_ansatz(data_qubit_num, **kwargs)
    elif algorithm == "s_ghz":
        return s_ghz(data_qubit_num)
    elif algorithm == "s_hamiltonian_simulation":
        return s_hamiltonian_simulation(data_qubit_num, **kwargs)
    elif algorithm == "s_mermin_bell":
        return s_mermin_bell(data_qubit_num)
    elif algorithm == "bv_hard":
        return bv_hard(data_qubit_num)
    else:
        raise ValueError(f"Unknown program name: {algorithm}")
