import networkx as nx
from random import sample
import matplotlib.pyplot as plt
from ..utils.utils import find_longest_topo_distance


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

        recent_qubit_info = {i: {"recent_role": None, "recent_ctrls": [], "recent_other": None} for i in range(data_qubit_num)}
        self.gate_depend_graph: nx.DiGraph = nx.DiGraph()
        for gate in logic_gates:
            if isinstance(gate, Barrier):
                self.gate_depend_graph.add_node(gate.id, gate=gate)
                qubit = gate.qubit
                if recent_qubit_info[qubit]["recent_role"]:
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
                if recent_qubit_info[control]["recent_role"]:
                    if recent_qubit_info[control]["recent_role"] == "control":
                        if recent_qubit_info[control]["recent_other"]:
                            self.gate_depend_graph.add_edge(recent_qubit_info[control]["recent_other"], gate.id)
                    else:
                        self.gate_depend_graph.add_edge(recent_qubit_info[control]["recent_other"], gate.id)
                recent_qubit_info[control]["recent_role"] = "control"
                recent_qubit_info[control]["recent_ctrls"].append(gate.id)
                target = gate.target_qubit
                if recent_qubit_info[target]["recent_role"]:
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

        assert nx.is_directed_acyclic_graph(self.gate_depend_graph)
    
    def __repr__(self):
        return f"{self.algo_name} program: " + ", ".join([str(self.gate_depend_graph.nodes[id]["gate"]) for id in nx.topological_sort(self.gate_depend_graph)])

    def draw(self, 
        marginrate: float = 0.2, 
        markersize: float = 5, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
        show: bool = True, 
        save_path: str = None, 
    ):
        """
        Draw the gate dependency graph to present the program.
        """
        pos = nx.spring_layout(self.gate_depend_graph)
        figsize = (find_longest_topo_distance(self.gate_depend_graph), find_longest_topo_distance(self.gate_depend_graph))
        plt.figure(figsize=figsize)
        nx.draw(
            self.gate_depend_graph, 
            pos, 
            with_labels=False, 
            arrows=True, 
            node_size=markersize, 
            width=linewidth_t,
            node_color='black', 
            edge_color='black', 
        )
        labels = {node: str(node) for node in self.gate_depend_graph.nodes()}
        label_offset = (-0.04, 0.03)
        label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(
            self.gate_depend_graph,
            label_pos,
            labels=labels,
            font_size=10,
            font_color='black'
        )
        plt.title(f"Program {self.algo_name}-{self.data_qubit_num}")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

def VQE(data_qubit_num: int) -> Program:
    """
    Generate a VQE program.

    Args:
        data_qubit_num (int): Number of data qubits.

    Returns:
        Program: A quantum program with VQE logic gates.
    """
    Gate._counter = 0
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
        control_gates = [Control(i, j) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program("VQE", data_qubit_num, logic_gates)

def QFT(data_qubit_num: int) -> Program:
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
    return Program("QFT", data_qubit_num, logic_gates)

def QAOA(data_qubit_num: int, percentage: float = 0.5, blocklayer: int = 1) -> Program:
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
    return Program("QAOA", data_qubit_num, logic_gates)

def BV(data_qubit_num: int, percentage: float = 0.5) -> Program:
    """
    Generate a BV program.

    Args:
        data_qubit_num (int): Number of data qubits.
        percentage (float): Percentage of control gates to be selected.

    Returns:
        Program: A quantum program with BV logic gates.
    """
    Gate._counter = 0
    all_possible_gates = [(0, i) for i in range(1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    control_gates.sort()
    logic_gates = []
    for gate in control_gates:
        logic_gates.append(Control(*gate))
    return Program("BV", data_qubit_num, logic_gates)
