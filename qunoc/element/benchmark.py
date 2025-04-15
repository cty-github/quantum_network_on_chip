import networkx as nx
from random import sample
import matplotlib.pyplot as plt


class Gate:
    _counter = 0

    def __init__(self):
        self.id = Gate._counter
        Gate._counter += 1

class Control(Gate):
    def __init__(self, control_qubit, target_qubit):
        super().__init__()
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.control_qubit}_{self.target_qubit}"
    
class Barrier(Gate):
    def __init__(self, qubit):
        super().__init__()
        self.qubit = qubit

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.qubit}"


class Program:
    def __init__(self, algo_name, data_qubit_num, logic_gates: list[Gate]):
        self.algo_name = algo_name
        self.data_qubit_num = data_qubit_num
        self.logic_gates = logic_gates

        recent_qubit_info = {i: {"recent_role": None, "recent_ctrls": [], "recent_other": None} for i in range(data_qubit_num)}
        self.gate_depend_graph = nx.DiGraph()
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
        return ", ".join([str(self.gate_depend_graph.nodes[id]["gate"]) for id in nx.topological_sort(self.gate_depend_graph)])

    def draw(self):
        pos = nx.spring_layout(self.gate_depend_graph)
        plt.figure()
        nx.draw(self.gate_depend_graph, pos, with_labels=True, arrows=True, node_size=1000, node_color='lightblue', edge_color='gray')
        plt.title(f"Program {self.algo_name}-{self.data_qubit_num}")


def VQE(data_qubit_num):
    Gate._counter = 0
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
        control_gates = [Control(i, j) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program("vqe", data_qubit_num, logic_gates)

def QFT(data_qubit_num):
    Gate._counter = 0
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
        control_gates = [Control(j, i) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program("qft", data_qubit_num, logic_gates)

def QAOA(data_qubit_num, percentage=0.5, blocklayer=1):
    Gate._counter = 0
    all_possible_gates = [(i, j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    logic_gates = []
    for i, gate in enumerate(control_gates):
        logic_gates.append(Control(*gate))
        if (i+1) % (data_qubit_num*blocklayer) == 0 and (i+1) != len(control_gates):
            logic_gates.extend([Barrier(j) for j in range(data_qubit_num-1)])
    return Program("qaoa", data_qubit_num, logic_gates)

def BV(data_qubit_num, percentage=0.5):
    Gate._counter = 0
    all_possible_gates = [(0, i) for i in range(1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    control_gates.sort()
    logic_gates = []
    for gate in control_gates:
        logic_gates.append(Control(*gate))
    return Program("bv", data_qubit_num, logic_gates)
