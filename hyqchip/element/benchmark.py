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
        return f"control_{self.control_qubit}_{self.target_qubit}"
    
class Barrier(Gate):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "barrier"


class Program:
    def __init__(self, algo_name, data_qubit_num, logic_gates: list[Gate]):
        self.algo_name = algo_name
        self.data_qubit_num = data_qubit_num
        self.logic_gates = logic_gates

        recent_qubit_info = {i: {"operated": False, "all_depend": -1, "ctrl_depend": -1} for i in range(data_qubit_num)}
        self.gate_depend_graph = nx.DiGraph()
        for gate in logic_gates:
            if isinstance(gate, Barrier):
                self.gate_depend_graph.add_node(gate.id, gate=gate)
                for i in range(data_qubit_num):
                    if recent_qubit_info[i]["operated"]:
                        self.gate_depend_graph.add_edge(recent_qubit_info[i]["all_depend"], gate.id)
                for i in range(data_qubit_num):
                    recent_qubit_info[i]["operated"] = True
                    recent_qubit_info[i]["all_depend"] = gate.id
                    recent_qubit_info[i]["ctrl_depend"] = gate.id
            elif isinstance(gate, Control):
                control = gate.control_qubit
                target = gate.target_qubit
                self.gate_depend_graph.add_node(gate.id, gate=gate)
                if recent_qubit_info[control]["ctrl_depend"] != -1:
                    self.gate_depend_graph.add_edge(recent_qubit_info[control]["ctrl_depend"], gate.id)
                if recent_qubit_info[target]["operated"]:
                    self.gate_depend_graph.add_edge(recent_qubit_info[target]["all_depend"], gate.id)
                recent_qubit_info[control]["operated"] = True
                recent_qubit_info[control]["all_depend"] = gate.id
                recent_qubit_info[target]["operated"] = True
                recent_qubit_info[target]["all_depend"] = gate.id
                recent_qubit_info[target]["ctrl_depend"] = gate.id
            else:
                raise ValueError("Unknown gate type")
            
        for node in self.gate_depend_graph:
            if isinstance(self.gate_depend_graph.nodes[node]["gate"], Barrier):
                pred = list(self.gate_depend_graph.predecessors(node))
                succ = list(self.gate_depend_graph.successors(node))
                for p in pred:
                    for s in succ:
                        if p != s:
                            self.gate_depend_graph.add_edge(p, s)
                self.gate_depend_graph.remove_node(node)
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
        logic_gates.append(Barrier())
        control_gates = [Control(i, j) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program("vqe", data_qubit_num, logic_gates)

def QFT(data_qubit_num):
    Gate._counter = 0
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.append(Barrier())
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
            logic_gates.append(Barrier())
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
