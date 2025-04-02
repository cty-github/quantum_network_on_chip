from random import sample
from collections import Counter, defaultdict


class Program:
    def __init__(self, data_qubit_num, logic_gates):
        self.data_qubit_num = data_qubit_num
        self.logic_gates = logic_gates


def VQE(data_qubit_num):
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.append('barrier')
        control_gates = [(i, j) for j in range(i+1, data_qubit_num)]
        logic_gates.extend(control_gates)
    return Program(data_qubit_num, logic_gates)

def QFT(data_qubit_num):
    logic_gates = []
    for i in range(data_qubit_num-1):
        logic_gates.append('barrier')
        control_gates = [(j, i) for j in range(i+1, data_qubit_num)]
        logic_gates.append(control_gates)
    return Program(data_qubit_num, logic_gates)

def QAOA(data_qubit_num, percentage=0.5, blocklayer=1):
    all_possible_gates = [(i, j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    logic_gates = []
    for i, val in enumerate(control_gates):
        logic_gates.append(val)
        if (i+1) % (data_qubit_num*blocklayer) == 0 and (i+1) != len(control_gates):
            logic_gates.append('barrier')
    return Program(data_qubit_num, logic_gates)

def BV(data_qubit_num, percentage=0.5):
    all_possible_gates = [(0, i) for i in range(1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    control_gates.sort()
    return Program(data_qubit_num, control_gates)
