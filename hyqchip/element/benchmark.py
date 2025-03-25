from random import sample
from collections import Counter, defaultdict


def VQE(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        program.append('barrier')
        control_gates = [(i, j) for j in range(i+1, data_qubit_num)]
        program.extend(control_gates)
    return program

def QFT(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        program.append('barrier')
        control_gates = [(j, i) for j in range(i+1, data_qubit_num)]
        program.append(control_gates)
    return program

def QAOA(data_qubit_num, percentage=0.5, blocklayer=1):
    all_possible_gates = [(i, j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    program = []
    for i, val in enumerate(control_gates):
        program.append(val)
        if (i+1) % (data_qubit_num*blocklayer) == 0 and (i+1) != len(control_gates):
            program.append('barrier')
    return program

def BV(data_qubit_num, percentage=0.5):
    all_possible_gates = [(0, i) for i in range(1, data_qubit_num)]
    control_gates = sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))
    control_gates.sort()
    program = control_gates
    return program
