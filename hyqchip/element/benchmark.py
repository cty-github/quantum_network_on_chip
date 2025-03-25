from random import sample
from collections import Counter, defaultdict


class ControlBlock:
    def __init__(self, control, target_counter=Counter()):
        self.control = control
        self.target_counter = target_counter


def VQE(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        control_block = ControlBlock(i, Counter(list(range(i+1, data_qubit_num))))
        program.append(control_block)
    return program

def QFT(data_qubit_num):
    program = []
    for i in range(data_qubit_num-1):
        control_block = ControlBlock(i, Counter(list(range(i+1, data_qubit_num))*2))
        program.append(control_block)
    return program

def QAOA(data_qubit_num, percentage=0.5):
    all_possible_gates = [(i,j) for i in range(data_qubit_num) for j in range(i+1, data_qubit_num)]
    gates = list(sample(all_possible_gates, k=int(len(all_possible_gates) * percentage)))
    gates.sort()
    print('{}/{} gates are selected: {}...{}'.format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))

    program = []
    control_target_dict = defaultdict(list)
    for control, target in gates:
        control_target_dict[control].append(target)
    for control, target_list in control_target_dict.items():
        control_block = ControlBlock(control, Counter(target_list*2))
        program.append(control_block)
    return program

def BV(data_qubit_num, percentage=0.5):
    all_possible_gates = [(0,i) for i in range(1, data_qubit_num)]
    gates = list(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage)))
    gates.sort()
    
    program = []
    control_target_dict = defaultdict(list)
    for control, target in gates:
        control_target_dict[control].append(target)
    for control, target_list  in control_target_dict.items():
        control_block = ControlBlock(control, Counter(target_list))
        program.append(control_block)
    return program
