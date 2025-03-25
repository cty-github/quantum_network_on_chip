import numpy as np
from copy import deepcopy
from random import shuffle
from collections import defaultdict, deque


class CommutableBlock:
    def __init__(self):
        self.cnot_gate_list = []
        self.control_qubits = []
        self.target_qubits = []


class TeleBlock:
    def __init__(self):
        self.tele_gate_list = []
        self.control_qubits = []
        self.target_qubits = []


class MoveBlock:
    def __init__(self):
        self.move_gate_list = []
        self.control_qubits = []
        self.target_qubits = []


class Circuit: 
    def __init__(self, qubit_num):
        '''
        Structure of physical quantum circuit with enhanced quantum state transfer
        '''
        self.qubit_num = qubit_num
        self.circuit_lines = defaultdict(list) # each line contains a list of (node, role)
        self.commutable_blocks = []
        self.compute_blocks = []
        self.circuit_stages = []
        