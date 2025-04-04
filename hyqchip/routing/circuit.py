import numpy as np
from copy import deepcopy
from random import shuffle
from collections import defaultdict, deque


class Circuit: 
    def __init__(self, package, logic_qubit_num):
        '''
        Structure of physical quantum circuit with enhanced quantum state transfer.
        Circuit with enhanced quantum state transfer is a list interleaved with operating stage and boarding stage.
        Operation stage: 1) transport on transfer die; 2) local compute on compute die; 3) routing to stops.
        Boarding stage: 1) boarding on/off transfer die; 2) GHZ state preparation; 3) transfer between transfer dies.
        '''
        
        self.package = package
        self.logic_qubit_num = logic_qubit_num
        self.circuit_stages: list[CircuitStage] = []
    
    def estimate_fidelity(self):
        pass
    
    def estimate_depth(self):
        pass


class CircuitAction:
    def __init__(self):
        pass


class OperatingAction(CircuitAction):
    def __init__(self):
        pass


class BoardingAction(CircuitAction):
    def __init__(self):
        pass


class CircuitStage:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.id = len(self.circuit.circuit_stages)
        self.occupied_nodes = []
    
    def add_occupied_nodes(self, occupied_node):
        self.occupied_nodes.append(occupied_node)


class OperatingStage(CircuitStage):
    def __init__(self, circuit: Circuit):
        super().__init__(circuit)
        self.operating_actions: list[OperatingAction] = []
        self.phy_qubit_operate_layer = []
    
    def add_operating_action(self, action: OperatingAction):
        self.operating_actions.append(action)
        if isinstance(action, LocalCompute):
            stage_layer = action.stage_layer
            self.phy_qubit_operate_layer[stage_layer].extend(action.physical_nodes)
            


class BoardingStage(CircuitStage):
    def __init__(self, circuit: Circuit):
        super().__init__(circuit)
        self.boarding_actions: list[BoardingAction] = []


class TransportOnRail(OperatingAction):
    def __init__(self, logic_qubit, source_physical_qubit, target_physical_qubit):
        '''
        Action to transport a logic qubit inside the transfer die.

        logic_qubit: int, the logic qubit to transport
        source_physical_qubit: node, where the transport start
        target_physical_qubit: node, where the transport end
        '''
        self.logic_qubit: int = logic_qubit
        self.source_physical_qubit: tuple[int] = source_physical_qubit
        self.target_physical_qubit: tuple[int] = target_physical_qubit


class RouteToStop(OperatingAction):
    def __init__(self, logic_qubit, stop, swap_gates, enter_flag):
        '''
        Action of routing a data qubit to stop.

        logic_qubit: int, the data qubit to route
        stop: node, the physical qubit on compute die to route to
        swap_gates: list[edge], the swap gate list used to route
        enter_flag: bool, True if the data qubit is needed to route onto the stop
        '''
        self.logic_qubit: int = logic_qubit
        self.stop: tuple[int] = stop
        self.swap_gates = swap_gates
        self.enter_flag: bool = enter_flag


class Teleport(OperatingAction):
    def __init__(self, data_qubit, ancilla_qubit, stop, control_flag):
        '''
        Action to complete the teleporation given prepared GHZ state.

        data_qubit: int, the data qubit to teleport
        ancilla_qubit: int, the ancilla_qubit in GHZ state for teleportation
        stop: node, the physical qubit where the ancilla_qubit would be routed to
        control_flag: bool, True if the data qubit is the control qubit
        '''
        self.data_qubit: int = data_qubit
        self.ancilla_qubit: int = ancilla_qubit
        self.stop: tuple[int] = stop
        self.control_flag: bool = control_flag


class LocalCompute(OperatingAction):
    def __init__(self, logic_gate, physical_nodes, stage_layer):
        '''
        Action of control gate within a compute area.

        logic_gate: (int, int), the logical control gate
        physical_nodes: list[node], the physical qubits along which the local compute happens, include start and end.
        stage_layer: int, the layer num where the opt happens in the stage
        '''

        self.logic_gate: tuple[int] = logic_gate
        self.physical_nodes = physical_nodes
        self.stage_layer = stage_layer


class GhzPreparation(BoardingAction):
    def __init__(self, qubit_num, logic_qubits, source_physical_qubits, target_physical_qubits):
        '''
        Action to prepare GHZ state.

        qubit_num: int, number of ancilla qubits in the GHZ state
        logic_qubits: list[int], the data qubits that the GHZ state is serving for
        source_physical_qubits: list[node], the list of physical qubits where the GHZ state is prepared
        target_physical_qubits: list[node], the list of physical qubits where the GHZ state will be routed to
        '''
        self.qubit_num: int = qubit_num
        self.logic_qubits = logic_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits


class OnOffBoard(BoardingAction):
    def __init__(self, logic_qubit, stop, on_off_flag):
        '''
        Action of logic qubit on/off boarding the transfer die.

        logic_qubit: int, the logic qubit to board
        stop: node, the physical qubit on compute die where the logic qubit is boarding
        on_off_flag: bool, True if the logic qubit is going on board
        '''
        self.logic_qubit: int = logic_qubit
        self.stop: tuple[int] = stop
        self.on_off_flag: bool = on_off_flag


class TransferBetweenDies(BoardingAction):
    def __init__(self, logic_qubit, interchange):
        '''
        Action to transfer logic qubit between two transfer dies.

        logic_qubit: int, the logic qubit to transfer
        interchange: edge, the edge between two transfer dies for transfer
        '''
        self.logic_qubit: int = logic_qubit
        self.interchange: tuple[tuple[int]] = interchange


class TransferDieRoute:
    def __init__(self, logic_qubit, source_physical_qubit, target_physical_qubit, actions):
        '''
        Actions to transport a logic qubit among the transfer die system.

        logic_qubit: int, the logic qubit to transport
        source_physical_qubit: node, where the transport start
        target_physical_qubit: node, where the transport end
        actions: list[CircuitAction], actions interleaved with TransportOnDie and TransferBetweenDies
        '''
        self.logic_qubit: int = logic_qubit
        self.source_physical_qubit: tuple[int] = source_physical_qubit
        self.target_physical_qubit: tuple[int] = target_physical_qubit
        self.actions = actions
