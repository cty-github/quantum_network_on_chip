import networkx as nx
import numpy as np
from copy import deepcopy


class CircuitAction:
    _counter = 0
    def __init__(self):
        self.id = CircuitAction._counter
        CircuitAction._counter += 1


class BoardingAction(CircuitAction):
    def __init__(self):
        super().__init__()


class OperatingAction(CircuitAction):
    def __init__(self):
        super().__init__()


class GhzPreparation(BoardingAction):
    def __init__(self, qubit_num, logic_qubits, source_physical_qubits, target_physical_qubits):
        '''
        Action to prepare GHZ state.

        qubit_num: int, number of ancilla qubits in the GHZ state
        logic_qubits: list[int], the data qubits that the GHZ state is serving for
        source_physical_qubits: list[node], the list of physical qubits where the GHZ state is prepared
        target_physical_qubits: list[node], the list of physical qubits where the GHZ state will be routed to
        '''

        super().__init__()
        self.qubit_num: int = qubit_num
        self.logic_qubits = logic_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits


class OnOffBoard(BoardingAction):
    def __init__(self, logic_qubit, station, on_off_flag):
        '''
        Action of logic qubit on/off boarding the transfer die.

        logic_qubit: int, the logic qubit to board
        stop: node, the physical qubit on compute die where the logic qubit is boarding
        on_off_flag: bool, True if the logic qubit is going on board
        '''

        super().__init__()
        self.logic_qubit: int = logic_qubit
        self.station: tuple[int] = station
        self.on_off_flag: bool = on_off_flag


class TransferBetweenDies(BoardingAction):
    def __init__(self, logic_qubit, interchange):
        '''
        Action to transfer logic qubit between two transfer dies.

        logic_qubit: int, the logic qubit to transfer
        interchange: edge, the edge between two transfer dies for transfer
        '''

        super().__init__()
        self.logic_qubit: int = logic_qubit
        self.interchange: tuple[tuple[int]] = interchange
            

class TransportOnRail(OperatingAction):
    def __init__(self, logic_qubit, source_physical_qubit, target_physical_qubit):
        '''
        Action to transport a logic qubit inside the transfer die.

        logic_qubit: int, the logic qubit to transport
        source_physical_qubit: node, where the transport start
        target_physical_qubit: node, where the transport end
        '''

        super().__init__()
        self.logic_qubit: int = logic_qubit
        self.source_physical_qubit: tuple[int] = source_physical_qubit
        self.target_physical_qubit: tuple[int] = target_physical_qubit


class RouteToPlat(OperatingAction):
    def __init__(self, logic_qubit, platform, enter_flag):
        '''
        Action of routing a data qubit to stop.

        logic_qubit: int, the data qubit to route
        stop: node, the physical qubit on compute die to route to
        enter_flag: bool, True if the data qubit is needed to route onto the stop
        '''

        super().__init__()
        self.logic_qubit: int = logic_qubit
        self.platform: tuple[int] = platform
        self.enter_flag: bool = enter_flag


class Teleport(OperatingAction):
    def __init__(self, logic_qubit, platform, control_flag):
        '''
        Action to complete the teleporation given prepared GHZ state.

        logic_qubit: int, the logic qubit to teleport
        platform: node, the physical qubit on compute die where the ancilla qubit is located
        control_flag: bool, True if the data qubit is the control qubit
        '''

        super().__init__()
        self.logic_qubit: int = logic_qubit
        self.platform: tuple[int] = platform
        self.control_flag: bool = control_flag


class LocalCompute(OperatingAction):
    def __init__(self, logic_gate, area):
        '''
        Action of control gate within a compute area.

        logic_gate: (int, int), the logical control gate
        area: dict, the compute area where the control gate is executed
        '''

        super().__init__()
        self.logic_gate: tuple[int] = logic_gate
        self.area = area


# class TransferDieRoute:
#     def __init__(self, logic_qubit, source_physical_qubit, target_physical_qubit, actions):
#         '''
#         Actions to transport a logic qubit among the transfer die system.

#         logic_qubit: int, the logic qubit to transport
#         source_physical_qubit: node, where the transport start
#         target_physical_qubit: node, where the transport end
#         actions: list[CircuitAction], actions interleaved with TransportOnDie and TransferBetweenDies
#         '''

#         self.logic_qubit: int = logic_qubit
#         self.source_physical_qubit: tuple[int] = source_physical_qubit
#         self.target_physical_qubit: tuple[int] = target_physical_qubit
#         self.actions = actions


class CircuitStage:
    _counter = 0

    def __init__(self):
        self.id = CircuitStage._counter
        CircuitStage._counter += 1
        self.occupied_nodes = []
    
    def add_occupied_nodes(self, occupied_node):
        self.occupied_nodes.append(occupied_node)


class BoardingStage(CircuitStage):
    def __init__(self):
        super().__init__()
        assert self.id % 2 == 0, "Boarding stage id should be even"
        self.boarding_actions: list[BoardingAction] = []


class OperatingStage(CircuitStage):
    def __init__(self):
        super().__init__()
        assert self.id % 2 == 1, "Operating stage id should be odd"
        self.operating_actions: list[OperatingAction] = []
    
    def add_operating_action(self, action: OperatingAction, depend_action_id: list[int]):
        self.operating_actions.append(action)


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
        self.stages: list[CircuitStage] = []
    
    def add_stage(self, stage: CircuitStage):
        assert stage.id == len(self.stages), "Stage id should be equal to the length of circuit stages"
        if stage.id % 2 == 0:
            assert isinstance(stage, BoardingStage), "Even stage id should be a boarding stage"
        else:
            assert isinstance(stage, OperatingStage), "Odd stage id should be an operating stage"
        self.stages.append(stage)
    
    def extend_two_stages(self):
        self.add_stage(BoardingStage())
        self.add_stage(OperatingStage())
    
    def estimate_fidelity(self):
        pass
    
    def estimate_depth(self):
        pass
