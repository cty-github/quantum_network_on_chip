import networkx as nx
from copy import deepcopy

from hyqchip.element.benchmark import *


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
    def __init__(self, qubit_num: int, logic_qubits: list[int], 
                 source_physical_qubits: list[tuple[int, int]], 
                 target_physical_qubits: list[tuple[int, int]]):
        '''
        Action to prepare GHZ state.

        qubit_num: int, number of ancilla qubits in the GHZ state
        logic_qubits: list[int], the data qubits that the GHZ state is serving for
        source_physical_qubits: list[node], the list of physical qubits where the GHZ state is prepared
        target_physical_qubits: list[node], the list of physical qubits where the GHZ state will be routed to
        '''

        super().__init__()
        self.qubit_num = qubit_num
        self.logic_qubits = logic_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits


class Board(BoardingAction):
    def __init__(self, logic_qubit: int, station: tuple[int, int], platform: tuple[int, int]):
        '''
        Action of logic qubit boarding the transfer die.

        logic_qubit: int, the logic qubit to board
        station: node, the physical qubit on transfer die where the logic qubit is boarding
        platform: node, the physical qubit on compute die where the logic qubit is boarding
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.station = station
        self.platform = platform


class Alight(BoardingAction):
    def __init__(self, logic_qubit: int, station: tuple[int, int], platform: tuple[int, int], exit_flag: bool):
        '''
        Action of logic qubit alighting the transfer die.

        logic_qubit: int, the logic qubit to alight
        station: node, the physical qubit on transfer die where the logic qubit is alighting
        paltform: node, the physical qubit on compute die where the logic qubit is alighting
        exit_flag: bool, True if the logic qubit is needed to exit from the platform
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.station = station
        self.platform = platform
        self.exit_flag = exit_flag


class Interchange(BoardingAction):
    def __init__(self, logic_qubit: int, interchange: tuple[tuple[int, int], tuple[int, int]]):
        '''
        Action to transfer logic qubit between two transfer dies.

        logic_qubit: int, the logic qubit to transfer
        interchange: edge, the edge between two transfer dies for transfer
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.interchange = interchange
            

class Transport(OperatingAction):
    def __init__(self, logic_qubit: int, source_physical_qubit: tuple[int, int], target_physical_qubit: tuple[int, int]):
        '''
        Action to transport a logic qubit inside the transfer die.

        logic_qubit: int, the logic qubit to transport
        source_physical_qubit: node, where the transport start
        target_physical_qubit: node, where the transport end
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.source_physical_qubit = source_physical_qubit
        self.target_physical_qubit = target_physical_qubit


class RouteToPlat(OperatingAction):
    def __init__(self, logic_qubit: int, platform: tuple[int, int], enter_flag: bool):
        '''
        Action of routing a data qubit to platform.

        logic_qubit: int, the data qubit to route
        platform: node, the physical qubit on compute die to route to
        enter_flag: bool, True if the data qubit is needed to route onto the platform
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.enter_flag = enter_flag


class Teleport(OperatingAction):
    def __init__(self, logic_qubit: int, platform: tuple[int, int], control_flag: bool):
        '''
        Action to complete the teleporation given prepared GHZ state.

        logic_qubit: int, the logic qubit to teleport
        platform: node, the physical qubit on compute die where the ancilla qubit is located
        control_flag: bool, True if the data qubit is the control qubit
        '''

        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.control_flag = control_flag


class LocalGate(OperatingAction):
    def __init__(self, logic_gate: Control, area: dict):
        '''
        Action of control gate within a compute area.

        logic_gate: Control, the logical control gate
        area: dict, the compute area where the control gate is executed
        '''

        super().__init__()
        self.logic_gate = logic_gate
        self.area = area


class CircuitStage:
    _counter = 0

    def __init__(self, init_occupied_balises: set[tuple[int, int]], init_occupied_platforms: set[tuple[int, int]], holding_ancilla_plats: set[tuple[int, int]]):
        self.id = CircuitStage._counter
        CircuitStage._counter += 1

        self.init_occupied_balises = init_occupied_balises
        self.curr_occupied_balises = deepcopy(init_occupied_balises)
        self.init_occupied_platforms = init_occupied_platforms
        self.curr_occupied_platforms = deepcopy(init_occupied_platforms)
        self.holding_ancilla_plats = holding_ancilla_plats


class BoardingStage(CircuitStage):
    def __init__(self, init_occupied_balises, init_occupied_platforms, holding_ancilla_plats):
        super().__init__(init_occupied_balises, init_occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 0, "Boarding stage id should be even"
        self.boarding_actions: list[BoardingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises = set()
        self.potential_free_platforms = dict()
        self.paired_platforms = set()

    def add_alight_action(self, action: Alight) -> bool:
        '''
        Add an Alight to the boarding stage.
        The action is added to the end of the boarding actions list.
        '''

        assert isinstance(action, Alight), "Action should be Alight"
        assert action.station in self.curr_occupied_balises, "Data qubit to alight should be in station."

        if action.platform in self.curr_occupied_platforms:
            if action.platform in self.potential_free_platforms and self.potential_free_platforms[action.platform] == action.station:
                self.potential_free_platforms.pop(action.platform)
                self.paired_platforms.add(action.platform)
            elif action.station in self.paired_balises:
                self.paired_balises.remove(action.station)
            else:
                self.potential_free_balises[action.station] = action.platform
                return False
            if action.exit_flag:
                self.curr_occupied_platforms.remove(action.platform)
            else:
                self.holding_ancilla_plats.add(action.platform)
        else:
            self.curr_occupied_balises.remove(action.station)
            if not action.exit_flag:
                self.curr_occupied_platforms.add(action.platform)
                self.holding_ancilla_plats.add(action.platform)

        self.boarding_actions.append(action)
        return True
    
    def add_interchange_action(self, action: Interchange) -> bool:
        '''
        Add an Interchange to the boarding stage.
        The action is added to the end of the boarding actions list.
        '''

        assert isinstance(action, Interchange), "Action should be Interchange"
        assert action.interchange[0] in self.curr_occupied_balises, "Data qubit to interchange should be in source interchange."

        if action.interchange[1] in self.curr_occupied_balises:
            if action.interchange[1] in self.potential_free_balises and self.potential_free_balises[action.interchange[1]] == action.interchange[0]:
                self.potential_free_balises.pop(action.interchange[1])
                self.paired_balises.add(action.interchange[1])
            elif action.interchange[0] in self.paired_balises:
                self.paired_balises.remove(action.interchange[0])
            else:
                self.potential_free_balises[action.interchange[0]] = action.interchange[1]
                return False
        else:
            self.curr_occupied_balises.remove(action.interchange[0])
            self.curr_occupied_balises.add(action.interchange[1])

        self.boarding_actions.append(action)
        return True
    
    def add_board_action(self, action: Board) -> bool:
        '''
        Add a Board to the boarding stage.
        The action is added to the end of the boarding actions list.
        '''

        assert isinstance(action, Board), "Action should be Board"
        assert action.platform in self.curr_occupied_platforms or action.platform in self.paired_platforms, "Data qubit to board should be in platform."
        
        if action.station in self.curr_occupied_balises:
            if action.station in self.potential_free_balises and self.potential_free_balises[action.station] == action.platform:
                self.potential_free_balises.pop(action.station)
                self.paired_balises.add(action.station)
            elif action.platform in self.paired_platforms:
                self.paired_platforms.remove(action.platform)
            else:
                self.potential_free_platforms[action.platform] = action.station
                return False
        else:
            self.curr_occupied_platforms.remove(action.platform)
            self.curr_occupied_balises.add(action.station)

        self.boarding_actions.append(action)
        return True
    
    def add_ghz_action(self, action: GhzPreparation) -> bool:
        '''
        Add a GhzPreparation to the boarding stage.
        The action is added to the end of the boarding actions list.
        '''

        assert isinstance(action, GhzPreparation), "Action should be GhzPreparation"

        for pysical_qubit in action.source_physical_qubits:
            if pysical_qubit in self.curr_occupied_balises:
                return False
        self.curr_occupied_balises.update(action.source_physical_qubits)

        self.boarding_actions.append(action)
        return True


class OperatingStage(CircuitStage):
    def __init__(self, init_occupied_balises, init_occupied_platforms, holding_ancilla_plats):
        super().__init__(init_occupied_balises, init_occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 1, "Operating stage id should be odd"
        self.operating_actions: list[OperatingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises = set()
    
    def add_transport_action(self, action: Transport) -> bool:
        '''
        Add a Transport to the operating stage.
        The action is added to the end of the operating actions list.
        '''

        assert isinstance(action, Transport), "Action should be Transport"
        assert action.source_physical_qubit in self.curr_occupied_balises, "Data qubit to transport should be in source balise."

        if action.target_physical_qubit == action.source_physical_qubit:
            pass
        elif action.target_physical_qubit in self.curr_occupied_balises:
            if action.target_physical_qubit in self.potential_free_balises and self.potential_free_balises[action.target_physical_qubit] == action.source_physical_qubit:
                self.potential_free_balises.pop(action.target_physical_qubit)
                self.paired_balises.add(action.target_physical_qubit)
            elif action.source_physical_qubit in self.paired_balises:
                self.paired_balises.remove(action.source_physical_qubit)
            else:
                self.potential_free_balises[action.source_physical_qubit] = action.target_physical_qubit
                return False
        else:
            self.curr_occupied_balises.remove(action.source_physical_qubit)
            self.curr_occupied_balises.add(action.target_physical_qubit)
        
        self.operating_actions.append(action)
        return True
    
    def add_route_action(self, action: RouteToPlat) -> bool:
        '''
        Add a RouteToPlat to the operating stage.
        The action is added to the end of the operating actions list.
        '''

        assert isinstance(action, RouteToPlat), "Action should be RouteToPlat"

        if action.enter_flag:
            if action.platform in self.curr_occupied_platforms:
                return False
            self.curr_occupied_platforms.add(action.platform)
            
        self.operating_actions.append(action)
        return True
    
    def add_teleport_action(self, action: Teleport) -> bool:
        '''
        Add a Teleport to the operating stage.
        The action is added to the end of the operating actions list.
        '''

        assert isinstance(action, Teleport), "Action should be Teleport"

        self.curr_occupied_platforms.remove(action.platform)
        self.holding_ancilla_plats.remove(action.platform)

        self.operating_actions.append(action)
        return True

    def add_local_action(self, action: LocalGate) -> bool:
        '''
        Add a LocalGate to the operating stage.
        The action is added to the end of the operating actions list.
        '''

        assert isinstance(action, LocalGate), "Action should be LocalGate"

        self.operating_actions.append(action)
        return True


class Circuit: 
    def __init__(self, package, logic_qubit_num, init_l2p_map: dict, init_p2l_map: dict, action_depend_graph: nx.DiGraph):
        '''
        Structure of physical quantum circuit with enhanced quantum state transfer.
        Circuit with enhanced quantum state transfer is a list interleaved with operating stage and boarding stage.
        Operation stage: 1) transport on transfer die; 2) local compute on compute die; 3) routing to platforms; 4) teleportation.
        Boarding stage: 1) alighting transfer die; 2) boarding transfer die; 3) GHZ state preparation; 4) transfer between transfer dies. 
        '''
        
        self.package = package
        self.logic_qubit_num = logic_qubit_num
        self.init_l2p_map = init_l2p_map
        self.init_p2l_map = init_p2l_map
        self.action_depend_graph = action_depend_graph

        self.stages: list[CircuitStage] = [] # only use extend_stage to add new stage

        self.curr_l2p_map = deepcopy(init_l2p_map)
        self.curr_p2l_map = deepcopy(init_p2l_map)
        self.hold_ancilla_plats = set()
    
    def extend_stage(self, board_flag: bool):
        '''
        Extend the circuit with a new stage.
        The new stage is added to the end of the circuit.
        boarding_flag: bool, True if the new stage is a boarding stage
        '''

        if board_flag:
            assert len(self.stages) % 2 == 0, "Boarding stage should have even id"
        else:
            assert len(self.stages) % 2 == 1, "Operating stage should have odd id"
        curr_occupied_balises = self.stages[-1].curr_occupied_balises if len(self.stages) else set()
        curr_occupied_platforms = self.stages[-1].curr_occupied_platforms if len(self.stages) else set()
        if board_flag:
            new_stage = BoardingStage(curr_occupied_balises, curr_occupied_platforms, self.hold_ancilla_plats)
        else:
            new_stage = OperatingStage(curr_occupied_balises, curr_occupied_platforms, self.hold_ancilla_plats)
        assert new_stage.id == len(self.stages), "Stage id should be equal to the length of existing stages"
        self.stages.append(new_stage)
        return new_stage
    
    def estimate_fidelity(self):
        '''
        Estimate the fidelity of the circuit.
        '''

        # TODO: complete the function
        pass

    def estimate_average_qst_fide(self):
        '''
        Estimate the average fidelity of each quantum state transfer in the circuit.
        '''

        # TODO: complete the function
        pass
    
    def estimate_depth(self):
        '''
        Estimate the depth of the circuit.
        '''

        # TODO: complete the function
        pass
