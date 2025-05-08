import networkx as nx
from copy import deepcopy

from ..element.benchmark import *
from ..element.chiplet import ComputeModule
from ..element.package import Package


class CircuitAction:
    """
    Represents the action in the circuit.

    Attributes:
        id (int): Unique identifier for the action.
    """
    _counter = 0

    def __init__(self):
        """
        Initialize a circuit action.
        """
        self.id = CircuitAction._counter
        CircuitAction._counter += 1

class BoardingAction(CircuitAction):
    """
    Represents the boarding action in the circuit.
    """
    
    def __init__(self):
        """
        Initialize a boarding action.
        """
        super().__init__()

class RidingAction(CircuitAction):
    """
    Represents the riding action in the circuit.
    """

    def __init__(self):
        """
        Initialize a riding action.
        """
        super().__init__()

class GhzPreparation(BoardingAction):
    """
    Represents the action to prepare GHZ state.

    Attributes:
        qubit_num (int): Number of ancilla qubits in the GHZ state.
        logic_qubits (list[int]): The data qubits that the GHZ state is serving for.
        source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
        target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
    """

    def __init__(self, 
            qubit_num: int, 
            logic_qubits: list[int], 
            source_physical_qubits: list[tuple[int, int]], 
            target_physical_qubits: list[tuple[int, int]], 
        ):
        """
        Initialize the GHZ preparation action.

        Args:
            qubit_num (int): Number of ancilla qubits in the GHZ state.
            logic_qubits (list[int]): The data qubits that the GHZ state is serving for.
            source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
            target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
        """
        super().__init__()
        self.qubit_num = qubit_num
        self.logic_qubits = logic_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits

class Board(BoardingAction):
    """
    Represents the action of a logic qubit boarding the transfer chiplet.
    
    Attributes:
        logic_qubit (int): The logic qubit to board.
        station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is boarding.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is boarding.
    """

    def __init__(self, logic_qubit: int, station: tuple[int, int], platform: tuple[int, int]):
        """
        Initialize the boarding action.

        Args:
            logic_qubit (int): The logic qubit to board.
            station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is boarding.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is boarding.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.station = station
        self.platform = platform

class Alight(BoardingAction):
    """
    Represents the action of a logic qubit alighting the transfer chiplet.
    
    Attributes:
        logic_qubit (int): The logic qubit to alight.
        station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is alighting.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is alighting.
        exit_flag (bool): The flag marking if the logic qubit is needed to exit from the platform.
    """

    def __init__(self, logic_qubit: int, station: tuple[int, int], platform: tuple[int, int], exit_flag: bool):
        """
        Initialize the alighting action.

        Args:
            logic_qubit (int): The logic qubit to alight.
            station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is alighting.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is alighting.
            exit_flag (bool): The flag marking if the logic qubit is needed to exit from the platform.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.station = station
        self.platform = platform
        self.exit_flag = exit_flag

class Interchange(BoardingAction):
    """
    Represents the action of transferring a logic qubit between two transfer chiplets.
    
    Attributes:
        logic_qubit (int): The logic qubit to transfer.
        interchange (tuple[tuple[int, int], tuple[int, int]]): The edge between two transfer chiplets for transfer.
    """

    def __init__(self, logic_qubit: int, interchange: tuple[tuple[int, int], tuple[int, int]]):
        """
        Initialize the interchange action.
        
        Args:
            logic_qubit (int): The logic qubit to transfer.
            interchange (tuple[tuple[int, int], tuple[int, int]]): The edge between two transfer chiplets for transfer.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.interchange = interchange
            
class Transport(RidingAction):
    """
    Represents the action to transport a logic qubit inside the transfer chiplet.
    
    Attributes:
        logic_qubit (int): The logic qubit to transport.
        source_physical_qubit (tuple[int, int]): The node where the transport starts.
        target_physical_qubit (tuple[int, int]): The node where the transport ends.
    """

    def __init__(self, logic_qubit: int, source_physical_qubit: tuple[int, int], target_physical_qubit: tuple[int, int]):
        """
        Initialize the transport action.

        Args:
            logic_qubit (int): The logic qubit to transport.
            source_physical_qubit (tuple[int, int]): The node where the transport starts.
            target_physical_qubit (tuple[int, int]): The node where the transport ends.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.source_physical_qubit = source_physical_qubit
        self.target_physical_qubit = target_physical_qubit

class RouteToPlat(RidingAction):
    """
    Represents the action of routing a data qubit to a platform.
    
    Attributes:
        logic_qubit (int): The data qubit to route.
        platform (tuple[int, int]): The physical qubit on compute chiplet to route to.
        enter_flag (bool): The flag marking if the data qubit is needed to route onto the platform.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], enter_flag: bool):
        """
        Initialize the routing action.

        Args:
            logic_qubit (int): The data qubit to route.
            platform (tuple[int, int]): The physical qubit on compute chiplet to route to.
            enter_flag (bool): The flag marking if the data qubit is needed to route onto the platform.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.enter_flag = enter_flag

class Teleport(RidingAction):
    """
    Represents the action to teleport a logic qubit using a prepared GHZ state.
    
    Attributes:
        logic_qubit (int): The logic qubit to teleport.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
        control_flag (bool): The flag marking if the data qubit is the control qubit.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], control_flag: bool):
        """
        Initialize the teleportation action.

        Args:
            logic_qubit (int): The logic qubit to teleport.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
            control_flag (bool): The flag marking if the data qubit is the control qubit.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.control_flag = control_flag

class LocalGate(RidingAction):
    """
    Represents the action of applying a local gate on a compute module.

    Attributes:
        logic_gate (Control): The logical control gate.
        module (ComputeModule): The compute module where the control gate is executed.
    """
    
    def __init__(self, logic_gate: Control, module: ComputeModule):
        """
        Initialize the local gate action.
        
        Args:
            logic_gate (Control): The logical control gate.
            module (ComputeModule): The compute module where the control gate is executed.
        """
        super().__init__()
        self.logic_gate = logic_gate
        self.module = module


class CircuitStage:
    """
    Represents a stage in the quantum circuit.
    
    Attributes:
        id (int): Unique identifier for the stage.
    """
    _counter = 0

    def __init__(self, init_occupied_balises: set[tuple[int, int]], init_occupied_platforms: set[tuple[int, int]], holding_ancilla_plats: set[tuple[int, int]]):
        """
        Initialize a circuit stage.
        
        Args:
            init_occupied_balises (set[tuple[int, int]]): The initial occupied balises.
            init_occupied_platforms (set[tuple[int, int]]): The initial occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): The platforms holding ancilla qubits.
        """
        self.id = CircuitStage._counter
        CircuitStage._counter += 1

        self.init_occupied_balises = init_occupied_balises
        self.curr_occupied_balises = deepcopy(init_occupied_balises)
        self.init_occupied_platforms = init_occupied_platforms
        self.curr_occupied_platforms = deepcopy(init_occupied_platforms)
        self.holding_ancilla_plats = holding_ancilla_plats

class BoardingStage(CircuitStage):
    """
    Represents the boarding stage in the quantum circuit.

    Attributes:
        boarding_actions (list[BoardingAction]): The list of boarding actions in the stage.
        potential_free_balises (dict): The dictionary of potential free balises.
        paired_balises (set): The set of paired balises.
        potential_free_platforms (dict): The dictionary of potential free platforms.
        paired_platforms (set): The set of paired platforms.
    """
    
    def __init__(self, init_occupied_balises: set[tuple[int, int]], init_occupied_platforms: set[tuple[int, int]], holding_ancilla_plats: set[tuple[int, int]]):
        """
        Initialize the boarding stage.
        
        Args:
            init_occupied_balises (set[tuple[int, int]]): The initial occupied balises.
            init_occupied_platforms (set[tuple[int, int]]): The initial occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): The platforms holding ancilla qubits.
        """
        super().__init__(init_occupied_balises, init_occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 0, "Boarding stage id should be even"
        self.boarding_actions: list[BoardingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises = set()
        self.potential_free_platforms = dict()
        self.paired_platforms = set()

    def add_alight_action(self, action: Alight) -> bool:
        """
        Add an Alight to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Alight): The alighting action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
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
        """
        Add an Interchange to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Interchange): The interchange action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
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
        """
        Add a Board to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Board): The boarding action to add.
        
        Returns:
            bool: Whether the action is successfully added.
        """
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
        """
        Add a GhzPreparation to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (GhzPreparation): The GHZ preparation action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, GhzPreparation), "Action should be GhzPreparation"

        for pysical_qubit in action.source_physical_qubits:
            if pysical_qubit in self.curr_occupied_balises:
                return False
        self.curr_occupied_balises.update(action.source_physical_qubits)

        self.boarding_actions.append(action)
        return True

class RidingStage(CircuitStage):
    """
    Represents the riding stage in the quantum circuit.
    
    Attributes:
        operating_actions (list[RidingAction]): The list of riding actions in the stage.
        potential_free_balises (dict): The dictionary of potential free balises.
        paired_balises (set): The set of paired balises.
    """

    def __init__(self, init_occupied_balises: set[tuple[int, int]], init_occupied_platforms: set[tuple[int, int]], holding_ancilla_plats: set[tuple[int, int]]):
        """
        Initialize the riding stage.
        
        Args:
            init_occupied_balises (set[tuple[int, int]]): The initial occupied balises.
            init_occupied_platforms (set[tuple[int, int]]): The initial occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): The platforms holding ancilla qubits.
        """
        super().__init__(init_occupied_balises, init_occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 1, "Operating stage id should be odd"
        self.operating_actions: list[RidingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises = set()
    
    def add_transport_action(self, action: Transport) -> bool:
        """
        Add a Transport to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Transport): The transport action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
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
        """
        Add a RouteToPlat to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (RouteToPlat): The routing action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, RouteToPlat), "Action should be RouteToPlat"

        if action.enter_flag:
            if action.platform in self.curr_occupied_platforms:
                return False
            self.curr_occupied_platforms.add(action.platform)
            
        self.operating_actions.append(action)
        return True
    
    def add_teleport_action(self, action: Teleport) -> bool:
        """
        Add a Teleport to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Teleport): The teleportation action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Teleport), "Action should be Teleport"

        self.curr_occupied_platforms.remove(action.platform)
        self.holding_ancilla_plats.remove(action.platform)

        self.operating_actions.append(action)
        return True

    def add_local_action(self, action: LocalGate) -> bool:
        """
        Add a LocalGate to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (LocalGate): The local gate action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, LocalGate), "Action should be LocalGate"

        self.operating_actions.append(action)
        return True


class Circuit:
    """
    Represents a physical quantum circuit with enhanced quantum state transfer.
    Circuit with enhanced quantum state transfer is a list interleaved with boarding stage and riding stage.
    Boarding stage: 1) alighting transfer chiplet; 2) boarding transfer chiplet; 3) GHZ state preparation; 4) transfer between transfer chiplets. 
    Riding stage: 1) transport on transfer chiplet; 2) local compute on compute chiplet; 3) routing to platforms; 4) teleportation.

    Attributes:
        package (Package): The package to perform the quantum circuit.
        logic_qubit_num (int): The number of logic qubits in the circuit.
        init_l2p_map (dict): The initial mapping from logical qubits to physical qubits.
        init_p2l_map (dict): The initial mapping from physical qubits to logical qubits.
        action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.

        stages (list[CircuitStage]): The list of stages in the circuit.
        
        curr_l2p_map (dict): The current mapping from logical qubits to physical qubits.
        curr_p2l_map (dict): The current mapping from physical qubits to logical qubits.
        hold_ancilla_plats (set): The set of platforms holding ancilla qubits.
    """
    
    def __init__(self, package: Package, logic_qubit_num: int, init_l2p_map: dict, init_p2l_map: dict, action_depend_graph: nx.DiGraph):
        """
        Initialize the circuit.

        Args:
            package (Package): The package to perform the quantum circuit.
            logic_qubit_num (int): The number of logic qubits in the circuit.
            init_l2p_map (dict): The initial mapping from logical qubits to physical qubits.
            init_p2l_map (dict): The initial mapping from physical qubits to logical qubits.
            action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.
        """
        self.package = package
        self.logic_qubit_num = logic_qubit_num
        self.init_l2p_map = init_l2p_map
        self.init_p2l_map = init_p2l_map
        self.action_depend_graph = action_depend_graph

        self.single_time: float = package.fab_model.single_time
        self.intra_time: float = package.fab_model.intra_time
        self.inter_time: float = package.fab_model.inter_time
        self.qst_time_constant: float = package.fab_model.qst_time_constant
        self.measure_time: float = package.fab_model.measure_time
        self.reset_time: float = package.fab_model.reset_time

        self.stages: list[CircuitStage] = [] # only use extend_stage to add new stage

        self.curr_l2p_map = deepcopy(init_l2p_map)
        self.curr_p2l_map = deepcopy(init_p2l_map)
        self.hold_ancilla_plats = set()
    
    def extend_stage(self, board_flag: bool) -> CircuitStage:
        """
        Extend the circuit with a new stage.
        
        Note: The new stage is added to the end of the circuit.

        Args:
            board_flag (bool): The flag marking if the new stage is a boarding stage.

        Returns:
            CircuitStage: The new stage added to the circuit.
        """
        if board_flag:
            assert len(self.stages) % 2 == 0, "Boarding stage should have even id"
        else:
            assert len(self.stages) % 2 == 1, "Operating stage should have odd id"
        curr_occupied_balises = self.stages[-1].curr_occupied_balises if len(self.stages) else set()
        curr_occupied_platforms = self.stages[-1].curr_occupied_platforms if len(self.stages) else set()
        if board_flag:
            new_stage = BoardingStage(curr_occupied_balises, curr_occupied_platforms, self.hold_ancilla_plats)
        else:
            new_stage = RidingStage(curr_occupied_balises, curr_occupied_platforms, self.hold_ancilla_plats)
        assert new_stage.id == len(self.stages), "Stage id should be equal to the length of existing stages"
        self.stages.append(new_stage)
        return new_stage
    
    def estimate_fidelity(self) -> float:
        """
        Estimate the fidelity of the entire circuit.

        Returns:
            float: The estimated fidelity of the circuit.
        """
        # TODO: complete the function
        pass

    def estimate_average_qst_fide(self) -> float:
        """
        Estimate the average fidelity of each quantum state transfer in the circuit.

        Returns:
            float: The estimated average fidelity of quantum state transfer.
        """
        # TODO: complete the function
        pass
    
    def estimate_depth(self) -> int:
        """
        Estimate the depth of the circuit.

        Returns:
            int: The estimated depth of the circuit.
        """
        depth = 0
        for stage in self.stages:
            if isinstance(stage, BoardingStage):
                inter_chiplet_flag = False
                ghz_flag = False
                for action in stage.boarding_actions:
                    if isinstance(action, Board) or isinstance(action, Alight) or isinstance(action, Interchange):
                        inter_chiplet_flag = True
                    elif isinstance(action, GhzPreparation):
                        ghz_flag = True
                inter_chiplet_depth = 2 * self.inter_time + 3 * self.single_time if inter_chiplet_flag else 0
                ghz_depth = 2 * self.qst_time_constant + 2 * self.single_time if ghz_flag else 0
                stage_depth = max(inter_chiplet_depth, ghz_depth)
            elif isinstance(stage, RidingStage):
                transport_actions = []
                route_actions = []
                teleport_control_flag = False
                teleport_target_flag = False
                local_actions = []
                for action in stage.operating_actions:
                    if isinstance(action, Transport):
                        transport_actions.append(action)
                    elif isinstance(action, RouteToPlat):
                        route_actions.append(action)
                    elif isinstance(action, Teleport):
                        if action.control_flag:
                            teleport_control_flag = True
                        else:
                            teleport_target_flag = True
                    elif isinstance(action, LocalGate):
                        local_actions.append(action)
                transport_depth = len(transport_actions) * self.intra_time
                route_depth = len(route_actions) * self.intra_time
                teleport_depth = self.intra_time * (int(teleport_control_flag) + int(teleport_target_flag))
                local_depth = len(local_actions) * self.intra_time
                stage_depth = max(transport_depth, route_depth, teleport_depth, local_depth)
            depth += stage_depth
        return depth
