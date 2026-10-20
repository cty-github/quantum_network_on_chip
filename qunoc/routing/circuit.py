import pickle
import logging
from copy import deepcopy

from ..utils.config import config
from ..element.benchmark import *
from ..element.chiplet import OperationConfig, TransferChiplet
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
    
    def init_logging(self):
        """
        Initialize logging for block debugging.
        """
        if config.debug_block:
            logging.info(f"Transformed {self}")

    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        """
        Get the physical qubits involved in the action.

        Returns:
            set[tuple[int, int]]: The set of physical qubits involved in the action.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

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
        ancilla_qubits (list[int]): The ancilla qubits used to prepare the GHZ state.
        data_qubits (list[int]): The data qubits that the GHZ state is serving for.
        source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
        target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
        tele_gate_ids (list[int]): The list of teleportation gate ids that the GHZ state is serving for.

        ancilla_qubit_nodes (dict[int, list[tuple[int, int]]]): The dictionary recording the ancilla nodes used for each ancilla qubit.
    """

    def __init__(self, 
            qubit_num: int, 
            ancilla_qubits: list[int],
            data_qubits: list[int], 
            source_physical_qubits: list[tuple[int, int]], 
            target_physical_qubits: list[tuple[int, int]], 
            ghz_balise_graph: list[tuple[tuple[int, int], tuple[int, int]]], 
            tele_gate_ids: list[int]
        ):
        """
        Initialize the GHZ preparation action.

        Args:
            qubit_num (int): Number of ancilla qubits in the GHZ state.
            ancilla_qubits (list[int]): The ancilla qubits used in the GHZ state.
            data_qubits (list[int]): The data qubits that the GHZ state is serving for.
            source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
            target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
            ghz_balise_graph (list[tuple[tuple[int, int], tuple[int, int]]]): The list of physical qubits where the GHZ state is prepared and how it is linked to the subgraph.
            tele_gate_ids (list[int]): The list of teleportation gate ids that the GHZ state is serving for.
        """
        super().__init__()
        self.qubit_num = qubit_num
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits
        self.ghz_balise_graph = ghz_balise_graph
        self.tele_gate_ids = tele_gate_ids
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Prepare {self.qubit_num}-GHZ state for logic qubits {self.data_qubits} on physical qubits {self.source_physical_qubits}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return set(self.source_physical_qubits)

    def record_ancilla_qubit_nodes(self, ancilla_qubit: int, ancilla_nodes: list[tuple[int, int]]):
        """
        Record the ancilla nodes used for the GHZ state.

        Args:
            ancilla_qubit (int): The ancilla qubit.
            ancilla_nodes (list[tuple[int, int]]): The list of physical qubits where the ancilla qubit is located.
        """
        if not hasattr(self, "ancilla_qubit_nodes"):
            self.ancilla_qubit_nodes = {}
        self.ancilla_qubit_nodes[ancilla_qubit] = ancilla_nodes
    
    def get_ancilla_qubit_nodes(self) -> dict[int, list[tuple[int, int]]]:
        """
        Get the ancilla nodes used for the GHZ state.

        Returns:
            dict[int, list[tuple[int, int]]]: The dictionary recording the ancilla nodes used for each ancilla qubit.
        """
        if not hasattr(self, "ancilla_qubit_nodes"):
            self.ancilla_qubit_nodes = {}
        return self.ancilla_qubit_nodes

    def get_depth(self) -> int:
        """
        Get the depth of contructing the GHZ state.

        Returns:
            int: The CX depth.
        """
        balise_depth_dict = dict()
        for balise, parent in self.ghz_balise_graph:
            if parent is None:
                balise_depth_dict[balise] = 0
            else:
                assert parent in balise_depth_dict
                balise_depth_dict[balise] = balise_depth_dict[parent] + 1
        return max(balise_depth_dict.values())
    
    def get_max_degree(self) -> int:
        """
        Get the maximum degree of the GHZ state.

        Returns:
            int: The maximum degree of the GHZ state.
        """
        balise_degree_dict = {balise: 0 for balise, _ in self.ghz_balise_graph}
        for balise, parent in self.ghz_balise_graph:
            if parent is not None:
                balise_degree_dict[balise] += 1
                balise_degree_dict[parent] += 1
        return max(balise_degree_dict.values())

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
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Board logic qubit {self.logic_qubit} from physical qubit {self.platform} to physical qubit {self.station}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.station, self.platform}

class Alight(BoardingAction):
    """
    Represents the action of a logic qubit alighting the transfer chiplet.
    
    Attributes:
        logic_qubit (int): The logic qubit to alight.
        station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is alighting.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is alighting.
    """

    def __init__(self, logic_qubit: int, station: tuple[int, int], platform: tuple[int, int]):
        """
        Initialize the alighting action.

        Args:
            logic_qubit (int): The logic qubit to alight.
            station (tuple[int, int]): The physical qubit on transfer chiplet where the logic qubit is alighting.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is alighting.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.station = station
        self.platform = platform
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Alight logic qubit {self.logic_qubit} from physical qubit {self.station} to physical qubit {self.platform}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.station, self.platform}

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
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Interchange logic qubit {self.logic_qubit} between physical qubits {self.interchange}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.interchange[0], self.interchange[1]}

# class BellPreparation(BoardingAction):
#     """
#     Represents the action to prepare a Bell state.

#     Attributes:
#         data_qubit (int): The data qubit requiring the Bell state.
#         ancilla_qubits (list[int]): The ancilla qubits used in the Bell state. 
#         platform (tuple[int, int]): The physical qubit on compute chiplet where the Bell state is prepared.
#         station (tuple[int, int]): The physical qubit on transfer chiplet where the Bell state is prepared.
#     """

#     def __init__(self, data_qubit: int, ancilla_qubits: list[int], platform: tuple[int, int], station: tuple[int, int]):
#         """
#         Initialize the Bell preparation action.

#         Args:
#             data_qubit (int): The data qubit requiring the Bell state.
#             ancilla_qubits (list[int]): The ancilla qubits used in the Bell state.
#             platform (tuple[int, int]): The physical qubit on compute chiplet where the Bell state is prepared.
#             station (tuple[int, int]): The physical qubit on transfer chiplet where the Bell state is prepared.
#         """
#         super().__init__()
#         self.data_qubit = data_qubit
#         self.ancilla_qubits = ancilla_qubits
#         self.platform = platform
#         self.station = station
#         self.init_logging()

#     def __repr__(self):
#         return f"{self.__class__.__name__} {self.id}: Prepare Bell state for logic qubit {self.data_qubit} on physical qubit {self.platform}."

#     def get_involved_nodes(self) -> set[tuple[int, int]]:
#         return {self.platform, self.station}

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
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Transport logic qubit {self.logic_qubit} from physical qubit {self.source_physical_qubit} to physical qubit {self.target_physical_qubit}."

    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        if dummy:
            if self.source_physical_qubit[0] == self.target_physical_qubit[0]:
                return set([(self.source_physical_qubit[0], i) for i in range(min(self.source_physical_qubit[1], self.target_physical_qubit[1]), max(self.source_physical_qubit[1], self.target_physical_qubit[1]) + 1)])
            elif self.source_physical_qubit[1] == self.target_physical_qubit[1]:
                return set([(i, self.source_physical_qubit[1]) for i in range(min(self.source_physical_qubit[0], self.target_physical_qubit[0]), max(self.source_physical_qubit[0], self.target_physical_qubit[0]) + 1)])
        else:
            return {self.source_physical_qubit, self.target_physical_qubit}

    def get_distance(self) -> int:
        """
        Get the distance of QST.

        Returns:
            int: The distance of QST.
        """
        return abs(self.source_physical_qubit[0] - self.target_physical_qubit[0]) + abs(self.source_physical_qubit[1] - self.target_physical_qubit[1])

class Teleport(RidingAction):
    """
    Represents the action to teleport a data qubit using a prepared GHZ state.
    
    Attributes:
        ancilla_qubit (int): The ancilla qubit used for teleportation.
        data_qubit (int): The data qubit to be teleported.
        ancilla_node (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
        data_node (tuple[int, int]): The physical qubit on compute chiplet where the data qubit is located.
        control_flag (bool): The flag marking if the data qubit is the control qubit.
        inter_flag (bool): The flag marking if the action is performed on inter coupler.
    """

    def __init__(self, ancilla_qubit: int, data_qubit: int, ancilla_node: tuple[int, int], data_node: tuple[int, int], control_flag: bool, inter_flag: bool):
        """
        Initialize the teleportation action.

        Args:
            ancilla_qubit (int): The ancilla qubit used for teleportation.
            data_qubit (int): The data qubit to be teleported.
            ancilla_node (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
            data_node (tuple[int, int]): The physical qubit on compute chiplet where the data qubit is located.
            control_flag (bool): The flag marking if the data qubit is the control qubit.
            inter_flag (bool): The flag marking if the action is performed on inter coupler.
        """
        super().__init__()
        self.ancilla_qubit = ancilla_qubit
        self.data_qubit = data_qubit
        self.ancilla_node = ancilla_node
        self.data_node = data_node
        self.control_flag = control_flag
        self.inter_flag = inter_flag
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Teleport data qubit {self.data_qubit} with ancilla qubit {self.ancilla_qubit} on physical qubit {self.ancilla_node}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.ancilla_node, self.data_node}

class Enter(RidingAction):
    """
    Represents the action of entering a platform with a logic qubit.

    Attributes:
        logic_qubit (int): The logic qubit to enter.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
        register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
        inter_flag (bool): The flag marking if the action is performed on inter coupler.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], register: tuple[int, int], inter_flag: bool):
        """
        Initialize the enter action.

        Args:
            logic_qubit (int): The logic qubit to enter.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
            register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
            inter_flag (bool): The flag marking if the action is performed on inter coupler.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.register = register
        self.inter_flag = inter_flag
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Enter logic qubit {self.logic_qubit} to physical qubit {self.platform} from physical qubit {self.register}."
    
    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.platform, self.register}
    
class Exit(RidingAction):
    """
    Represents the action of exiting a platform with a logic qubit.

    Attributes:
        logic_qubit (int): The logic qubit to exit.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
        register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
        inter_flag (bool): The flag marking if the action is performed on inter coupler.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], register: tuple[int, int], inter_flag: bool):
        """
        Initialize the exit action.

        Args:
            logic_qubit (int): The logic qubit to exit.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
            register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
            inter_flag (bool): The flag marking if the action is performed on inter coupler.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.register = register
        self.inter_flag = inter_flag
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Exit logic qubit {self.logic_qubit} from physical qubit {self.platform} to physical qubit {self.register}."

    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.platform, self.register}

class SwapGate(RidingAction):
    """
    Represents the action of applying a swap gate on a compute module.

    Attributes:
        logic_qubit (int): The logic qubit routed by the swap gate.
        qubit_u (tuple[int, int]): The first qubit of the swap gate.
        qubit_v (tuple[int, int]): The second qubit of the swap gate.
        inter_flag (bool): The flag marking if the action is performed on inter coupler.
    """
    
    def __init__(self, logic_qubit: int, qubit_u: tuple[int, int], qubit_v: tuple[int, int], inter_flag: bool):
        """
        Initialize the swap gate action.

        Args:
            logic_qubit (int): The logic qubit routed by the swap gate.
            qubit_u (tuple[int, int]): The first qubit of the swap gate.
            qubit_v (tuple[int, int]): The second qubit of the swap gate.
            inter_flag (bool): The flag marking if the action is performed on inter coupler.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.qubit_u = qubit_u
        self.qubit_v = qubit_v
        self.inter_flag = inter_flag
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Route logic qubit {self.logic_qubit} with swap gate between physical qubits {self.qubit_u} and {self.qubit_v}."

    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.qubit_u, self.qubit_v}

class DirectGate(RidingAction):
    """
    Represents the action of applying a local gate on a compute module.

    Attributes:
        logic_gate (Control): The logical control gate.
        phy_ctrl_qubit (tuple[int, int]): The physical qubit where the control qubit is located.
        phy_targ_qubit (tuple[int, int]): The physical qubit where the target qubit is located.
        inter_flag (bool): The flag marking if the action is performed on inter coupler.
    """
    
    def __init__(self, logic_gate: Control, phy_ctrl_qubit: tuple[int, int], phy_targ_qubit: tuple[int, int], inter_flag: bool):
        """
        Initialize the local gate action.
        
        Args:
            logic_gate (Control): The logical control gate.
            phy_ctrl_qubit (tuple[int, int]): The physical qubit where the control qubit is located.
            phy_targ_qubit (tuple[int, int]): The physical qubit where the target qubit is located.
            inter_flag (bool): The flag marking if the action is performed on inter coupler.
        """
        super().__init__()
        self.logic_gate = logic_gate
        self.phy_ctrl_qubit = phy_ctrl_qubit
        self.phy_targ_qubit = phy_targ_qubit
        self.inter_flag = inter_flag
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: {self.logic_gate}."

    def get_involved_nodes(self, dummy: bool = False) -> set[tuple[int, int]]:
        return {self.phy_ctrl_qubit, self.phy_targ_qubit}

# class TeleState(RidingAction):
#     """
#     Represents the action of completing a teleportation on a compute module.

#     Attributes:
#         data_qubit (int): The data qubit to be teleported.
#         ancilla_qubits (list[int]): The ancilla qubits used in the teleportation.
#         register (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is completed.
#         source_platform (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is initiated.
#         target_platform (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is routed to.
#     """
    
#     def __init__(self, data_qubit: int, ancilla_qubits: list[int], register: tuple[int, int], source_platform: tuple[int, int], target_platform: tuple[int, int]):
#         """
#         Initialize the teleportation state action.

#         Args:
#             data_qubit (int): The data qubit to be teleported.
#             ancilla_qubits (list[int]): The ancilla qubits used in the teleportation.
#             register (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is completed.
#             source_platform (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is initiated.
#             target_platform (tuple[int, int]): The physical qubit on compute chiplet where the teleportation is routed to.
#         """
#         super().__init__()
#         self.data_qubit = data_qubit
#         self.ancilla_qubits = ancilla_qubits
#         self.register = register
#         self.source_platform = source_platform
#         self.target_platform = target_platform
#         self.init_logging()

#     def __repr__(self):
#         return f"{self.__class__.__name__} {self.id}: Teleport data qubit {self.data_qubit} with ancilla qubits {self.ancilla_qubits} from physical qubit {self.source_platform} to physical qubit {self.target_platform}."
    
#     def get_involved_nodes(self) -> set[tuple[int, int]]:
#         return {self.register, self.source_platform, self.target_platform}


class CircuitStage:
    """
    Represents a stage in the quantum circuit.
    
    Attributes:
        id (int): Unique identifier for the stage.
        curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
        curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
        occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
        occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
        active_ancilla_qubit_nodes (dict[int, list[tuple[int, int]]]): Recording the active ancilla qubit and nodes it will occupy.

        node_action_dict (dict[tuple[int, int], list[tuple[int, int]]]): Recording the actions and its time for each node.
    """
    _counter = 0

    def __init__(self, 
        curr_l2p_map: dict[int, tuple[int, int]], 
        curr_p2l_map: dict[tuple[int, int], int], 
        occupied_balises: set[tuple[int, int]], 
        occupied_platforms: set[tuple[int, int]], 
        active_ancilla_qubit_nodes: dict[int, list[tuple[int, int]]],
    ):
        """
        Initialize a circuit stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            active_ancilla_qubit_nodes (dict[int, list[tuple[int, int]]]): Recording the active ancilla qubit and nodes it will occupy.
        """
        self.id = CircuitStage._counter
        CircuitStage._counter += 1

        self.curr_l2p_map = curr_l2p_map
        self.curr_p2l_map = curr_p2l_map
        self.occupied_balises = occupied_balises
        self.occupied_platforms = occupied_platforms
        self.active_ancilla_qubit_nodes = active_ancilla_qubit_nodes

        self.circuit_actions: list[CircuitAction] = []
        self.node_action_ids: dict[tuple[int, int], list[int]] = {}

    def get_action_num(self) -> int:
        """
        Get the number of actions in the stage.

        Returns:
            int: The number of actions in the stage.
        """
        return len(self.circuit_actions)
    
    def get_action_depend_graph(self) -> nx.DiGraph:
        """
        Get the action dependency graph of the stage.

        Returns:
            nx.DiGraph: The directed graph representing the dependencies between actions.
        """
        action_depend_graph = nx.DiGraph()
        for action in self.circuit_actions:
            action_depend_graph.add_node(action.id, action=action)
        for node, action_ids in self.node_action_ids.items():
            for i in range(len(action_ids) - 1):
                action_depend_graph.add_edge(action_ids[i], action_ids[i + 1])
        for action in self.circuit_actions:
            if isinstance(action, Transport):
                action_depend_graph.remove_node(action.id)
        return action_depend_graph
    
    def add_action_immediate(self, action: CircuitAction) -> bool:
        """
        Try to add action into the stage.

        Args:
            action (CircuitAction): The action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def add_action_logging(self, action: CircuitAction):
        """
        Add logging for the action added to the stage.

        Args:
            action (CircuitAction): The action to log.
        """
        if config.debug_action:
            logging.info(f"Added {action.__class__.__name__} {action.id} to stage {self.id}.")

class BoardingStage(CircuitStage):
    """
    Represents the boarding stage in the quantum circuit.

    Attributes:
        boarding_actions (list[BoardingAction]): The list of boarding actions in the stage.
        potential_free_balises (dict): The dictionary of potential free balises.
        paired_balises_to_free (set): The set of paired balises to free.
        potential_free_platforms (dict): The dictionary of potential free platforms.
        paired_platforms_to_free (set): The set of paired platforms to free.
    """
    
    def __init__(self, 
        curr_l2p_map: dict[int, tuple[int, int]], 
        curr_p2l_map: dict[tuple[int, int], int], 
        occupied_balises: set[tuple[int, int]], 
        occupied_platforms: set[tuple[int, int]], 
        active_ancilla_qubit_nodes: dict[int, list[tuple[int, int]]]
    ):
        """
        Initialize the boarding stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            active_ancilla_qubit_nodes (dict[int, list[tuple[int, int]]]): Recording the active ancilla qubit and nodes it will occupy.
        """
        super().__init__(curr_l2p_map, curr_p2l_map, occupied_balises, occupied_platforms, active_ancilla_qubit_nodes)
        assert self.id % 2 == 0, "Boarding stage id should be even"

        self.potential_free_balises = dict()
        self.paired_balises_to_free = set()
        self.potential_free_platforms = dict()
        self.paired_platforms_to_free = set()

    def add_action_immediate(self, action: CircuitAction) -> bool:
        if isinstance(action, RidingAction):
            return False
        elif isinstance(action, GhzPreparation):
            return self.add_ghz_action_immediate(action)
        elif isinstance(action, Board):
            return self.add_board_action_immediate(action)
        elif isinstance(action, Alight):
            return self.add_alight_action_immediate(action)
        elif isinstance(action, Interchange):
            return self.add_interchange_action_immediate(action)

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

        for physical_qubit in action.source_physical_qubits:
            if physical_qubit in self.occupied_balises:
                return False
        active_ancilla_nodes = set().union(*self.active_ancilla_qubit_nodes.values())
        ancilla_nodes = set().union(*action.get_ancilla_qubit_nodes().values())
        if active_ancilla_nodes.intersection(ancilla_nodes):
            return False
        self.occupied_balises.update(action.source_physical_qubits)
        for ancilla_qubit, physical_qubit in zip(action.ancilla_qubits, action.source_physical_qubits):
            self.curr_l2p_map[ancilla_qubit] = physical_qubit
            self.curr_p2l_map[physical_qubit] = ancilla_qubit
        self.active_ancilla_qubit_nodes.update(action.get_ancilla_qubit_nodes())

        self.circuit_actions.append(action)
        self.add_action_logging(action)
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
        assert action.platform in self.occupied_platforms, f"In action {action.id}, data qubit {action.logic_qubit} to board should be in platform {action.platform}."

        if action.station in self.occupied_balises:
            if action.station in self.potential_free_balises and self.potential_free_balises[action.station] == action.platform:
                self.potential_free_balises.pop(action.station)
                self.paired_balises_to_free.add(action.station)
            elif action.platform in self.paired_platforms_to_free:
                self.paired_platforms_to_free.remove(action.platform)
            else:
                self.potential_free_platforms[action.platform] = action.station
                return False
        else:
            self.occupied_platforms.remove(action.platform)
            self.occupied_balises.add(action.station)
            self.curr_p2l_map[action.platform] = -1
        self.curr_l2p_map[action.logic_qubit] = action.station
        self.curr_p2l_map[action.station] = action.logic_qubit

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

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
        assert action.station in self.occupied_balises, f"In action {action.id}, data qubit {action.logic_qubit} to alight should be in station {action.station}."

        if action.platform in self.occupied_platforms:
            if action.platform in self.potential_free_platforms and self.potential_free_platforms[action.platform] == action.station:
                self.potential_free_platforms.pop(action.platform)
                self.paired_platforms_to_free.add(action.platform)
            elif action.station in self.paired_balises_to_free:
                self.paired_balises_to_free.remove(action.station)
            else:
                self.potential_free_balises[action.station] = action.platform
                return False
        else:
            self.occupied_balises.remove(action.station)
            self.occupied_platforms.add(action.platform)
            self.curr_p2l_map[action.station] = -1
        self.curr_l2p_map[action.logic_qubit] = action.platform
        self.curr_p2l_map[action.platform] = action.logic_qubit

        self.circuit_actions.append(action)
        self.add_action_logging(action)
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
        assert action.interchange[0] in self.occupied_balises, f"In action {action.id}, data qubit {action.logic_qubit} to interchange should be in source interchange {action.interchange[0]}."

        if action.interchange[1] in self.occupied_balises:
            if action.interchange[1] in self.potential_free_balises and self.potential_free_balises[action.interchange[1]] == action.interchange[0]:
                self.potential_free_balises.pop(action.interchange[1])
                self.paired_balises_to_free.add(action.interchange[1])
            elif action.interchange[0] in self.paired_balises_to_free:
                self.paired_balises_to_free.remove(action.interchange[0])
            else:
                self.potential_free_balises[action.interchange[0]] = action.interchange[1]
                return False
        else:
            self.occupied_balises.remove(action.interchange[0])
            self.occupied_balises.add(action.interchange[1])
            self.curr_p2l_map[action.interchange[0]] = -1
        self.curr_l2p_map[action.logic_qubit] = action.interchange[1]
        self.curr_p2l_map[action.interchange[1]] = action.logic_qubit

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_ghz_action_immediate(self, action: GhzPreparation) -> bool:
        """
        Add a GhzPreparation to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (GhzPreparation): The GHZ preparation action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, GhzPreparation), "Action should be GhzPreparation"
        for physical_qubit in action.source_physical_qubits:
            assert self.curr_p2l_map[physical_qubit] == -1, f"In action {action.id}, physical qubit {physical_qubit} should be free for GHZ preparation."
        
        for ancilla_qubit, physical_qubit in zip(action.ancilla_qubits, action.source_physical_qubits):
            self.curr_l2p_map[ancilla_qubit] = physical_qubit
            self.curr_p2l_map[physical_qubit] = ancilla_qubit
            self.node_action_ids.setdefault(physical_qubit, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_board_action_immediate(self, action: Board) -> bool:
        """
        Add a Board to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Board): The boarding action to add.
        
        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Board), "Action should be Board"
        assert self.curr_p2l_map[action.platform] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to board should be in platform {action.platform}."

        alight_qubit = self.curr_p2l_map[action.station]
        self.curr_l2p_map[action.logic_qubit] = action.station
        self.curr_p2l_map[action.station] = action.logic_qubit
        if alight_qubit != -1:
            self.curr_l2p_map[alight_qubit] = action.platform
        self.curr_p2l_map[action.platform] = alight_qubit
        self.node_action_ids.setdefault(action.station, []).append(action.id)
        self.node_action_ids.setdefault(action.platform, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_alight_action_immediate(self, action: Alight) -> bool:
        """
        Add an Alight to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Alight): The alighting action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Alight), "Action should be Alight"
        assert self.curr_p2l_map[action.station] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to alight should be in station {action.station}."

        board_qubit = self.curr_p2l_map[action.platform]
        self.curr_l2p_map[action.logic_qubit] = action.platform
        self.curr_p2l_map[action.platform] = action.logic_qubit
        if board_qubit != -1:
            self.curr_l2p_map[board_qubit] = action.station
        self.curr_p2l_map[action.station] = board_qubit
        self.node_action_ids.setdefault(action.station, []).append(action.id)
        self.node_action_ids.setdefault(action.platform, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_interchange_action_immediate(self, action: Interchange) -> bool:
        """
        Add an Interchange to the boarding stage.

        Note: The action is added to the end of the boarding actions list.

        Args:
            action (Interchange): The interchange action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Interchange), "Action should be Interchange"
        assert self.curr_p2l_map[action.interchange[0]] == action.logic_qubit, f"In action {action.id}, data qubit {action.logic_qubit} to interchange should be in source interchange {action.interchange[0]}."

        interchanged_qubit = self.curr_p2l_map[action.interchange[1]]
        self.curr_l2p_map[action.logic_qubit] = action.interchange[1]
        self.curr_p2l_map[action.interchange[1]] = action.logic_qubit
        if interchanged_qubit != -1:
            self.curr_l2p_map[interchanged_qubit] = action.interchange[0]
        self.curr_p2l_map[action.interchange[0]] = interchanged_qubit
        self.node_action_ids.setdefault(action.interchange[0], []).append(action.id)
        self.node_action_ids.setdefault(action.interchange[1], []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

class RidingStage(CircuitStage):
    """
    Represents the riding stage in the quantum circuit.
    
    Attributes:
        riding_actions (list[RidingAction]): The list of riding actions in the stage.
        potential_free_balises (dict): The dictionary of potential free balises.
        paired_balises_to_free (set): The set of paired balises to free.
        potential_free_platforms (dict): The dictionary of potential free platforms.
        paired_platforms_to_free (set): The set of paired platforms to free.
        potential_free_registers (dict): The dictionary of potential free registers.
        paired_registers_to_free (set): The set of paired registers to free.
    """

    def __init__(self, 
        curr_l2p_map: dict[int, tuple[int, int]], 
        curr_p2l_map: dict[tuple[int, int], int], 
        occupied_balises: set[tuple[int, int]], 
        occupied_platforms: set[tuple[int, int]], 
        active_ancilla_qubit_nodes: dict[int, list[tuple[int, int]]]
    ):
        """
        Initialize the riding stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            active_ancilla_qubit_nodes (dict[int, list[tuple[int, int]]]): Recording the active ancilla qubit and nodes it will occupy.
        """
        super().__init__(curr_l2p_map, curr_p2l_map, occupied_balises, occupied_platforms, active_ancilla_qubit_nodes)
        assert self.id % 2 == 1, "Operating stage id should be odd"

        self.potential_free_balises = dict()
        self.paired_balises_to_free = set()
        self.potential_free_platforms = dict()
        self.paired_platforms_to_free = set()
        self.potential_free_registers = dict()
        self.paired_registers_to_free = set()

        self.locked_balises: set[tuple[int, int]] = set()
        # Stage-owned operands survive shared actions / 阶段独立记录真实 SWAP 操作数.
        self.swap_logical_ids: dict[int, tuple[int, int]] = {}

    def add_action_immediate(self, action: CircuitAction) -> bool:
        if isinstance(action, BoardingAction):
            return False
        elif isinstance(action, Transport):
            return self.add_transport_action_immediate(action)
        elif isinstance(action, Teleport):
            return self.add_teleport_action_immediate(action)
        elif isinstance(action, Enter):
            return self.add_enter_action_immediate(action)
        elif isinstance(action, Exit):
            return self.add_exit_action_immediate(action)
        elif isinstance(action, SwapGate):
            return self.add_swap_action_immediate(action)
        elif isinstance(action, DirectGate):
            return self.add_direct_action_immediate(action)

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
        assert action.source_physical_qubit in self.occupied_balises, f"In action {action.id}, data qubit {action.logic_qubit} to transport should be in source balise {action.source_physical_qubit}."

        if action.target_physical_qubit == action.source_physical_qubit:
            pass
        elif action.target_physical_qubit in self.occupied_balises:
            if action.target_physical_qubit in self.potential_free_balises and self.potential_free_balises[action.target_physical_qubit] == action.source_physical_qubit:
                self.potential_free_balises.pop(action.target_physical_qubit)
                self.paired_balises_to_free.add(action.target_physical_qubit)
            elif action.source_physical_qubit in self.paired_balises_to_free:
                self.paired_balises_to_free.remove(action.source_physical_qubit)
            else:
                self.potential_free_balises[action.source_physical_qubit] = action.target_physical_qubit
                return False
        else:
            self.occupied_balises.remove(action.source_physical_qubit)
            self.occupied_balises.add(action.target_physical_qubit)
            self.curr_p2l_map[action.source_physical_qubit] = -1
        self.curr_l2p_map[action.logic_qubit] = action.target_physical_qubit
        self.curr_p2l_map[action.target_physical_qubit] = action.logic_qubit
        
        self.circuit_actions.append(action)
        self.add_action_logging(action)
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
        assert action.ancilla_node in self.occupied_platforms, f"In action {action.id}, ancilla qubit {action.ancilla_qubit} for teleportation should be in platform {action.ancilla_node}."
        assert self.curr_p2l_map[action.data_node] == action.data_qubit, f"In action {action.id}, data qubit {action.data_qubit} to teleport should be in register {action.data_node}."

        self.occupied_platforms.remove(action.ancilla_node)
        self.curr_l2p_map.pop(action.ancilla_qubit)
        self.curr_p2l_map[action.ancilla_node] = -1
        self.active_ancilla_qubit_nodes.pop(action.ancilla_qubit)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_enter_action(self, action: Enter) -> bool:
        """
        Add an Enter to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Enter): The enter action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Enter), "Action should be Enter"
        assert self.curr_p2l_map[action.register] == action.logic_qubit, f"In action {action.id}, data qubit {action.logic_qubit} to enter should be in register {action.register}."

        if action.platform in self.occupied_platforms:
            if action.platform in self.potential_free_platforms and self.potential_free_platforms[action.platform] == action.register:
                self.potential_free_platforms.pop(action.platform)
                self.paired_platforms_to_free.add(action.platform)
            elif action.register in self.paired_registers_to_free:
                self.paired_registers_to_free.remove(action.register)
            else:
                self.potential_free_registers[action.register] = action.platform
                return False
        else:
            self.occupied_platforms.add(action.platform)
            self.curr_p2l_map[action.register] = -1
        self.curr_l2p_map[action.logic_qubit] = action.platform
        self.curr_p2l_map[action.platform] = action.logic_qubit
        
        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_exit_action(self, action: Exit) -> bool:
        """
        Add an Exit to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Exit): The exit action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Exit), "Action should be Exit"
        assert action.platform in self.occupied_platforms, f"In action {action.id}, data qubit {action.logic_qubit} to exit should be in platform {action.platform}."

        if self.curr_p2l_map[action.register] == -1:
            self.occupied_platforms.remove(action.platform)
            self.curr_p2l_map[action.platform] = -1
        else:
            if action.register in self.potential_free_registers and self.potential_free_registers[action.register] == action.platform:
                self.potential_free_registers.pop(action.register)
                self.paired_registers_to_free.add(action.register)
            elif action.platform in self.paired_platforms_to_free:
                self.paired_platforms_to_free.remove(action.platform)
            else:
                self.potential_free_platforms[action.platform] = action.register
                return False
        self.curr_l2p_map[action.logic_qubit] = action.register
        self.curr_p2l_map[action.register] = action.logic_qubit

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_swap_action(self, action: SwapGate) -> bool:
        """
        Add a SwapGate to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (SwapGate): The routing action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, SwapGate), "Action should be SwapGate"
        assert self.curr_p2l_map[action.qubit_u] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} should can be routed by the swap gate."

        logic_qubit_u = self.curr_p2l_map[action.qubit_u]
        logic_qubit_v = self.curr_p2l_map[action.qubit_v]
        if not hasattr(self, "swap_logical_ids"):
            self.swap_logical_ids = {}
        self.swap_logical_ids[action.id] = (logic_qubit_u, logic_qubit_v)
        self.curr_p2l_map[action.qubit_u] = logic_qubit_v
        self.curr_p2l_map[action.qubit_v] = logic_qubit_u
        if logic_qubit_u != -1:
            self.curr_l2p_map[logic_qubit_u] = action.qubit_v
        if logic_qubit_v != -1:
            self.curr_l2p_map[logic_qubit_v] = action.qubit_u

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_direct_action(self, action: DirectGate) -> bool:
        """
        Add a LocalGate to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (LocalGate): The local gate action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, DirectGate), "Action should be LocalGate"
        assert self.curr_p2l_map[action.phy_ctrl_qubit] == action.logic_gate.control_qubit, \
            f"In action {action.id}, control qubit {action.logic_gate.control_qubit} should be in physical qubit {action.phy_ctrl_qubit}."
        assert self.curr_p2l_map[action.phy_targ_qubit] == action.logic_gate.target_qubit, \
            f"In action {action.id}, target qubit {action.logic_gate.target_qubit} should be in physical qubit {action.phy_targ_qubit}."

        physical_qubit_control = self.curr_l2p_map[action.logic_gate.control_qubit]
        physical_qubit_target = self.curr_l2p_map[action.logic_gate.target_qubit]
        assert (physical_qubit_control[0] == physical_qubit_target[0] and abs(physical_qubit_control[1] - physical_qubit_target[1]) == 1) or \
               (physical_qubit_control[1] == physical_qubit_target[1] and abs(physical_qubit_control[0] - physical_qubit_target[0]) == 1), \
               f"In action {action.id}, control and target qubits {action.logic_gate.control_qubit} and {action.logic_gate.target_qubit} should be adjacent on the compute module."

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_transport_action_immediate(self, action: Transport) -> bool:
        """
        Add a Transport to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Transport): The transport action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Transport), "Action should be Transport"
        assert self.curr_p2l_map[action.source_physical_qubit] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to transport should be in source balise {action.source_physical_qubit}."

        if action.target_physical_qubit in self.locked_balises:
            return False
        exchanged_qubit = self.curr_p2l_map[action.target_physical_qubit]
        self.curr_l2p_map[action.logic_qubit] = action.target_physical_qubit
        self.curr_p2l_map[action.target_physical_qubit] = action.logic_qubit
        if exchanged_qubit != -1:
            self.curr_l2p_map[exchanged_qubit] = action.source_physical_qubit
        self.curr_p2l_map[action.source_physical_qubit] = exchanged_qubit
        self.locked_balises.add(action.target_physical_qubit)
        
        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_teleport_action_immediate(self, action: Teleport) -> bool:
        """
        Add a Teleport to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Teleport): The teleportation action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Teleport), "Action should be Teleport"
        assert self.curr_p2l_map[action.ancilla_node] == action.ancilla_qubit, f"In action {action.id}, ancilla qubit {action.ancilla_qubit} for teleportation should be in platform {action.ancilla_node}."
        assert self.curr_p2l_map[action.data_node] == action.data_qubit, f"In action {action.id}, data qubit {action.data_qubit} to teleport should be in register {action.data_node}."

        self.curr_l2p_map.pop(action.ancilla_qubit)
        self.curr_p2l_map[action.ancilla_node] = -1
        self.node_action_ids.setdefault(action.ancilla_node, []).append(action.id)
        self.node_action_ids.setdefault(action.data_node, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_enter_action_immediate(self, action: Enter) -> bool:
        """
        Add an Enter to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Enter): The enter action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Enter), "Action should be Enter"
        assert self.curr_p2l_map[action.register] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to enter should be in register {action.register}."

        exit_qubit = self.curr_p2l_map[action.platform]
        self.curr_l2p_map[action.logic_qubit] = action.platform
        self.curr_p2l_map[action.platform] = action.logic_qubit
        if exit_qubit != -1:
            self.curr_l2p_map[exit_qubit] = action.register
        self.curr_p2l_map[action.register] = exit_qubit
        self.node_action_ids.setdefault(action.platform, []).append(action.id)
        self.node_action_ids.setdefault(action.register, []).append(action.id)
        
        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_exit_action_immediate(self, action: Exit) -> bool:
        """
        Add an Exit to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (Exit): The exit action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, Exit), "Action should be Exit"
        assert self.curr_p2l_map[action.platform] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to exit should be in platform {action.platform}."

        enter_qubit = self.curr_p2l_map[action.register]
        self.curr_l2p_map[action.logic_qubit] = action.register
        self.curr_p2l_map[action.register] = action.logic_qubit
        if enter_qubit != -1:
            self.curr_l2p_map[enter_qubit] = action.platform
        self.curr_p2l_map[action.platform] = enter_qubit
        self.node_action_ids.setdefault(action.platform, []).append(action.id)
        self.node_action_ids.setdefault(action.register, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_swap_action_immediate(self, action: SwapGate) -> bool:
        """
        Add a SwapGate to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (SwapGate): The routing action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, SwapGate), "Action should be SwapGate"
        assert self.curr_p2l_map[action.qubit_u] == action.logic_qubit, \
            f"In action {action.id}, logic qubit {action.logic_qubit} should can be routed by the swap gate between {action.qubit_u} and {action.qubit_v}."

        logic_qubit_u = self.curr_p2l_map[action.qubit_u]
        logic_qubit_v = self.curr_p2l_map[action.qubit_v]
        if logic_qubit_u != -1:
            self.curr_l2p_map[logic_qubit_u] = action.qubit_v
        self.curr_p2l_map[action.qubit_u] = logic_qubit_v
        if logic_qubit_v != -1:
            self.curr_l2p_map[logic_qubit_v] = action.qubit_u
        self.curr_p2l_map[action.qubit_v] = logic_qubit_u
        self.node_action_ids.setdefault(action.qubit_u, []).append(action.id)
        self.node_action_ids.setdefault(action.qubit_v, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True

    def add_direct_action_immediate(self, action: DirectGate) -> bool:
        """
        Add a LocalGate to the operating stage.

        Note: The action is added to the end of the operating actions list.

        Args:
            action (LocalGate): The local gate action to add.

        Returns:
            bool: Whether the action is successfully added.
        """
        assert isinstance(action, DirectGate), "Action should be LocalGate"
        actual_ctrl_qubit = self.curr_p2l_map[action.phy_ctrl_qubit]
        actual_targ_qubit = self.curr_p2l_map[action.phy_targ_qubit]
        if actual_ctrl_qubit != action.logic_gate.control_qubit:
            raise AssertionError(
                f"In action {action.id}, control qubit "
                f"{action.logic_gate.control_qubit} should be in physical qubit "
                f"{action.phy_ctrl_qubit}, actual={actual_ctrl_qubit}, "
                f"control_current_node="
                f"{self.curr_l2p_map.get(action.logic_gate.control_qubit)}, "
                f"target_current_node="
                f"{self.curr_l2p_map.get(action.logic_gate.target_qubit)}."
            )
        if actual_targ_qubit != action.logic_gate.target_qubit:
            raise AssertionError(
                f"In action {action.id}, target qubit "
                f"{action.logic_gate.target_qubit} should be in physical qubit "
                f"{action.phy_targ_qubit}, actual={actual_targ_qubit}, "
                f"control_current_node="
                f"{self.curr_l2p_map.get(action.logic_gate.control_qubit)}, "
                f"target_current_node="
                f"{self.curr_l2p_map.get(action.logic_gate.target_qubit)}."
            )

        self.node_action_ids.setdefault(action.phy_ctrl_qubit, []).append(action.id)
        self.node_action_ids.setdefault(action.phy_targ_qubit, []).append(action.id)

        self.circuit_actions.append(action)
        self.add_action_logging(action)
        return True


class Circuit:
    """
    Represents a physical quantum circuit with enhanced quantum state transfer.
    Circuit with enhanced quantum state transfer is a list interleaved with boarding stage and riding stage.
    Boarding stage: 1) alighting transfer chiplet; 2) boarding transfer chiplet; 3) GHZ state preparation; 4) transfer between transfer chiplets. 
    Riding stage: 1) transport on transfer chiplet; 2) local compute on compute chiplet; 3) routing to platforms; 4) teleportation.

    Attributes:
        logic_qubit_num (int): The number of logic qubits in the circuit.
        init_l2p_map (dict): The initial mapping from logical qubits to physical qubits.
        init_p2l_map (dict): The initial mapping from physical qubits to logical qubits.
        action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.
        op_config (OperationConfig): The configuration for operations, including time constants for different operations.

        stages (list[CircuitStage]): The list of stages in the circuit.
        
        curr_l2p_map (dict): The current mapping from logical qubits to physical qubits.
        curr_p2l_map (dict): The current mapping from physical qubits to logical qubits.
        occupied_balises (set): The set of balises occupied by logic qubits.
        occupied_platforms (set): The set of platforms occupied by logic qubits.
        hold_ancilla_plats (set): The set of platforms holding ancilla qubits.
    """

    def __init__(self, logic_qubit_num: int, package_shape: tuple[int, int], init_l2p_map: dict[int, tuple[int, int]], init_p2l_map: dict[tuple[int, int], int], action_depend_graph: nx.DiGraph, op_config: OperationConfig):
        """
        Initialize the circuit.

        Args:
            logic_qubit_num (int): The number of logic qubits in the circuit.
            package_shape (tuple[int, int]): The shape of the package, representing the number of rows and columns of physical qubits.
            init_l2p_map (dict[int, tuple[int, int]]): The initial mapping from logical qubits to physical qubits.
            init_p2l_map (dict[tuple[int, int], int]): The initial mapping from physical qubits to logical qubits.
            action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.
            op_config (OperationConfig): The configuration for operations, including time constants for different operations.
        """
        self.logic_qubit_num = logic_qubit_num
        self.package_shape = package_shape
        self.init_l2p_map = init_l2p_map
        self.init_p2l_map = init_p2l_map
        self.action_depend_graph = action_depend_graph
        self.routing_mode = "immediate" if action_depend_graph is None else "concurrent"
        self.op_config = op_config
        
        self.stages: list[CircuitStage] = [] # only use extend_stage to add new stage
        CircuitStage._counter = 0  # reset the stage counter
        
        self.curr_l2p_map = deepcopy(init_l2p_map)
        self.curr_p2l_map = deepcopy(init_p2l_map)
        self.occupied_balises = set()
        self.occupied_platforms = set()
        self.active_ancilla_qubit_nodes = dict()
        self.recent_node_active_stage = {(x, y): 0 for x in range(self.package_shape[0]) for y in range(self.package_shape[1])}
        if config.debug_action:
            logging.info("-" * 80)
            logging.info(f"Initialized circuit with {self.logic_qubit_num} logic qubits.")
            logging.info(f"Current qubit mapping: {self.curr_l2p_map}")

    def __repr__(self):
        if not hasattr(self, "string_repr"):
            self.string_repr = str()
            for stage in self.stages:
                self.string_repr += "=" * 20 + str(stage.__class__.__name__) + "=" * 20 + "\n"
                if isinstance(stage, BoardingStage):
                    for action in stage.circuit_actions:
                        self.string_repr += str(action) + "\n"
                elif isinstance(stage, RidingStage):
                    for action in stage.circuit_actions:
                        self.string_repr += str(action) + "\n"
        return self.string_repr

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
        if len(self.stages) > 1:
            assert self.stages[-1].get_action_num() + self.stages[-2].get_action_num() > 0, "The last two stages should have at least one action."
        if board_flag:
            new_stage = BoardingStage(self.curr_l2p_map, self.curr_p2l_map, self.occupied_balises, self.occupied_platforms, self.active_ancilla_qubit_nodes)
        else:
            new_stage = RidingStage(self.curr_l2p_map, self.curr_p2l_map, self.occupied_balises, self.occupied_platforms, self.active_ancilla_qubit_nodes)
        assert new_stage.id == len(self.stages), "Stage id should be equal to the length of existing stages"
        self.stages.append(new_stage)
        if config.debug_action:
            logging.info("-" * 80)
            logging.info(f"Extended circuit with {new_stage.__class__.__name__} {new_stage.id}.")
            # logging.info(f"Current qubit mapping: {new_stage.curr_l2p_map}")
        return new_stage

    def add_action_into_earliest_stage(self, action: CircuitAction) -> int:
        """
        Add the action into the earlist stage.

        Args:
            action (CircuitAction): The action to be added.

        Returns:
            int: The stage id the action added to.
        """
        earliest_stage_id = 0
        for node in action.get_involved_nodes():
            earliest_stage_id = max(earliest_stage_id, self.recent_node_active_stage[node])
        while True:
            if earliest_stage_id < len(self.stages):
                stage = self.stages[earliest_stage_id]
            else:
                stage = self.extend_stage(bool(len(self.stages) % 2 == 0))
            if stage.add_action_immediate(action):
                for node in action.get_involved_nodes():
                    self.recent_node_active_stage[node] = earliest_stage_id
                return earliest_stage_id
            else:
                earliest_stage_id += 1

    def get_action_duration(self, action: CircuitAction) -> float:
        """
        Get the duration of the action in the circuit.

        Args:
            action (CircuitAction): The action to get the duration for.

        Returns:
            float: The duration of the action in nanoseconds.
        """
        if isinstance(action, GhzPreparation):
            return action.get_max_degree() * self.op_config.intra_time
        elif isinstance(action, (Board, Alight, Interchange)):
            return self.op_config.inter_time * 3
        elif isinstance(action, Transport):
            return action.get_distance() * self.op_config.qst_time_constant
            return action.get_distance() ** 0.5 * self.op_config.qst_time_constant
        elif isinstance(action, Teleport):
            return self.op_config.inter_time if action.inter_flag else self.op_config.intra_time
        elif isinstance(action, (SwapGate, Enter, Exit)):
            return (self.op_config.inter_time if action.inter_flag else self.op_config.intra_time) * 3
        elif isinstance(action, DirectGate):
            return self.op_config.inter_time if action.inter_flag else self.op_config.intra_time

    def _replay_stage_mappings(self):
        """Replay finalized history, yielding independent Riding boundary maps.
        重放最终动作历史，输出独立的 Riding 阶段边界映射，不读取阶段共享映射。

        Inputs / 输入: initial map and stored stage/action order / 初始映射及最终动作顺序.
        Yields / 输出: (stage index, start l2p, end l2p) / 阶段索引及始末逻辑映射.
        Raises / 异常: ValueError for ambiguous or inconsistent history / 历史歧义或不一致.
        """
        mode = getattr(self, "routing_mode", None)
        if mode is None:
            mode = "immediate" if self.action_depend_graph is None else "concurrent"
        if mode not in {"immediate", "concurrent"}:
            raise ValueError(f"Unknown circuit routing mode: {mode!r}")
        l2p = dict(self.init_l2p_map)
        residents = {}
        for logical_id, node in l2p.items():
            residents.setdefault(node, set()).add(logical_id)

        def validate_boundary(label):
            """Check unique positions / 检查边界位置唯一性.
            Input / 输入: boundary label / 边界名. Output / 输出: None; raises on collision / 冲突时抛错.
            """
            collisions = {node: sorted(ids) for node, ids in residents.items() if len(ids) > 1}
            if collisions:
                raise ValueError(f"EQST history has duplicate logical positions at {label}: {collisions}; recompile the circuit")

        def place(logical_id, node):
            """Move one known ID / 移动已知逻辑 ID.
            Inputs / 输入: logical ID and target / 逻辑 ID 与目标. Output / 输出: None; updates replay / 更新重放映射.
            """
            if logical_id not in l2p:
                raise ValueError(f"EQST history references missing logical ID {logical_id}; recompile the circuit")
            old_node = l2p[logical_id]
            residents[old_node].remove(logical_id)
            if not residents[old_node]:
                del residents[old_node]
            l2p[logical_id] = node
            residents.setdefault(node, set()).add(logical_id)

        def occupant(node, action):
            """Invert one unambiguous endpoint / 反查无歧义端点.
            Inputs / 输入: node and action / 端点与动作. Output / 输出: resident ID or -1 / 驻留 ID 或 -1.
            """
            ids = residents.get(node, set())
            if len(ids) > 1:
                raise ValueError(
                    f"Ambiguous SwapGate {action.id} at {node}: missing stage-owned pre-swap logical IDs; "
                    "recompile the circuit"
                )
            return next(iter(ids), -1)

        validate_boundary("initial mapping")
        for stage_index, stage in enumerate(self.stages):
            before = dict(l2p) if isinstance(stage, RidingStage) else None
            for action in stage.circuit_actions:
                if isinstance(action, GhzPreparation):
                    if len(action.ancilla_qubits) != len(action.source_physical_qubits):
                        raise ValueError(f"Invalid GHZ operands in action {action.id}; recompile the circuit")
                    for logical_id, node in zip(action.ancilla_qubits, action.source_physical_qubits):
                        if logical_id in l2p:
                            raise ValueError(f"GHZ action {action.id} recreates live logical ID {logical_id}")
                        if mode == "immediate" and residents.get(node):
                            raise ValueError(f"GHZ action {action.id} overwrites occupied node {node}")
                        l2p[logical_id] = node
                        residents.setdefault(node, set()).add(logical_id)
                    continue
                if isinstance(action, Teleport):
                    if action.ancilla_qubit not in l2p:
                        raise ValueError(f"Teleport {action.id} releases missing ancilla {action.ancilla_qubit}")
                    node = l2p.pop(action.ancilla_qubit)
                    residents[node].remove(action.ancilla_qubit)
                    if not residents[node]:
                        del residents[node]
                    continue
                if isinstance(action, DirectGate):
                    continue
                if isinstance(action, SwapGate):
                    pair = getattr(stage, "swap_logical_ids", {}).get(action.id) if mode == "concurrent" else None
                    if pair is None:
                        if mode == "concurrent" and any(len(ids) > 1 for ids in residents.values()):
                            raise ValueError(
                                f"Ambiguous SwapGate {action.id}: missing stage-owned pre-swap logical IDs "
                                "while concurrent positions overlap; recompile the circuit"
                            )
                        pair = (occupant(action.qubit_u, action), occupant(action.qubit_v, action))
                        if pair[0] != action.logic_qubit:
                            raise ValueError(
                                f"Ambiguous SwapGate {action.id}: cannot infer pre-swap logical IDs; recompile the circuit"
                            )
                    if len(pair) != 2 or pair[0] != action.logic_qubit or (pair[0] != -1 and pair[0] == pair[1]):
                        raise ValueError(f"Invalid pre-swap logical IDs for SwapGate {action.id}; recompile the circuit")
                    if pair[0] != -1:
                        place(pair[0], action.qubit_v)
                    if pair[1] != -1:
                        place(pair[1], action.qubit_u)
                    continue
                if isinstance(action, Transport):
                    source, target = action.source_physical_qubit, action.target_physical_qubit
                elif isinstance(action, Board):
                    source, target = action.platform, action.station
                elif isinstance(action, Alight):
                    source, target = action.station, action.platform
                elif isinstance(action, Interchange):
                    source, target = action.interchange
                elif isinstance(action, Enter):
                    source, target = action.register, action.platform
                elif isinstance(action, Exit):
                    source, target = action.platform, action.register
                else:
                    raise ValueError(f"Unsupported EQST history action: {type(action).__name__}")
                if mode == "immediate":
                    if l2p.get(action.logic_qubit) != source:
                        raise ValueError(
                            f"EQST history action {action.id} source disagrees with logical ID {action.logic_qubit}; "
                            "recompile the circuit"
                        )
                    exchanged = occupant(target, action)
                    place(action.logic_qubit, target)
                    if exchanged != -1 and exchanged != action.logic_qubit:
                        place(exchanged, source)
                else:
                    # Directed concurrent requests are not consecutive endpoint swaps.
                    # 并发请求按逻辑 ID 更新，允许同阶段尚未配对的临时位置重叠。
                    place(action.logic_qubit, target)
            validate_boundary(f"stage {stage_index}")
            if before is not None:
                yield stage_index, before, dict(l2p)
        if l2p != self.curr_l2p_map:
            raise ValueError("Final EQST replay disagrees with Circuit.curr_l2p_map; recompile the circuit")
        expected_p2l = {node: logical_id for logical_id, node in l2p.items()}
        actual_p2l = {node: logical_id for node, logical_id in self.curr_p2l_map.items() if logical_id != -1}
        if expected_p2l != actual_p2l:
            raise ValueError(
                "Final EQST replay disagrees with Circuit.curr_p2l_map; historical physical-map maintenance "
                "may be missing; recompile the circuit"
            )

    def estimate_eqst_batches(self, package: Package) -> list[dict]:
        """Estimate each Riding/die net map once, paired by every resident logical ID.
        按阶段与传输裸片估计净映射；所有驻留逻辑 ID（含静止态）一一配对。

        Args / 输入: package supplies actual chiplets and full chain geometry / 封装提供裸片与完整链几何.
        Returns / 输出: batch dictionaries with frozen boundary tuples, time and distance / 批次边界、时间及等效距离.
        This is net-map resynthesis, not a replay of physical trajectories / 本结果为净映射重合成成本.
        """
        if package is None:
            raise ValueError("Hamiltonian EQST estimation requires a package")
        settings = self.op_config.get_eqst_config()
        node_dies = {}

        def die_at(node):
            """Resolve a physical die within this call / 本次调用内查询节点裸片.
            Input / 输入: physical node / 物理节点. Output / 输出: owning chiplet / 所属裸片.
            """
            if node not in node_dies:
                node_dies[node] = package.get_node_chiplet(node)
            return node_dies[node]

        def by_die(mapping):
            """Group logical positions by actual die / 按实际裸片分组逻辑位置.
            Input / 输入: l2p map / 逻辑映射. Output / 输出: die-local position dictionaries / 裸片局部位置字典.
            """
            result = {}
            for logical_id, node in mapping.items():
                die = die_at(node)
                if isinstance(die, TransferChiplet):
                    if not die.has_node(node):
                        raise ValueError(f"Physical node {node} is outside its transfer die {die}")
                    result.setdefault(die, {})[logical_id] = node[die.axis] - die.start
            return result

        batches = []
        for stage_index, before, after in self._replay_stage_mappings():
            starts, ends = by_die(before), by_die(after)
            transports = {}
            for action in self.stages[stage_index].circuit_actions:
                if isinstance(action, Transport):
                    die = die_at(action.source_physical_qubit)
                    if not isinstance(die, TransferChiplet) or die_at(action.target_physical_qubit) is not die:
                        raise ValueError(f"Transport {action.id} must stay within one TransferChiplet; use Interchange between dies")
                    transports.setdefault(die, []).append(action.id)
            dies = starts.keys() | ends.keys() | transports.keys()
            for die in sorted(dies, key=lambda item: (item.axis, item.pos, item.start, item.end)):
                start_positions, end_positions = starts.get(die, {}), ends.get(die, {})
                if start_positions.keys() != end_positions.keys():
                    raise ValueError(
                        f"Riding stage {stage_index} changes resident logical IDs on {die}; "
                        "die entry/exit belongs in a BoardingStage"
                    )
                logical_ids = tuple(sorted(start_positions))
                batches.append({
                    "stage_index": stage_index,
                    "die": (die.axis, die.pos, die.start, die.end),
                    "logical_ids": logical_ids,
                    "sources": tuple(start_positions[q] for q in logical_ids),
                    "targets": tuple(end_positions[q] for q in logical_ids),
                    "chain_length": die.end - die.start + 1,
                    "transport_action_ids": tuple(transports.get(die, ())),
                })
        # Exhaust replay (including final-map checks) before invoking any planner.
        # 完整验证重放终态后再调用规划器；不按可变 Circuit 身份缓存。
        from ..hamiltonian import estimate_transfer
        for batch in batches:
            result = estimate_transfer(
                batch["sources"], batch["targets"], chain_length=batch["chain_length"],
                reference_time_ns=settings["reference_time_ns"],
                reference_distance=settings["reference_distance"], g_max=settings["g_max"],
                max_segment_steps=settings["max_segment_steps"], beam_width=settings["beam_width"],
                ordered=True,
            )
            batch["transfer_time_ns"] = result["transfer_time_ns"]
            batch["effective_distance_edges"] = result["effective_distance_edges"]
        return batches

    def _estimate_hamiltonian_duration(self, package: Package, eqst_batches=None) -> float:
        """Contract same-die Transport nodes while preserving all external dependencies.
        收缩同裸片 Transport 节点并保留外部依赖；内部旧传输腿不重复计时。

        Inputs / 输入: package and optional freshly computed batches / 封装及可选本次批次.
        Returns / 输出: sum of stage critical paths in ns / 阶段关键路径之和(ns).
        Raises / 异常: ValueError if contraction hides an observed intermediate state / 收缩掩盖中间态时拒绝.
        """
        batches = self.estimate_eqst_batches(package) if eqst_batches is None else eqst_batches
        by_stage = {}
        for batch in batches:
            by_stage.setdefault(batch["stage_index"], []).append(batch)
        duration = 0.0
        for stage_index, stage in enumerate(self.stages):
            actions = {action.id: action for action in stage.circuit_actions}
            if self.action_depend_graph is not None:
                original = self.action_depend_graph.subgraph(actions).copy()
            else:
                original = stage.get_action_depend_graph()
                # Keep any explicitly recorded Transport dependencies before contraction.
                # 收缩前保留已有节点记录中的传输依赖；正常阶段中传输与计算并行。
                for action_ids in stage.node_action_ids.values():
                    original.add_edges_from(zip(action_ids, action_ids[1:]))
            original.add_nodes_from(actions)
            if not nx.is_directed_acyclic_graph(original):
                raise ValueError(f"Action dependency graph is cyclic in stage {stage_index}")
            graph = nx.DiGraph()
            replacement = {}
            weights = {}
            for batch in by_stage.get(stage_index, ()):
                node = ("eqst", stage_index, tuple(batch["die"]))
                weights[node] = batch["transfer_time_ns"]
                for action_id in batch["transport_action_ids"]:
                    replacement[action_id] = node
            for action_id, action in actions.items():
                if isinstance(action, Transport):
                    if action_id not in replacement:
                        raise ValueError(f"Missing Hamiltonian batch for Transport {action_id}")
                else:
                    replacement[action_id] = action_id
                    weights[action_id] = self.get_action_duration(action)
            graph.add_nodes_from(weights)
            for source, target in original.edges:
                source, target = replacement[source], replacement[target]
                if source != target:
                    graph.add_edge(source, target)
            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError(
                    f"Hamiltonian batch contraction in stage {stage_index} creates a dependency cycle: "
                    "a non-Transport operation observes an intermediate transfer state; cannot resynthesize one net batch"
                )
            depths = dict(weights)
            for node in nx.topological_sort(graph):
                for successor in graph.successors(node):
                    depths[successor] = max(depths[successor], depths[node] + weights[successor])
            duration += max(depths.values(), default=0.0)
        return duration

    def estimate_duration(self, package: Package = None, eqst_batches=None) -> float:
        """
        Estimate duration using the configured EQST model / 按所选 EQST 模型估计时长.

        Args / 输入:
            package: Physical package (needed for transfer timing) / 物理封装.
            eqst_batches: Optional batches freshly computed for the same circuit / 可选本次计算的同电路批次.
        Returns / 输出:
            float: The estimated duration in nanoseconds / 估计总时长(ns).
        """
        if self.op_config.get_eqst_config()["model"] == "hamiltonian":
            return self._estimate_hamiltonian_duration(package, eqst_batches)
        duration = 0
        for stage in self.stages:
            if len(stage.circuit_actions) == 0:
                continue
            if self.action_depend_graph is not None:
                stage_action_depend_graph = self.action_depend_graph.subgraph([action.id for action in stage.circuit_actions])
            else:
                stage_action_depend_graph = stage.get_action_depend_graph()
            stage_action_depth = {action_id: self.get_action_duration(stage_action_depend_graph.nodes[action_id]["action"]) for action_id in stage_action_depend_graph.nodes}
            for action_id in nx.topological_sort(stage_action_depend_graph):
                for succ_action_id in stage_action_depend_graph.successors(action_id):
                    if stage_action_depth[succ_action_id] < stage_action_depth[action_id] + self.get_action_duration(stage_action_depend_graph.nodes[succ_action_id]["action"]):
                        stage_action_depth[succ_action_id] = stage_action_depth[action_id] + self.get_action_duration(stage_action_depend_graph.nodes[succ_action_id]["action"])
            if len(stage_action_depth) == 0:
                stage_action_duration = 0
            else:
                stage_action_duration = max(stage_action_depth.values())
            if isinstance(stage, RidingStage):
                transport_chiplet_dict = {}
                for action in stage.circuit_actions:
                    if isinstance(action, Transport):
                        transport_chiplet = package.get_node_chiplet(action.source_physical_qubit)
                        if transport_chiplet in transport_chiplet_dict:
                            # transport_chiplet_dict[transport_chiplet] += self.get_action_duration(action)
                            transport_chiplet_dict[transport_chiplet] = max(transport_chiplet_dict[transport_chiplet], self.get_action_duration(action))
                        else:
                            transport_chiplet_dict[transport_chiplet] = self.get_action_duration(action)
                if len(list(transport_chiplet_dict.values())) != 0:
                    # print(max(transport_chiplet_dict.values()))
                    stage_action_duration = max(stage_action_duration, max(transport_chiplet_dict.values()))
            duration += stage_action_duration
        return duration

    @classmethod
    def from_pickle(cls, filepath: str):
        """
        Load a circuit from a pickle file.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Unpickled object is not a {cls.__name__}")
        return obj

    def to_pickle(self, filepath: str):
        """
        Save the circuit to a pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

class DummyCircuit:
    """
    Represents a dummy circuit for testing purposes.
    
    Attributes:
        logic_qubit_num (int): The number of logic qubits in the circuit.
        init_l2p_map (dict): The initial mapping from logical qubits to physical qubits.
        init_p2l_map (dict): The initial mapping from physical qubits to logical qubits.
        op_config (OperationConfig): The configuration for operations, including time constants for different operations.
        action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.
    """

    def __init__(self, logic_qubit_num: int, package_shape: tuple[int, int], init_l2p_map: dict[int, tuple[int, int]], init_p2l_map: dict[tuple[int, int], int], op_config: OperationConfig):
        """
        Initialize the dummy circuit.

        Args:
            logic_qubit_num (int): The number of logic qubits in the circuit.
            package_shape (tuple[int, int]): The shape of the package, representing the number of rows and columns of physical qubits.
            init_l2p_map (dict[int, tuple[int, int]]): The initial mapping from logical qubits to physical qubits.
            init_p2l_map (dict[tuple[int, int], int]): The initial mapping from physical qubits to logical qubits.
            op_config (OperationConfig): The configuration for operations, including time constants for different operations.
        """
        self.logic_qubit_num = logic_qubit_num
        self.package_shape = package_shape
        self.init_l2p_map = init_l2p_map
        self.init_p2l_map = init_p2l_map
        self.op_config = op_config

        self.curr_l2p_map = deepcopy(init_l2p_map)
        self.curr_p2l_map = deepcopy(init_p2l_map)
        self.action_depend_graph = nx.DiGraph()
        self.action_depend_graph.add_node(-1, action=None)
        self.recent_node_active_action = {(x, y): -1 for x in range(self.package_shape[0]) for y in range(self.package_shape[1])}
        if config.debug_action:
            logging.info("-" * 80)
            logging.info(f"Initialized circuit with {self.logic_qubit_num} logic qubits.")
            logging.info(f"Current qubit mapping: {self.curr_l2p_map}")
    
    def add_action(self, action: CircuitAction):
        """
        Add an action to the dummy circuit.

        Args:
            action (CircuitAction): The action to be added.
        """
        if isinstance(action, GhzPreparation):
            for physical_qubit in action.source_physical_qubits:
                assert self.curr_p2l_map[physical_qubit] == -1, f"In action {action.id}, physical qubit {physical_qubit} should be free for GHZ preparation."

            for ancilla_qubit, physical_qubit in zip(action.ancilla_qubits, action.source_physical_qubits):
                self.curr_l2p_map[ancilla_qubit] = physical_qubit
                self.curr_p2l_map[physical_qubit] = ancilla_qubit
        elif isinstance(action, Board):
            assert self.curr_p2l_map[action.platform] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to board should be in platform {action.platform}."

            alight_qubit = self.curr_p2l_map[action.station]
            self.curr_l2p_map[action.logic_qubit] = action.station
            self.curr_p2l_map[action.station] = action.logic_qubit
            if alight_qubit != -1:
                self.curr_l2p_map[alight_qubit] = action.platform
            self.curr_p2l_map[action.platform] = alight_qubit
        elif isinstance(action, Alight):
            assert self.curr_p2l_map[action.station] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to alight should be in station {action.station}."

            board_qubit = self.curr_p2l_map[action.platform]
            self.curr_l2p_map[action.logic_qubit] = action.platform
            self.curr_p2l_map[action.platform] = action.logic_qubit
            if board_qubit != -1:
                self.curr_l2p_map[board_qubit] = action.station
            self.curr_p2l_map[action.station] = board_qubit
        elif isinstance(action, Interchange):
            assert self.curr_p2l_map[action.interchange[0]] == action.logic_qubit, f"In action {action.id}, data qubit {action.logic_qubit} to interchange should be in source interchange {action.interchange[0]}."

            interchanged_qubit = self.curr_p2l_map[action.interchange[1]]
            self.curr_l2p_map[action.logic_qubit] = action.interchange[1]
            self.curr_p2l_map[action.interchange[1]] = action.logic_qubit
            if interchanged_qubit != -1:
                self.curr_l2p_map[interchanged_qubit] = action.interchange[0]
            self.curr_p2l_map[action.interchange[0]] = interchanged_qubit
        elif isinstance(action, Transport):
            assert self.curr_p2l_map[action.source_physical_qubit] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to transport should be in source balise {action.source_physical_qubit}."

            exchanged_qubit = self.curr_p2l_map[action.target_physical_qubit]
            self.curr_l2p_map[action.logic_qubit] = action.target_physical_qubit
            self.curr_p2l_map[action.target_physical_qubit] = action.logic_qubit
            if exchanged_qubit != -1:
                self.curr_l2p_map[exchanged_qubit] = action.source_physical_qubit
            self.curr_p2l_map[action.source_physical_qubit] = exchanged_qubit
        elif isinstance(action, Teleport):
            assert self.curr_p2l_map[action.ancilla_node] == action.ancilla_qubit, f"In action {action.id}, ancilla qubit {action.ancilla_qubit} for teleportation should be in platform {action.ancilla_node}."
            assert self.curr_p2l_map[action.data_node] == action.data_qubit, f"In action {action.id}, data qubit {action.data_qubit} to teleport should be in register {action.data_node}."

            self.curr_l2p_map.pop(action.ancilla_qubit)
            self.curr_p2l_map[action.ancilla_node] = -1
        elif isinstance(action, Enter):
            assert self.curr_p2l_map[action.register] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to enter should be in register {action.register}."

            exit_qubit = self.curr_p2l_map[action.platform]
            self.curr_l2p_map[action.logic_qubit] = action.platform
            self.curr_p2l_map[action.platform] = action.logic_qubit
            if exit_qubit != -1:
                self.curr_l2p_map[exit_qubit] = action.register
            self.curr_p2l_map[action.register] = exit_qubit
        elif isinstance(action, Exit):
            assert self.curr_p2l_map[action.platform] == action.logic_qubit, f"In action {action.id}, logic qubit {action.logic_qubit} to exit should be in platform {action.platform}."

            enter_qubit = self.curr_p2l_map[action.register]
            self.curr_l2p_map[action.logic_qubit] = action.register
            self.curr_p2l_map[action.register] = action.logic_qubit
            if enter_qubit != -1:
                self.curr_l2p_map[enter_qubit] = action.platform
            self.curr_p2l_map[action.platform] = enter_qubit
        elif isinstance(action, SwapGate):
            assert self.curr_p2l_map[action.qubit_u] == action.logic_qubit, \
                f"In action {action.id}, logic qubit {action.logic_qubit} should can be routed by the swap gate between {action.qubit_u} and {action.qubit_v}."

            logic_qubit_u = self.curr_p2l_map[action.qubit_u]
            logic_qubit_v = self.curr_p2l_map[action.qubit_v]
            if logic_qubit_u != -1:
                self.curr_l2p_map[logic_qubit_u] = action.qubit_v
            self.curr_p2l_map[action.qubit_u] = logic_qubit_v
            if logic_qubit_v != -1:
                self.curr_l2p_map[logic_qubit_v] = action.qubit_u
            self.curr_p2l_map[action.qubit_v] = logic_qubit_u
        elif isinstance(action, DirectGate):
            assert self.curr_p2l_map[action.phy_ctrl_qubit] == action.logic_gate.control_qubit, \
                f"In action {action.id}, control qubit {action.logic_gate.control_qubit} should be in physical qubit {action.phy_ctrl_qubit}."
            assert self.curr_p2l_map[action.phy_targ_qubit] == action.logic_gate.target_qubit, \
                f"In action {action.id}, target qubit {action.logic_gate.target_qubit} should be in physical qubit {action.phy_targ_qubit}."
        else:
            raise ValueError("Unknown action type.")

        self.action_depend_graph.add_node(action.id, action=action)
        for node in action.get_involved_nodes(dummy=True):
            self.action_depend_graph.add_edge(self.recent_node_active_action[node], action.id)
            self.recent_node_active_action[node] = action.id

    def get_action_duration(self, action: CircuitAction) -> float:
        """
        Get the duration of the action in the circuit.

        Args:
            action (CircuitAction): The action to get the duration for.

        Returns:
            float: The duration of the action in nanoseconds.
        """
        if isinstance(action, GhzPreparation):
            return action.get_max_degree() * self.op_config.intra_time
        elif isinstance(action, (Board, Alight, Interchange)):
            return self.op_config.inter_time * 3
        elif isinstance(action, Transport):
            return action.get_distance() * self.op_config.qst_time_constant
            return action.get_distance() ** 0.5 * self.op_config.qst_time_constant
        elif isinstance(action, Teleport):
            return self.op_config.inter_time if action.inter_flag else self.op_config.intra_time
        elif isinstance(action, (SwapGate, Enter, Exit)):
            return (self.op_config.inter_time if action.inter_flag else self.op_config.intra_time) * 3
        elif isinstance(action, DirectGate):
            return self.op_config.inter_time if action.inter_flag else self.op_config.intra_time
        elif action is None:
            return 0

    def estimate_duration(self) -> float:
        """
        Estimate the duration of the circuit.

        Returns:
            float: The estimated duration of the circuit in nanoseconds.
        """
        durations = {action_id: self.get_action_duration(self.action_depend_graph.nodes[action_id]["action"]) for action_id in self.action_depend_graph}
        longest_time = {action_id: 0 for action_id in self.action_depend_graph}

        for action_id in nx.topological_sort(self.action_depend_graph):
            preds = list(self.action_depend_graph.predecessors(action_id))
            if preds:
                longest_time[action_id] = max(longest_time[p] for p in preds) + durations[action_id]
            else:
                longest_time[action_id] = durations[action_id]

        return max(longest_time.values(), default=0)

    @classmethod
    def from_pickle(cls, filepath: str):
        """
        Load a circuit from a pickle file.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Unpickled object is not a {cls.__name__}")
        return obj

    def to_pickle(self, filepath: str):
        """
        Save the circuit to a pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
