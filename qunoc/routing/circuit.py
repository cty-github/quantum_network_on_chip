import logging
import pickle
import networkx as nx
from copy import deepcopy

from ..element.benchmark import *
from ..element.chiplet import OperationConfig


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
        ancilla_qubits (list[int]): The ancilla qubits used to prepare the GHZ state.
        data_qubits (list[int]): The data qubits that the GHZ state is serving for.
        source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
        target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
    """

    def __init__(self, 
            qubit_num: int, 
            ancilla_qubits: list[int],
            data_qubits: list[int], 
            source_physical_qubits: list[tuple[int, int]], 
            target_physical_qubits: list[tuple[int, int]], 
        ):
        """
        Initialize the GHZ preparation action.

        Args:
            qubit_num (int): Number of ancilla qubits in the GHZ state.
            ancilla_qubits (list[int]): The ancilla qubits used in the GHZ state.
            data_qubits (list[int]): The data qubits that the GHZ state is serving for.
            source_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state is prepared.
            target_physical_qubits (list[tuple[int, int]]): The list of physical qubits where the GHZ state will be routed to.
        """
        super().__init__()
        self.qubit_num = qubit_num
        self.ancilla_qubits = ancilla_qubits
        self.data_qubits = data_qubits
        self.source_physical_qubits = source_physical_qubits
        self.target_physical_qubits = target_physical_qubits
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Prepare {self.qubit_num}-GHZ state for logic qubits {self.data_qubits} on physical qubits {self.source_physical_qubits}."

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
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Board logic qubit {self.logic_qubit} from physical qubit {self.platform} to physical qubit {self.station}."

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
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Alight logic qubit {self.logic_qubit} from physical qubit {self.station} to physical qubit {self.platform}."

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
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Interchange logic qubit {self.logic_qubit} between physical qubits {self.interchange}."

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
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Transport logic qubit {self.logic_qubit} from physical qubit {self.source_physical_qubit} to physical qubit {self.target_physical_qubit}."

class Teleport(RidingAction):
    """
    Represents the action to teleport a data qubit using a prepared GHZ state.
    
    Attributes:
        ancilla_qubit (int): The ancilla qubit used for teleportation.
        data_qubit (int): The data qubit to be teleported.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
        control_flag (bool): The flag marking if the data qubit is the control qubit.
    """

    def __init__(self, ancilla_qubit: int, data_qubit: int, platform: tuple[int, int], control_flag: bool):
        """
        Initialize the teleportation action.

        Args:
            ancilla_qubit (int): The ancilla qubit used for teleportation.
            data_qubit (int): The data qubit to be teleported.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the ancilla qubit is located.
            control_flag (bool): The flag marking if the data qubit is the control qubit.
        """
        super().__init__()
        self.ancilla_qubit = ancilla_qubit
        self.data_qubit = data_qubit
        self.platform = platform
        self.control_flag = control_flag
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Teleport data qubit {self.data_qubit} with ancilla qubit {self.ancilla_qubit} on physical qubit {self.platform}."

class Enter(RidingAction):
    """
    Represents the action of entering a platform with a logic qubit.

    Attributes:
        logic_qubit (int): The logic qubit to enter.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
        register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], register: tuple[int, int]):
        """
        Initialize the enter action.

        Args:
            logic_qubit (int): The logic qubit to enter.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
            register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.register = register
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Enter logic qubit {self.logic_qubit} to physical qubit {self.platform} from physical qubit {self.register}."
    
class Exit(RidingAction):
    """
    Represents the action of exiting a platform with a logic qubit.

    Attributes:
        logic_qubit (int): The logic qubit to exit.
        platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
        register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
    """

    def __init__(self, logic_qubit: int, platform: tuple[int, int], register: tuple[int, int]):
        """
        Initialize the exit action.

        Args:
            logic_qubit (int): The logic qubit to exit.
            platform (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is exiting.
            register (tuple[int, int]): The physical qubit on compute chiplet where the logic qubit is entering.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.platform = platform
        self.register = register
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Exit logic qubit {self.logic_qubit} from physical qubit {self.platform} to physical qubit {self.register}."

class SwapGate(RidingAction):
    """
    Represents the action of applying a swap gate on a compute module.

    Attributes:
        logic_qubit (int): The logic qubit routed by the swap gate.
        qubit_u (tuple[int, int]): The first qubit of the swap gate.
        qubit_v (tuple[int, int]): The second qubit of the swap gate.
    """
    
    def __init__(self, logic_qubit: int, qubit_u: tuple[int, int], qubit_v: tuple[int, int]):
        """
        Initialize the swap gate action.

        Args:
            logic_qubit (int): The logic qubit routed by the swap gate.
            qubit_u (tuple[int, int]): The first qubit of the swap gate.
            qubit_v (tuple[int, int]): The second qubit of the swap gate.
        """
        super().__init__()
        self.logic_qubit = logic_qubit
        self.qubit_u = qubit_u
        self.qubit_v = qubit_v
        logging.info(f"Transformed {self}")
        
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Route logic qubit {self.logic_qubit} with swap gate between physical qubits {self.qubit_u} and {self.qubit_v}."

class DirectGate(RidingAction):
    """
    Represents the action of applying a local gate on a compute module.

    Attributes:
        logic_gate (Control): The logical control gate.
    """
    
    def __init__(self, logic_gate: Control):
        """
        Initialize the local gate action.
        
        Args:
            logic_gate (Control): The logical control gate.
        """
        super().__init__()
        self.logic_gate = logic_gate
        logging.info(f"Transformed {self}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: {self.logic_gate}."


class CircuitStage:
    """
    Represents a stage in the quantum circuit.
    
    Attributes:
        id (int): Unique identifier for the stage.
        curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
        curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
        occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
        occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
        holding_ancilla_plats (set[tuple[int, int]]): Recording the platforms holding ancilla qubits.
    """
    _counter = 0

    def __init__(self, 
        curr_l2p_map: dict[int, tuple[int, int]], 
        curr_p2l_map: dict[tuple[int, int], int], 
        occupied_balises: set[tuple[int, int]], 
        occupied_platforms: set[tuple[int, int]], 
        holding_ancilla_plats: set[tuple[int, int]]
    ):
        """
        Initialize a circuit stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): Recording the platforms holding ancilla qubits.
        """
        self.id = CircuitStage._counter
        CircuitStage._counter += 1

        self.curr_l2p_map = curr_l2p_map
        self.curr_p2l_map = curr_p2l_map
        self.occupied_balises = occupied_balises
        self.occupied_platforms = occupied_platforms
        self.holding_ancilla_plats = holding_ancilla_plats

    def get_action_num(self) -> int:
        """
        Get the number of actions in the stage.

        Returns:
            int: The number of actions in the stage.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

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
        holding_ancilla_plats: set[tuple[int, int]]
    ):
        """
        Initialize the boarding stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): Recording the platforms holding ancilla qubits.
        """
        super().__init__(curr_l2p_map, curr_p2l_map, occupied_balises, occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 0, "Boarding stage id should be even"
        self.boarding_actions: list[BoardingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises_to_free = set()
        self.potential_free_platforms = dict()
        self.paired_platforms_to_free = set()

    def get_action_num(self) -> int:
        return len(self.boarding_actions)

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
        self.occupied_balises.update(action.source_physical_qubits)
        for ancilla_qubit, physical_qubit in zip(action.ancilla_qubits, action.source_physical_qubits):
            self.curr_l2p_map[ancilla_qubit] = physical_qubit
            self.curr_p2l_map[physical_qubit] = ancilla_qubit

        self.boarding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert action.platform in self.occupied_platforms, "Data qubit to board should be in platform."
        
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

        self.boarding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert action.station in self.occupied_balises, "Data qubit to alight should be in station."

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

        self.boarding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert action.interchange[0] in self.occupied_balises, "Data qubit to interchange should be in source interchange."

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

        self.boarding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        holding_ancilla_plats: set[tuple[int, int]]
    ):
        """
        Initialize the riding stage.
        
        Args:
            curr_l2p_map (dict[int, tuple[int, int]]): Recording the current mapping from logical qubits to physical qubits.
            curr_p2l_map (dict[tuple[int, int], int]): Recording the current mapping from physical qubits to logical qubits.
            occupied_balises (set[tuple[int, int]]): Recording the occupied balises.
            occupied_platforms (set[tuple[int, int]]): Recording the occupied platforms.
            holding_ancilla_plats (set[tuple[int, int]]): Recording the platforms holding ancilla qubits.
        """
        super().__init__(curr_l2p_map, curr_p2l_map, occupied_balises, occupied_platforms, holding_ancilla_plats)
        assert self.id % 2 == 1, "Operating stage id should be odd"
        self.riding_actions: list[RidingAction] = []

        self.potential_free_balises = dict()
        self.paired_balises_to_free = set()
        self.potential_free_platforms = dict()
        self.paired_platforms_to_free = set()
        self.potential_free_registers = dict()
        self.paired_registers_to_free = set()

    def get_action_num(self) -> int:
        return len(self.riding_actions)

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
        assert action.source_physical_qubit in self.occupied_balises, f"Data qubit on {action.source_physical_qubit} to transport should be in source balise: {self.occupied_balises}."

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
        
        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert action.platform in self.occupied_platforms, "Ancilla qubit for teleportation should be in platform."

        self.occupied_platforms.remove(action.platform)
        self.curr_l2p_map.pop(action.ancilla_qubit)
        self.curr_p2l_map[action.platform] = -1

        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert self.curr_p2l_map[action.register] == action.logic_qubit, "Data qubit to enter should be in register."

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
        
        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        assert action.platform in self.occupied_platforms, "Data qubit to exit should be in platform."

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

        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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

        logic_qubit_u = self.curr_p2l_map[action.qubit_u]
        logic_qubit_v = self.curr_p2l_map[action.qubit_v]
        self.curr_p2l_map[action.qubit_u] = logic_qubit_v
        self.curr_p2l_map[action.qubit_v] = logic_qubit_u
        if logic_qubit_u != -1:
            self.curr_l2p_map[logic_qubit_u] = action.qubit_v
        if logic_qubit_v != -1:
            self.curr_l2p_map[logic_qubit_v] = action.qubit_u

        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
        
        physical_qubit_control = self.curr_l2p_map[action.logic_gate.control_qubit]
        physical_qubit_target = self.curr_l2p_map[action.logic_gate.target_qubit]
        assert (physical_qubit_control[0] == physical_qubit_target[0] and abs(physical_qubit_control[1] - physical_qubit_target[1]) == 1) or \
               (physical_qubit_control[1] == physical_qubit_target[1] and abs(physical_qubit_control[0] - physical_qubit_target[0]) == 1), \
               "Control and target qubits should be adjacent on the compute module."

        self.riding_actions.append(action)
        logging.info(f"Added {action.__class__.__name__} {action.id}.")
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
    
    def __init__(self, logic_qubit_num: int, init_l2p_map: dict, init_p2l_map: dict, action_depend_graph: nx.DiGraph, op_config: OperationConfig):
        """
        Initialize the circuit.

        Args:
            logic_qubit_num (int): The number of logic qubits in the circuit.
            init_l2p_map (dict): The initial mapping from logical qubits to physical qubits.
            init_p2l_map (dict): The initial mapping from physical qubits to logical qubits.
            action_depend_graph (nx.DiGraph): The directed acyclic graph representing dependencies between actions.
            op_config (OperationConfig): The configuration for operations, including time constants for different operations.
        """
        self.logic_qubit_num = logic_qubit_num
        self.init_l2p_map = init_l2p_map
        self.init_p2l_map = init_p2l_map
        self.action_depend_graph = action_depend_graph
        self.op_config = op_config
        
        self.stages: list[CircuitStage] = [] # only use extend_stage to add new stage
        CircuitStage._counter = 0  # reset the stage counter
        
        self.curr_l2p_map = deepcopy(init_l2p_map)
        self.curr_p2l_map = deepcopy(init_p2l_map)
        self.occupied_balises = set()
        self.occupied_platforms = set()
        self.holding_ancilla_plats = set()

    def __repr__(self):
        if not hasattr(self, "string_repr"):
            self.string_repr = str()
            for stage in self.stages:
                self.string_repr += "=" * 20 + str(stage.__class__.__name__) + "=" * 20 + "\n"
                if isinstance(stage, BoardingStage):
                    for action in stage.boarding_actions:
                        self.string_repr += str(action) + "\n"
                elif isinstance(stage, RidingStage):
                    for action in stage.riding_actions:
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
            new_stage = BoardingStage(self.curr_l2p_map, self.curr_p2l_map, self.occupied_balises, self.occupied_platforms, self.holding_ancilla_plats)
        else:
            new_stage = RidingStage(self.curr_l2p_map, self.curr_p2l_map, self.occupied_balises, self.occupied_platforms, self.holding_ancilla_plats)
        assert new_stage.id == len(self.stages), "Stage id should be equal to the length of existing stages"
        self.stages.append(new_stage)
        logging.info("-"*50)
        logging.info(f"Extended circuit with {new_stage.__class__.__name__} {new_stage.id}.")
        logging.info(f"Current qubit mapping: {new_stage.curr_l2p_map}")
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
                inter_chiplet_depth = 2 * self.op_config.inter_time + 3 * self.op_config.single_time if inter_chiplet_flag else 0
                ghz_depth = 2 * self.op_config.qst_time_constant + 2 * self.op_config.single_time if ghz_flag else 0
                stage_depth = max(inter_chiplet_depth, ghz_depth)
            elif isinstance(stage, RidingStage):
                transfer_actions = []
                compute_actions = []
                for action in stage.riding_actions:
                    if isinstance(action, Transport):
                        transfer_actions.append(action)
                    else:
                        assert isinstance(action, (Teleport, SwapGate, DirectGate, Enter, Exit)), "Action should be Teleport, SwapGate, DirectGate, Enter or Exit"
                        compute_actions.append(action)
                transfer_depth = len(transfer_actions) * self.op_config.qst_time_constant
                compute_action_graph = self.action_depend_graph.subgraph(compute_actions)
                computer_depth = nx.dag_longest_path_length(compute_action_graph) * self.op_config.intra_time if compute_action_graph else 0
                stage_depth = max(transfer_depth, computer_depth)
            depth += stage_depth
        return depth

    def to_pickle(self, filepath: str):
        """
        Save the circuit to a pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_pickle(cls, filepath: str):
        """
        Load a circuit from a pickle file.
        """
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Unpickled object is not a {cls.__name__}")
        return obj
