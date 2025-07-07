import logging
import statistics
from copy import deepcopy

from ..utils.config import config
from ..element.package import *
from ..element.benchmark import *
from .circuit import *


class ComputeBlock:
    """
    Represents a compute block in the quantum routing process.

    Attributes:
        id (int): The id of the compute block.
    """
    _counter = 0

    def __init__(self):
        """
        Initialize a compute block.
        """
        self.id = ComputeBlock._counter
        ComputeBlock._counter += 1
    
    def init_logging(self):
        """
        Initialize logging for the block debugging.
        """
        if config.debug_block:
            logging.info("-" * 80)
            logging.info(f"Constructed {self}")

class LocalBlock(ComputeBlock):
    """
    Represents a compute block including a logical gate can be executed locally with high fidelity.

    Attributes:
        ctrl_gate (Control): The control gate in the local block.
        module (ComputeModule): The compute module where the local block is executed.
    """

    def __init__(self, ctrl_gate: Control, module: ComputeModule):
        """
        Initialize a LocalBlock.

        Args:
            ctrl_gate (Control): The control gate in the local block.
            module (ComputeModule): The compute module where the local block is executed.
        """
        super().__init__()
        self.ctrl_gate = ctrl_gate
        self.module = module
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Local {self.ctrl_gate} on {self.module}."

class TeleBlock(ComputeBlock):
    """
    Represents a teleportation block in which gates are executed by teleportation.

    Attributes:
        tele_gates (list[Control]): The gates to be executed in the teleportation block.
        control_qubit (int): The control qubit of the teleportation block.
        target_qubits (list[int]): The target qubits of the teleportation block.
        nearest_platforms (dict[int, tuple[int, int]]): The nearest platforms for each involved qubit.
    """

    def __init__(self, tele_gates: list[Control], control_qubit: int, target_qubits: list[int], nearest_platforms: dict[int, tuple[int, int]]):
        """
        Initialize a TeleBlock.

        Args:
            tele_gates (list[Control]): The gates to be executed in the teleportation block.
            control_qubit (int): The control qubit of the teleportation block.
            target_qubits (list[int]): The target qubits of the teleportation block.
            nearest_platforms (dict[int, tuple[int, int]]): The nearest platforms for each involved qubit.
        """
        super().__init__()
        self.tele_gates = tele_gates
        self.control_qubit = control_qubit
        self.target_qubits = target_qubits
        self.nearest_platforms = nearest_platforms
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Teleport qubit {self.control_qubit} to {self.target_qubits}."

class MoveBlock(ComputeBlock):
    """
    Represents a move block in which gates are executed by moving qubits.

    Attributes:
        ctrl_gates (list[Control]): The control gates in the move block.
        chiplet (ComputeChiplet): The chiplet where the move block is executed.
    """

    def __init__(self, ctrl_gates: list[Control], chiplet: ComputeChiplet):
        """
        Initialize a MoveBlock.

        Args:
            ctrl_gates (list[Control]): The control gates in the move block.
            chiplet (ComputeChiplet): The chiplet where the move block is executed.
        """
        super().__init__()
        self.ctrl_gates = ctrl_gates
        self.chiplet = chiplet
        self.init_logging()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.id}: Move {self.ctrl_gates} to {self.chiplet}."

    def get_involved_qubits(self) -> set[int]:
        """
        Get the list of involved qubits in the move block.

        Returns:
            set[int]: A set of involved qubits in the move block.
        """
        if not hasattr(self, "involved_qubits"):
            self.involved_qubits = set()
            for gate in self.ctrl_gates:
                self.involved_qubits.add(gate.control_qubit)
                self.involved_qubits.add(gate.target_qubit)
        return self.involved_qubits


class Router:
    """
    Represents a router for routing quantum programs on a package.

    Attributes:
        package (Package): The package to be routed on.
        program (Program): The quantum program to be routed.
        result_dir (str): The directory to save the routing results.

        package_graph (nx.Graph): The graph representing the package topology.
        metro_graph (nx.Graph): The subgraph of the package topology representing the metro network.
        gate_depend_graph (nx.DiGraph): The dependency graph of gates in the quantum program.
        ancilla_qubit_id (int): The new id of the ancilla qubit in the quantum program.
        gate_actions (dict[int, list[int]]): The actions for each gate in the quantum program.

        block_history (list[plt.Figure]): The history of compute blocks in the routing process.
        action_history (list[plt.Figure]): The history of actions in the routing process.

        init_l2p_map (dict[int, tuple[int, int]]): The initial mapping of logic qubits to physical qubits.
        init_p2l_map (dict[tuple[int, int], int]): The initial mapping of physical qubits to logic qubits.
        curr_l2p_map (dict[int, tuple[int, int]]): The current mapping of logic qubits to physical qubits.
        curr_p2l_map (dict[tuple[int, int], int]): The current mapping of physical qubits to logic qubits.

        block_depend_graph (nx.DiGraph): The dependency graph of compute blocks.
        action_depend_graph (nx.DiGraph): The dependency graph of actions.
        local_gate_freq (dict[int, int]): The dictionary recording frequency of local gates for each logic qubit.
        recent_logic_qubit_action_dict (dict[int, int]): The dictionary recording the most recent action for each logic qubit.
        recent_physical_qubit_action_dict (dict[tuple[int, int], int]): The dictionary recording the most recent action for each physical qubit.
        last_ghz_action_id (int): The id of the last GHZ action in the routing process.

        circuit (Circuit): The physical quantum circuit after routing.
    """
    
    def __init__(self, package: Package, program: Program, result_dir: str = None):
        """
        Initialize a Router.

        Args:
            package (Package): The package to be routed on.
            program (Program): The quantum program to be routed.
            result_dir (str): The directory to save the routing results, default is None.
        """
        self.package = package
        self.program = program
        self.result_dir = result_dir

        self.package_graph = package.package_graph
        self.metro_graph: nx.Graph = package.package_graph.subgraph(self.package.transfer_nodes)

        self.gate_depend_graph = program.gate_depend_graph
        self.ancilla_qubit_id = program.data_qubit_num
        self.gate_actions: dict[int, list[int]] = dict()

        ComputeBlock._counter = 0
        self.block_history: list[plt.Figure] = []
        self.action_history: list[plt.Figure] = []

    def route_program_on_package(self) -> Circuit:
        """
        Route the program on the package.
        1) map logic qubits to physical qubits;
        2) determine compute blocks;
        3) transform compute blocks into circuit actions;
        4) schedule actions to circuit.

        Returns:
            Circuit: The physical quantum circuit after routing.
        """
        self.map_qubits()
        self.compile_blocks_and_actions()
        self.schedule_actions_to_circuit()
        return self.circuit

    def map_qubits(self):
        """
        Map logic qubits to physical qubits which play as registers.
        """
        self.init_l2p_map: dict[int, tuple[int, int]] = dict()
        self.init_p2l_map: dict[tuple[int, int], int] = {phy_qubit: -1 for phy_qubit in self.package.compute_nodes.union(self.package.transfer_nodes)}
        logic_data_qubit_num = self.program.data_qubit_num
        physical_data_qubit_set = set(self.package.get_registers())
        if logic_data_qubit_num > len(physical_data_qubit_set):
            raise ValueError("No enough physical qubits to map logic qubits.")
        for i in range(logic_data_qubit_num):
            self.init_l2p_map[i] = physical_data_qubit_set.pop()
            self.init_p2l_map[self.init_l2p_map[i]] = i
        self.curr_l2p_map = deepcopy(self.init_l2p_map)
        self.curr_p2l_map = deepcopy(self.init_p2l_map)

        if config.debug_mapping:
            logging.info(f"Initial qubit mapping: {self.init_l2p_map}")
        fig = self.draw_qubit_mapping("Initial Qubit Mapping")
        self.block_history.append(fig)
        self.action_history.append(fig)

    # Functions for compiling compute blocks and actions
    def compile_blocks_and_actions(self):
        """
        Compile control gates into compute blocks, either TeleBlock or MoveBlock. Transform the compute blocks into circuit actions.

        Compute criteria: sequentially check no dependency gates
        1) LocalBlock: collect a single gate whose control qubit and target qubit are in the same compute area and can realize with higher fidelity without transfer line.
        2) TeleBlock: collect following gates having the same control qubit with the first gate, if #gate > 2, then form a TeleBlock.
        3) MoveBlock: collect following gates, if #gate > #qubits*(#qubits-1)//4 and #gate < max(chiplet_capacity). If #gate > 2, then form a MoveBlock.
        4) Otherwise, form a TeleBlock.
        """
        self.block_depend_graph = nx.DiGraph()
        self.block_depend_graph.add_node(-1, block=None)
        self.gate_depend_graph.nodes[-1]["block"] = -1
        self.action_depend_graph = nx.DiGraph()
        self.action_depend_graph.add_node(-1, action=None)
        self.local_gate_freq = {logic_qubit: 0 for logic_qubit in range(self.program.data_qubit_num)}
        self.recent_logic_qubit_action_dict = {logic_qubit: -1 for logic_qubit in range(self.program.data_qubit_num)}
        self.recent_physical_qubit_action_dict = {phy_qubit: -1 for phy_qubit in self.package.compute_nodes} # queried for dependency when try to use a vacant physical qubit
        self.last_ghz_action_id = -1

        for gate_id in self.gate_depend_graph.nodes:
            self.gate_depend_graph.nodes[gate_id]["pred"] = set(self.gate_depend_graph.predecessors(gate_id))
        no_depend_gate_ids: set[int] = set()
        for gate_id in self.gate_depend_graph.nodes:
            if len(self.gate_depend_graph.nodes[gate_id]["pred"]) == 0:
                no_depend_gate_ids.add(gate_id)
        no_depend_gate_ids.remove(-1)
        self.update_no_depend_gate_ids(no_depend_gate_ids, -1)
        
        while len(no_depend_gate_ids) > 0:
            gate_id = no_depend_gate_ids.pop()
            self.update_no_depend_gate_ids(no_depend_gate_ids, gate_id)
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            module = self.check_local_gate(gate)
            if module:
                self.local_gate_freq[gate.control_qubit] += 1
                self.local_gate_freq[gate.target_qubit] += 1
                local_block = self.construct_local_block(gate, module)
                self.transform_local_block(local_block)
            else:
                tele_gates = [gate]
                ctrl_qubit = gate.control_qubit
                while True:
                    add_gate = None
                    for cand_gate_id in no_depend_gate_ids:
                        cand_gate = self.gate_depend_graph.nodes[cand_gate_id]["gate"]
                        assert isinstance(cand_gate, Control)
                        if cand_gate.control_qubit == ctrl_qubit:
                            add_gate = cand_gate
                            break
                    if add_gate:
                        tele_gates.append(add_gate)
                        no_depend_gate_ids.remove(add_gate.id)
                        self.update_no_depend_gate_ids(no_depend_gate_ids, add_gate.id)
                    else:
                        break
                if len(tele_gates) > 2:
                    block_type = "tele"
                else:
                    move_gates = tele_gates.copy()
                    while True:
                        add_gate = None
                        for cand_gate_id in no_depend_gate_ids:
                            cand_gate = self.gate_depend_graph.nodes[cand_gate_id]["gate"]
                            assert isinstance(cand_gate, Control)
                            if self.check_move_gate(cand_gate, move_gates):
                                add_gate = cand_gate
                                break
                        if add_gate:
                            move_gates.append(add_gate)
                            no_depend_gate_ids.remove(add_gate.id)
                            self.update_no_depend_gate_ids(no_depend_gate_ids, add_gate.id)
                        else:
                            break
                    if len(move_gates) > 2:
                        block_type = "move"
                    else:
                        block_type = "tele"
                        if len(move_gates) == 2 and len(tele_gates) == 1:
                            restore_gate = move_gates[1]
                            no_depend_gate_ids.add(move_gates[1].id)
                            for succ_gate in self.gate_depend_graph.successors(restore_gate.id):
                                self.gate_depend_graph.nodes[succ_gate]["pred"].add(restore_gate.id)
                                no_depend_gate_ids.discard(succ_gate)
                if block_type == "tele":
                    self.schedule_tele_gates_no_contention(tele_gates)
                else:
                    move_block = self.construct_move_block(move_gates)
                    self.transform_move_block(move_block)

        assert nx.is_directed_acyclic_graph(self.block_depend_graph), "Block dependency graph is not a DAG."
        assert nx.is_directed_acyclic_graph(self.action_depend_graph), "Action dependency graph is not a DAG."

    def update_no_depend_gate_ids(self, no_depend_gate_ids: set[int], removed_gate_id: int) -> set[int]:
        """
        Update the no_depend_gate_ids set by checking removed gate's successors.

        Args:
            no_depend_gate_ids (set[int]): The original set of gate ids with no dependencies.
            removed_gate_id (int): The id of the removed gate.

        Returns:
            set[int]: The updated set of gate ids with no dependencies.
        """
        for succ_gate in self.gate_depend_graph.successors(removed_gate_id):
            self.gate_depend_graph.nodes[succ_gate]["pred"].remove(removed_gate_id)
            if len(self.gate_depend_graph.nodes[succ_gate]["pred"]) == 0:
                no_depend_gate_ids.add(succ_gate)
        return no_depend_gate_ids

    def check_local_gate(self, gate: Control) -> ComputeModule | None:
        """
        Check if the gate is a local gate.

        Local criteria: transfer chiplet involved in the shortest path.

        Args:
            gate (Control): The control gate to be checked.

        Returns:
            ComputeModule | None: The compute module to which the gate belongs if the gate is a local gate, otherwise None.
        """
        shortest_path, _ = self.package.get_shortest_path(self.curr_l2p_map[gate.control_qubit], self.curr_l2p_map[gate.target_qubit])
        if len(shortest_path.roads) == 1:
            return self.package.get_node_area(self.curr_l2p_map[gate.control_qubit])
        else:
            return None
    
    def construct_local_block(self, local_gate: Control, module: ComputeModule) -> LocalBlock:
        """
        Construct a local block for the given local gate and add it to the block dependency graph.

        Args:
            local_gate (Control): The local gate to be added to the block.
            module (ComputeModule): The compute module where the local block is executed.

        Returns:
            LocalBlock: The constructed local block.
        """
        local_block = LocalBlock(local_gate, module)
        self.block_depend_graph.add_node(local_block.id, block=local_block)
        self.gate_depend_graph.nodes[local_gate.id]["block"] = local_block.id
        for pred_gate in self.gate_depend_graph.predecessors(local_gate.id):
            self.block_depend_graph.add_edge(self.gate_depend_graph.nodes[pred_gate]["block"], local_block.id)
        
        fig = self.draw_qubit_mapping_block([local_gate.control_qubit, local_gate.target_qubit], title=f"Construct LocalBlock {local_block.id}")
        self.block_history.append(fig)
        return local_block

    def transform_local_block(self, local_block: LocalBlock):
        """
        Transform local block into circuit actions.
        1) route qubits together -> SwapGate;
        2) execute logic gate -> DirectGate.

        Action dependency: SwapGate -> DirectGate.
        """
        self.block_depend_graph.nodes[local_block.id]["actions"] = []
        local_gate = local_block.ctrl_gate
        self.gate_actions[local_gate.id] = []
        compute_module = local_block.module
        swap_path = compute_module.get_swap_path(self.curr_l2p_map[local_gate.control_qubit], self.curr_l2p_map[local_gate.target_qubit])
        for i in range(len(swap_path)-1):
            if i < len(swap_path)-2:
                phy_qubit_u, phy_qubit_v = swap_path[i], swap_path[i+1]
                assert self.curr_p2l_map[phy_qubit_u] == local_gate.control_qubit, f"phy_qubit_u: {phy_qubit_u}, local_gate.control_qubit: {local_gate.control_qubit}"
                swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                swap_gate_act = SwapGate(local_gate.control_qubit, phy_qubit_u, phy_qubit_v)
                self.gate_actions[local_gate.id].append(swap_gate_act.id)
                self.block_depend_graph.nodes[local_block.id]["actions"].append(swap_gate_act.id)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.control_qubit] = swap_gate_act.id
                self.curr_l2p_map[local_gate.control_qubit] = phy_qubit_v
                self.curr_p2l_map[phy_qubit_v] = local_gate.control_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                else:
                    self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[phy_qubit_v], swap_gate_act.id)
                    self.recent_physical_qubit_action_dict[phy_qubit_u] = swap_gate_act.id
                self.curr_p2l_map[phy_qubit_u] = swapped_qubit
            else:
                assert self.curr_p2l_map[swap_path[i]] == local_gate.control_qubit, f"phy_qubit_u: {swap_path[i]}, local_gate.control_qubit: {local_gate.control_qubit}"
                assert self.curr_p2l_map[swap_path[i+1]] == local_gate.target_qubit, f"phy_qubit_v: {swap_path[i+1]}, local_gate.target_qubit: {local_gate.target_qubit}"
                direct_gate_act = DirectGate(local_gate, swap_path[i], swap_path[i+1])
                self.gate_actions[local_gate.id].append(direct_gate_act.id)
                self.block_depend_graph.nodes[local_block.id]["actions"].append(direct_gate_act.id)
                self.action_depend_graph.add_node(direct_gate_act.id, action=direct_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], direct_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.control_qubit] = direct_gate_act.id
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.target_qubit], direct_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.target_qubit] = direct_gate_act.id
        
        fig = self.draw_qubit_mapping_block([local_gate.control_qubit, local_gate.target_qubit], title=f"Transform LocalBlock {local_block.id}")
        self.block_history.append(fig)
    
    def schedule_tele_gates_no_contention(self, tele_gates: list[Control]):
        """
        Schedule tele gates into TeleBlocks without platform contention.

        Args:
            tele_gates (list[Control]): The list of control gates to be scheduled into TeleBlocks.
        """
        ctrl_qubit = tele_gates[0].control_qubit
        ctrl_module = self.package.get_node_area(self.curr_l2p_map[ctrl_qubit])
        assert isinstance(ctrl_module, ComputeModule)
        nearest_ctrl_platform = None
        highest_ctrl_fide = 0
        for platform in ctrl_module.get_platforms():
            _, fide = self.package.get_shortest_no_transfer_path(self.curr_l2p_map[ctrl_qubit], platform)
            if nearest_ctrl_platform is None or fide > highest_ctrl_fide:
                nearest_ctrl_platform = platform
                highest_ctrl_fide = fide
        
        while len(tele_gates) > 0:
            nearest_platforms = {ctrl_qubit: nearest_ctrl_platform}
            scheduled_tele_gates = []
            for tele_gate in tele_gates:
                targ_qubit = tele_gate.target_qubit
                if targ_qubit in nearest_platforms:
                    scheduled_tele_gates.append(tele_gate)
                    continue
                compute_module = self.package.get_node_area(self.curr_l2p_map[targ_qubit])
                assert isinstance(compute_module, ComputeModule)
                nearest_platform = None
                highest_fide = 0
                for platform in compute_module.get_platforms():
                    if platform in nearest_platforms.values():
                        continue
                    _, fide = self.package.get_shortest_no_transfer_path(self.curr_l2p_map[targ_qubit], platform)
                    if nearest_platform is None or fide > highest_fide:
                        nearest_platform = platform
                        highest_fide = fide
                if nearest_platform:
                    nearest_platforms[targ_qubit] = nearest_platform
                    scheduled_tele_gates.append(tele_gate)
            assert len(scheduled_tele_gates) > 0, "No scheduled tele gates found."
            tele_block = self.construct_tele_block(scheduled_tele_gates, nearest_platforms)
            self.transform_tele_block(tele_block)
            for tele_gate in scheduled_tele_gates:
                tele_gates.remove(tele_gate)

    def construct_tele_block(self, tele_gates: list[Control], nearest_platforms: dict[int, tuple[int, int]]) -> TeleBlock:
        """
        Construct a teleportation block for the given tele gates and add it to the block dependency graph.

        Args:
            tele_gates (list[Control]): The list of control gates in the teleportation block.
            nearest_platforms (dict[int, tuple[int, int]]): The nearest platforms for each involved qubit.

        Returns:
            TeleBlock: The constructed teleportation block.
        """
        ctrl_qubit = tele_gates[0].control_qubit
        target_qubits = [tele_gate.target_qubit for tele_gate in tele_gates]
        assert set([ctrl_qubit] + target_qubits) == set(nearest_platforms.keys()), "Control qubit and target qubits must match the nearest platforms."
        tele_block = TeleBlock(tele_gates, ctrl_qubit, target_qubits, nearest_platforms)
        self.block_depend_graph.add_node(tele_block.id, block=tele_block)
        depend_block_ids = set()
        tele_ids = [tele_gate.id for tele_gate in tele_gates]
        for tele_gate in tele_gates:
            self.gate_depend_graph.nodes[tele_gate.id]["block"] = tele_block.id
            for pred_gate in self.gate_depend_graph.predecessors(tele_gate.id):
                if pred_gate not in tele_ids:
                    depend_block_ids.add(self.gate_depend_graph.nodes[pred_gate]["block"])
        for depend_block_id in depend_block_ids:
            assert depend_block_id != tele_block.id
            self.block_depend_graph.add_edge(depend_block_id, tele_block.id)

        fig = self.draw_qubit_mapping_block([ctrl_qubit] + target_qubits, title=f"Construct TeleBlock {tele_block.id}")
        self.block_history.append(fig)
        return tele_block

    def find_nearest_platform(self, physical_qubit: tuple[int, int]) -> tuple[int, int]:
        """
        Find the nearest unoccupied platform to route in or out for a physical qubit.

        Args:
            physical_qubit (tuple[int, int]): The coordinate of the physical qubit.

        Return:
            tuple[int, int]: The coordinate of the nearest platform.
        """
        compute_module = self.package.get_node_area(physical_qubit)
        assert isinstance(compute_module, ComputeModule)
        nearest_platform = None
        highest_fide = 0
        for platform in compute_module.get_platforms():
            _, fide = self.package.get_shortest_no_transfer_path(physical_qubit, platform)
            if nearest_platform is None or fide > highest_fide:
                nearest_platform = platform
                highest_fide = fide
        return nearest_platform
    
    def find_nearest_platforms(self, physical_qubits: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Find the nearest unoccupied platforms for a list of physical qubits.

        Args:
            physical_qubits (list[tuple[int, int]]): The list of coordinates of the physical qubits.

        Return:
            list[tuple[int, int]]: The list of coordinates of the nearest platforms (no repeat if possible).
        """
        nearest_platforms = []
        for physical_qubit in physical_qubits:
            compute_module = self.package.get_node_area(physical_qubit)
            assert isinstance(compute_module, ComputeModule)
            nearest_platform = None
            highest_fide = 0
            nearest_platform_vacant = None
            highest_fide_vacant = 0
            for platform in compute_module.get_platforms():
                _, fide = self.package.get_shortest_no_transfer_path(physical_qubit, platform)
                if nearest_platform is None or fide > highest_fide:
                    nearest_platform = platform
                    highest_fide = fide
                if platform not in nearest_platforms:
                    if nearest_platform_vacant is None or fide > highest_fide_vacant:
                        nearest_platform_vacant = platform
                        highest_fide_vacant = fide
            if nearest_platform_vacant:
                nearest_platforms.append(nearest_platform_vacant)
            else:
                nearest_platforms.append(nearest_platform)
        return nearest_platforms

    def get_balise_nearest_metro_subgraph(self, balise_id: tuple[int, int], node_num: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Get the nearest metro subgraph for a balise.

        Args:
            balise_id (tuple[int, int]): The coordinate of the balise.
            node_num (int): The number of nodes in the subgraph.

        Return:
            list[tuple[tuple[int, int], tuple[int, int]]]: The list of nodes in the nearest metro subgraph and how it is linked to the subgraph.
        """
        visited_balise: set[tuple[int, int]] = set()
        visited_balise_graph: list[tuple[tuple[int, int], tuple[int, int]]] = []
        candidate_balise_queue = [(balise_id, None)]
        while candidate_balise_queue and len(visited_balise_graph) < node_num:
            node, parent = candidate_balise_queue.pop(0)
            if node not in visited_balise:
                visited_balise.add(node)
                visited_balise_graph.append((node, parent))
                candidate_balise_queue.extend([(nbr, node) for nbr in self.metro_graph.neighbors(node) if nbr not in visited_balise])
        assert len(visited_balise) == len(visited_balise_graph), "Repeated balise found in the visited balise graph."
        return visited_balise_graph

    def locate_ghz_state(self, involved_qubits: list[int], platforms: list[tuple[int, int]], tele_gate_ids: list[int]) -> GhzPreparation:
        """
        Allocate GHZ resource for a list of logic qubits for teleportation.
        Extend the ghz graph from the station of control qubit, try to ensure the total transfer distance is minimized.

        Args:
            involved_qubits (list[int]): The list of logic qubits to be teleported.
            platforms (list[tuple[int, int]]): The list of platforms to which the qubits will be teleported.

        Return:
            GhzPreparation: The action to allocate GHZ resource.
        """
        ancilla_qubit_num = len(involved_qubits)
        stations = [self.package.platform_stations_dict[platform][0] for platform in platforms]
        ghz_balise_graph = self.get_balise_nearest_metro_subgraph(stations[0], ancilla_qubit_num)
        source_physical_qubits = [ghz_balise for ghz_balise, _ in ghz_balise_graph]
        target_physical_qubits = stations
        ghz_action = GhzPreparation(ancilla_qubit_num, [self.ancilla_qubit_id + i for i in range(ancilla_qubit_num)], involved_qubits, source_physical_qubits, target_physical_qubits, ghz_balise_graph, tele_gate_ids)
        self.ancilla_qubit_id += ancilla_qubit_num
        return ghz_action

    def route_on_transfer_chiplets(self, logic_qubit: int, source_physical_qubit: tuple[int, int], target_physical_qubit: tuple[int, int]) -> list[Transport | Interchange]:
        """
        Route the qubit on transfer chiplets with least transfer algorithm.

        Args:
            logic_qubit (int): The logic qubit to be routed.
            source_physical_qubit (tuple[int, int]): The source physical qubit.
            target_physical_qubit (tuple[int, int]): The target physical qubit.

        Return:
            list[Transport | Interchange]: The action list to route the qubit on transfer chiplets.
        """
        actions = []
        transfer_path, _ = self.package.get_shortest_transfer_path(source_physical_qubit, target_physical_qubit)
        for road in transfer_path.roads:
            if isinstance(road, IntraTransferRoad):
                if road.source != road.target:
                    actions.append(Transport(logic_qubit, road.source, road.target))
            elif isinstance(road, InterChipletRoad):
                actions.append(Interchange(logic_qubit, (road.source, road.target)))
        return actions

    def transform_tele_block(self, tele_block: TeleBlock):
        """
        Transform tele block into circuit actions.
        1) route nearest platform for all involved qubits -> SwapGate;
        2) allocate GHZ resource -> GhzPreparation;
        3) route on transfer chiplets -> TransferChipletRoute.
        4) ancilla qubits get off chiplet -> Alight;
        5) teleport qubits -> Teleport.

        Action dependency: GhzPreparation -> TransferChipletRoute & SwapGate -> Alight -> Teleport.
        """
        self.block_depend_graph.nodes[tele_block.id]["actions"] = []
        involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits
        platforms = [tele_block.nearest_platforms[qubit] for qubit in involved_qubits]
        ghz_action = self.locate_ghz_state(involved_qubits, platforms, [tele_gate.id for tele_gate in tele_block.tele_gates])
        for tele_gate in tele_block.tele_gates:
            self.gate_actions[tele_gate.id] = [ghz_action.id]
        self.block_depend_graph.nodes[tele_block.id]["actions"].append(ghz_action.id)
        self.action_depend_graph.add_node(ghz_action.id, action=ghz_action)
        self.action_depend_graph.add_edge(self.last_ghz_action_id, ghz_action.id)
        self.last_ghz_action_id = ghz_action.id
        for data_qubit, ancilla_qubit, source_physical_qubit in zip(involved_qubits, ghz_action.ancilla_qubits, ghz_action.source_physical_qubits):
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[data_qubit], ghz_action.id)
            self.curr_l2p_map[ancilla_qubit] = source_physical_qubit
            self.curr_p2l_map[source_physical_qubit] = ancilla_qubit

        fig = self.draw_qubit_mapping_block(involved_qubits, ancilla_qubits=list(zip(ghz_action.source_physical_qubits, ghz_action.ancilla_qubits)), title=f"Transform TeleBlock {tele_block.id}")
        self.block_history.append(fig)

        for i in range(len(involved_qubits)):
            data_qubit = involved_qubits[i]
            qubit_actions = []
            route_source_phy_qubit = self.curr_l2p_map[data_qubit]
            route_target_phy_qubit = platforms[i]
            module = self.package.get_node_area(route_source_phy_qubit)
            assert isinstance(module, ComputeModule)
            swap_path = module.get_swap_path(route_source_phy_qubit, route_target_phy_qubit)
            for j in range(len(swap_path)-2):
                phy_qubit_u, phy_qubit_v = swap_path[j], swap_path[j+1]
                assert self.curr_p2l_map[phy_qubit_u] == data_qubit, f"phy_qubit_u: {phy_qubit_u}, logic_qubit: {data_qubit}"
                swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                swap_gate_act = SwapGate(data_qubit, phy_qubit_u, phy_qubit_v)
                qubit_actions.append(swap_gate_act.id)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[data_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[data_qubit] = swap_gate_act.id
                self.curr_l2p_map[data_qubit] = phy_qubit_v
                self.curr_p2l_map[phy_qubit_v] = data_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                else:
                    self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[phy_qubit_v], swap_gate_act.id)
                    self.recent_physical_qubit_action_dict[phy_qubit_u] = swap_gate_act.id
                self.curr_p2l_map[phy_qubit_u] = swapped_qubit
            
            ancilla_qubit = ghz_action.ancilla_qubits[i]
            tele_source_phy_qubit = ghz_action.source_physical_qubits[i]
            tele_target_phy_qubit = ghz_action.target_physical_qubits[i]
            ancilla_nodes = [tele_source_phy_qubit]
            transfer_on_metro_acts = self.route_on_transfer_chiplets(ancilla_qubit, tele_source_phy_qubit, tele_target_phy_qubit)
            for j in range(len(transfer_on_metro_acts)):
                qubit_actions.append(transfer_on_metro_acts[j].id)
                self.action_depend_graph.add_node(transfer_on_metro_acts[j].id, action=transfer_on_metro_acts[j])
                if j == 0:
                    self.action_depend_graph.add_edge(ghz_action.id, transfer_on_metro_acts[j].id)
                else:
                    self.action_depend_graph.add_edge(transfer_on_metro_acts[j-1].id, transfer_on_metro_acts[j].id)
                if isinstance(transfer_on_metro_acts[j], Transport):
                    ancilla_nodes.append(transfer_on_metro_acts[j].target_physical_qubit)
                elif isinstance(transfer_on_metro_acts[j], Interchange):
                    ancilla_nodes.append(transfer_on_metro_acts[j].interchange[1])
            alight_act = Alight(ancilla_qubit, tele_target_phy_qubit, route_target_phy_qubit)
            qubit_actions.append(alight_act.id)
            self.action_depend_graph.add_node(alight_act.id, action=alight_act)
            if len(transfer_on_metro_acts) == 0:
                self.action_depend_graph.add_edge(ghz_action.id, alight_act.id)
            else:
                self.action_depend_graph.add_edge(transfer_on_metro_acts[-1].id, alight_act.id)
            ancilla_nodes.append(route_target_phy_qubit)
            ghz_action.record_ancilla_qubit_nodes(ancilla_qubit, ancilla_nodes)
            teleport_act = Teleport(ancilla_qubit, data_qubit, route_target_phy_qubit, swap_path[-2], data_qubit==tele_block.control_qubit)
            qubit_actions.append(teleport_act.id)
            self.action_depend_graph.add_node(teleport_act.id, action=teleport_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[data_qubit], teleport_act.id)
            self.recent_logic_qubit_action_dict[data_qubit] = teleport_act.id
            self.action_depend_graph.add_edge(alight_act.id, teleport_act.id)
            self.curr_l2p_map.pop(ancilla_qubit)
            self.curr_p2l_map[tele_source_phy_qubit] = -1
            if i == 0:
                for tele_gate in tele_block.tele_gates:
                    self.gate_actions[tele_gate.id].extend(qubit_actions)
                tele_ctrl_act_id = teleport_act.id
            else:
                assert tele_block.tele_gates[i-1].target_qubit == data_qubit
                self.gate_actions[tele_block.tele_gates[i-1].id].extend(qubit_actions)
                self.action_depend_graph.add_edge(tele_ctrl_act_id, teleport_act.id)
            self.block_depend_graph.nodes[tele_block.id]["actions"].extend(qubit_actions)
        
        fig = self.draw_qubit_mapping_block(involved_qubits, ancilla_qubits=list(zip(platforms, ghz_action.ancilla_qubits)), title=f"Transform TeleBlock {tele_block.id}")
        self.block_history.append(fig)
        
    def check_move_gate(self, gate: Control, move_gates: list[Control]) -> bool:
        """
        Check if the gate can be inserted into the move gates.

        Concentrate criteria: #gate > #qubits*(#qubits-1)//4 and #qubit < max(chiplet_capacity)

        Args:
            gate (Control): The control gate to be checked.
            move_gates (list[Control]): The list of integrated move gates.

        Returns:
            bool: True if the gate can be inserted into the move gates, otherwise False.
        """
        involved_qubits = set()
        for move_gate in move_gates:
            involved_qubits.add(move_gate.control_qubit)
            involved_qubits.add(move_gate.target_qubit)
        if gate.control_qubit not in involved_qubits and gate.target_qubit not in involved_qubits:
            return False
        involved_qubits.add(gate.control_qubit)
        involved_qubits.add(gate.target_qubit)
        
        max_chiplet_capacity = 0
        for involved_qubit in involved_qubits:
            physical_qubit = self.curr_l2p_map[involved_qubit]
            chiplet: ComputeChiplet = self.package_graph.nodes[physical_qubit]["qubit"].chiplet
            max_chiplet_capacity = max(max_chiplet_capacity, chiplet.get_capacity())

        if len(move_gates)+1 < len(involved_qubits) * (len(involved_qubits) - 1) // 3:
            return False
        if len(involved_qubits) > max_chiplet_capacity:
            return False
        return True

    def construct_move_block(self, move_gates: list[Control]) -> MoveBlock:
        """
        Construct a move block for the given move gates and add it to the block dependency graph.

        Args:
            move_gates (list[Control]): The list of control gates in the move block.

        Returns:
            MoveBlock: The constructed move block.
        """
        involved_qubits = set()
        for move_gate in move_gates:
            involved_qubits.add(move_gate.control_qubit)
            involved_qubits.add(move_gate.target_qubit)
        chiplet = self.find_chiplet_for_move_block(involved_qubits)
        move_block = MoveBlock(move_gates, chiplet)
        self.block_depend_graph.add_node(move_block.id, block=move_block)
        depend_block_ids = set()
        move_ids = [move_gate.id for move_gate in move_gates]
        for move_gate in move_gates:
            self.gate_depend_graph.nodes[move_gate.id]["block"] = move_block.id
            for pred_gate in self.gate_depend_graph.predecessors(move_gate.id):
                if pred_gate not in move_ids:
                    depend_block_ids.add(self.gate_depend_graph.nodes[pred_gate]["block"])
        for depend_block_id in depend_block_ids:
            assert depend_block_id != move_block.id
            self.block_depend_graph.add_edge(depend_block_id, move_block.id)
        
        fig = self.draw_qubit_mapping_block(list(involved_qubits), title=f"Construct MoveBlock {move_block.id}")
        self.block_history.append(fig)
        return move_block

    def find_chiplet_for_move_block(self, involved_qubits: list[int]) -> ComputeChiplet:
        """
        Determine the chiplet to move the qubits to.

        Args:
            involved_qubits (list[int]): The list of involved qubits in the move block.

        Returns:
            ComputeChiplet: The chiplet to which the qubits will be moved.
        """
        chiplet_cnt = dict()
        for qubit in involved_qubits:
            physical_qubit = self.curr_l2p_map[qubit]
            chiplet: ComputeChiplet = self.package_graph.nodes[physical_qubit]["qubit"].chiplet
            chiplet_capacity = chiplet.get_capacity()
            if chiplet_capacity > len(involved_qubits):
                if chiplet in chiplet_cnt:
                    chiplet_cnt[chiplet] += 1
                else:
                    chiplet_cnt[chiplet] = 1
        chiplet = max(chiplet_cnt, key=chiplet_cnt.get)
        return chiplet

    def transform_move_block(self, move_block: MoveBlock):
        """
        Transform move block into circuit actions.
        1) route nearest platform for all involved qubits -> SwapGate & Enter;
        2) data qubits get on chiplet -> Board;
        3) route on transfer chiplets -> TransferChipletRoute;
        4) data qubits get off chiplet -> Alight;
        5) route involved qubits to vacant physical qubits -> Exit & SwapGate;
        6) route local gates -> SwapGate & DirectGate.

        Action dependency: SwapGate -> Enter -> Board -> TransferChipletRoute -> Alight -> Exit -> SwapGate -> DirectGate.
        """
        self.block_depend_graph.nodes[move_block.id]["actions"] = []
        involved_qubits = move_block.get_involved_qubits()
        immigate_qubits: list[int] = []
        native_qubits: list[int] = []
        for qubit in involved_qubits:
            if self.package_graph.nodes[self.curr_l2p_map[qubit]]["qubit"].chiplet != move_block.chiplet:
                immigate_qubits.append(qubit)
            else:
                native_qubits.append(qubit)
        while len(immigate_qubits) > 0:
            immigate_qubit = immigate_qubits.pop()
            qubit_actions = []
            depart_node = self.curr_l2p_map[immigate_qubit]
            depart_module = self.package.get_node_area(depart_node)
            assert isinstance(depart_module, ComputeModule)
            depart_platform = self.find_nearest_platform(depart_node)
            depart_swap_path = depart_module.get_swap_path(depart_node, depart_platform)
            depart_station = self.package.platform_stations_dict[depart_platform][0]

            arrive_node = self.find_node_for_immigation(move_block.chiplet, native_qubits)
            expelled_qubit = self.curr_p2l_map[arrive_node]
            arrive_module = self.package.get_node_area(arrive_node)
            assert isinstance(arrive_module, ComputeModule)
            arrive_platform = self.find_nearest_platform(arrive_node)
            arrive_swap_path = arrive_module.get_swap_path(arrive_node, arrive_platform)
            arrive_station = self.package.platform_stations_dict[arrive_platform][0]

            # Swap the immigate_qubit from depart_node to depart_platform
            for j in range(len(depart_swap_path)-2):
                assert self.curr_p2l_map[depart_swap_path[j]] == immigate_qubit, f"qubit_u: {depart_swap_path[j]}, immigate_qubit: {immigate_qubit}"
                swapped_qubit = self.curr_p2l_map[depart_swap_path[j+1]]
                swap_gate_act = SwapGate(immigate_qubit, depart_swap_path[j], depart_swap_path[j+1])
                qubit_actions.append(swap_gate_act.id)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[immigate_qubit] = swap_gate_act.id
                self.curr_l2p_map[immigate_qubit] = depart_swap_path[j+1]
                self.curr_p2l_map[depart_swap_path[j+1]] = immigate_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = depart_swap_path[j]
                else:
                    self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[depart_swap_path[j+1]], swap_gate_act.id)
                    self.recent_physical_qubit_action_dict[depart_swap_path[j]] = swap_gate_act.id
                self.curr_p2l_map[depart_swap_path[j]] = swapped_qubit
            immigate_enter_act = Enter(immigate_qubit, depart_platform, depart_swap_path[-2])
            qubit_actions.append(immigate_enter_act.id)
            self.action_depend_graph.add_node(immigate_enter_act.id, action=immigate_enter_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_enter_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_enter_act.id
            self.curr_l2p_map[immigate_qubit] = depart_platform
            self.curr_p2l_map[depart_platform] = immigate_qubit
            self.recent_physical_qubit_action_dict[depart_swap_path[-2]] = immigate_enter_act.id
            self.curr_p2l_map[depart_swap_path[-2]] = -1

            # Swap the expelled_qubit from arrive_node to arrive_platform
            for j in range(len(arrive_swap_path)-2):
                assert self.curr_p2l_map[arrive_swap_path[j]] == expelled_qubit, f"qubit_u: {arrive_swap_path[j]}, expelled_qubit: {expelled_qubit}"
                swapped_qubit = self.curr_p2l_map[arrive_swap_path[j+1]]
                if expelled_qubit == -1 and swapped_qubit == -1:
                    continue
                swap_gate_act = SwapGate(expelled_qubit, arrive_swap_path[j], arrive_swap_path[j+1])
                qubit_actions.append(swap_gate_act.id)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                if expelled_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[expelled_qubit] = swap_gate_act.id
                    self.curr_l2p_map[expelled_qubit] = arrive_swap_path[j+1]
                else:
                    self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[arrive_swap_path[j]], swap_gate_act.id)
                    self.recent_physical_qubit_action_dict[arrive_swap_path[j+1]] = swap_gate_act.id
                self.curr_p2l_map[arrive_swap_path[j+1]] = expelled_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = arrive_swap_path[j]
                else:
                    self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[arrive_swap_path[j+1]], swap_gate_act.id)
                    self.recent_physical_qubit_action_dict[arrive_swap_path[j]] = swap_gate_act.id
                self.curr_p2l_map[arrive_swap_path[j]] = swapped_qubit
            if expelled_qubit != -1:
                expelled_enter_act = Enter(expelled_qubit, arrive_platform, arrive_swap_path[-2])
                qubit_actions.append(expelled_enter_act.id)
                self.action_depend_graph.add_node(expelled_enter_act.id, action=expelled_enter_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_enter_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_enter_act.id
                self.curr_l2p_map[expelled_qubit] = arrive_platform
                self.curr_p2l_map[arrive_platform] = expelled_qubit
                self.recent_physical_qubit_action_dict[arrive_swap_path[-2]] = expelled_enter_act.id
                self.curr_p2l_map[arrive_swap_path[-2]] = -1

            # Transfer the immigate_qubit from depart_platform to arrive_platform
            immigate_board_act = Board(immigate_qubit, depart_station, depart_platform)
            qubit_actions.append(immigate_board_act.id)
            self.action_depend_graph.add_node(immigate_board_act.id, action=immigate_board_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_board_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_board_act.id
            immigate_transfer_acts = self.route_on_transfer_chiplets(immigate_qubit, depart_station, arrive_station)
            for i in range(len(immigate_transfer_acts)):
                qubit_actions.append(immigate_transfer_acts[i].id)
                self.action_depend_graph.add_node(immigate_transfer_acts[i].id, action=immigate_transfer_acts[i])
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_transfer_acts[i].id)
                self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_transfer_acts[i].id
            immigate_alight_act = Alight(immigate_qubit, arrive_station, arrive_platform)
            qubit_actions.append(immigate_alight_act.id)
            self.action_depend_graph.add_node(immigate_alight_act.id, action=immigate_alight_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_alight_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_alight_act.id
            
            if expelled_qubit != -1:
                # Transfer the expelled_qubit from arrive_platform to depart_platform
                expelled_board_act = Board(expelled_qubit, arrive_station, arrive_platform)
                qubit_actions.append(expelled_board_act.id)
                self.action_depend_graph.add_node(expelled_board_act.id, action=expelled_board_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_board_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_board_act.id
                expelled_transfer_acts = self.route_on_transfer_chiplets(expelled_qubit, arrive_station, depart_station)
                for i in range(len(expelled_transfer_acts)):
                    qubit_actions.append(expelled_transfer_acts[i].id)
                    self.action_depend_graph.add_node(expelled_transfer_acts[i].id, action=expelled_transfer_acts[i])
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_transfer_acts[i].id)
                    self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_transfer_acts[i].id
                expelled_alight_act = Alight(expelled_qubit, depart_station, depart_platform)
                qubit_actions.append(expelled_alight_act.id)
                self.action_depend_graph.add_node(expelled_alight_act.id, action=expelled_alight_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_alight_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_alight_act.id

                # self.action_depend_graph.add_edge(expelled_board_act.id, immigate_alight_act.id)
                # self.action_depend_graph.add_edge(immigate_board_act.id, expelled_alight_act.id)

            self.curr_l2p_map[immigate_qubit] = arrive_platform
            self.curr_p2l_map[arrive_platform] = immigate_qubit
            if expelled_qubit != -1:
                self.curr_l2p_map[expelled_qubit] = depart_platform
            self.curr_p2l_map[depart_platform] = expelled_qubit

            immigate_exit_act = Exit(immigate_qubit, arrive_platform, arrive_swap_path[len(arrive_swap_path)-2])
            qubit_actions.append(immigate_exit_act.id)
            self.action_depend_graph.add_node(immigate_exit_act.id, action=immigate_exit_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_exit_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_exit_act.id
            self.curr_l2p_map[immigate_qubit] = arrive_swap_path[len(arrive_swap_path)-2]
            self.curr_p2l_map[arrive_swap_path[len(arrive_swap_path)-2]] = immigate_qubit
            # self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[arrive_swap_path[len(arrive_swap_path)-2]], immigate_exit_act.id)
            self.curr_p2l_map[arrive_platform] = -1
            if expelled_qubit != -1:
                expelled_exit_act = Exit(expelled_qubit, depart_platform, depart_swap_path[len(depart_swap_path)-2])
                qubit_actions.append(expelled_exit_act.id)
                self.action_depend_graph.add_node(expelled_exit_act.id, action=expelled_exit_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_exit_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_exit_act.id
                self.curr_l2p_map[expelled_qubit] = depart_swap_path[len(depart_swap_path)-2]
                self.curr_p2l_map[depart_swap_path[len(depart_swap_path)-2]] = expelled_qubit
                # self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[depart_swap_path[len(depart_swap_path)-2]], expelled_exit_act.id)
                self.curr_p2l_map[depart_platform] = -1
            
            for ctrl_gate in move_block.ctrl_gates:
                if ctrl_gate.control_qubit == immigate_qubit or ctrl_gate.target_qubit == immigate_qubit:
                    if ctrl_gate.id not in self.gate_actions:
                        self.gate_actions[ctrl_gate.id] = []
                    self.gate_actions[ctrl_gate.id].extend(qubit_actions)
            self.block_depend_graph.nodes[move_block.id]["actions"].extend(qubit_actions)
            native_qubits.append(immigate_qubit)

        # Route the local gates in the move block
        compute_module = self.package.get_node_area(move_block.chiplet.get_nodes()[0])
        assert isinstance(compute_module, ComputeModule)
        for local_gate in move_block.ctrl_gates:
            if local_gate.id not in self.gate_actions:
                self.gate_actions[local_gate.id] = []
            swap_path = compute_module.get_swap_path(self.curr_l2p_map[local_gate.control_qubit], self.curr_l2p_map[local_gate.target_qubit])
            for i in range(len(swap_path)-1):
                if i < len(swap_path)-2:
                    phy_qubit_u, phy_qubit_v = swap_path[i], swap_path[i+1]
                    assert self.curr_p2l_map[phy_qubit_u] == local_gate.control_qubit, f"phy_qubit_u: {phy_qubit_u}, local_gate.control_qubit: {local_gate.control_qubit}"
                    swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                    swap_gate_act = SwapGate(local_gate.control_qubit, phy_qubit_u, phy_qubit_v)
                    self.gate_actions[local_gate.id].append(swap_gate_act.id)
                    self.block_depend_graph.nodes[move_block.id]["actions"].append(swap_gate_act.id)
                    self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.control_qubit] = swap_gate_act.id
                    self.curr_l2p_map[local_gate.control_qubit] = phy_qubit_v
                    self.curr_p2l_map[phy_qubit_v] = local_gate.control_qubit
                    if swapped_qubit != -1:
                        self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                        self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                        self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                    else:
                        self.action_depend_graph.add_edge(self.recent_physical_qubit_action_dict[phy_qubit_v], swap_gate_act.id)
                        self.recent_physical_qubit_action_dict[phy_qubit_u] = swap_gate_act.id
                    self.curr_p2l_map[phy_qubit_u] = swapped_qubit
                else:
                    assert self.curr_p2l_map[swap_path[i]] == local_gate.control_qubit, f"phy_qubit_u: {swap_path[i]}, local_gate.control_qubit: {local_gate.control_qubit}"
                    assert self.curr_p2l_map[swap_path[i+1]] == local_gate.target_qubit, f"phy_qubit_v: {swap_path[i+1]}, local_gate.target_qubit: {local_gate.target_qubit}"
                    direct_gate_act = DirectGate(local_gate, swap_path[i], swap_path[i+1])
                    self.gate_actions[local_gate.id].append(direct_gate_act.id)
                    self.block_depend_graph.nodes[move_block.id]["actions"].append(direct_gate_act.id)
                    self.action_depend_graph.add_node(direct_gate_act.id, action=direct_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], direct_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.control_qubit] = direct_gate_act.id
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.target_qubit], direct_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.target_qubit] = direct_gate_act.id
        
        fig = self.draw_qubit_mapping_block(involved_qubits, title=f"Transform MoveBlock {move_block.id}")
        self.block_history.append(fig)

    def find_node_for_immigation(self, chiplet: ComputeChiplet, native_qubits: list[int]) -> tuple[int, int]:
        """
        Find the node (physical qubit) on target chiplet for immigation, which has less control gate with qubits in the same chiplet.
        
        Args:
            chiplet (ComputeChiplet): The target chiplet to which the qubit will be moved.
            immigate_qubit (int): The qubit to be moved in.
            native_qubits (list[int]): The list of qubits that are already on the target chiplet.

        Returns:
            tuple[int, int]: The physical qubit to accept the logic qubit.
        """
        candidate_empty_nodes: list[tuple[int, int]] = []
        candidate_occupied_freq: dict[tuple[int, int], int] = dict()
        for i in range(chiplet.left, chiplet.right + 1):
            for j in range(chiplet.top, chiplet.bottom + 1):
                if self.package_graph.nodes[(i, j)]["qubit"].role == QubitRole.Register:
                    cand_qubit = self.curr_p2l_map[(i, j)]
                    if cand_qubit == -1:
                        candidate_empty_nodes.append((i, j))
                    elif cand_qubit not in native_qubits:
                        candidate_occupied_freq[(i, j)] = self.local_gate_freq[cand_qubit]
        if len(candidate_empty_nodes) > 0:
            return candidate_empty_nodes[0]
        target_node = min(candidate_occupied_freq, key=candidate_occupied_freq.get)
        self.local_gate_freq[target_node] = 0
        return target_node

    def find_nodes_for_immigation(self, chiplet: ComputeChiplet, native_qubits: list[int], target_node_num: int) -> list[tuple[int, int]]:
        """
        Find the nodes (physical qubits) on target chiplet for immigation, which has less control gate with qubits in the same chiplet.
        
        Args:
            chiplet (ComputeChiplet): The target chiplet to which the qubits will be moved.
            native_qubits (list[int]): The list of qubits that are already on the target chiplet.
            target_node_num (int): The number of target nodes to be found.

        Returns:
            list[tuple[int, int]]: The list of physical qubits to accept the logic qubits.
        """
        candidate_empty_nodes: list[tuple[int, int]] = []
        candidate_occupied_nodes_freq: dict[tuple[int, int], int] = dict()
        for i in range(chiplet.left, chiplet.right + 1):
            for j in range(chiplet.top, chiplet.bottom + 1):
                if self.package_graph.nodes[(i, j)]["qubit"].role == QubitRole.Register:
                    cand_qubit = self.curr_p2l_map[(i, j)]
                    if cand_qubit == -1:
                        candidate_empty_nodes.append((i, j))
                    elif cand_qubit not in native_qubits:
                        candidate_occupied_nodes_freq[(i, j)] = self.local_gate_freq[cand_qubit]
        target_nodes = candidate_empty_nodes + sorted(candidate_occupied_nodes_freq, key=candidate_occupied_nodes_freq.get)
        target_nodes = target_nodes[:target_node_num]
        self.local_gate_freq.update({target_node: 0 for target_node in target_nodes})
        return target_nodes
    
    # Functions for scheduling actions to circuit
    def schedule_actions_to_circuit(self):
        """
        Schedule the actions to the circuit.
        """
        self.circuit = Circuit(self.program.data_qubit_num, self.init_l2p_map, self.init_p2l_map, self.action_depend_graph, self.package.op_config)

        for action_id in self.action_depend_graph.nodes:
            self.action_depend_graph.nodes[action_id]["pred"] = set(self.action_depend_graph.predecessors(action_id))
        no_depend_action_ids: set[int] = set()
        for action_id in self.action_depend_graph.nodes:
            if len(self.action_depend_graph.nodes[action_id]["pred"]) == 0:
                no_depend_action_ids.add(action_id)
        self.update_no_depend_action_ids(no_depend_action_ids, -1)
        
        board_flag = True
        while len(no_depend_action_ids) > 0:
            new_stage = self.circuit.extend_stage(board_flag)
            if board_flag:
                assert isinstance(new_stage, BoardingStage)
                # schedule Alight, Interchange, and Board actions
                while True:
                    added_action_ids = set()
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, Alight) and new_stage.add_alight_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, Interchange) and new_stage.add_interchange_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, Board) and new_stage.add_board_action(action):
                            added_action_ids.add(action_id)
                    if not added_action_ids:
                        break
                    else:
                        self.update_no_depend_action_ids(no_depend_action_ids, added_action_ids)
                        fig = self.draw_qubit_mapping_action([self.action_depend_graph.nodes[action_id]["action"] for action_id in added_action_ids], f"Boarding Stage {new_stage.id}")
                        self.action_history.append(fig)
                # schedule GhzPreparation actions
                ghz_visited_action_ids = set()
                while True:
                    added_action_ids = set()
                    for action_id in no_depend_action_ids - ghz_visited_action_ids:
                        ghz_visited_action_ids.add(action_id)
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, GhzPreparation) and new_stage.add_ghz_action(action):
                            added_action_ids.add(action_id)
                    if not added_action_ids:
                        break
                    else:
                        self.update_no_depend_action_ids(no_depend_action_ids, added_action_ids)
                        fig = self.draw_qubit_mapping_action([self.action_depend_graph.nodes[action_id]["action"] for action_id in added_action_ids], f"Boarding Stage {new_stage.id}")
                        self.action_history.append(fig)
            else:
                assert isinstance(new_stage, RidingStage)
                # schedule Transport actions
                while True:
                    added_action_ids = set()
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, Transport) and new_stage.add_transport_action(action):
                            added_action_ids.add(action_id)
                    if not added_action_ids:
                        break
                    else:
                        self.update_no_depend_action_ids(no_depend_action_ids, added_action_ids)
                        fig = self.draw_qubit_mapping_action([self.action_depend_graph.nodes[action_id]["action"] for action_id in added_action_ids], f"Riding Stage {new_stage.id}")
                        self.action_history.append(fig)
                # schedule SwapGate, DirectGate, Teleport, Exit, and Enter actions
                while True:
                    added_action_ids = set()
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, SwapGate) and new_stage.add_swap_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, DirectGate) and new_stage.add_direct_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, Teleport) and new_stage.add_teleport_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, Exit) and new_stage.add_exit_action(action):
                            added_action_ids.add(action_id)
                        elif isinstance(action, Enter) and new_stage.add_enter_action(action):
                            added_action_ids.add(action_id)
                    if not added_action_ids:
                        break
                    else:
                        self.update_no_depend_action_ids(no_depend_action_ids, added_action_ids)
                        fig = self.draw_qubit_mapping_action([self.action_depend_graph.nodes[action_id]["action"] for action_id in added_action_ids], f"Riding Stage {new_stage.id}")
                        self.action_history.append(fig)
            board_flag = not board_flag

    def update_no_depend_action_ids(self, no_depend_action_ids: set[int], removed_action_id: int | set[int]) -> set[int]:
        """
        Update the no_depend_action_ids set by checking removed action's successors.

        Args:
            no_depend_action_ids (set[int]): The original set of action ids with no dependencies.
            removed_action_id (int | set[int]): The id of the removed action or a set of action ids.

        Returns:
            set[int]: The updated set of action ids with no dependencies.
        """
        if isinstance(removed_action_id, int):
            removed_action_id = {removed_action_id}
        # if debug_action:
        #     logging.info(f"Scheduled actions: {[self.action_depend_graph.nodes[action_id]['action'] for action_id in removed_action_id]}.")
        for action_id in removed_action_id:
            no_depend_action_ids.remove(action_id)
            for succ_action in self.action_depend_graph.successors(action_id):
                self.action_depend_graph.nodes[succ_action]["pred"].remove(action_id)
                if len(self.action_depend_graph.nodes[succ_action]["pred"]) == 0:
                    no_depend_action_ids.add(succ_action)
        if config.debug_action:
            logging.info(f"Updated no depend actions: {no_depend_action_ids}.")
        return no_depend_action_ids

    def estimate_gate_fides(self) -> dict[int, float]:
        """
        Estimate the fidelity for each gate in the circuit.

        Returns:
            dict[int, float]: A dictionary mapping gate ids to their estimated fidelities.
        """
        if not hasattr(self, "gate_fides"):
            self.gate_fides: dict[int, float] = dict()
            for gate_id, action_ids in self.gate_actions.items():
                self.gate_fides[gate_id] = 1.0
                for action_id in action_ids:
                    action = self.action_depend_graph.nodes[action_id]["action"]
                    if isinstance(action, GhzPreparation):
                        ghz_edge = action.ghz_balise_graph[action.tele_gate_ids.index(gate_id) + 1]
                        self.gate_fides[gate_id] *= self.package.get_coupler_fide(*ghz_edge)
                    elif isinstance(action, (Board, Alight, Interchange)):
                        self.gate_fides[gate_id] *= self.package.fab_model.inter_fide ** 3
                    elif isinstance(action, Transport):
                        qst_dist = abs(action.source_physical_qubit[0] - action.target_physical_qubit[0]) + abs(action.source_physical_qubit[1] - action.target_physical_qubit[1])
                        self.gate_fides[gate_id] *= self.package.fab_model.enhanced_fide ** qst_dist
                    elif isinstance(action, Teleport):
                        phy_qubit_u = action.platform
                        phy_qubit_v = action.register
                        self.gate_fides[gate_id] *= self.package.get_coupler_fide(phy_qubit_u, phy_qubit_v)
                    elif isinstance(action, (Enter, Exit, SwapGate, DirectGate)):
                        if isinstance(action, (Enter, Exit)):
                            phy_qubit_u = action.platform
                            phy_qubit_v = action.register
                        elif isinstance(action, SwapGate):
                            phy_qubit_u = action.qubit_u
                            phy_qubit_v = action.qubit_v
                        elif isinstance(action, DirectGate):
                            phy_qubit_u = action.phy_ctrl_qubit
                            phy_qubit_v = action.phy_targ_qubit
                        self.gate_fides[gate_id] *= self.package.get_coupler_fide(phy_qubit_u, phy_qubit_v) ** 3
        return self.gate_fides

    def statistic_op_num_fide(self) -> tuple[int, int, int, float]:
        """
        Statistic the number of intra and inter gates, the distance of quantum state transfer, and the total fidelity of the circuit.

        Returns:
            tuple[int, int, int, float]: A tuple containing the number of intra gates, inter gates, quantum state transfer distance, and total fidelity.
        """
        # TODO: The paired actions may be statisticed multiple times, which may lead to an overestimation of the fidelity.
        intra_gate_num = 0
        inter_gate_num = 0
        qst_distance = 0
        total_fidelity = 1.0
        for action_id in self.action_depend_graph.nodes:
            action = self.action_depend_graph.nodes[action_id]["action"]
            if isinstance(action, GhzPreparation):
                for balise, parent in action.ghz_balise_graph:
                    if parent:
                        if self.package_graph.edges[balise, parent]["type"] == EdgeType.Intra:
                            intra_gate_num += 1
                        else:
                            inter_gate_num += 1
                        total_fidelity *= self.package.get_coupler_fide(balise, parent)
            elif isinstance(action, (Board, Alight, Interchange)):
                inter_gate_num += 3
                total_fidelity *= self.package.fab_model.inter_fide ** 3
            elif isinstance(action, Transport):
                qst_distance += abs(action.source_physical_qubit[0] - action.target_physical_qubit[0]) + abs(action.source_physical_qubit[1] - action.target_physical_qubit[1])
                total_fidelity *= self.package.fab_model.enhanced_fide ** (abs(action.source_physical_qubit[0] - action.target_physical_qubit[0]) + abs(action.source_physical_qubit[1] - action.target_physical_qubit[1]))
            elif isinstance(action, Teleport):
                phy_qubit_u = action.platform
                phy_qubit_v = action.register
                if self.package_graph.edges[phy_qubit_u, phy_qubit_v]["type"] == EdgeType.Intra:
                    intra_gate_num += 1
                else:
                    inter_gate_num += 1
                total_fidelity *= self.package.get_coupler_fide(phy_qubit_u, phy_qubit_v)
            elif isinstance(action, (Enter, Exit, SwapGate, DirectGate)):
                if isinstance(action, (Enter, Exit)):
                    phy_qubit_u = action.platform
                    phy_qubit_v = action.register
                elif isinstance(action, SwapGate):
                    phy_qubit_u = action.qubit_u
                    phy_qubit_v = action.qubit_v
                elif isinstance(action, DirectGate):
                    phy_qubit_u = action.phy_ctrl_qubit
                    phy_qubit_v = action.phy_targ_qubit
                if self.package_graph.edges[phy_qubit_u, phy_qubit_v]["type"] == EdgeType.Intra:
                    intra_gate_num += 3
                else:
                    inter_gate_num += 3
                total_fidelity *= self.package.get_coupler_fide(phy_qubit_u, phy_qubit_v) ** 3
        return intra_gate_num, inter_gate_num, qst_distance, total_fidelity

    def estimate_qunoc_circuit_fide(self) -> float:
        """
        Estimate the fidelity of the circuit.

        Return:
            float, the estimated fidelity of the circuit.
        """
        _, _, _, total_fidelity = self.statistic_op_num_fide()
        return total_fidelity

    def estimate_average_gate_fide(self) -> float:
        """
        Estimate the average fidelity of each quantum gate in the circuit.

        Returns:
            float: The estimated average fidelity of quantum gate operations.
        """
        _, _, _, total_fidelity = self.statistic_op_num_fide()
        return total_fidelity ** (1 / self.program.gate_depend_graph.number_of_nodes())
    
    def estimate_depth(self) -> int:
        """
        Estimate the depth of the circuit.

        Return:
            int, the estimated depth of the circuit.
        """
        return self.circuit.estimate_depth()

    def draw_qubit_mapping(self, 
        title: str = "",
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
    ) -> plt.Figure:
        """
        Draw the logic qubit mapped to the package graph.
        """
        fig, ax = self.package.draw_qubits()
        for logic_qubit in range(self.program.data_qubit_num):
            phy_qubit = self.curr_l2p_map[logic_qubit]
            assert phy_qubit in self.package_graph.nodes, f"Logic qubit {logic_qubit} is not mapped to a physical qubit."
            ax.plot(phy_qubit[0], phy_qubit[1], "ko", markersize=4*markersize)
            ax.text(phy_qubit[0], phy_qubit[1], str(logic_qubit), fontsize=10, color="white", ha="center", va="center")
        ax.set_title(title, fontsize=14)
        ax.axis("on")
        ax.xaxis.set_ticks_position('top')
        return fig

    def draw_qubit_mapping_block(self,
        involved_qubits: list[int],
        ancilla_qubits: list[tuple[tuple[int, int], int]] = [],
        title: str = "",
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
    ):
        """
        Draw the logic qubit mapping to the package graph with highlighted involved qubits.

        Args:
            involved_qubits (list[int]): The list of involved qubits to be highlighted.
            ancilla_qubits (list[tuple[tuple[int, int], int]]): The list of ancilla qubits, each represented as a tuple of (node, qubit).
        """
        fig, ax = self.package.draw_qubits()
        for logic_qubit in range(self.program.data_qubit_num):
            phy_qubit = self.curr_l2p_map[logic_qubit]
            assert phy_qubit in self.package_graph.nodes, f"Logic qubit {logic_qubit} is not mapped to a physical qubit."
            if logic_qubit in involved_qubits:
                ax.plot(phy_qubit[0], phy_qubit[1], "ro", markersize=4*markersize)
                ax.text(phy_qubit[0], phy_qubit[1], str(logic_qubit), fontsize=10, color="white", ha="center", va="center")
            else:
                ax.plot(phy_qubit[0], phy_qubit[1], "ko", markersize=4*markersize)
                ax.text(phy_qubit[0], phy_qubit[1], str(logic_qubit), fontsize=10, color="white", ha="center", va="center")
        for ancilla_node, ancilla_qubit in ancilla_qubits:
            assert ancilla_node in self.package_graph.nodes, f"Ancilla node {ancilla_node} is not in the package graph."
            ax.plot(ancilla_node[0], ancilla_node[1], "bo", markersize=4*markersize)
            ax.text(ancilla_node[0], ancilla_node[1], str(ancilla_qubit), fontsize=10, color="white", ha="center", va="center")
        ax.set_title(title, fontsize=14)
        ax.axis("on")
        ax.xaxis.set_ticks_position('top')
        return fig

    def draw_qubit_mapping_action(self,
        involved_actions: list[CircuitAction], 
        title: str = "", 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2,
    ):
        """
        Draw the qubit mapping action.

        Args:
            involved_actions (list[CircuitAction]): The list of involved actions to be highlighted.
        """
        fig, ax = self.package.draw_qubits()

        involved_qubits = set()
        for action in involved_actions:
            if isinstance(action, (Board, Alight, Interchange, Transport, Enter, Exit, SwapGate)):
                logic_qubit = action.logic_qubit
                if logic_qubit != -1:
                    phy_qubit = self.circuit.stages[-1].curr_l2p_map[logic_qubit]
                    ax.plot(phy_qubit[0], phy_qubit[1], "ro", markersize=4*markersize)
                    ax.text(phy_qubit[0], phy_qubit[1], str(logic_qubit), fontsize=10, color="white", ha="center", va="center")
                    involved_qubits.add(logic_qubit)
            elif isinstance(action, DirectGate):
                logic_qubit_u = action.logic_gate.control_qubit
                logic_qubit_v = action.logic_gate.target_qubit
                phy_qubit_u = self.circuit.stages[-1].curr_l2p_map[logic_qubit_u]
                phy_qubit_v = self.circuit.stages[-1].curr_l2p_map[logic_qubit_v]
                ax.plot([phy_qubit_u[0], phy_qubit_v[0]], [phy_qubit_u[1], phy_qubit_v[1]], "g-", linewidth=4*linewidth_b)
                ax.plot(phy_qubit_u[0], phy_qubit_u[1], "ro", markersize=4*markersize)
                ax.text(phy_qubit_u[0], phy_qubit_u[1], str(logic_qubit_u), fontsize=10, color="white", ha="center", va="center")
                ax.plot(phy_qubit_v[0], phy_qubit_v[1], "ro", markersize=4*markersize)
                ax.text(phy_qubit_v[0], phy_qubit_v[1], str(logic_qubit_v), fontsize=10, color="white", ha="center", va="center")
                involved_qubits.add(logic_qubit_u)
                involved_qubits.add(logic_qubit_v)
            elif isinstance(action, Teleport):
                ancilla_qubit = action.ancilla_qubit
                data_qubit = action.data_qubit
                ancilla_node = action.platform
                data_node = self.circuit.stages[-1].curr_l2p_map[data_qubit]
                ax.plot([ancilla_node[0], data_node[0]], [ancilla_node[1], data_node[1]], "g-", linewidth=4*linewidth_b)
                ax.plot(ancilla_node[0], ancilla_node[1], "bo", markersize=4*markersize)
                ax.text(ancilla_node[0], ancilla_node[1], str(ancilla_qubit), fontsize=10, color="white", ha="center", va="center")
                ax.plot(data_node[0], data_node[1], "ro", markersize=4*markersize)
                ax.text(data_node[0], data_node[1], str(data_qubit), fontsize=10, color="white", ha="center", va="center")
                involved_qubits.add(data_qubit)
        
        for logic_qubit, phy_qubit in self.circuit.stages[-1].curr_l2p_map.items():
            assert phy_qubit in self.package_graph.nodes, f"Logic qubit {logic_qubit} is not mapped to a physical qubit."
            if logic_qubit not in involved_qubits:
                if logic_qubit < self.program.data_qubit_num:
                    ax.plot(phy_qubit[0], phy_qubit[1], "ko", markersize=4*markersize)
                else:
                    ax.plot(phy_qubit[0], phy_qubit[1], "bo", markersize=4*markersize)
                ax.text(phy_qubit[0], phy_qubit[1], str(logic_qubit), fontsize=10, color="white", ha="center", va="center")
        
        ax.set_title(title, fontsize=14)
        ax.axis("on")
        ax.xaxis.set_ticks_position('top')
        return fig

    def save_block_history(self, save_jpg: bool = True, save_gif: bool = True, duration: int = 1000):
        """
        Save the block history.

        Args:
            save_jpg (bool): Whether to save the block history as JPG files.
            save_gif (bool): Whether to save the block history as a GIF file.
            duration (int): The duration of each frame in the GIF file in milliseconds.
        """
        if save_jpg:
            os.makedirs(os.path.join(self.result_dir, "block_history"), exist_ok=True)
            delete_files_with_prefix_and_suffix(os.path.join(self.result_dir, "block_history"), "block_history_", ".jpg")
            for i, fig in enumerate(self.block_history):
                fig.savefig(os.path.join(self.result_dir, "block_history", f"block_history_{i}.jpg"), bbox_inches="tight")
        if save_gif:
            if save_jpg:
                make_save_gif_file(os.path.join(self.result_dir, "block_history"), "block_history_", ".jpg", os.path.join(self.result_dir, "block_history.gif"), duration=duration)
            else:
                make_save_gif_figure(self.block_history, os.path.join(self.result_dir, "block_history.gif"), duration=duration)

    def save_action_history(self, save_jpg: bool = True, save_gif: bool = True, duration: int = 1000):
        """
        Save the action history.

        Args:
            save_jpg (bool): Whether to save the action history as JPG files.
            save_gif (bool): Whether to save the action history as a GIF file.
            duration (int): The duration of each frame in the GIF file in milliseconds.
        """
        if save_jpg:
            os.makedirs(os.path.join(self.result_dir, "action_history"), exist_ok=True)
            delete_files_with_prefix_and_suffix(os.path.join(self.result_dir, "action_history"), "action_history_", ".jpg")
            for i, fig in enumerate(self.action_history):
                fig.savefig(os.path.join(self.result_dir, "action_history", f"action_history_{i}.jpg"), bbox_inches="tight")
        if save_gif:
            if save_jpg:
                make_save_gif_file(os.path.join(self.result_dir, "action_history"), "action_history_", ".jpg", os.path.join(self.result_dir, "action_history.gif"), duration=duration)
            else:
                make_save_gif_figure(self.action_history, os.path.join(self.result_dir, "action_history.gif"), duration=duration)

    def get_block_depth(self) -> int:
        """
        Get the depth of the block dependency graph.

        Returns:
            int: The depth of the block dependency graph.
        """
        if not hasattr(self, "block_depth"):
            self.block_depth = nx.algorithms.dag_longest_path_length(self.block_depend_graph)
        return self.block_depth
    
    def get_action_depth(self) -> int:
        """
        Get the depth of the action dependency graph.

        Returns:
            int: The depth of the action dependency graph.
        """
        if not hasattr(self, "action_depth"):
            self.action_depth = nx.algorithms.dag_longest_path_length(self.action_depend_graph)
        return self.action_depth

    def draw_block_depend(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2,
    ) -> plt.Figure:
        """
        Draw the block dependency graph of the program.

        Return:
            plt.Figure: The figure containing the block dependency graph.
        """
        pos = nx.spring_layout(self.block_depend_graph)
        figsize = (self.get_block_depth(), self.get_block_depth())
        fig = plt.figure(figsize=figsize)
        nx.draw(
            self.block_depend_graph, 
            pos, 
            with_labels=False, 
            arrows=True, 
            node_size=markersize, 
            width=linewidth_t,
            node_color='black', 
            edge_color='black', 
        )
        labels = {node: str(node) for node in self.block_depend_graph.nodes()}
        label_offset = (-0.04, 0.03)
        label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(
            self.block_depend_graph,
            label_pos,
            labels=labels,
            font_size=10,
            font_color='black'
        )
        plt.close()
        return fig

    def draw_action_depend(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2,
    ) -> plt.Figure:
        """
        Draw the action dependency graph of the program.

        Returns:
            plt.Figure: The figure containing the action dependency graph.
        """
        pos = nx.spring_layout(self.action_depend_graph)
        figsize = (self.get_action_depth(), self.get_action_depth())
        fig = plt.figure(figsize=figsize)
        nx.draw(
            self.action_depend_graph, 
            pos, 
            with_labels=False, 
            arrows=True, 
            node_size=markersize, 
            width=linewidth_t,
            node_color='black', 
            edge_color='black', 
        )
        labels = {node: str(node) for node in self.action_depend_graph.nodes()}
        label_offset = (-0.04, 0.03)
        label_pos = {k: (v[0] + label_offset[0], v[1] + label_offset[1]) for k, v in pos.items()}
        nx.draw_networkx_labels(
            self.action_depend_graph,
            label_pos,
            labels=labels,
            font_size=10,
            font_color='black'
        )
        plt.close()
        return fig
