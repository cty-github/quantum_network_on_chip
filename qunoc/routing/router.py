from copy import deepcopy

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

class TeleBlock(ComputeBlock):
    """
    Represents a teleportation block in which gates are executed by teleportation.

    Attributes:
        control_qubit (int): The control qubit of the teleportation block.
        target_qubits (list[int]): The target qubits of the teleportation block.
    """

    def __init__(self, control_qubit: int, target_qubits: list[int]):
        """
        Initialize a TeleBlock.

        Args:
            control_qubit (int): The control qubit of the teleportation block.
            target_qubits (list[int]): The target qubits of the teleportation block.
        """
        super().__init__()
        self.control_qubit = control_qubit
        self.target_qubits = target_qubits

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

        init_l2p_map (dict[int, tuple[int, int]]): The initial mapping of logic qubits to physical qubits.
        init_p2l_map (dict[tuple[int, int], int]): The initial mapping of physical qubits to logic qubits.

        block_depend_graph (nx.DiGraph): The dependency graph of compute blocks.
        action_depend_graph (nx.DiGraph): The dependency graph of actions.
        curr_l2p_map (dict[int, tuple[int, int]]): The current mapping of logic qubits to physical qubits.
        curr_p2l_map (dict[tuple[int, int], int]): The current mapping of physical qubits to logic qubits.
        local_gate_freq (dict[int, int]): The dictionary recording frequency of local gates for each logic qubit.
        recent_logic_qubit_action_dict (dict[int, int]): The dictionary recording the most recent action for each logic qubit.

        circuit (Circuit): The physical quantum circuit after routing.
    """

    def __init__(self, package: Package, program: Program):
        """
        Initialize a Router.

        Args:
            package (Package): The package to be routed on.
            program (Program): The quantum program to be routed.
        """
        self.package = package
        self.program = program

        self.package_graph = package.package_graph
        self.metro_graph: nx.Graph = package.package_graph.subgraph(self.package.transfer_nodes)
        self.gate_depend_graph = program.gate_depend_graph

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
        self.init_p2l_map: dict[tuple[int, int], int] = defaultdict(lambda: -1)
        logic_data_qubit_num = self.program.data_qubit_num
        physical_data_qubit_set = set(self.package.get_registers())
        if logic_data_qubit_num > len(physical_data_qubit_set):
            raise ValueError("No enough physical qubits to map logic qubits.")
        for i in range(logic_data_qubit_num):
            self.init_l2p_map[i] = physical_data_qubit_set.pop()
            self.init_p2l_map[self.init_l2p_map[i]] = i

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
        self.action_depend_graph = nx.DiGraph()
        self.action_depend_graph.add_node(-1, action=None)
        self.curr_l2p_map = deepcopy(self.init_l2p_map)
        self.curr_p2l_map = deepcopy(self.init_p2l_map)
        self.local_gate_freq = {i: 0 for i in range(self.program.data_qubit_num)}
        self.recent_logic_qubit_action_dict = {i: -1 for i in range(self.program.data_qubit_num)}

        for gate_id in self.gate_depend_graph.nodes:
            self.gate_depend_graph.nodes[gate_id]["pred"] = set(self.gate_depend_graph.predecessors(gate_id))
        no_depend_gate_ids: set[int] = set()
        for gate_id in self.gate_depend_graph.nodes:
            if len(self.gate_depend_graph.nodes[gate_id]["pred"]) == 0:
                no_depend_gate_ids.add(gate_id)
        
        while len(no_depend_gate_ids) > 0:
            gate_id = no_depend_gate_ids.pop()
            no_depend_gate_ids = self.update_no_depend_gate_ids(no_depend_gate_ids, gate_id)
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
                        no_depend_gate_ids = self.update_no_depend_gate_ids(no_depend_gate_ids, add_gate.id)
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
                            no_depend_gate_ids = self.update_no_depend_gate_ids(no_depend_gate_ids, add_gate.id)
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
                    tele_block = self.construct_tele_block(tele_gates)
                    self.transform_tele_block(tele_block)
                else:
                    move_block = self.construct_move_block(move_gates)
                    self.transform_move_block(move_block)

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
            self.gate_depend_graph.nodes[succ_gate]["pred"].discard(removed_gate_id)
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
        self.gate_depend_graph.nodes[local_gate.id]["block"] = local_block.id
        self.block_depend_graph.add_node(local_block.id, block=local_block)
        for pred_gate in self.gate_depend_graph.predecessors(local_gate.id):
            self.block_depend_graph.add_edge(self.gate_depend_graph.nodes[pred_gate]["block"], local_block.id)
        return local_block

    def transform_local_block(self, local_block: LocalBlock):
        """
        Transform local block into circuit actions.
        1) route qubits together -> SwapGate;
        2) execute logic gate -> DirectGate.

        Action dependency: SwapGate -> DirectGate.
        """
        local_gate = local_block.ctrl_gate
        compute_module = local_block.module
        swap_path = compute_module.get_swap_path(self.curr_l2p_map[local_gate.control_qubit], self.curr_l2p_map[local_gate.target_qubit])
        for i in range(len(swap_path)-1):
            if i < len(swap_path)-2:
                phy_qubit_u, phy_qubit_v = swap_path[i], swap_path[i+1]
                assert self.curr_p2l_map[phy_qubit_u] == local_gate.control_qubit, f"phy_qubit_u: {phy_qubit_u}, local_gate.control_qubit: {local_gate.control_qubit}"
                swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                swap_gate_act = SwapGate(phy_qubit_u, phy_qubit_v)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.control_qubit] = swap_gate_act.id
                self.curr_l2p_map[local_gate.control_qubit] = phy_qubit_v
                self.curr_p2l_map[phy_qubit_v] = local_gate.control_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                self.curr_p2l_map[phy_qubit_u] = swapped_qubit
            else:
                assert self.curr_p2l_map[swap_path[i]] == local_gate.control_qubit, f"phy_qubit_u: {swap_path[i]}, local_gate.control_qubit: {local_gate.control_qubit}"
                assert self.curr_p2l_map[swap_path[i+1]] == local_gate.target_qubit, f"phy_qubit_v: {swap_path[i+1]}, local_gate.target_qubit: {local_gate.target_qubit}"
                direct_gate_act = DirectGate(local_gate, compute_module)
                self.action_depend_graph.add_node(direct_gate_act.id, action=direct_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], direct_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.control_qubit] = direct_gate_act.id
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.target_qubit], direct_gate_act.id)
                self.recent_logic_qubit_action_dict[local_gate.target_qubit] = direct_gate_act.id
    
    def construct_tele_block(self, tele_gates: list[Control]) -> TeleBlock:
        """
        Construct a teleportation block for the given tele gates and add it to the block dependency graph.

        Args:
            tele_gates (list[Control]): The list of control gates in the teleportation block.

        Returns:
            TeleBlock: The constructed teleportation block.
        """
        ctrl_qubit = tele_gates[0].control_qubit
        target_qubits = [tele_gate.target_qubit for tele_gate in tele_gates]
        tele_block = TeleBlock(ctrl_qubit, target_qubits)
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

    def get_balise_nearest_metro_subgraph(self, balise_id: tuple[int, int], node_num: int) -> nx.Graph:
        """
        Get the nearest metro subgraph for a balise.

        Args:
            balise_id (tuple[int, int]): The coordinate of the balise.
            node_num (int): The number of nodes in the subgraph.

        Return:
            nx.Graph: The subgraph of the metro topology.
        """
        visited = set()
        queue = [balise_id]
        while queue and len(visited) < node_num:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend([nbr for nbr in self.metro_graph.neighbors(node) if nbr not in visited])
        subgraph = self.metro_graph.subgraph(visited).copy()
        return subgraph

    def locate_ghz_state(self, involved_qubits: list[int], platforms: list[tuple[int, int]]) -> GhzPreparation:
        """
        Allocate GHZ resource for a list of logic qubits for teleportation.
        Ensure the total transfer distance is minimized.

        Args:
            involved_qubits (list[int]): The list of logic qubits to be teleported.
            platforms (list[tuple[int, int]]): The list of platforms to which the qubits will be teleported.

        Return:
            GhzPreparation: The action to allocate GHZ resource.
        """
        ancilla_qubit_num = len(involved_qubits)
        stations = [self.package.platform_stations_dict[platform][0] for platform in platforms]
        best_interchange_depth = None
        best_ghz_balises = None
        for balise_id in self.metro_graph.nodes:
            ghz_balise_graph = self.get_balise_nearest_metro_subgraph(balise_id, ancilla_qubit_num)
            ghz_balises = list(ghz_balise_graph.nodes)
            interchange_depth = max([self.package.get_inter_chiplet_distance(ghz_balises[i], stations[i]) for i in range(ancilla_qubit_num)])
            if best_interchange_depth == None or interchange_depth < best_interchange_depth:
                best_interchange_depth = interchange_depth
                best_ghz_balises = ghz_balises
        source_physical_qubits = []
        target_physical_qubits = []
        for i in range(ancilla_qubit_num):
            source_physical_qubits.append(best_ghz_balises[i])
            target_physical_qubits.append(stations[i])
        return GhzPreparation(ancilla_qubit_num, involved_qubits, source_physical_qubits, target_physical_qubits)

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
        involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits
        platforms = [self.find_nearest_platform(self.curr_l2p_map[qubit]) for qubit in involved_qubits]
        ghz_action = self.locate_ghz_state(involved_qubits, platforms)
        self.action_depend_graph.add_node(ghz_action.id, action=ghz_action)
        for i in range(len(involved_qubits)):
            logic_qubit = involved_qubits[i]
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[logic_qubit], ghz_action.id)
            self.recent_logic_qubit_action_dict[logic_qubit] = ghz_action.id
            route_source_phy_qubit = self.curr_l2p_map[logic_qubit]
            route_target_phy_qubit = platforms[i]
            module = self.package.get_node_area(route_source_phy_qubit)
            assert isinstance(module, ComputeModule)
            swap_path = module.get_swap_path(route_source_phy_qubit, route_target_phy_qubit)
            for j in range(len(swap_path)-2):
                phy_qubit_u, phy_qubit_v = swap_path[j], swap_path[j+1]
                assert self.curr_p2l_map[phy_qubit_u] == logic_qubit, f"phy_qubit_u: {phy_qubit_u}, logic_qubit: {logic_qubit}"
                swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                swap_gate_act = SwapGate(phy_qubit_u, phy_qubit_v)
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[logic_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[logic_qubit] = swap_gate_act.id
                self.curr_l2p_map[logic_qubit] = phy_qubit_v
                self.curr_p2l_map[phy_qubit_v] = logic_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                self.curr_p2l_map[phy_qubit_u] = swapped_qubit
            tele_source_phy_qubit = ghz_action.source_physical_qubits[i]
            tele_target_phy_qubit = ghz_action.target_physical_qubits[i]
            transfer_on_metro_acts = self.route_on_transfer_chiplets(logic_qubit, tele_source_phy_qubit, tele_target_phy_qubit)
            for j in range(len(transfer_on_metro_acts)):
                self.action_depend_graph.add_node(transfer_on_metro_acts[j].id, action=transfer_on_metro_acts[j])
                if j == 0:
                    self.action_depend_graph.add_edge(ghz_action.id, transfer_on_metro_acts[j].id)
                else:
                    self.action_depend_graph.add_edge(transfer_on_metro_acts[j-1].id, transfer_on_metro_acts[j].id)
            alight_act = Alight(logic_qubit, tele_target_phy_qubit, platforms[i])
            self.action_depend_graph.add_node(alight_act.id, action=alight_act)
            self.action_depend_graph.add_edge(transfer_on_metro_acts[-1].id, alight_act.id)
            teleport_act = Teleport(logic_qubit, platforms[i], logic_qubit==tele_block.control_qubit)
            self.action_depend_graph.add_node(teleport_act.id, action=teleport_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[logic_qubit], teleport_act.id)
            self.action_depend_graph.add_edge(alight_act.id, teleport_act.id)
            if i == 0:
                tele_ctrl_act_id = teleport_act.id
            else:
                self.action_depend_graph.add_edge(tele_ctrl_act_id, teleport_act.id)
            self.recent_logic_qubit_action_dict[logic_qubit] = teleport_act.id
        
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
        involved_qubits = move_block.get_involved_qubits()
        immigate_qubits: list[int] = []
        native_qubits: list[int] = []
        for qubit in involved_qubits:
            if self.package_graph.nodes[self.curr_l2p_map[qubit]]["qubit"].chiplet != move_block.chiplet:
                immigate_qubits.append(qubit)
            else:
                native_qubits.append(qubit)
        arrive_nodes = self.find_nodes_for_immigation(move_block.chiplet, immigate_qubits, native_qubits)
        for i in range(len(immigate_qubits)):
            immigate_qubit = immigate_qubits[i]
            depart_node = self.curr_l2p_map[immigate_qubit]
            depart_module = self.package.get_node_area(depart_node)
            assert isinstance(depart_module, ComputeModule)
            depart_platform = self.find_nearest_platform(depart_node)
            depart_swap_path = depart_module.get_swap_path(depart_node, depart_platform)
            depart_station = self.package.platform_stations_dict[depart_platform][0]

            arrive_node = arrive_nodes[i]
            expelled_qubit = self.curr_p2l_map[arrive_node]
            arrive_module = self.package.get_node_area(arrive_node)
            assert isinstance(arrive_module, ComputeModule)
            arrive_platform = self.find_nearest_platform(arrive_node)
            arrive_swap_path = arrive_module.get_swap_path(arrive_node, arrive_platform)
            arrive_station = self.package.platform_stations_dict[arrive_platform][0]

            # Swap the immigate_qubit from depart_node to depart_platform
            for j in range(len(depart_swap_path)-1):
                swapped_qubit = self.curr_p2l_map[depart_swap_path[j+1]]
                if j == len(depart_swap_path)-2:
                    swap_gate_act = Enter(immigate_qubit, depart_platform, depart_swap_path[j])
                else:
                    swap_gate_act = SwapGate(depart_swap_path[j], depart_swap_path[j+1])
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[immigate_qubit] = swap_gate_act.id
                self.curr_l2p_map[immigate_qubit] = depart_swap_path[j+1]
                self.curr_p2l_map[depart_swap_path[j+1]] = immigate_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = depart_swap_path[j]
                self.curr_p2l_map[depart_swap_path[j]] = swapped_qubit

            # Transfer the immigate_qubit from depart_platform to arrive_platform
            immigate_board_act = Board(immigate_qubit, depart_station, depart_platform)
            self.action_depend_graph.add_node(immigate_board_act.id, action=immigate_board_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_board_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_board_act.id
            immigate_transfer_acts = self.route_on_transfer_chiplets(immigate_qubit, depart_station, arrive_station)
            for i in range(len(immigate_transfer_acts)):
                self.action_depend_graph.add_node(immigate_transfer_acts[i].id, action=immigate_transfer_acts[i])
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_transfer_acts[i].id)
                self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_transfer_acts[i].id
            immigate_alight_act = Alight(immigate_qubit, arrive_station, arrive_platform)
            self.action_depend_graph.add_node(immigate_alight_act.id, action=immigate_alight_act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], immigate_alight_act.id)
            self.recent_logic_qubit_action_dict[immigate_qubit] = immigate_alight_act.id

            if expelled_qubit != -1:
                # Swap the expelled_qubit from arrive_node to arrive_platform
                for j in range(len(arrive_swap_path)-1):
                    swapped_qubit = self.curr_p2l_map[arrive_swap_path[j+1]]
                    if j == len(arrive_swap_path)-2:
                        swap_gate_act = Enter(expelled_qubit, arrive_platform, arrive_swap_path[j])
                    else:
                        swap_gate_act = SwapGate(arrive_swap_path[j], arrive_swap_path[j+1])
                    self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[expelled_qubit] = swap_gate_act.id
                    self.curr_l2p_map[expelled_qubit] = arrive_swap_path[j+1]
                    self.curr_p2l_map[arrive_swap_path[j+1]] = expelled_qubit
                    if swapped_qubit != -1:
                        self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                        self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                        self.curr_l2p_map[swapped_qubit] = arrive_swap_path[j]
                    self.curr_p2l_map[arrive_swap_path[j]] = swapped_qubit
                
                # Transfer the expelled_qubit from arrive_platform to depart_platform
                expelled_board_act = Board(expelled_qubit, arrive_station, arrive_platform)
                self.action_depend_graph.add_node(expelled_board_act.id, action=expelled_board_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_board_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_board_act.id
                expelled_transfer_acts = self.route_on_transfer_chiplets(expelled_qubit, arrive_station, depart_station)
                for i in range(len(expelled_transfer_acts)):
                    self.action_depend_graph.add_node(expelled_transfer_acts[i].id, action=expelled_transfer_acts[i])
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_transfer_acts[i].id)
                    self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_transfer_acts[i].id
                expelled_alight_act = Alight(expelled_qubit, depart_station, depart_platform, True)
                self.action_depend_graph.add_node(expelled_alight_act.id, action=expelled_alight_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], expelled_alight_act.id)
                self.recent_logic_qubit_action_dict[expelled_qubit] = expelled_alight_act.id

                self.action_depend_graph.add_edge(expelled_board_act.id, immigate_alight_act.id)
                self.action_depend_graph.add_edge(immigate_board_act.id, expelled_alight_act.id)

            self.curr_l2p_map[immigate_qubit] = arrive_platform
            self.curr_p2l_map[arrive_platform] = immigate_qubit
            if expelled_qubit != -1:
                self.curr_l2p_map[expelled_qubit] = depart_platform
            self.curr_p2l_map[depart_platform] = expelled_qubit

            # Swap the immigate_qubit from arrive_platform to arrive_node
            for j in range(len(arrive_swap_path)-1):
                swapped_qubit = arrive_swap_path[len(arrive_swap_path)-1-j]
                if j == 0:
                    swap_gate_act = Exit(immigate_qubit, arrive_platform, arrive_swap_path[len(arrive_swap_path)-2-j])
                else:
                    swap_gate_act = SwapGate(arrive_swap_path[len(arrive_swap_path)-1-j], arrive_swap_path[len(arrive_swap_path)-2-j])
                self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[immigate_qubit], swap_gate_act.id)
                self.recent_logic_qubit_action_dict[immigate_qubit] = swap_gate_act.id
                self.curr_l2p_map[immigate_qubit] = arrive_swap_path[len(arrive_swap_path)-2-j]
                self.curr_p2l_map[arrive_swap_path[len(arrive_swap_path)-2-j]] = immigate_qubit
                if swapped_qubit != -1:
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                    self.curr_l2p_map[swapped_qubit] = arrive_swap_path[len(arrive_swap_path)-1-j]
                self.curr_p2l_map[arrive_swap_path[len(arrive_swap_path)-1-j]] = swapped_qubit
            
            if expelled_qubit != -1:
                # Swap the expelled_qubit from depart_platform to depart_node
                for j in range(len(depart_swap_path)-1):
                    swapped_qubit = depart_swap_path[len(depart_swap_path)-1-j]
                    if j == 0:
                        swap_gate_act = Exit(expelled_qubit, depart_platform, depart_swap_path[len(depart_swap_path)-2-j])
                    else:
                        swap_gate_act = SwapGate(depart_swap_path[len(depart_swap_path)-1-j], depart_swap_path[len(depart_swap_path)-2-j])
                    self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[expelled_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[expelled_qubit] = swap_gate_act.id
                    self.curr_l2p_map[expelled_qubit] = depart_swap_path[len(depart_swap_path)-2-j]
                    self.curr_p2l_map[depart_swap_path[len(depart_swap_path)-2-j]] = expelled_qubit
                    if swapped_qubit != -1:
                        self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                        self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                        self.curr_l2p_map[swapped_qubit] = depart_swap_path[len(depart_swap_path)-1-j]
                    self.curr_p2l_map[depart_swap_path[len(depart_swap_path)-1-j]] = swapped_qubit

        # Route the local gates in the move block
        compute_module = self.package.get_node_area(move_block.chiplet.get_nodes()[0])
        assert isinstance(compute_module, ComputeModule)
        for local_gate in move_block.ctrl_gates:
            swap_path = compute_module.get_swap_path(self.curr_l2p_map[local_gate.control_qubit], self.curr_l2p_map[local_gate.target_qubit])
            for i in range(len(swap_path)-1):
                if i < len(swap_path)-2:
                    phy_qubit_u, phy_qubit_v = swap_path[i], swap_path[i+1]
                    assert self.curr_p2l_map[phy_qubit_u] == local_gate.control_qubit, f"phy_qubit_u: {phy_qubit_u}, local_gate.control_qubit: {local_gate.control_qubit}"
                    swapped_qubit = self.curr_p2l_map[phy_qubit_v]
                    swap_gate_act = SwapGate(phy_qubit_u, phy_qubit_v)
                    self.action_depend_graph.add_node(swap_gate_act.id, action=swap_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], swap_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.control_qubit] = swap_gate_act.id
                    self.curr_l2p_map[local_gate.control_qubit] = phy_qubit_v
                    self.curr_p2l_map[phy_qubit_v] = local_gate.control_qubit
                    if swapped_qubit != -1:
                        self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[swapped_qubit], swap_gate_act.id)
                        self.recent_logic_qubit_action_dict[swapped_qubit] = swap_gate_act.id
                        self.curr_l2p_map[swapped_qubit] = phy_qubit_u
                    self.curr_p2l_map[phy_qubit_u] = swapped_qubit
                else:
                    assert self.curr_p2l_map[swap_path[i]] == local_gate.control_qubit, f"phy_qubit_u: {swap_path[i]}, local_gate.control_qubit: {local_gate.control_qubit}"
                    assert self.curr_p2l_map[swap_path[i+1]] == local_gate.target_qubit, f"phy_qubit_v: {swap_path[i+1]}, local_gate.target_qubit: {local_gate.target_qubit}"
                    direct_gate_act = DirectGate(local_gate, compute_module)
                    self.action_depend_graph.add_node(direct_gate_act.id, action=direct_gate_act)
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.control_qubit], direct_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.control_qubit] = direct_gate_act.id
                    self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[local_gate.target_qubit], direct_gate_act.id)
                    self.recent_logic_qubit_action_dict[local_gate.target_qubit] = direct_gate_act.id

    def find_nodes_for_immigation(self, chiplet: ComputeChiplet, immigate_qubits: list[int], native_qubits: list[int]) -> list[tuple[int, int]]:
        """
        Find the nodes (physical qubits) on target chiplet for immigation, which has less control gate with qubits in the same chiplet.
        
        Args:
            chiplet (ComputeChiplet): The target chiplet to which the qubits will be moved.
            immigate_qubits (list[int]): The list of qubits to be moved in.
            native_qubits (list[int]): The list of qubits that are already on the target chiplet.

        Returns:
            list[tuple[int, int]]: The list of physical qubits to accept the logic qubits.
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
        target_qubits = candidate_empty_nodes + sorted(candidate_occupied_freq, key=candidate_occupied_freq.get)
        target_qubits = target_qubits[:len(immigate_qubits)]
        self.local_gate_freq = {i: 0 for i in range(self.program.data_qubit_num)}
        return target_qubits
    
    # Functions for scheduling actions to circuit
    def schedule_actions_to_circuit(self):
        """
        Schedule the actions to the circuit.
        """
        self.circuit = Circuit(self.package, self.program.data_qubit_num, self.init_l2p_map, self.init_p2l_map, self.action_depend_graph)

        for action_id in self.action_depend_graph.nodes:
            self.action_depend_graph.nodes[action_id]["pred"] = set(self.action_depend_graph.predecessors(action_id))
        no_depend_action_ids: set[int] = set()
        for action_id in self.action_depend_graph.nodes:
            if len(self.action_depend_graph.nodes[action_id]["pred"]) == 0:
                no_depend_action_ids.add(action_id)
        
        board_flag = True
        while len(no_depend_action_ids) > 0:
            new_stage = self.circuit.extend_stage(board_flag)
            if board_flag:
                assert isinstance(new_stage, BoardingStage)
                # schedule Alight, Interchange, and Board actions
                while True:
                    action_added = False
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, Alight) and new_stage.add_alight_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Interchange) and new_stage.add_interchange_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Board) and new_stage.add_board_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                    if not action_added:
                        break
                # schedule GhzPreparation actions
                ghz_visited_action_ids = set()
                while True:
                    action_added = False
                    for action_id in no_depend_action_ids - ghz_visited_action_ids:
                        ghz_visited_action_ids.add(action_id)
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, GhzPreparation) and new_stage.add_ghz_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                    if not action_added:
                        break
            else:
                assert isinstance(new_stage, RidingStage)
                # schedule Transport actions
                for action_id in no_depend_action_ids:
                    action = self.action_depend_graph.nodes[action_id]["action"]
                    if isinstance(action, Transport) and new_stage.add_transport_action(action):
                        no_depend_action_ids.remove(action_id)
                        no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                # schedule SwapGate, DirectGate, Teleport, Exit, and Enter actions
                while True:
                    action_added = False
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, SwapGate) and new_stage.add_swap_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, DirectGate) and new_stage.add_direct_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Teleport) and new_stage.add_teleport_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Exit) and new_stage.add_exit_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Enter) and new_stage.add_enter_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                    if not action_added:
                        break
            board_flag = not board_flag
    
    def update_no_depend_action_ids(self, no_depend_action_ids: set[int], removed_action_id: int) -> set[int]:
        """
        Update the no_depend_action_ids set by checking removed action's successors.

        Args:
            no_depend_action_ids (set[int]): The original set of action ids with no dependencies.
            removed_action_id (int): The id of the removed action.

        Returns:
            set[int]: The updated set of action ids with no dependencies.
        """
        for succ_action in self.action_depend_graph.successors(removed_action_id):
            self.action_depend_graph.nodes[succ_action]["pred"].discard(removed_action_id)
            if len(self.action_depend_graph.nodes[succ_action]["pred"]) == 0:
                no_depend_action_ids.add(succ_action)
        return no_depend_action_ids

    def estimate_fidelity(self) -> float:
        """
        Estimate the fidelity of the circuit.

        Return:
            float, the estimated fidelity of the circuit.
        """
        return self.circuit.estimate_fidelity()

    def estimate_average_qst_fide(self) -> float:
        """
        Estimate the average fidelity of each quantum state transfer in the circuit.

        Returns:
            float: The estimated average fidelity of quantum state transfer.
        """
        return self.circuit.estimate_average_qst_fide()
    
    def estimate_depth(self) -> int:
        """
        Estimate the depth of the circuit.

        Return:
            int, the estimated depth of the circuit.
        """
        return self.circuit.estimate_depth()

    def draw_program(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
        show: bool = True, 
        save_path: str = None, 
    ):
        """
        Draw the gate dependency graph of the program.
        """
        self.program.draw_depend(
            marginrate=marginrate, 
            markersize=markersize, 
            linewidth_t=linewidth_t, 
            linewidth_b=linewidth_b, 
            show=show, 
            filepath=save_path
        )

    def draw_block(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
        show: bool = True, 
        save_path: str = None, 
    ):
        """
        Draw the block dependency graph of the program.
        """
        pos = nx.spring_layout(self.block_depend_graph)
        figsize = (find_longest_topo_distance(self.block_depend_graph), find_longest_topo_distance(self.block_depend_graph))
        plt.figure(figsize=figsize)
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
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()
