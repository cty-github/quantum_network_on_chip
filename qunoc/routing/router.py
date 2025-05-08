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
    Represents a compute block in which gates are executed locally.

    Attributes:
        ctrl_gates (list[Control]): The control gates in the local block.
        module (ComputeModule): The compute module where the local block is executed.
    """

    def __init__(self, ctrl_gates: list[Control], module: ComputeModule):
        """
        Initialize a LocalBlock.

        Args:
            ctrl_gates (list[Control]): The control gates in the local block.
            module (ComputeModule): The compute module where the local block is executed.
        """
        super().__init__()
        self.ctrl_gates = ctrl_gates
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
        involved_qubits (list[int]): The involved qubits in the move block.
        chiplet (ComputeChiplet): The chiplet where the move block is executed.
        swap_logic_pairs (list[tuple[int, int]]): The logic qubit pairs to be swapped.
    """

    def __init__(self, 
        ctrl_gates: list[Control], 
        involved_qubits: list[int], 
        chiplet: ComputeChiplet, 
        swap_logic_pairs: list[tuple[int, int]], 
    ):
        """
        Initialize a MoveBlock.

        Args:
            ctrl_gates (list[Control]): The control gates in the move block.
            involved_qubits (list[int]): The involved qubits in the move block.
            chiplet (ComputeChiplet): The chiplet where the move block is executed.
            swap_logic_pairs (list[tuple[int, int]]): The logic qubit pairs to be swapped.
        """
        super().__init__()
        self.ctrl_gates = ctrl_gates
        self.involved_qubits = involved_qubits
        self.chiplet = chiplet
        self.swap_logic_pairs = swap_logic_pairs


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
        circuit (Circuit): The physical quantum circuit after routing.

        curr_l2p_map (dict[int, tuple[int, int]]): The current mapping of logic qubits to physical qubits.
        curr_p2l_map (dict[tuple[int, int], int]): The current mapping of physical qubits to logic qubits.
        local_gate_freq (dict[int, int]): The dictionary recording frequency of local gates for each logic qubit.
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

        # self.qubit_topology = package.qubit_topology
        # self.chiplet_topology = package.chiplet_topology
        # self.metro_topology = package.metro_topology
        # self.platform_list = package.get_platform_list()
        # self.station_list = package.get_station_list()
        # self.interchange_list = package.get_interchange_list()

        self.package_graph = package.package_graph
        self.metro_graph: nx.Graph = package.package_graph.subgraph(self.package.transfer_nodes)
        self.gate_depend_graph = program.gate_depend_graph

        self.init_l2p_map: dict[int, tuple[int, int]] = dict()
        self.init_p2l_map: dict[tuple[int, int], int] = dict()
        self.block_depend_graph = nx.DiGraph()
        self.action_depend_graph = nx.DiGraph()
        self.circuit: Circuit = None

        self.curr_l2p_map: dict[int, tuple[int, int]] = None
        self.curr_p2l_map: dict[tuple[int, int], int] = None
        self.local_gate_freq = {i: 0 for i in range(program.data_qubit_num)}

        # self.commutable_blocks = []
        # self.compute_blocks = []

    def route_program_on_package(self):
        """
        Route the program on the package.
        1) map logic qubits to physical qubits;
        2) determine compute blocks;
        3) transform compute blocks into circuit actions;
        4) schedule actions to circuit.
        """
        self.map_qubits()
        self.curr_l2p_map = deepcopy(self.init_l2p_map)
        self.curr_p2l_map = deepcopy(self.init_p2l_map)
        self.determine_compute_blocks()
        self.transform_compute_blocks()
        self.circuit = Circuit(self.package, self.program.data_qubit_num, self.init_l2p_map, self.init_p2l_map, self.action_depend_graph)
        self.schedule_action_to_circuit()

    def map_qubits(self):
        """
        Map logic qubits to physical qubits which play as registers.
        """
        logic_data_qubit_num = self.program.data_qubit_num
        physical_data_qubit_list = self.package.get_registers()
        if logic_data_qubit_num > len(physical_data_qubit_list):
            raise ValueError("No enough physical qubits to map logic qubits.")
        for i in range(logic_data_qubit_num):
            self.init_l2p_map[i] = physical_data_qubit_list[i]
            self.init_p2l_map[physical_data_qubit_list[i]] = i

    # Functions for determining compute blocks
    def update_no_depend_gate_ids(self, no_depend_gate_ids: list[int], removed_gate_id: int) -> list[int]:
        """
        Update the no_depend_gate_ids list by checking removed gate's successors.

        Args:
            no_depend_gate_ids (list[int]): The original list of gate ids with no dependencies.
            removed_gate_id (int): The id of the removed gate.

        Returns:
            list[int]: The updated list of gate ids with no dependencies.
        """
        for succ_gate in self.gate_depend_graph.successors(removed_gate_id):
            self.gate_depend_graph.nodes[succ_gate]["pred"].remove(removed_gate_id)
            if len(self.gate_depend_graph.nodes[succ_gate]["pred"]) == 0:
                no_depend_gate_ids.append(succ_gate)
        return no_depend_gate_ids

    def determine_compute_blocks(self):
        """
        Schedule control gates into compute blocks, either TeleBlock or MoveBlock.

        Compute criteria: sequentially check gates in commutable block, 
        1) LocalBlock: collect gates whose control qubit and target qubit are in the same compute area and can realize with higher fidelity without transfer line.
        2) TeleBlock: collect following gates having the same control qubit with the first gate, if #gate > 2, then form a TeleBlock.
        3) MoveBlock: collect following gates, if #gate > #qubits*(#qubits-1)//4 and #gate < max(chiplet_capacity). If #gate > 2, then form a MoveBlock.
        4) Otherwise, form a TeleBlock.

        Note: If all gates in the commutable block is collected, then check the next commutable block immediately.
        """
        for gate_id in self.gate_depend_graph.nodes:
            self.gate_depend_graph.nodes[gate_id]["pred"] = list(self.gate_depend_graph.predecessors(gate_id))
        no_depend_gate_ids = []
        for gate_id in self.gate_depend_graph.nodes:
            if len(self.gate_depend_graph.nodes[gate_id]["pred"]) == 0:
                no_depend_gate_ids.append(gate_id)

        while len(no_depend_gate_ids) > 0:
            gate_id = no_depend_gate_ids.pop(0)
            no_depend_gate_ids = self.update_no_depend_gate_ids(no_depend_gate_ids, gate_id)
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            area = self.check_local_gate(gate)
            if area:
                local_gates = [gate]
                self.local_gate_freq[gate.control_qubit] += 1
                self.local_gate_freq[gate.target_qubit] += 1
                while True:
                    add_gate = None
                    for cand_gate_id in no_depend_gate_ids:
                        cand_gate = self.gate_depend_graph.nodes[cand_gate_id]["gate"]
                        assert isinstance(cand_gate, Control)
                        if self.check_local_gate(cand_gate) == area:
                            add_gate = cand_gate
                            break
                    if add_gate:
                        local_gates.append(add_gate)
                        self.local_gate_freq[add_gate.control_qubit] += 1
                        self.local_gate_freq[add_gate.target_qubit] += 1
                        no_depend_gate_ids.remove(add_gate.id)
                        no_depend_gate_ids = self.update_no_depend_gate_ids(no_depend_gate_ids, add_gate.id)
                    else:
                        break
                local_block = LocalBlock(local_gates, area)
                self.block_depend_graph.add_node(local_block.id, block=local_block)
                depend_block_ids = set()
                local_ids = [local_gate.id for local_gate in local_gates]
                for local_gate in local_block.ctrl_gates:
                    self.gate_depend_graph.nodes[local_gate.id]["block"] = local_block.id
                    for pred_gate in self.gate_depend_graph.predecessors(local_gate.id):
                        if pred_gate not in local_ids:
                            depend_block_ids.add(self.gate_depend_graph.nodes[pred_gate]["block"])
                for depend_block_id in depend_block_ids:
                    assert depend_block_id != local_block.id
                    self.block_depend_graph.add_edge(depend_block_id, local_block.id)
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
                    no_depend_gate_ids.append(move_gates[1].id)
                    for succ_gate in self.gate_depend_graph.successors(restore_gate.id):
                        self.gate_depend_graph.nodes[succ_gate]["pred"].append(restore_gate.id)
                        no_depend_gate_ids.remove(succ_gate)
                if block_type == "tele":
                    tele_block = TeleBlock(ctrl_qubit, [tele_gate.target_qubit for tele_gate in tele_gates])
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
                else:
                    involved_qubits = set()
                    for move_gate in move_gates:
                        involved_qubits.add(move_gate.control_qubit)
                        involved_qubits.add(move_gate.target_qubit)
                    chiplet = self.determine_chiplet(involved_qubits)
                    swap_logic_pairs = self.move_qubits_in_block(involved_qubits, chiplet)
                    move_block = MoveBlock(move_gates, involved_qubits, chiplet, swap_logic_pairs)
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

    def determine_chiplet(self, involved_qubits: list[int]) -> ComputeChiplet:
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
            chiplet_capacity = chiplet.get_capacity(chiplet)
            if chiplet_capacity > len(involved_qubits):
                if chiplet in chiplet_cnt:
                    chiplet_cnt[chiplet] += 1
                else:
                    chiplet_cnt[chiplet] = 1
        chiplet = max(chiplet_cnt, key=chiplet_cnt.get)
        return chiplet

    def find_qubits_to_swap(self, chiplet: ComputeChiplet, in_qubits: list[int], stand_qubits: list[int]) -> list[int]:
        """
        Find the logic qubits on target chiplet to replace, which has less control gate with qubits in the same chiplet.
        
        Args:
            chiplet (ComputeChiplet): The target chiplet to which the qubits will be moved.
            in_qubits (list[int]): The list of qubits to be moved in.
            stand_qubits (list[int]): The list of qubits that are already on the target chiplet.

        Returns:
            list[int]: The list of qubits to be swapped out.
        """
        candidate_local_freq = dict()
        for i in range(chiplet.left, chiplet.right + 1):
            for j in range(chiplet.top, chiplet.bottom + 1):
                if self.package_graph.nodes[(i, j)]["qubit"].role == QubitRole.Register:
                    cand_qubit = self.curr_p2l_map[(i, j)]
                    if cand_qubit not in stand_qubits:
                        candidate_local_freq[cand_qubit] = self.local_gate_freq[cand_qubit]
        out_qubits = sorted(candidate_local_freq, key=candidate_local_freq.get)
        out_qubits = out_qubits[:len(in_qubits)]
        self.local_gate_freq = {i: 0 for i in range(self.program.data_qubit_num)}
        return out_qubits

    def move_qubits_in_block(self, involved_qubits: list[int], chiplet: ComputeChiplet) -> list[tuple[int, int]]:
        """
        Move qubits in the block to the target chiplet.

        Args:
            involved_qubits (list[int]): The list of involved qubits in the move block.
            chiplet (ComputeChiplet): The target chiplet to which the qubits will be moved.

        Returns:
            list[tuple[int, int]]: The list of qubit pairs to be swapped.
        """
        in_qubits: list[int] = []
        stand_qubits: list[int] = []
        for qubit in involved_qubits:
            if self.package_graph.nodes[self.curr_l2p_map[qubit]]["qubit"].chiplet != chiplet:
                in_qubits.append(qubit)
            else:
                stand_qubits.append(qubit)
        out_qubits = self.find_qubits_to_swap(chiplet, in_qubits, stand_qubits)
        swap_logic_pairs: list[tuple[int, int]] = []
        for i in len(in_qubits):
            swap_logic_pairs.append((in_qubits[i], out_qubits[i]))
            in_physical_qubit = self.curr_l2p_map[in_qubits[i]]
            out_physical_qubit = self.curr_l2p_map[out_qubits[i]]
            self.curr_l2p_map[in_qubits[i]] = out_physical_qubit
            self.curr_l2p_map[out_qubits[i]] = in_physical_qubit
            self.curr_p2l_map[out_physical_qubit] = in_qubits[i]
            self.curr_p2l_map[in_physical_qubit] = out_qubits[i]
        return swap_logic_pairs

    # Functions for transforming compute blocks into circuit actions
    def update_no_depend_block_ids(self, no_depend_block_ids: list[int], removed_block_id: int) -> list[int]:
        """
        Update the no_depend_block_ids list by checking removed block's successors.

        Args:
            no_depend_block_ids (list[int]): The original list of block ids with no dependencies.
            removed_block_id (int): The id of the removed block.

        Returns:
            list[int]: The updated list of block ids with no dependencies.
        """
        for succ_block in self.block_depend_graph.successors(removed_block_id):
            self.block_depend_graph.nodes[succ_block]["pred"].remove(removed_block_id)
            if len(self.block_depend_graph.nodes[succ_block]["pred"]) == 0:
                no_depend_block_ids.append(succ_block)
        return no_depend_block_ids

    def transform_compute_blocks(self):
        """
        Transform compute blocks into circuit actions.
        """
        self.recent_logic_qubit_action_dict = {i: None for i in range(self.program.data_qubit_num)}
        for block_id in self.block_depend_graph:
            self.block_depend_graph.nodes[block_id]["pred"] = list(self.block_depend_graph.predecessors(block_id))
        no_depend_block_ids = []
        for block_id in self.block_depend_graph.nodes:
            if len(self.block_depend_graph.nodes[block_id]["pred"]) == 0:
                no_depend_block_ids.append(block_id)
        while len(no_depend_block_ids) > 0:
            block_id = no_depend_block_ids.pop(0)
            no_depend_block_ids = self.update_no_depend_block_ids(no_depend_block_ids, block_id)
            block = self.block_depend_graph.nodes[block_id]["block"]
            assert isinstance(block, ComputeBlock)
            if isinstance(block, LocalBlock):
                self.transform_local_block(block)
            elif isinstance(block, TeleBlock):
                self.transform_tele_block(block)
            elif isinstance(block, MoveBlock):
                self.transform_move_block(block)
            else:
                raise ValueError("Unknown block type.")

    def transform_local_block(self, local_block: LocalBlock):
        """
        Transform local block into circuit actions.
        1) route local gates -> LocalGate.

        Action dependency: LocalGate.
        """
        local_gate_ids = [local_gate.id for local_gate in local_block.ctrl_gates]
        local_gate_depent: nx.DiGraph = self.gate_depend_graph.subgraph(local_gate_ids)
        local_gates_topo_sort = list(nx.topological_sort(local_gate_depent))
        compute_area = local_block.module
        for gate_id in local_gates_topo_sort:
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            local_action = LocalGate(gate, compute_area)
            self.action_depend_graph.add_node(local_action.id, action=local_action)
            if self.recent_logic_qubit_action_dict[gate.control_qubit]:
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[gate.control_qubit], local_action.id)
            self.recent_logic_qubit_action_dict[gate.control_qubit] = local_action.id
            if self.recent_logic_qubit_action_dict[gate.target_qubit]:
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[gate.target_qubit], local_action.id)
            self.recent_logic_qubit_action_dict[gate.target_qubit] = local_action.id
    
    def transform_tele_block(self, tele_block: TeleBlock):
        """
        Transform tele block into circuit actions.
        1) route nearest platform for all involved qubits -> RouteToPlat;
        2) allocate GHZ resource -> GhzPreparation;
        3) route on transfer chiplets -> TransferChipletRoute.
        4) ancilla qubits get off chiplet -> Alight;
        5) teleport qubits -> Teleport.

        Action dependency: GhzPreparation -> TransferChipletRoute & RouteToPlat -> Alight -> Teleport.
        """
        involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits
        route_to_plat_acts = self.route_nearest_platforms(involved_qubits, False)

        platforms = [act.platform for act in route_to_plat_acts]
        ghz_action = self.locate_ghz_state(involved_qubits, platforms)

        self.action_depend_graph.add_node(ghz_action.id, action=ghz_action)
        for logic_qubit in involved_qubits:
            if self.recent_logic_qubit_action_dict[logic_qubit]:
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[logic_qubit], ghz_action.id)
                
        route_to_plat_act_ids = dict()
        for act in route_to_plat_acts:
            self.action_depend_graph.add_node(act.id, action=act)
            self.action_depend_graph.add_edge(ghz_action.id, act.id)
            route_to_plat_act_ids[act.logic_qubit] = act.id

        for i in range(ghz_action.qubit_num):
            logic_qubit = ghz_action.logic_qubits[i]
            source_phy_qubit = ghz_action.source_physical_qubits[i]
            target_phy_qubit = ghz_action.target_physical_qubits[i]
            transfer_on_metro_acts = self.route_on_transfer_chiplets(logic_qubit, source_phy_qubit, target_phy_qubit)
            for i in range(len(transfer_on_metro_acts)):
                self.action_depend_graph.add_node(transfer_on_metro_acts[i].id, action=transfer_on_metro_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(ghz_action.id, transfer_on_metro_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(transfer_on_metro_acts[i-1].id, transfer_on_metro_acts[i].id)
            alight_act = Alight(logic_qubit, target_phy_qubit, False)
            self.action_depend_graph.add_node(alight_act.id, action=alight_act)
            self.action_depend_graph.add_edge(transfer_on_metro_acts[-1].id, alight_act.id)
            teleport_act = Teleport(logic_qubit, platforms[i], logic_qubit==tele_block.control_qubit)
            self.action_depend_graph.add_node(teleport_act.id, action=teleport_act)
            self.action_depend_graph.add_edge(route_to_plat_act_ids[logic_qubit], teleport_act.id)
            self.action_depend_graph.add_edge(alight_act.id, teleport_act.id)
            self.recent_logic_qubit_action_dict[logic_qubit] = teleport_act.id

    def transform_move_block(self, move_block: MoveBlock):
        """
        Transform move block into circuit actions.
        1) route nearest platform for all involved qubits -> RouteToPlat;
        2) data qubits get on chiplet -> Board;
        3) route on transfer chiplets -> TransferChipletRoute;
        4) data qubits get off chiplet -> Alight;
        5) route local gates -> LocalGate.

        Action dependency: RouteToPlat -> Board -> TransferChipletRoute -> Alight -> LocalGate.
        """
        in_qubits = [swap_pair[0] for swap_pair in move_block.swap_logic_pairs]
        out_qubits = [swap_pair[1] for swap_pair in move_block.swap_logic_pairs]
        route_to_plat_acts = self.route_nearest_platforms(in_qubits+out_qubits, True)
        route_to_plat_act_ids = dict()
        for act in route_to_plat_acts:
            self.action_depend_graph.add_node(act.id, action=act)
            self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[act.logic_qubit], act.id)
            route_to_plat_act_ids[act.logic_qubit] = act.id

        in_alight_act_ids = []
        for swap_pair in move_block.swap_logic_pairs:
            in_qubit = swap_pair[0]
            out_qubit = swap_pair[1]
            in_platform = route_to_plat_acts[route_to_plat_act_ids[in_qubit]].platform
            out_platform = route_to_plat_acts[route_to_plat_act_ids[out_qubit]].platform
            in_station = self.qubit_topology.nodes[in_platform]["station"]
            out_station = self.qubit_topology.nodes[out_platform]["station"]
            in_board_act = Board(in_qubit, in_station, in_platform)
            self.action_depend_graph.add_node(in_board_act.id, action=in_board_act)
            self.action_depend_graph.add_edge(route_to_plat_act_ids[in_qubit], in_board_act.id)
            out_board_act = Board(out_qubit, out_station, out_platform)
            self.action_depend_graph.add_node(out_board_act.id, action=out_board_act)
            self.action_depend_graph.add_edge(route_to_plat_act_ids[out_qubit], out_board_act.id)
            in_transfer_acts = self.route_on_transfer_chiplets(in_qubit, in_station, out_station)
            for i in range(len(in_transfer_acts)):
                self.action_depend_graph.add_node(in_transfer_acts[i].id, action=in_transfer_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(in_board_act.id, in_transfer_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(in_transfer_acts[i-1].id, in_transfer_acts[i].id)
            out_transfer_acts = self.route_on_transfer_chiplets(out_qubit, out_station, in_station)
            for i in range(len(out_transfer_acts)):
                self.action_depend_graph.add_node(out_transfer_acts[i].id, action=out_transfer_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(out_board_act.id, out_transfer_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(out_transfer_acts[i-1].id, out_transfer_acts[i].id)
            in_alight_act = Alight(in_qubit, out_station, out_platform, True)
            self.action_depend_graph.add_node(in_alight_act.id, action=in_alight_act)
            self.action_depend_graph.add_edge(in_transfer_acts[-1].id, in_alight_act.id)
            self.action_depend_graph.add_edge(out_board_act.id, in_alight_act.id)
            in_alight_act_ids.append(in_alight_act.id)
            out_alight_act = Alight(out_qubit, in_station, in_platform, True)
            self.action_depend_graph.add_node(out_alight_act.id, action=out_alight_act)
            self.action_depend_graph.add_edge(out_transfer_acts[-1].id, out_alight_act.id)
            self.action_depend_graph.add_edge(in_board_act.id, out_alight_act.id)
            self.recent_logic_qubit_action_dict[in_qubit] = in_alight_act.id
            self.recent_logic_qubit_action_dict[out_qubit] = out_alight_act.id

        local_gate_ids = [local_gate.id for local_gate in move_block.ctrl_gates]
        local_gate_depent: nx.DiGraph = self.gate_depend_graph.subgraph(local_gate_ids)
        local_gates_topo_sort = list(nx.topological_sort(local_gate_depent))
        compute_area = self.package.get_node_area((move_block.chiplet.split("_")[1], move_block.chiplet.split("_")[2]))
        for gate_id in local_gates_topo_sort:
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            local_action = LocalGate(gate, compute_area)
            self.action_depend_graph.add_node(local_action.id, action=local_action)
            if self.recent_logic_qubit_action_dict[gate.control_qubit]:
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[gate.control_qubit], local_action.id)
            self.recent_logic_qubit_action_dict[gate.control_qubit] = local_action.id
            if self.recent_logic_qubit_action_dict[gate.target_qubit]:
                self.action_depend_graph.add_edge(self.recent_logic_qubit_action_dict[gate.target_qubit], local_action.id)
            self.recent_logic_qubit_action_dict[gate.target_qubit] = local_action.id

    def route_nearest_platforms(self, logic_qubits: list[int], enter_flag: bool) -> list[RouteToPlat]:
        """
        Find the nearest unoccupied platform for logic qubits and route to it.

        Args:
            logic_qubits (list[int]): The list of logic qubits to be routed.
            enter_flag (bool): The flag marking whether the qubits are entering the platform.

        Return:
            list[RouteToPlat]: The list of actions to route the qubits to the platform.
        """
        route_to_plat_acts = []
        occupied_platforms = []
        for logic_qubit in logic_qubits:
            physical_qubit = self.curr_l2p_map[logic_qubit]
            compute_module = self.package.get_node_area(physical_qubit)
            assert isinstance(compute_module, ComputeModule)
            nearest_platform = None
            highest_fide = 0
            for platform in compute_module.get_platforms():
                if platform not in occupied_platforms:
                    _, fide = self.package.get_shortest_no_transfer_path(physical_qubit, platform)
                    if nearest_platform is None or fide > highest_fide:
                        nearest_platform = platform
                        highest_fide = fide
            route_to_plat_acts.append(RouteToPlat(logic_qubit, platform, enter_flag))
            occupied_platforms.append(platform)
        return route_to_plat_acts

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
                actions.append(Interchange(logic_qubit, road.source, road.target))
        return actions

    # Functions for scheduling actions to circuit
    def update_no_depend_action_ids(self, no_depend_action_ids: list[int], removed_action_id: int) -> list[int]:
        """
        Update the no_depend_action_ids list by checking removed action's successors.

        Args:
            no_depend_action_ids (list[int]): The original list of action ids with no dependencies.
            removed_action_id (int): The id of the removed action.

        Returns:
            list[int]: The updated list of action ids with no dependencies.
        """
        for succ_action in self.action_depend_graph.successors(removed_action_id):
            self.action_depend_graph.nodes[succ_action]["pred"].remove(removed_action_id)
            if len(self.action_depend_graph.nodes[succ_action]["pred"]) == 0:
                no_depend_action_ids.append(succ_action)
        return no_depend_action_ids

    def schedule_action_to_circuit(self):
        """
        Schedule the actions to the circuit.
        """
        for action_id in self.action_depend_graph.nodes:
            self.action_depend_graph.nodes[action_id]["pred"] = list(self.action_depend_graph.predecessors(action_id))
        no_depend_action_ids = []
        for action_id in self.action_depend_graph.nodes:
            if len(self.action_depend_graph.nodes[action_id]["pred"]) == 0:
                no_depend_action_ids.append(action_id)
        board_flag = True
        while len(no_depend_action_ids) > 0:
            new_stage = self.circuit.extend_stage(board_flag)
            if board_flag:
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
                # schedule Transport actions
                for action_id in no_depend_action_ids:
                    action = self.action_depend_graph.nodes[action_id]["action"]
                    if isinstance(action, Transport) and new_stage.add_transport_action(action):
                        no_depend_action_ids.remove(action_id)
                        no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                # schedule RouteToPlat, Teleport, and LocalGate actions
                while True:
                    action_added = False
                    for action_id in no_depend_action_ids:
                        action = self.action_depend_graph.nodes[action_id]["action"]
                        if isinstance(action, RouteToPlat) and new_stage.add_route_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, Teleport) and new_stage.add_teleport_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                        elif isinstance(action, LocalGate) and new_stage.add_local_action(action):
                            no_depend_action_ids.remove(action_id)
                            no_depend_action_ids = self.update_no_depend_action_ids(no_depend_action_ids, action_id)
                            action_added = True
                            break
                    if not action_added:
                        break
            board_flag = not board_flag
    
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
