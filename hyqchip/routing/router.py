import copy

from ..element.package import *
from ..element.benchmark import *
from .circuit import *


class ComputeBlock:
    _counter = 0

    def __init__(self):
        self.id = ComputeBlock._counter
        ComputeBlock._counter += 1


class LocalBlock(ComputeBlock):
    def __init__(self, ctrl_gates: list[Control], area):
        super().__init__()
        self.ctrl_gates = ctrl_gates
        self.area = area


class TeleBlock(ComputeBlock):
    def __init__(self, control_qubit, target_qubits):
        super().__init__()
        self.control_qubit = control_qubit
        self.target_qubits = target_qubits


class MoveBlock(ComputeBlock):
    def __init__(self, ctrl_gates: list[Control], involved_qubits, die, swap_logic_pairs):
        super().__init__()
        self.ctrl_gates = ctrl_gates
        self.involved_qubits = involved_qubits
        self.die = die
        self.swap_logic_pairs: list[tuple[int, int]] = swap_logic_pairs


class Router:
    def __init__(self, package: Package, program: Program):
        '''
        Class for routing a quantum program on a package.

        package: Package, the package to be routed on.
        program: Program, the quantum program to be routed.

        l2p_map: dict, map from logic qubit index to physical qubit index
        p2l_map: dict, map from physical qubit index to logic qubit index
        circuit: Circuit, the routing results with circuit level actions in stages.
        '''

        self.package = package
        self.program = program

        self.qubit_topology = package.qubit_topology
        self.die_topology = package.die_topology
        self.metro_graph = package.metro_graph
        self.analysis_metro()

        self.platform_list = package.get_platform_list()
        self.station_list = package.get_station_list()
        self.interchange_list = package.get_interchange_list()

        self.gate_depend_graph = program.gate_depend_graph
        self.block_depend_graph = nx.DiGraph()
        self.action_depend_graph = nx.DiGraph()

        self.l2p_map = dict()
        self.p2l_map = dict()
        self.local_gate_freq = {i: 0 for i in range(program.data_qubit_num)}

        # self.commutable_blocks = []
        # self.compute_blocks = []
        self.circuit = Circuit(package, program.data_qubit_num)

    def analysis_metro(self):
        self.balise_station_distance_dict = {balise: {station: {"interchange": 0, "path": None} for station in self.station_list} for balise in self.metro_graph.nodes}
        for balise in self.metro_graph.nodes:
            for station in self.station_list:
                shortest_path, interchange_num = self.package.get_shortest_metro_path(balise, station)
                self.balise_station_distance_dict[balise][station]["interchange"] = interchange_num
                self.balise_station_distance_dict[balise][station]["path"] = shortest_path
        pass

    def map_qubits(self):
        '''
        Map logic qubits to physical qubits.
        Left qubits beside the die gaps, and take rest qubits in compute dies as compute qubits.
        '''
        
        logic_data_qubit_num = self.program.data_qubit_num
        physical_data_qubit_list = self.package.get_data_qubit_list()
        if logic_data_qubit_num > len(physical_data_qubit_list):
            raise ValueError("No enough physical qubits to map logic qubits.")
        for i in range(logic_data_qubit_num):
            self.l2p_map[i] = physical_data_qubit_list[i]
            self.p2l_map[physical_data_qubit_list[i]] = i

    # @staticmethod
    # def check_commutable_gate(gate, control_qubits, target_qubits) -> bool:
    #     '''
    #     Check if the gate is commutable with previous gates.
    #     Commutable criteria: 1) no common involved qubits; 2) share the same control qubit.
    #     '''

    #     involved_qubits = control_qubits + target_qubits
    #     if gate[0] not in involved_qubits and gate[1] not in involved_qubits:
    #         return True
    #     elif gate[0] in control_qubits and gate[0] not in target_qubits:
    #         return True
    #     else:
    #         return False
    
    # def check_commutable_block(self):
    #     '''
    #     Check the commutable blocks in the program.
    #     Commutable criteria: 1) no common involved qubits; 2) share the same control qubit.

    #     self.commutable_blocks: list[list[tuple]], list of commutable blocks, each commutable blocks contains commutable control gates.
    #     '''

    #     program_copy = copy.deepcopy(self.program)
    #     commute_block = []
    #     while len(program_copy) > 0:
    #         gate = program_copy.pop(0)
    #         if gate == 'barrier':
    #             continue
    #         else:
    #             commute_block.append(gate)
    #             checked_control_qubits = [gate[0]]
    #             checked_target_qubits = [gate[1]]
    #             pop_list = []
    #             for i in range(len(program_copy)):
    #                 if program_copy[i] == 'barrier':
    #                     pop_list.append(i)
    #                     break
    #                 else:
    #                     if self.check_commutable_gate(program_copy[i], checked_control_qubits, checked_target_qubits):
    #                         commute_block.append(program_copy[i])
    #                         pop_list.append(i)
    #                     if program_copy[i][0] not in checked_control_qubits:
    #                         checked_control_qubits.append(program_copy[i][0])
    #                     if program_copy[i][1] not in checked_target_qubits:
    #                         checked_target_qubits.append(program_copy[i][1])
    #             for i in sorted(pop_list, reverse=True):
    #                 program_copy.pop(i)
    #             self.commutable_blocks.append(commute_block)
    #             commute_block = []

    def check_local_gate(self, gate: Control) -> dict:
        '''
        Check if the gate is a local gate.
        Local criteria: transfer die involved in the shortest path.
        '''
        
        shortest_path, _ = self.package.get_shortest_path((self.l2p_map[gate.control_qubit], self.l2p_map[gate.target_qubit]))
        for node in shortest_path:
            if self.qubit_topology.nodes[node]["type"] == CellType.Transfer:
                return None
        return self.package.get_physical_qubit_area(self.l2p_map[gate.control_qubit])
    
    def check_move_gate(self, gate: Control, move_gates: list[Control]) -> bool:
        '''
        Check if the gate can be inserted into the move gates.
        Concentrate criteria: #gate > #qubits*(#qubits-1)//4 and #qubit < max(die_capacity)
        '''
        involved_qubits = set()
        for move_gate in move_gates:
            involved_qubits.add(move_gate.control_qubit)
            involved_qubits.add(move_gate.target_qubit)
        if gate.control_qubit not in involved_qubits and gate.target_qubit not in involved_qubits:
            return False
        involved_qubits.add(gate.control_qubit)
        involved_qubits.add(gate.target_qubit)
        
        max_die_capacity = 0
        for involved_qubit in involved_qubits:
            physical_qubit = self.l2p_map[involved_qubit]
            die = self.qubit_topology.nodes[physical_qubit]["die"]
            die_capacity = self.package.get_die_capacity(die)
            if die_capacity > max_die_capacity:
                max_die_capacity = die_capacity

        if len(move_gates)+1 < len(involved_qubits) * (len(involved_qubits) - 1) // 3:
            return False
        if len(involved_qubits) > max_die_capacity:
            return False
        return True

    def determine_die(self, involved_qubits):
        '''
        Determine the die to move the qubits to.
        '''

        die_cnt = dict()
        for qubit in involved_qubits:
            physical_qubit = self.l2p_map[qubit]
            die = self.qubit_topology.nodes[physical_qubit]["die"]
            die_capacity = self.package.get_die_capacity(die)
            if die_capacity > len(involved_qubits):
                if die in die_cnt:
                    die_cnt[die] += 1
                else:
                    die_cnt[die] = 1
        die = max(die_cnt, key=die_cnt.get)
        return die

    def find_qubits_to_swap(self, die, in_qubits, stand_qubits):
        '''
        Find the logic qubits on target die to replace, which has less control gate with qubits in the same die.
        
        Return:
        list, the logic qubits on target die to replace.
        '''

        die_info = die.split("_")
        die_left = int(die_info[1])
        die_top = int(die_info[2])
        die_right = int(die_info[3])
        die_bottom = int(die_info[4])
        candidate_local_freq = dict()
        for i in range(die_left, die_right + 1):
            for j in range(die_top, die_bottom + 1):
                if self.qubit_topology.nodes[(i, j)]["role"] == RoleType.Data:
                    cand_qubit = self.p2l_map[(i, j)]
                    if cand_qubit not in stand_qubits:
                        candidate_local_freq[cand_qubit] = self.local_gate_freq[cand_qubit]
        out_qubits = sorted(candidate_local_freq, key=candidate_local_freq.get)
        out_qubits = out_qubits[:len(in_qubits)]
        self.local_gate_freq = {i: 0 for i in range(self.program.data_qubit_num)}
        return out_qubits

    def move_qubits_in_block(self, involved_qubits, die):
        '''
        Move qubits in the block to the target die.
        '''

        in_qubits = []
        stand_qubits = []
        for qubit in involved_qubits:
            if self.qubit_topology.nodes[self.l2p_map[qubit]]["die"] != die:
                in_qubits.append(qubit)
            else:
                stand_qubits.append(qubit)
        out_qubits = self.find_qubits_to_swap(die, in_qubits, stand_qubits)
        swap_logic_pairs = []
        for i in len(in_qubits):
            swap_logic_pairs.append((in_qubits[i], out_qubits[i]))
            in_physical_qubit = self.l2p_map[in_qubits[i]]
            out_physical_qubit = self.l2p_map[out_qubits[i]]
            self.l2p_map[in_qubits[i]] = out_physical_qubit
            self.l2p_map[out_qubits[i]] = in_physical_qubit
            self.p2l_map[out_physical_qubit] = in_qubits[i]
            self.p2l_map[in_physical_qubit] = out_qubits[i]
        return swap_logic_pairs

    # def scan_local_block(self, commute_block):
    #     '''
    #     Scan the commutable block and find local blocks.
    #     '''

    #     pop_list = []
    #     local_block_dict = dict()
    #     for i in range(len(commute_block)):
    #         area = self.check_local_gate(commute_block[i])
    #         if area:
    #             if area in local_block_dict:
    #                 local_block_dict[area].append(commute_block[i])
    #             else:
    #                 local_block_dict[area] = [commute_block[i]]
    #             if commute_block[i][0] not in self.local_gate_freq:
    #                 self.local_gate_freq[commute_block[i][0]] = 1
    #             else:
    #                 self.local_gate_freq[commute_block[i][0]] += 1
    #             if commute_block[i][1] not in self.local_gate_freq:
    #                 self.local_gate_freq[commute_block[i][1]] = 1
    #             else:
    #                 self.local_gate_freq[commute_block[i][1]] += 1
    #             pop_list.append(i)
    #     local_blocks = []
    #     for key, value in local_block_dict.items():
    #         local_block = LocalBlock(value, key)
    #         local_blocks.append(local_block)
    #     return local_blocks, pop_list
    
    # def scan_tele_block(self, commute_block):
    #     '''
    #     Scan the commutable block and find tele blocks.
    #     '''

    #     tele_blocks = []
    #     pop_list = []
    #     while len(commute_block) > 0:
    #         gate = commute_block.pop(0)
    #         ctrl_qubit = gate[0]
    #         common_ctrl_target_qubits = [gate[1]]
    #         for i in range(len(commute_block)):
    #             if commute_block[i][0] == ctrl_qubit:
    #                 common_ctrl_target_qubits.append(commute_block[i][1])
    #                 pop_list.append(i)
    #         for i in sorted(pop_list, reverse=True):
    #             commute_block.pop(i)
    #         tele_block = TeleBlock(ctrl_qubit, common_ctrl_target_qubits)
    #         tele_blocks.append(tele_block)
    #     return tele_blocks

    # def extend_to_tele_block(self, tele_block, commute_block):
    #     '''
    #     Extend the tele block with next commute block as tele block to include more gates.
    #     '''

    #     ctrl_qubit = tele_block.control_qubit
    #     extend_list = []
    #     for i in range(len(commute_block)):
    #         if self.check_local_gate(commute_block[i]):
    #             continue
    #         if commute_block[i][0] == ctrl_qubit:
    #             extend_list.append(i)
    #     return extend_list

    # def extend_to_move_block(self, tele_block, commute_block):
    #     '''
    #     Extend the tele block with next commute block as move block to include more gates.
    #     '''

    #     gate_cluster = [(tele_block.control_qubit, _) for _ in tele_block.target_qubits]
    #     involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits
    #     extend_list = []
    #     for i in range(len(commute_block)):
    #         if self.check_local_gate(commute_block[i]):
    #             continue
    #         if self.check_move_gate(commute_block[i], gate_cluster, involved_qubits):
    #             extend_list.append(i)
    #             gate_cluster.append(commute_block[i])
    #             if commute_block[i][0] not in involved_qubits:
    #                 involved_qubits.append(commute_block[i][0])
    #             if commute_block[i][1] not in involved_qubits:
    #                 involved_qubits.append(commute_block[i][1])
    #     return extend_list, gate_cluster, involved_qubits

    def update_no_depend_gate_ids(self, no_depend_gate_ids, removed_gate_id):
        '''
        Update the no_depend_gate_ids list by checking removed gate's successors.
        '''

        for succ_gate in self.gate_depend_graph.successors(removed_gate_id):
            self.gate_depend_graph.nodes[succ_gate]["pred"].remove(removed_gate_id)
            if len(self.gate_depend_graph.nodes[succ_gate]["pred"]) == 0:
                no_depend_gate_ids.append(succ_gate)
        return no_depend_gate_ids

    def determine_compute_blocks(self):
        '''
        Schedule control gates into compute blocks, either TeleBlock or MoveBlock.
        Compute criteria: sequentially check gates in commutable block
        1) LocalBlock: collect gates whose control qubit and target qubit are in the same compute area and can realize with higher fidelity without transfer line.
        2) TeleBlock: collect following gates having the same control qubit with the first gate, if #gate > 2, then form a TeleBlock.
        3) MoveBlock: collect following gates, if #gate > #qubits*(#qubits-1)//4 and #gate < max(die_capacity). If #gate > 2, then form a MoveBlock.
        4) Otherwise, form a TeleBlock.
        Note: if all left gates in the commutable block is collected, then check the next commutable block.

        self.compute_blocks: list[TeleBlock or MoveBlock], list of compute blocks.
        '''

        for node in self.gate_depend_graph:
            self.gate_depend_graph.nodes[node]["pred"] = list(self.gate_depend_graph.predecessors(node))
        no_depend_gate_ids = []
        for node in self.gate_depend_graph.nodes:
            if len(self.gate_depend_graph.nodes[node]["pred"]) == 0:
                no_depend_gate_ids.append(node)

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
                    die = self.determine_die(involved_qubits)
                    swap_logic_pairs = self.move_qubits_in_block(involved_qubits, die)
                    move_block = MoveBlock(move_gates, involved_qubits, die, swap_logic_pairs)
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
                
        # commute_blocks_copy = copy.deepcopy(self.commutable_blocks)
        # last_tele_blocks = None
        # while len(commute_blocks_copy) > 0:
        #     commute_block = commute_blocks_copy.pop(0)

        #     if last_tele_blocks:
        #         max_extend_list = []
        #         max_extend_block_index = -1
        #         max_extend_type = None
        #         max_gate_cluster = None
        #         max_involved_qubits = None
        #         for i in len(last_tele_blocks):
        #             extend_list_tele = self.extend_to_tele_block(last_tele_blocks[i], commute_block)
        #             if len(extend_list_tele) > len(max_extend_list):
        #                 max_extend_list = extend_list_tele
        #                 max_extend_block_index = i
        #                 max_extend_type = 'tele'
        #             extend_list_move, gate_cluster, involved_qubits = self.extend_to_move_block(last_tele_blocks[i], commute_block)
        #             if len(extend_list_move) > len(max_extend_list):
        #                 max_extend_list = extend_list_move
        #                 max_extend_block_index = i
        #                 max_extend_type = 'move'
        #                 max_gate_cluster = gate_cluster
        #                 max_involved_qubits = involved_qubits
        #         for i in len(last_tele_blocks):
        #             if i != max_extend_block_index:
        #                 self.compute_blocks.append(last_tele_blocks[i])
        #         if len(max_extend_list) > 0:
        #             if max_extend_type == 'tele':
        #                 tele_block = last_tele_blocks[max_extend_block_index]
        #                 tele_block.target_qubits.extend([commute_block[i][1] for i in max_extend_list])
        #                 self.compute_blocks.append(tele_block)
        #                 for i in sorted(max_extend_list, reverse=True):
        #                     commute_block.pop(i)
        #             elif max_extend_type == 'move':
        #                 die = self.determine_die(max_involved_qubits)
        #                 move_block = MoveBlock(max_gate_cluster, max_involved_qubits, die)
        #                 self.compute_blocks.append(move_block)
        #                 for i in sorted(max_extend_list, reverse=True):
        #                     commute_block.pop(i)
        #                 self.move_qubits_in_block(max_involved_qubits, die)
                    
        #     local_blocks, pop_list = self.scan_local_block(commute_block)
        #     for i in sorted(pop_list, reverse=True):
        #         commute_block.pop(i)
        #     self.compute_blocks.extend(local_blocks)
        #     last_tele_blocks = self.scan_tele_block(commute_block)

        # for i in range(len(last_tele_blocks)):
        #     self.compute_blocks.append(last_tele_blocks[i])

    def update_no_depend_block_ids(self, no_depend_block_ids, removed_block_id):
        '''
        Update the no_depend_block_ids list by checking removed block's successors.
        '''

        for succ_block in self.block_depend_graph.successors(removed_block_id):
            self.block_depend_graph.nodes[succ_block]["pred"].remove(removed_block_id)
            if len(self.block_depend_graph.nodes[succ_block]["pred"]) == 0:
                no_depend_block_ids.append(succ_block)
        return no_depend_block_ids

    def transform_compute_blocks(self):
        '''
        Transform compute blocks into circuit actions.
        '''

        for node in self.block_depend_graph:
            self.block_depend_graph.nodes[node]["pred"] = list(self.block_depend_graph.predecessors(node))
            self.block_depend_graph.nodes[node]["actions"] = []
        no_depend_block_ids = []
        for node in self.block_depend_graph.nodes:
            if len(self.block_depend_graph.nodes[node]["pred"]) == 0:
                no_depend_block_ids.append(node)
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
        '''
        Transform local block into circuit actions.
        1) route local gates -> LocalCompute.

        LocalCompute.
        '''

        depend_block_ids = self.block_depend_graph.predecessors(local_block.id)
        depend_block_action_ids = []
        for depend_block_id in depend_block_ids:
            depend_block_action_ids.extend(self.block_depend_graph.nodes[depend_block_id]["actions"])
        
        local_gate_ids = [local_gate.id for local_gate in local_block.ctrl_gates]
        local_gate_depent: nx.DiGraph = self.gate_depend_graph.subgraph(local_gate_ids)
        local_gates_topo_sort = list(nx.topological_sort(local_gate_depent))
        compute_area = local_block.area
        
        local_gate_act_dict = dict()
        for gate_id in local_gates_topo_sort:
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            local_action = LocalCompute(gate, compute_area)
            self.action_depend_graph.add_node(local_action.id, action=local_action)
            local_gate_act_dict[gate_id] = local_action
            if len(local_gate_depent.predecessors(gate_id)) == 0:
                for depend_action_id in depend_block_action_ids:
                    self.action_depend_graph.add_edge(depend_action_id, local_action.id)
            else:
                for pred_gate in local_gate_depent.predecessors(gate_id):
                    self.action_depend_graph.add_edge(local_gate_act_dict[pred_gate], local_action.id)
    
    def transform_tele_block(self, tele_block: TeleBlock):
        '''
        Transform tele block into circuit actions.
        1) route nearest stop for all involved qubits -> RouteToStop;
        2) allocate GHZ resource -> GhzPreparation;
        3) route on transfer dies -> TransferDieRoute.
        4) on/off board qubits -> OnOffBoard;
        5) teleport qubits -> Teleport.

        GhzPreparation -> TransferDieRoute & RouteToStop -> OnOffBoard -> Teleport.
        '''

        depend_block_ids = self.block_depend_graph.predecessors(tele_block.id)
        depend_block_action_ids = []
        for depend_block_id in depend_block_ids:
            depend_block_action_ids.extend(self.block_depend_graph.nodes[depend_block_id]["actions"])

        involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits

        route_to_plat_acts = self.route_nearest_platform(involved_qubits, False)
        route_to_plat_act_ids = dict()
        for act in route_to_plat_acts:
            self.action_depend_graph.add_node(act.id, action=act)
            route_to_plat_act_ids[act.logic_qubit] = act.id
            for depend_action_id in depend_block_action_ids:
                self.action_depend_graph.add_edge(depend_action_id, act.id)
        platforms = [act.platform for act in route_to_plat_acts]
        ghz_action = self.locate_ghz_state(involved_qubits, platforms)
        self.action_depend_graph.add_node(ghz_action.id, action=ghz_action)
        for depend_action_id in depend_block_action_ids:
            self.action_depend_graph.add_edge(depend_action_id, ghz_action.id)
        for i in range(ghz_action.qubit_num):
            logic_qubit = ghz_action.logic_qubits[i]
            source_phy_qubit = ghz_action.source_physical_qubits[i]
            target_phy_qubit = ghz_action.target_physical_qubits[i]
            transfer_on_metro_acts = self.route_on_transfer_dies(logic_qubit, source_phy_qubit, target_phy_qubit)
            for i in range(len(transfer_on_metro_acts)):
                self.action_depend_graph.add_node(transfer_on_metro_acts[i].id, action=transfer_on_metro_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(ghz_action.id, transfer_on_metro_acts[i].id)
                    self.action_depend_graph.add_edge(route_to_plat_act_ids[logic_qubit], transfer_on_metro_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(transfer_on_metro_acts[i-1].id, transfer_on_metro_acts[i].id)
            alight_act = self.on_off_board(logic_qubit, target_phy_qubit, False)
            self.action_depend_graph.add_node(alight_act.id, action=alight_act)
            self.action_depend_graph.add_edge(transfer_on_metro_acts[-1].id, alight_act.id)
            teleport_act = self.teleport(logic_qubit, platforms[i], logic_qubit==tele_block.control_qubit)
            self.action_depend_graph.add_node(teleport_act.id, action=teleport_act)
            self.action_depend_graph.add_edge(alight_act.id, teleport_act.id)

    def transform_move_block(self, move_block: MoveBlock):
        '''
        Transform move block into circuit actions.
        1) find qubit to swap;
        2) route nearest stop for all involved qubits -> RouteToStop;
        3) on/off board qubits -> OnOffBoard;
        3) route on transfer dies -> TransferDieRoute.
        4) route local gates -> LocalCompute.

        RouteToStop -> OnOffBoard -> TransferDieRoute -> OnOffBoard -> LocalCompute.
        '''

        depend_block_ids = self.block_depend_graph.predecessors(move_block.id)
        depend_block_action_ids = []
        for depend_block_id in depend_block_ids:
            depend_block_action_ids.extend(self.block_depend_graph.nodes[depend_block_id]["actions"])
        
        in_qubits = [swap_pair[0] for swap_pair in move_block.swap_logic_pairs]
        out_qubits = [swap_pair[1] for swap_pair in move_block.swap_logic_pairs]

        route_to_plat_acts = self.route_nearest_platform(in_qubits+out_qubits, True)
        route_to_plat_act_ids = dict()
        for act in route_to_plat_acts:
            self.action_depend_graph.add_node(act.id, action=act)
            route_to_plat_act_ids[act.logic_qubit] = act.id
            for depend_action_id in depend_block_action_ids:
                self.action_depend_graph.add_edge(depend_action_id, act.id)
        in_alight_act_ids = []
        for swap_pair in move_block.swap_logic_pairs:
            in_qubit = swap_pair[0]
            out_qubit = swap_pair[1]
            in_platform = route_to_plat_acts[route_to_plat_act_ids[in_qubit]].platform
            out_platform = route_to_plat_acts[route_to_plat_act_ids[out_qubit]].platform
            in_station = self.qubit_topology.nodes[in_platform]["station"]
            out_station = self.qubit_topology.nodes[out_platform]["station"]
            in_board_act = self.on_off_board(in_qubit, in_station, True)
            self.action_depend_graph.add_node(in_board_act.id, action=in_board_act)
            self.action_depend_graph.add_edge(route_to_plat_act_ids[in_qubit], in_board_act.id)
            out_board_act = self.on_off_board(out_qubit, out_station, True)
            self.action_depend_graph.add_node(out_board_act.id, action=out_board_act)
            self.action_depend_graph.add_edge(route_to_plat_act_ids[out_qubit], out_board_act.id)
            in_transfer_acts = self.route_on_transfer_dies(in_qubit, in_station, out_station)
            for i in range(len(in_transfer_acts)):
                self.action_depend_graph.add_node(in_transfer_acts[i].id, action=in_transfer_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(in_board_act.id, in_transfer_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(in_transfer_acts[i-1].id, in_transfer_acts[i].id)
            out_transfer_acts = self.route_on_transfer_dies(out_qubit, out_station, in_station)
            for i in range(len(out_transfer_acts)):
                self.action_depend_graph.add_node(out_transfer_acts[i].id, action=out_transfer_acts[i])
                if i == 0:
                    self.action_depend_graph.add_edge(out_board_act.id, out_transfer_acts[i].id)
                else:
                    self.action_depend_graph.add_edge(out_transfer_acts[i-1].id, out_transfer_acts[i].id)
            in_alight_act = self.on_off_board(in_qubit, out_station, False)
            self.action_depend_graph.add_node(in_alight_act.id, action=in_alight_act)
            self.action_depend_graph.add_edge(in_transfer_acts[-1].id, in_alight_act.id)
            self.action_depend_graph.add_edge(out_board_act.id, in_alight_act.id)
            in_alight_act_ids.append(in_alight_act.id)
            out_alight_act = self.on_off_board(out_qubit, in_station, False)
            self.action_depend_graph.add_node(out_alight_act.id, action=out_alight_act)
            self.action_depend_graph.add_edge(out_transfer_acts[-1].id, out_alight_act.id)
            self.action_depend_graph.add_edge(in_board_act.id, out_alight_act.id)

        local_gate_ids = [local_gate.id for local_gate in move_block.ctrl_gates]
        local_gate_depent: nx.DiGraph = self.gate_depend_graph.subgraph(local_gate_ids)
        local_gates_topo_sort = list(nx.topological_sort(local_gate_depent))
        compute_area = self.package.get_physical_qubit_area((move_block.die.split("_")[1], move_block.die.split("_")[2]))
        
        local_gate_act_dict = dict()
        for gate_id in local_gates_topo_sort:
            gate = self.gate_depend_graph.nodes[gate_id]["gate"]
            assert isinstance(gate, Control)
            local_action = LocalCompute(gate, compute_area)
            self.action_depend_graph.add_node(local_action.id, action=local_action)
            local_gate_act_dict[gate_id] = local_action
            if len(local_gate_depent.predecessors(gate_id)) == 0:
                for depend_action_id in in_alight_act_ids:
                    self.action_depend_graph.add_edge(depend_action_id, local_action.id)
            else:
                for pred_gate in local_gate_depent.predecessors(gate_id):
                    self.action_depend_graph.add_edge(local_gate_act_dict[pred_gate], local_action.id)

    def route_nearest_platform(self, logic_qubits, enter_flag) -> list[RouteToPlat]:
        '''
        Find the nearest unoccupied stop for logic qubits and route to it.

        Return:
        RouteToStop, the action to route the qubit to the stop.
        '''

        route_to_plat_acts = []
        occupied_platforms = []
        for logic_qubit in logic_qubits:
            physical_qubit = self.l2p_map[logic_qubit]
            physical_qubit_area = self.package.get_physical_qubit_area(physical_qubit)
            area_graph = self.package.get_area_graph(physical_qubit_area)
            shortest_path_dict = nx.shortest_path(area_graph, physical_qubit)
            for target in shortest_path_dict:
                if self.qubit_topology.nodes[target]["role"] != RoleType.Platform:
                    shortest_path_dict.remove(target)
            shortest_path_dict = sorted(shortest_path_dict, key=lambda x: len(shortest_path_dict[x]))
            for platform in shortest_path_dict:
                if platform not in occupied_platforms:
                    route_to_plat_acts.append(RouteToPlat(logic_qubit, platform, enter_flag))
                    occupied_platforms.append(platform)
                    break
        return route_to_plat_acts

    def get_balise_nearest_metro_subgraph(self, balise_id, node_num) -> nx.Graph:
        '''
        Get the nearest metro subgraph for a balise.
        '''

        visited = set()
        queue = [balise_id]
        while queue and len(visited) < node_num:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend([nbr for nbr in self.metro_graph.neighbors(node) if nbr not in visited])
        subgraph = self.metro_graph.subgraph(visited).copy()
        return subgraph

    def locate_ghz_state(self, involved_qubits, platforms) -> GhzPreparation:
        '''
        Allocate GHZ resource for a list of logic qubits for teleportation.
        Ensure the total transfer distance is minimized.

        Return:
        GhzPreparation, the action to allocate GHZ resource.
        '''

        ancilla_qubit_num = len(involved_qubits)
        stations = [self.qubit_topology.nodes[platform]["station"] for platform in platforms]
        best_interchange_depth = None
        best_ghz_balises = None
        for balise_id in self.metro_graph.nodes:
            ghz_balise_graph = self.get_balise_nearest_metro_subgraph(balise_id, ancilla_qubit_num)
            ghz_balises = list(ghz_balise_graph.nodes)
            interchange_depth = max([self.balise_station_distance_dict[ghz_balises[i]][stations[i]]["interchange"] for i in range(ancilla_qubit_num)])
            if best_interchange_depth == None or interchange_depth < best_interchange_depth:
                best_interchange_depth = interchange_depth
                best_ghz_balises = ghz_balises
        source_physical_qubits = []
        target_physical_qubits = []
        for i in range(ancilla_qubit_num):
            source_physical_qubits.append(best_ghz_balises[i])
            target_physical_qubits.append(stations[i])
        return GhzPreparation(ancilla_qubit_num, involved_qubits, source_physical_qubits, target_physical_qubits)
    
    def on_off_board(self, logic_qubit, station, on_off_flag) -> OnOffBoard:
        '''
        Board or off-board a qubit on a stop.

        Return:
        OnOffBoard, the action to board or off-board the qubit.
        '''

        return OnOffBoard(logic_qubit, station, on_off_flag)

    def route_on_transfer_dies(self, logic_qubit, source_physical_qubit, target_physical_qubit) -> list[CircuitAction]:
        '''
        Route the qubit on transfer dies with least transfer algorithm.

        Return:
        list[CircuitAction], the action list to route the qubit on transfer dies, including TransportOnDie and TransferBetweenDies.
        '''

        actions = []
        transfer_path = self.balise_station_distance_dict[source_physical_qubit][target_physical_qubit]["path"]
        on_rail_source_qubit = transfer_path[0]
        for i in range(len(transfer_path)-1):
            if self.metro_graph.edges[transfer_path[i], transfer_path[i+1]]["role"] == RoleType.Interchange:
                if transfer_path[i] != on_rail_source_qubit:
                    actions.append(TransportOnRail(logic_qubit, on_rail_source_qubit, transfer_path[i]))
                actions.append(TransferBetweenDies(logic_qubit, (transfer_path[i], transfer_path[i+1])))
                on_rail_source_qubit = transfer_path[i+1]
        return actions

    def teleport(self, logic_qubit, platform, control_flag) -> Teleport:
        '''
        Teleport data qubit via ancilla qubit.

        Return:
        Teleport, the action to teleport the data qubit.
        '''

        return Teleport(logic_qubit, platform, control_flag)
