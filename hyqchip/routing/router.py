import copy

from ..element.package import *
from ..element.benchmark import *
from .circuit import *


class LocalBlock:
    def __init__(self, ctrl_gates, area):
        self.ctrl_gates = ctrl_gates
        self.area = area


class TeleBlock:
    def __init__(self, control_qubit, target_qubits):
        self.control_qubit = control_qubit
        self.target_qubits = target_qubits


class MoveBlock:
    def __init__(self, gate_cluster, invloved_qubits, die):
        self.gate_cluster = gate_cluster
        self.involved_qubits = invloved_qubits
        self.die = die


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
        self.qubit_topology = package.qubit_topology
        self.die_topology = package.die_topology
        self.program = program

        self.l2p_map = dict()
        self.p2l_map = dict()
        self.local_gate_freq = dict()

        self.commutable_blocks = []
        self.compute_blocks = []
        self.circuit = Circuit()

    def map_qubits(self):
        '''
        Map logic qubits to physical qubits.
        Left qubits beside the die gaps, and take rest qubits in compute dies as compute qubits.
        '''
        
        logic_data_qubit_num = self.program.data_qubit_num
        physical_data_qubit_list = self.package.get_data_qubit_list()
        if logic_data_qubit_num > len(physical_data_qubit_list):
            raise ValueError("Not enough physical qubits to map logic qubits.")
        for i in range(logic_data_qubit_num):
            self.l2p_map[i] = physical_data_qubit_list[i]
            self.p2l_map[physical_data_qubit_list[i]] = i

    @staticmethod
    def check_commutable_gate(gate, control_qubits, target_qubits) -> bool:
        '''
        Check if the gate is commutable with previous gates.
        Commutable criteria: 1) no common involved qubits; 2) share the same control qubit.
        '''

        involved_qubits = control_qubits + target_qubits
        if gate[0] not in involved_qubits and gate[1] not in involved_qubits:
            return True
        elif gate[0] in control_qubits and gate[0] not in target_qubits:
            return True
        else:
            return False
    
    def check_commutable_block(self):
        '''
        Check the commutable blocks in the program.
        Commutable criteria: 1) no common involved qubits; 2) share the same control qubit.

        self.commutable_blocks: list[list[tuple]], list of commutable blocks, each commutable blocks contains commutable control gates.
        '''

        program_copy = copy.deepcopy(self.program)
        commute_block = []
        while len(program_copy) > 0:
            gate = program_copy.pop(0)
            if gate == 'barrier':
                continue
            else:
                commute_block.append(gate)
                checked_control_qubits = [gate[0]]
                checked_target_qubits = [gate[1]]
                pop_list = []
                for i in range(len(program_copy)):
                    if program_copy[i] == 'barrier':
                        pop_list.append(i)
                        break
                    else:
                        if self.check_commutable_gate(program_copy[i], checked_control_qubits, checked_target_qubits):
                            commute_block.append(program_copy[i])
                            pop_list.append(i)
                        if program_copy[i][0] not in checked_control_qubits:
                            checked_control_qubits.append(program_copy[i][0])
                        if program_copy[i][1] not in checked_target_qubits:
                            checked_target_qubits.append(program_copy[i][1])
                for i in sorted(pop_list, reverse=True):
                    program_copy.pop(i)
                self.commutable_blocks.append(commute_block)
                commute_block = []

    def check_local_gate(self, gate) -> dict:
        '''
        Check if the gate is a local gate.
        Local criteria: 1) both control qubit and target qubit are in the same die; 2) no transfer die involved.
        '''
        
        control_area = self.package.get_physical_qubit_area(self.l2p_map[gate[0]])
        target_area = self.package.get_physical_qubit_area(self.l2p_map[gate[1]])
        if control_area != target_area:
            return None
        shortest_path, _ = self.package.get_shortest_path((self.l2p_map[gate[0]], self.l2p_map[gate[1]]))
        for node in shortest_path:
            if self.qubit_topology.nodes[node]["type"] == CellType.Transfer:
                return None
        return control_area
    
    def check_gate_insert_cluster(self, gate, gate_cluster, involved_qubits) -> bool:
        '''
        Check if the gate can be inserted into the gate cluster.
        Concentrate criteria: #gate > #qubits*(#qubits-1)//4 and #qubit < max(die_capacity)
        '''

        new_involved_qubits = involved_qubits.copy()
        if gate[0] not in new_involved_qubits:
            new_involved_qubits.append(gate[0])
        if gate[1] not in new_involved_qubits:
            new_involved_qubits.append(gate[1])

        max_die_capacity = 0
        for involved_qubit in new_involved_qubits:
            physical_qubit = self.l2p_map[involved_qubit]
            die = self.qubit_topology.nodes[physical_qubit]["die"]
            die_capacity = self.package.get_die_capacity(die)
            if die_capacity > max_die_capacity:
                max_die_capacity = die_capacity

        if len(gate_cluster)+1 < len(new_involved_qubits) * (len(new_involved_qubits) - 1) // 3:
            return False
        if len(new_involved_qubits) > max_die_capacity:
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
                    target_qubit = self.p2l_map[(i, j)]
                    if target_qubit not in stand_qubits:
                        candidate_local_freq[target_qubit] = self.local_gate_freq.get(target_qubit, 0)
        out_qubits = sorted(candidate_local_freq, key=candidate_local_freq.get)
        out_qubits = out_qubits[:len(in_qubits)]
        self.local_gate_freq = dict()
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
        for i in len(in_qubits):
            in_physical_qubit = self.l2p_map[in_qubits[i]]
            out_physical_qubit = self.l2p_map[out_qubits[i]]
            self.l2p_map[in_qubits[i]] = out_physical_qubit
            self.l2p_map[out_qubits[i]] = in_physical_qubit
            self.p2l_map[out_physical_qubit] = in_qubits[i]
            self.p2l_map[in_physical_qubit] = out_qubits[i]

    def scan_local_block(self, commute_block):
        '''
        Scan the commutable block and find local blocks.
        '''

        pop_list = []
        local_block_dict = dict()
        for i in range(len(commute_block)):
            area = self.check_local_gate(commute_block[i])
            if area:
                if area in local_block_dict:
                    local_block_dict[area].append(commute_block[i])
                else:
                    local_block_dict[area] = [commute_block[i]]
                if commute_block[i][0] not in self.local_gate_freq:
                    self.local_gate_freq[commute_block[i][0]] = 1
                else:
                    self.local_gate_freq[commute_block[i][0]] += 1
                if commute_block[i][1] not in self.local_gate_freq:
                    self.local_gate_freq[commute_block[i][1]] = 1
                else:
                    self.local_gate_freq[commute_block[i][1]] += 1
                pop_list.append(i)
        local_blocks = []
        for key, value in local_block_dict.items():
            local_block = LocalBlock(value, key)
            local_blocks.append(local_block)
        return local_blocks, pop_list
    
    def scan_tele_block(self, commute_block):
        '''
        Scan the commutable block and find tele blocks.
        '''

        tele_blocks = []
        pop_list = []
        while len(commute_block) > 0:
            gate = commute_block.pop(0)
            ctrl_qubit = gate[0]
            common_ctrl_target_qubits = [gate[1]]
            for i in range(len(commute_block)):
                if commute_block[i][0] == ctrl_qubit:
                    common_ctrl_target_qubits.append(commute_block[i][1])
                    pop_list.append(i)
            for i in sorted(pop_list, reverse=True):
                commute_block.pop(i)
            tele_block = TeleBlock(ctrl_qubit, common_ctrl_target_qubits)
            tele_blocks.append(tele_block)
        return tele_blocks

    def extend_to_tele_block(self, tele_block, commute_block):
        '''
        Extend the tele block with next commute block as tele block to include more gates.
        '''

        ctrl_qubit = tele_block.control_qubit
        extend_list = []
        for i in range(len(commute_block)):
            if self.check_local_gate(commute_block[i]):
                continue
            if commute_block[i][0] == ctrl_qubit:
                extend_list.append(i)
        return extend_list

    def extend_to_move_block(self, tele_block, commute_block):
        '''
        Extend the tele block with next commute block as move block to include more gates.
        '''

        gate_cluster = [(tele_block.control_qubit, _) for _ in tele_block.target_qubits]
        involved_qubits = [tele_block.control_qubit] + tele_block.target_qubits
        extend_list = []
        for i in range(len(commute_block)):
            if self.check_local_gate(commute_block[i]):
                continue
            if self.check_gate_insert_cluster(commute_block[i], gate_cluster, involved_qubits):
                extend_list.append(i)
                gate_cluster.append(commute_block[i])
                if commute_block[i][0] not in involved_qubits:
                    involved_qubits.append(commute_block[i][0])
                if commute_block[i][1] not in involved_qubits:
                    involved_qubits.append(commute_block[i][1])
        return extend_list, gate_cluster, involved_qubits

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

        commute_blocks_copy = copy.deepcopy(self.commutable_blocks)
        last_tele_blocks = None
        while len(commute_blocks_copy) > 0:
            commute_block = commute_blocks_copy.pop(0)

            if last_tele_blocks:
                max_extend_list = []
                max_extend_block_index = -1
                max_extend_type = None
                max_gate_cluster = None
                max_involved_qubits = None
                for i in len(last_tele_blocks):
                    extend_list_tele = self.extend_to_tele_block(last_tele_blocks[i], commute_block)
                    if len(extend_list_tele) > len(max_extend_list):
                        max_extend_list = extend_list_tele
                        max_extend_block_index = i
                        max_extend_type = 'tele'
                    extend_list_move, gate_cluster, involved_qubits = self.extend_to_move_block(last_tele_blocks[i], commute_block)
                    if len(extend_list_move) > len(max_extend_list):
                        max_extend_list = extend_list_move
                        max_extend_block_index = i
                        max_extend_type = 'move'
                        max_gate_cluster = gate_cluster
                        max_involved_qubits = involved_qubits
                for i in len(last_tele_blocks):
                    if i != max_extend_block_index:
                        self.compute_blocks.append(last_tele_blocks[i])
                if len(max_extend_list) > 0:
                    if max_extend_type == 'tele':
                        tele_block = last_tele_blocks[max_extend_block_index]
                        tele_block.target_qubits.extend([commute_block[i][1] for i in max_extend_list])
                        self.compute_blocks.append(tele_block)
                        for i in sorted(max_extend_list, reverse=True):
                            commute_block.pop(i)
                    elif max_extend_type == 'move':
                        die = self.determine_die(max_involved_qubits)
                        move_block = MoveBlock(max_gate_cluster, max_involved_qubits, die)
                        self.compute_blocks.append(move_block)
                        for i in sorted(max_extend_list, reverse=True):
                            commute_block.pop(i)
                        self.move_qubits_in_block(max_involved_qubits, die)
                    
            local_blocks, pop_list = self.scan_local_block(commute_block)
            for i in sorted(pop_list, reverse=True):
                commute_block.pop(i)
            self.compute_blocks.extend(local_blocks)
            last_tele_blocks = self.scan_tele_block(commute_block)

        for i in range(len(last_tele_blocks)):
            self.compute_blocks.append(last_tele_blocks[i])

    def schedule_compute_blocks(self):
        '''
        Transform compute blocks into circuit actions.
        '''

        # TODO: complete the function
        pass

    def schedule_local_block(self, local_block):
        '''
        Transform local block into circuit actions.
        1) route local gates -> LocalCompute.
        Insert actions into earliest available stage.
        LocalCompute.
        '''

        # TODO: complete the function
        pass
    
    def schedule_tele_block(self, tele_block):
        '''
        Transform tele block into circuit actions.
        1) route nearest stop for all involved qubits -> RouteToStop;
        2) allocate GHZ resource -> GhzPreparation;
        3) route on transfer dies -> TransferDieRoute.
        4) on/off board qubits -> OnOffBoard;
        5) teleport qubits -> Teleport.

        Insert actions into earliest available stage.
        GhzPreparation -> TransferDieRoute & RouteToStop -> OnOffBoard -> Teleport.
        '''

        # TODO: complete the function
        pass

    def schedule_move_block(self, move_block):
        '''
        Transform move block into circuit actions.
        1) find qubit to swap;
        2) route nearest stop for all involved qubits -> RouteToStop;
        3) on/off board qubits -> OnOffBoard;
        3) route on transfer dies -> TransferDieRoute.
        4) route local gates -> LocalCompute.

        Insert actions into earliest available stage.
        RouteToStop -> OnOffBoard -> TransferDieRoute -> OnOffBoard -> LocalCompute.
        '''

        # TODO: complete the function
        pass

    def route_nearest_stop(self, logic_qubit, enter_flag):
        '''
        Find the nearest unoccupied stop for a logic qubit and route to it.

        Return:
        RouteToStop, the action to route the qubit to the stop.
        '''

        # TODO: complete the function
        pass

    def allocate_ghz_resource(self, involved_qubits, stops):
        '''
        Allocate GHZ resource for a list of logic qubits for teleportation.
        Ensure the total transfer distance is minimized.

        Return:
        GhzPreparation, the action to allocate GHZ resource.
        '''

        # TODO: complete the function
        pass
    
    def on_off_board(self, logic_qubit, stop, on_off_flag):
        '''
        Board or off-board a qubit on a stop.

        Return:
        OnOffBoard, the action to board or off-board the qubit.
        '''

        # TODO: complete the function
        pass

    def route_on_transfer_dies(self, source_qubit, target_qubit):
        '''
        Route the qubit on transfer dies with least transfer algorithm.

        Return:
        TransferDieRoute, the action list to route the qubit on transfer dies, interleaved with TransportOnDie and TransferBetweenDies.
        '''

        # TODO: complete the function
        pass

    def teleport(self, data_qubit, ancilla_qubit, stop, control_flag):
        '''
        Teleport data qubit via ancilla qubit.

        Return:
        Teleport, the action to teleport the data qubit.
        '''

        # TODO: complete the function
        pass

    def route_local_gates(self, local_gates):
        '''
        Route local gates on the same die.

        Return:
        LocalCompute, the action to route local gates.
        '''

        # TODO: complete the function
        pass
