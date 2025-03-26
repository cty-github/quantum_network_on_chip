from ..element.package import *
from ..element.benchmark import *
from .circuit import *


class CommutableBlock:
    def __init__(self):
        self.cnot_gate_list = []
        self.control_qubits = []
        self.target_qubits = []


class LocalBlock:
    def __init__(self, cnot_gates_list):
        self.cnot_gates_list = cnot_gates_list


class TeleBlock:
    def __init__(self, control_qubit, target_qubits):
        self.control_qubit = control_qubit
        self.target_qubits = target_qubits


class MoveBlock:
    def __init__(self, move_gates_list):
        self.move_gates_list = move_gates_list
        self.involved_qubits = []
        for move_gate in move_gates_list:
            if move_gate[0] not in self.involved_qubits:
                self.involved_qubits.append(move_gate[0])
            if move_gate[1] not in self.involved_qubits:
                self.involved_qubits.append(move_gate[1])


class Router:
    def __init__(self, package: Package, program: list):
        '''
        Class for routing a quantum program on a package.

        package: Package, the package to be routed on.
        program: list, the quantum program to be routed.

        l2p_map: dict, map from logic qubit index to physical qubit index
        p2l_map: dict, map from physical qubit index to logic qubit index
        circuit: Circuit, the routing results with circuit level actions in stages.
        '''
        self.package = package
        self.die_topology = package.die_topology
        self.program = program

        self.l2p_map = dict()
        self.p2l_map = dict()

        self.commutable_blocks = []
        self.compute_blocks = []
        self.circuit = Circuit()

    def map_qubits(self):
        '''
        Map logic qubits to physical qubits.
        Left qubits beside the die gaps, and take rest qubits in compute dies as compute qubits.
        '''
        # TODO: complete the function
        pass

    def check_commutable_block(self):
        '''
        Check the commutable blocks in the program.
        Commutable criteria: 1) no common involved qubits; 2) share the same control qubit.

        self.commutable_blocks: list[list[tuple]], list of commutable blocks, each commutable blocks contains commutable control gates.
        '''
        # TODO: complete the function
        pass

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
        # TODO: complete the function
        pass

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

    def find_qubit_to_swap(self, logic_qubit):
        '''
        Find the logic qubit on target die to replace, which has less control gate with qubits in the same die.
        
        Return:
        int, the logic qubit on target die to replace.
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
