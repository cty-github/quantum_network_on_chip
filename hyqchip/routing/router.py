from ..element.package import *
from .circuit import *


class GhzPrepare:
    def __init__(self):
        pass


class ComputeDieRoute:
    def __init__(self):
        pass


class TransferDieRoute:
    def __init__(self):
        pass


class OnOffBoard:
    def __init__(self):
        pass


class LocalCompute:
    def __init__(self):
        pass


class Router:
    def __init__(self, package: Package, program: list):
        '''
        Class for routing a quantum program on a package.

        package: Package, the package to be routed on.
        program: list, the quantum program to be routed.

        circuit: Circuit, the routing results with circuit level actions in stages.
        '''
        self.package = package
        self.die_topology = package.die_topology
        self.program = program
        self.circuit = Circuit()
    
    def execute_move_comm(self, qubit, source, target):
        pass

    def execute_tele_comm(self, qubit, source, target):
        pass

    def execute_multi_tele_comm(self, qubit, source, target):
        pass
    
    def aggregate_comm(self):
        pass
