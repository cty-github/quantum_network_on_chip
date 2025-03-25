from ..element.package import *

class Router:
    def __init__(self, package: Package):
        self.package = package
        self.die_topology = package.die_topology
