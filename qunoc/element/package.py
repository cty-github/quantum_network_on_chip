import math
import random
from copy import deepcopy
from enum import Enum, auto

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from ..utils.utils import color_pool, get_full_space_tuple


class NodeType(Enum):
    """
    Enumerate of the types of the node in the package graph, each node corresponds to a potential qubit.
    """

    Empty = 0
    Compute = auto()
    Transfer = auto()

    def __repr__(self):
        return self.name

class EdgeType(Enum):
    """
    Enumerate of the types of the edge between two nodes, each edge corresponds to a potential coupler.
    """

    Empty = 0
    Intra = auto()
    Inter = auto()

    def __repr__(self):
        return self.name


class QubitRole(Enum):
    """
    Enumerate of the roles of the qubit in the package for routing.
    """

    Data = 0
    Platform = auto()
    Balise = auto()
    Station = auto()

    def __repr__(self):
        return self.name

class CouplerRole(Enum):
    """
    Enumerate of the roles of the coupler in the package for routing.
    """

    Rail = 0
    Interchange = auto()

    def __repr__(self):
        return self.name


class DesignTemplate(Enum):
    """
    Enumerate of predetermined templates used in layout design.
    """

    Strip = 0
    Radial = auto()
    Grid = auto()

    def __repr__(self):
        return self.name

class DesignOrientation(Enum):
    """
    Enumerate of orientations used in layout design.
    """

    X = 0
    Y = auto()

    def __repr__(self):
        return self.name


class FabricateModel:
    """
    Represents parameters of chiplet fabrication.

    Attributes:
        max_capacity (int): The max number of qubits in a chiplet.
        max_square_size (int): The size of max square chiplet.
        max_length (int): The max length of a qubit chain in the chiplet.
        compute_geometry (str): The qubit topology in a chiplet, i.e. 'square', 'hexagon', 'heavy square', 'heavy hexagon'.
        inter_sparsity (int): The sparsity of coupler between chiplets.
        single_fide (float): The fidelity of single gate.
        intra_fide_params (tuple[float, float]): The parameters of the intra_fide function.
        intra_fide_log_decay (function): The function to calculate intra_fide.
        enhanced_fide (float): The unit fidelity of enhanced qst.
        inter_fide (float): The fidelity of inter-chiplet CZ.
        yield_5 (float): The yield of 5-qubit chiplet.
        yield_decay (float): The decay rate of yield with the qubit number.
    """

    def __init__(self,
        max_capacity:int=100,
        max_length:int=50,
        compute_geometry:str="square",
        inter_sparsity:int=3,
        single_fide:float=0.9999,
        intra_fide_dict:dict[int,float]={27: 0.993, 65: 0.989, 127: 0.988},
        enhanced_fide:float=0.9999,
        inter_fide:float=0.99,
        yield_5:float=0.99,
        yield_decay:float=2,
    ):
        """
        Initialize a fabrication model.

        Args:
            max_capacity (int): The max number of qubits in a chiplet.
            max_length (int): The max length of a qubit chain in the chiplet.
            compute_geometry (str): The qubit topology in a chiplet, i.e. 'square', 'hexagon', 'heavy square', 'heavy hexagon'.
            inter_sparsity (int): The sparsity of coupler between chiplets.
            single_fide (float): The fidelity of single gate.
            intra_fide_dict (dict[int,float]): The fidelity of intra-chiplet CZ according to the qubit number.
            enhanced_fide (float): The unit fidelity of enhanced qst.
            inter_fide (float): The fidelity of inter-chiplet CZ.
            yield_5 (float): The yield of 5-qubit chiplet.
            yield_decay (float): The decay rate of yield with the qubit number.
        """
        self.max_capacity = max_capacity
        self.max_square_size = int(np.sqrt(max_capacity))
        self.max_length = max_length
        self.compute_geometry = compute_geometry
        self.inter_sparsity = inter_sparsity

        self.single_fide = single_fide
        def log_decay(x, a, b):
            return 1 - a * np.log(1 + b * x)
        self.intra_fide_params, _ = curve_fit(
            log_decay,
            np.array(list(intra_fide_dict.keys())),
            np.array(list(intra_fide_dict.values())),
            p0=[0.01, 0.01],
        )
        self.intra_fide_log_decay = log_decay
        self.enhanced_fide = enhanced_fide
        self.inter_fide = inter_fide

        self.yield_5 = yield_5
        self.yield_decay = yield_decay

    def get_intra_fide(self, qubit_num:int) -> float:
        """
        Get the fidelity of intra-chiplet CZ according to qubit num in the chiplet.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.
        
        Returns:
            float: The fidelity of intra-chiplet CZ.
        """
        return self.intra_fide_log_decay(qubit_num, *self.intra_fide_params)

    def get_yield(self, qubit_num:int) -> float:
        """
        Get the yield of the chiplet according to the qubit num.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.
        
        Returns:
            float: The yield of the chiplet.
        """
        return (self.yield_5 * (1 - np.tanh(self.yield_decay * (np.log10(qubit_num) - 2))) / 2)

    def get_cost(self, qubit_num:int) -> float:
        """
        Get the cost of the chiplet according to the qubit num.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.

        Returns:
            float: The cost of the chiplet.
        """
        return qubit_num / self.get_yield(qubit_num)

class Qubit:
    """
    Represents the physical qubit in the package.

    Attributes:
        coordinate (tuple[int, int]): The coordinate of the qubit.
        type (NodeType): The type of the chiplet that the qubit belongs to.
        fide (float): The single gate fidelity of the qubit.
        chiplet (Chiplet): The id of the chiplet that the qubit belongs to.
        role (QubitRole): The role of the qubit.
    """
    
    def __init__(self, coordinate:tuple[int,int], type:NodeType, fide:float):
        """
        Initialize a physcial qubit.

        Args:
            coordinate (tuple[int, int]): The coordinate of the qubit.
            type (NodeType): The type of the chiplet that the qubit belongs to.
            fide (float): The single gate fidelity of the qubit.
        """
        self.coordinate = coordinate
        self.type = type
        self.fide = fide
        self.chiplet: Chiplet = None
        self.role: QubitRole = None

    def __repr__(self):
        return f"{self.__class__.__name__}{self.coordinate}"

class Coupler:
    """
    Represents the coupler between two qubits in the package.
    
    Attributes:
        qubit_u (Qubit): The qubit u on one side of the coupler.
        qubit_v (Qubit): The qubit v on the other side of the coupler.
        type (EdgeType): The type of the coupler.
        fide (float): The CZ/EQST fidelity of the coupler.
        role (CouplerRole): The role of the coupler.
        gap (Gap): The id of the gap that the coupler belongs to.
    """
    
    def __init__(self, qubit_u:Qubit, qubit_v:Qubit, type:EdgeType, fide:float):
        """
        Initialize a coupler.

        Args:
            qubit_u (Qubit): The qubit u on one side of the coupler.
            qubit_v (Qubit): The qubit v on the other side of the coupler.
            type (EdgeType): The type of the coupler.
            fide (float): The CZ/EQST fidelity of the coupler.
        """
        self.qubit_u = qubit_u
        self.qubit_v = qubit_v
        self.type = type
        self.fide = fide
        self.gap: Gap = None
        self.role: CouplerRole = None

    def __repr__(self):
        return f"{self.__class__.__name__}{self.qubit_u.coordinate}{self.qubit_v.coordinate}"


class FuncArea:
    """
    Represents the function area in the package.

    Attributes:
        type (NodeType): The type of the function area.
        nodes (list[tuple[int, int]]): The list of nodes in the function area.
    """

    def __init__(self, type:NodeType):
        """
        Initialize a function area.

        Args:
            type (NodeType): The type of the function area.
        """
        self.type = type
    
    def get_nodes(self) -> list[tuple[int,int]]:
        """
        Get the nodes in the function region.

        Returns:
            nodes (list[tuple[int, int]]): The list of nodes in the function region.
        """
    
    def in_area(self, node:tuple[int,int]) -> bool:
        """
        Check if the node is in the function area.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            bool: True if the node is in the function area, False otherwise.
        """
        return node in self.get_nodes()

class ComputeModule(FuncArea):
    """
    Represents the compute module in the package.

    Attributes:
        left (int): The left boundary of the module.
        top (int): The top boundary of the module.
        right (int): The right boundary of the module.
        bottom (int): The bottom boundary of the module.
    """
    
    def __init__(self, left:int, top:int, right:int, bottom:int):
        """
        Initialize a compute module.

        Args:
            left (int): The left boundary of the module.
            top (int): The top boundary of the module.
            right (int): The right boundary of the module.
            bottom (int): The bottom boundary of the module.
        """
        super().__init__(NodeType.Compute)
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left}_{self.top}_{self.right}_{self.bottom}"
    
    def get_nodes(self) -> list[tuple[int,int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
        return self.nodes

class TransferLine(FuncArea):
    """
    Represents the transfer line in the package.

    Attributes:
        axis (int): The axis of the line.
        pos (int): The position of the line.
        start (int): The start point of the line.
        end (int): The end point of the line.
    """

    def __init__(self, axis:int, pos:int, start:int, end:int):
        """
        Initialize a transfer line.

        Args:
            axis (int): The axis of the line.
            pos (int): The position of the line.
            start (int): The start point of the line.
            end (int): The end point of the line.
        """
        super().__init__(NodeType.Transfer)
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start}_{self.end}"
    
    def get_nodes(self) -> list[tuple[int,int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start, self.end + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes


class Chiplet:
    """
    Represents the chiplet in the package.

    Attributes:
        type (NodeType): The type of the chiplet.
    """

    def __init__(self, type:NodeType):
        """
        Initialize a chiplet.

        Args:
            type (NodeType): The type of the chiplet.
        """
        self.type = type
    
    def get_nodes(self) -> list[tuple[int,int]]:
        """
        Get the nodes in the chiplet.

        Returns:
            nodes (list[tuple[int, int]]): the list of nodes in the chiplet
        """

    def get_capacity(self) -> int:
        """
        Get the capacity of the chiplet.

        Returns:
            capacity (int): the capacity of the chiplet
        """
    
    def get_coordinate(self) -> tuple[float,float]:
        """
        Get the center coordinate of the chiplet.

        Returns:
            coordinate (tuple[float, float]): the center coordinate of the chiplet
        """

class ComputeChiplet(Chiplet):
    """
    Represents the compute chiplet in the package.

    Attributes:
        left (int): The left boundary of the chiplet.
        top (int): The top boundary of the chiplet.
        right (int): The right boundary of the chiplet.
        bottom (int): The bottom boundary of the chiplet.
    """

    def __init__(self, left:int, top:int, right:int, bottom:int):
        """
        Initialize a compute chiplet.

        Args:
            left (int): The left boundary of the chiplet.
            top (int): The top boundary of the chiplet.
            right (int): The right boundary of the chiplet.
            bottom (int): The bottom boundary of the chiplet.
        """
        super().__init__(NodeType.Compute)
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left}_{self.top}_{self.right}_{self.bottom}"
    
    def get_nodes(self) -> list[tuple[int,int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
        return self.nodes
    
    def get_capacity(self) -> int:
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)
    
    def get_coordinate(self) -> tuple[float,float]:
        return (
            (self.left + self.right) / 2,
            (self.top + self.bottom) / 2,
        )

class TransferChiplet(Chiplet):
    """
    Represents the transfer chiplet in the package.
    
    Attributes:
        axis (int): The axis of the chiplet.
        pos (int): The position of the chiplet.
        start (int): The start point of the chiplet.
        end (int): The end point of the chiplet.
    """

    def __init__(self, axis:int, pos:int, start:int, end:int):
        """
        Initialize a transfer chiplet.

        Args:
            axis (int): The axis of the chiplet.
            pos (int): The position of the chiplet.
            start (int): The start point of the chiplet.
            end (int): The end point of the chiplet.
        """
        super().__init__(NodeType.Transfer)
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start}_{self.end}"
    
    def get_nodes(self) -> list[tuple[int,int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start, self.end + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes
    
    def get_capacity(self) -> int:
        return self.end - self.start + 1

    def get_coordinate(self) -> tuple[float,float]:
        return get_full_space_tuple(self.pos, self.axis, (self.start + self.end) / 2)

class Gap:
    """
    Represents the gap between two chiplets in the package.

    Attributes:
        chiplet_u (Chiplet): The chiplet u on one side of the gap.
        chiplet_v (Chiplet): The chiplet v on the other side of the gap.
        axis (int): The axis of the gap.
        edges (dict[int, tuple[tuple[int,int], tuple[int,int]]]): The edges in the gap.
    """

    def __init__(self, chiplet_u:Chiplet, chiplet_v:Chiplet, axis:int, edges:dict[int,tuple[tuple[int,int],tuple[int,int]]]):
        """
        Initialize a gap.

        Args:
            chiplet_u (Chiplet): The chiplet u on one side of the gap.
            chiplet_v (Chiplet): The chiplet v on the other side of the gap.
            axis (int): The axis of the gap.
            edges (dict[int, tuple[tuple[int,int], tuple[int,int]]]): The edges in the gap.
        """
        self.chiplet_u = chiplet_u
        self.chiplet_v = chiplet_v
        self.axis = axis
        self.edges = edges

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.chiplet_u}_{self.chiplet_v}"


class Package:
    """
    Represents the package of the quantum chiplet architecture (quantum network-on-chip architecture).

    Attributes:
        package_size (tuple[int, int]): The size of the package.
        node_num (int): The number of nodes in the package.
        fabricate_model (FabricateModel): The fabrication model of the package.
        design_scheme (str): The design scheme of the package, i.e. "top_down", "bottom_up".
        design_strategy (str): The design strategy of the package, i.e. "heuristic", "greedy", "rl".
        top_down_max_step (int): The max number of steps for top down design.

        package_graph (nx.Graph): The graph recording the grid structure of the package.
            node: tuple[int, int], {"type": NodeType, "qubit": Qubit}
            edge: tuple[node, node], {"type": EdgeType, "coupler": Coupler}
        compute_nodes (set[tuple[int, int]]): The set of nodes which are on the compute chiplets.
        transfer_nodes (set[tuple[int, int]]): The set of nodes which are on the transfer chiplets.
        compute_modules (list[ComputeModule]): The list of modules which are padded with compute chiplets.
        transfer_lines (list[TransferLine]): The list of lines which are padded with transfer chiplets.
        compute_chiplets (list[ComputeChiplet]): The list of chiplets, which are used for compute.
        transfer_chiplets (list[TransferChiplet]): The list of chiplets, which are used for transfer.
        gap_dict (dict[str, Gap]): The gaps between chiplets in the package.

        qubit_topology (nx.Graph): The subgraph view of package_graph recording the topology of physical qubits in the package.
        metro_topology (nx.Graph): The subgraph view of package_graph recording the topology of transfer qubits in the package.
        chiplet_topology (nx.Graph): The graph recording chiplet topology in the package.
            node: str, {"chiplet": Chiplet}
            edge: tuple[str, str], {"gap": Gap}
    """

    def __init__(self,
        package_size:tuple[int,int]=(50, 50),
        fabricate_model:FabricateModel=FabricateModel(),
        design_scheme:str="top_down",
        design_strategy:str="heuristic",
        top_down_max_step:int=10,
        bottom_up_init_type:np.ndarray=None,
    ):
        """
        Initialize a package.

        Args:
            package_size (tuple[int, int]): The size of the package.
            fabricate_model (FabricateModel): The fabrication model of the package.
            design_scheme (str): The design scheme of the package, i.e. "top_down", "bottom_up".
            design_strategy (str): The design strategy of the package, i.e. "heuristic", "greedy", "rl".
            top_down_max_step (int): The max number of steps for top down design.
            bottom_up_init_type (np.ndarray): The initial type of nodes used for bottom up design.
        """
        self.package_size = package_size
        self.node_num = package_size[0] * package_size[1]
        self.fabricate_model = fabricate_model
        self.design_scheme = design_scheme
        self.design_strategy = design_strategy
        self.top_down_max_step = top_down_max_step

        self.package_graph: nx.Graph = nx.grid_2d_graph(*self.package_size)
        self.compute_nodes: set[tuple[int, int]] = set()
        self.transfer_nodes: set[tuple[int, int]] = set()

        # determine compute rectangles and transfer lines
        if self.design_scheme == "bottom_up":
            if bottom_up_init_type is None:
                bottom_up_init_type = np.zeros(self.package_size, dtype=np.int8)
            else:
                assert bottom_up_init_type.shape == self.package_size
            for node in self.package_graph.nodes:
                self.package_graph.nodes[node]["type"] = NodeType(bottom_up_init_type[node[0], node[1]])
                if bottom_up_init_type[node[0], node[1]]:
                    self.package_graph.nodes[node]["qubit"] = Qubit(node, self.package_graph.nodes[node]["type"], self.fabricate_model.single_fide)
                    if bottom_up_init_type[node[0], node[1]] == NodeType.Compute.value:
                        self.compute_nodes.add(node)
                    elif bottom_up_init_type[node[0], node[1]] == NodeType.Transfer.value:
                        self.transfer_nodes.add(node)
                else:
                    self.package_graph.nodes[node]["qubit"] = None
            self.design_qunoc_bottom_up()
            self.compute_modules, self.transfer_lines = self.check_func_area()
        elif self.design_scheme == "top_down":
            self.compute_modules = [ComputeModule(0, 0, package_size[0] - 1, package_size[1] - 1)]
            self.transfer_lines: list[TransferLine] = []
            if self.design_strategy == "heuristic":
                old_compute_module = self.compute_modules[0]
                length_width_ratio = max(self.package_size) / min(self.package_size)
                if length_width_ratio > 1:
                    if self.package_size[0] > self.package_size[1]:
                        new_compute_modules, new_transfer_lines = self.design_module_template(
                            old_compute_module,
                            DesignTemplate.Strip,
                            DesignOrientation.Y,
                            min(self.package_size[1] - 1, (self.package_size[0] + 1) // 2 - 1),
                        )
                    else:
                        new_compute_modules, new_transfer_lines = self.design_module_template(
                            old_compute_module,
                            DesignTemplate.Strip,
                            DesignOrientation.X,
                            min(self.package_size[0] - 1, (self.package_size[1] + 1) // 2 - 1),
                        )
                    self.compute_modules.remove(old_compute_module)
                    self.compute_modules.extend(new_compute_modules)
                    self.transfer_lines.extend(new_transfer_lines)
                    old_compute_module = self.compute_modules[len(self.compute_modules) // 2]
                grid_round = len(self.compute_modules) - 1
                radial_round = (
                    min(self.package_size) - 2 * self.fabricate_model.max_square_size
                ) // (4 * self.fabricate_model.max_square_size)
                for i in range(radial_round):
                    new_compute_modules, new_transfer_lines = self.design_module_template(
                        old_compute_module,
                        DesignTemplate.Radial,
                        DesignOrientation.X,
                        2 * self.fabricate_model.max_square_size - 1,
                    )
                    self.compute_modules.remove(old_compute_module)
                    self.compute_modules.extend(new_compute_modules)
                    self.transfer_lines.extend(new_transfer_lines)
                    old_compute_module = self.compute_modules[-1]
                for i in range(grid_round):
                    old_compute_module = self.compute_modules[0]
                    if self.package_size[0] > self.package_size[1]:
                        new_compute_modules, new_transfer_lines = self.design_module_template(
                            old_compute_module,
                            DesignTemplate.Grid,
                            DesignOrientation.X,
                            min(
                                3 * self.fabricate_model.max_square_size - 1,
                                (min(self.package_size[1], old_compute_module.right - old_compute_module.left + 1) + 1) // 2 - 1,
                            ),
                        )
                    else:
                        new_compute_modules, new_transfer_lines = self.design_module_template(
                            old_compute_module,
                            DesignTemplate.Grid,
                            DesignOrientation.Y,
                            min(
                                3 * self.fabricate_model.max_square_size - 1,
                                (min(self.package_size[0], old_compute_module.bottom - old_compute_module.top + 1) + 1) // 2 - 1,
                            ),
                        )
                    self.compute_modules.remove(old_compute_module)
                    self.compute_modules.extend(new_compute_modules)
                    self.transfer_lines.extend(new_transfer_lines)
            elif self.design_strategy == "greedy":
                # TODO: Implement greedy strategy
                raise NotImplementedError("Greedy strategy is not implemented yet.")
            elif self.design_strategy == "rl":
                # TODO: Implement reinforcement learning strategy
                raise NotImplementedError("Reinforcement learning strategy is not implemented yet.")
            for compute_module in self.compute_modules:
                for node in compute_module.get_nodes():
                    self.package_graph.nodes[node]["type"] = compute_module.type
                    self.package_graph.nodes[node]["qubit"] = Qubit(node, compute_module.type, self.fabricate_model.single_fide)
                    self.compute_nodes.add(node)
            for transfer_line in self.transfer_lines:
                for node in transfer_line.get_nodes():
                    self.package_graph.nodes[node]["type"] = transfer_line.type
                    self.package_graph.nodes[node]["qubit"] = Qubit(node, transfer_line.type, self.fabricate_model.single_fide)
                    self.transfer_nodes.add(node)
        else:
            raise ValueError("Invalid design scheme: {}".format(design_scheme))

        self.complete_package()

    def design_qunoc_bottom_up(self):
        """
        Determine the transfer nodes to design the package.
        """
        if self.design_strategy == "heuristic":
            for node in self.package_graph.nodes:
                if node[1] == 0 or node[1] == self.package_size[1] - 1:
                    self.package_graph.nodes[node]["type"] = NodeType.Transfer
                else:
                    self.package_graph.nodes[node]["type"] = NodeType.Compute
        elif self.design_strategy == "greedy":
            raise NotImplementedError("Greedy strategy is not implemented yet.")
        elif self.design_strategy == "rl":
            raise NotImplementedError("Reinforcement learning strategy is not implemented yet.")
        else:
            raise ValueError("Invalid design strategy: {}".format(self.design_strategy))

    def update_node(self, node:tuple[int,int], new_type:NodeType):
        """
        Update the node type in the package.

        Args:
            node (tuple[int, int]): The coordinate of the node.
            new_type (NodeType): The new type of the node.
        """
        if self.package_graph.nodes[node]["type"] == new_type:
            return
        self.package_graph.nodes[node]["type"] = new_type
        if new_type:
            self.package_graph.nodes[node]["qubit"].type = new_type

        if new_type == NodeType.Compute:
            self.compute_nodes.add(node)
            self.transfer_nodes.remove(node)
        elif new_type == NodeType.Transfer:
            self.transfer_nodes.add(node)
            self.compute_nodes.remove(node)

    def update_all_nodes(self, new_type_matrix:np.ndarray):
        """
        Update all the node types in the package.

        Args:
            new_type_matrix (np.ndarray): The new type matrix of the package.
        """
        self.compute_nodes.clear()
        self.transfer_nodes.clear()
        for node in self.package_graph.nodes:
            self.package_graph.nodes[node]["type"] = NodeType(new_type_matrix[node[0], node[1]])
            if new_type_matrix[node[0], node[1]]:
                self.package_graph.nodes[node]["qubit"].type = self.package_graph.nodes[node]["type"]
            if new_type_matrix[node[0], node[1]] == NodeType.Compute.value:
                self.compute_nodes.add(node)
            elif new_type_matrix[node[0], node[1]] == NodeType.Transfer.value:
                self.transfer_nodes.add(node)

    def check_func_area(self) -> tuple[list[ComputeModule], list[TransferLine]]:
        """
        Find all the function areas in the package greedily.

        Returns:
            compute_modules (list[ComputeModule]): The list of compute modules.
            transfer_lines (list[TransferLine]): The list of transfer lines.
        """
        compute_modules = []
        transfer_lines = []
        check_flags = np.zeros(self.package_size, dtype=bool)
        for node in self.package_graph.nodes:
            if not check_flags[node[0], node[1]]:
                if self.package_graph.nodes[node]["type"] == NodeType.Empty:
                    check_flags[node[0], node[1]] = True
                    continue
                area_type = self.package_graph.nodes[node]["type"]
                area_left = node[0]
                area_top = node[1]
                area_right = node[0]
                area_bottom = node[1]
                check_flags[node[0], node[1]] = True
                for i in range(node[0] + 1, self.package_size[0]):
                    if self.package_graph.nodes[(i, node[1])]["type"] == area_type:
                        area_right = i
                        check_flags[i, node[1]] = True
                    else:
                        break
                for j in range(node[1] + 1, self.package_size[1]):
                    consec_flag = True
                    for i in range(area_left, area_right + 1):
                        if self.package_graph.nodes[(i, j)]["type"] != area_type:
                            consec_flag = False
                            break
                    if consec_flag:
                        area_bottom = j
                        for i in range(area_left, area_right + 1):
                            check_flags[i, j] = True
                    else:
                        break
                if area_type == NodeType.Compute:
                    compute_modules.append(ComputeModule(area_left, area_top, area_right, area_bottom))
                elif area_type == NodeType.Transfer:
                    if area_bottom == area_top:
                        transfer_lines.append(TransferLine(0, area_top, area_left, area_right))
                    elif area_right == area_left:
                        transfer_lines.append(TransferLine(1, area_left, area_top, area_bottom))
                    else:
                        raise ValueError("Invalid transfer line: left_{}, top_{}, right_{}, bottom_{}".format(area_left, area_top, area_right, area_bottom))
        return compute_modules, transfer_lines

    @staticmethod
    def is_valid_to_design( 
        module:ComputeModule, 
        template:DesignTemplate, 
        orient:DesignOrientation, 
        interval:int
    ) -> bool:
        """
        Check if the module is valid to design with the given template and arguments.

        Args:
            module (ComputeModule): The module to be designed.
            template (DesignTemplate): The template to be used.
            orient (DesignOrientation): The orientation of the template.
            interval (int): The interval between the compute modules.

        Returns:
            bool: Whether the module is valid to design.
        """
        if module.type != NodeType.Compute:
            raise ValueError("Invalid module type: {}".format(module.type))
        if not isinstance(template, DesignTemplate):
            raise ValueError("Invalid template type: {}".format(template))
        if not isinstance(orient, DesignOrientation):
            raise ValueError("Invalid orientation type: {}".format(orient))
        if interval < 1:
            return False
        if template == DesignTemplate.Strip:
            if orient == DesignOrientation.X:
                if module.bottom - module.top + 1 < 2 * interval + 1:
                    return False
            elif orient == DesignOrientation.Y:
                if module.right - module.left + 1 < 2 * interval + 1:
                    return False
        elif template == DesignTemplate.Radial:
            if module.bottom - module.top + 1 < 3 * interval + 2:
                return False
            if module.right - module.left + 1 < 3 * interval + 2:
                return False
        elif template == DesignTemplate.Radial:
            if module.bottom - module.top + 1 < 2 * interval + 1:
                return False
            if module.right - module.left + 1 < 2 * interval + 1:
                return False
        return True

    @staticmethod
    def design_module_template(
        module:ComputeModule, 
        template:DesignTemplate, 
        orient:DesignOrientation, 
        interval:int, 
        check_valid:bool=True
    ) -> tuple[list[ComputeModule], list[TransferLine]]:
        """
        Design the module with the given template and arguments.
        
        Args:
            module (ComputeModule): The module to be designed.
            template (DesignTemplate): The template to be used.
            orient (DesignOrientation): The orientation of the template.
            interval (int): The interval between the compute modules.
            check_valid (bool): Whether to check if the module is valid to design.
        
        Returns:
            tuple[list[ComputeModule], list[TransferLine]]: The new compute modules and transfer lines.
        """
        if check_valid:
            assert Package.is_valid_to_design(module, template, orient, interval), (
                f"Invalid module to design:\nmodule:{module}\ntemplate:{template}\ninterval:{interval}\norient:{orient}"
            )
        new_compute_modules = []
        new_transfer_lines = []
        if template == DesignTemplate.Strip:
            if orient == DesignOrientation.X:
                n_blocks = (module.bottom - module.top + 1) // (interval + 1)
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_modules.append(ComputeModule(module.left, module.top + i * (interval + 1), module.right, module.bottom))
                    else:
                        new_compute_modules.append(ComputeModule(module.left, module.top + i * (interval + 1), module.right, module.top + i * (interval + 1) + interval - 1))
                        new_transfer_lines.append(TransferLine(0, module.top + (i + 1) * (interval + 1) - 1, module.left, module.right))
            elif orient == DesignOrientation.Y:
                n_blocks = (module.right - module.left + 1) // (interval + 1)
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_modules.append(ComputeModule(module.left + i * (interval + 1), module.top, module.right, module.bottom))
                    else:
                        new_compute_modules.append(ComputeModule(module.left + i * (interval + 1), module.top, module.left + i * (interval + 1) + interval - 1, module.bottom))
                        new_transfer_lines.append(TransferLine(1, module.left + (i + 1) * (interval + 1) - 1, module.top, module.bottom))
        elif template == DesignTemplate.Radial:
            if orient == DesignOrientation.X:
                new_compute_modules.append(ComputeModule(module.left, module.top, module.right - interval - 1, module.top + interval - 1))
                new_transfer_lines.append(TransferLine(0, module.top + interval, module.left, module.right - interval - 1))
                new_compute_modules.append(ComputeModule(module.right - interval + 1, module.top, module.right, module.bottom - interval - 1))
                new_transfer_lines.append(TransferLine(1, module.right - interval, module.top, module.bottom - interval - 1))
                new_compute_modules.append(ComputeModule(module.left + interval + 1, module.bottom - interval + 1, module.right, module.bottom))
                new_transfer_lines.append(TransferLine(0, module.bottom - interval, module.left + interval + 1, module.right))
                new_compute_modules.append(ComputeModule(module.left, module.top + interval + 1, module.left + interval - 1, module.bottom))
                new_transfer_lines.append(TransferLine(1, module.left + interval, module.top + interval + 1, module.bottom))
                new_compute_modules.append(ComputeModule(module.left + interval + 1, module.top + interval + 1, module.right - interval - 1, module.bottom - interval - 1))
            elif orient == DesignOrientation.Y:
                new_compute_modules.append(ComputeModule(module.left, module.top, module.left + interval - 1, module.bottom - interval - 1))
                new_transfer_lines.append(TransferLine(1, module.left + interval, module.top, module.bottom - interval - 1))
                new_compute_modules.append(ComputeModule(module.left + interval + 1, module.top, module.right, module.top + interval - 1))
                new_transfer_lines.append(TransferLine(0, module.top + interval, module.left + interval + 1, module.right))
                new_compute_modules.append(ComputeModule(module.right - interval + 1, module.top + interval + 1, module.right, module.bottom))
                new_transfer_lines.append(TransferLine(1, module.right - interval, module.top + interval + 1, module.bottom))
                new_compute_modules.append(ComputeModule(module.left, module.bottom - interval + 1, module.right - interval - 1, module.bottom))
                new_transfer_lines.append(TransferLine(0, module.bottom - interval, module.left, module.right - interval - 1))
                new_compute_modules.append(ComputeModule(module.left + interval + 1, module.top + interval + 1, module.right - interval - 1, module.bottom - interval - 1))
        elif template == DesignTemplate.Grid:
            if orient == DesignOrientation.X:
                n_blocks = (module.bottom - module.top + 1) // (interval + 1)
                n_units = (module.right - module.left + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units):
                        if i == n_blocks - 1:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.right,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.left + j * (interval + 1) + interval - 1,
                                    module.bottom,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    1, 
                                    module.left + (j + 1) * (interval + 1) - 1,
                                    module.top + i * (interval + 1),
                                    module.bottom,
                                ))
                        else:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.right,
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    0,
                                    module.top + (i + 1) * (interval + 1) - 1,
                                    module.left,
                                    module.right,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.left + j * (interval + 1) + interval - 1,
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    1,
                                    module.left + (j + 1) * (interval + 1) - 1,
                                    module.top + i * (interval + 1),
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
            elif orient == DesignOrientation.Y:
                n_blocks = (module["right"] - module["left"] + 1) // (interval + 1)
                n_units = (module["bottom"] - module["top"] + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units):
                        if i == n_blocks - 1:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.right,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.right,
                                    module.top + j * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    0,
                                    module.top + (j + 1) * (interval + 1) - 1,
                                    module.left + i * (interval + 1),
                                    module.right,
                                ))
                        else:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                    module.bottom,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    1,
                                    module.left + (i + 1) * (interval + 1) - 1,
                                    module.top,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                    module.top + j * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    0,
                                    module.top + (j + 1) * (interval + 1) - 1,
                                    module.left + i * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                ))
        return new_compute_modules, new_transfer_lines

    def complete_package(self):
        """
        Complete the package graph after design the function area.

        1. Divide the compute modules into chiplets.
        2. Divide the transfer lines into chiplets.
        3. Determine the gaps between chiplets.
        """
        self.compute_chiplets = self.divide_compute_modules()
        self.transfer_chiplets = self.divide_transfer_lines()
        self.gap_dict = self.determine_inter_coupler()
        
        self.qubit_topology: nx.Graph = nx.Graph()
        for node in self.package_graph.nodes:
            if self.package_graph.nodes[node]["type"] != NodeType.Empty:
                self.qubit_topology.add_node(node, **self.package_graph.nodes[node])
                if self.qubit_topology.nodes[node]["type"] == NodeType.Compute:
                    self.qubit_topology.nodes[node]["qubit"].role = QubitRole.Data
                elif self.qubit_topology.nodes[node]["type"] == NodeType.Transfer:
                    self.qubit_topology.nodes[node]["qubit"].role = QubitRole.Balise
        for edge in self.package_graph.edges:
            if self.package_graph.edges[edge]["type"] != EdgeType.Empty:
                assert edge[0] in self.qubit_topology.nodes
                assert edge[1] in self.qubit_topology.nodes
                self.qubit_topology.add_edge(*edge, **self.package_graph.edges[edge])
        for transfer_chiplet in self.transfer_chiplets:
            for i in range(transfer_chiplet.start, transfer_chiplet.end + 1):
                node = get_full_space_tuple(transfer_chiplet.pos, transfer_chiplet.axis, i)
                neighbor_nodes = self.qubit_topology.neighbors(node)
                for neighbor_node in neighbor_nodes:
                    if self.qubit_topology.nodes[neighbor_node]["type"] == NodeType.Compute:
                        self.qubit_topology.nodes[node]["qubit"].role = QubitRole.Station
                        self.qubit_topology.nodes[neighbor_node]["qubit"].role = QubitRole.Platform
                # if transfer_chiplet["pos"] > 0:
                #     prev_node = get_full_space_tuple(transfer_chiplet["pos"] - 1, transfer_chiplet["axis"], i)
                #     self.qubit_topology.nodes[prev_node]["qubit"].role = QubitRole.Platform
                # if transfer_chiplet["pos"] < self.package_size[1 - transfer_chiplet["axis"]] - 1:
                #     next_node = get_full_space_tuple(transfer_chiplet["pos"] + 1, transfer_chiplet["axis"], i)
                #     self.qubit_topology.nodes[next_node]["qubit"].role = QubitRole.Platform
        
        self.metro_topology: nx.Graph = self.qubit_topology.subgraph(self.transfer_nodes)
        for edge in self.metro_topology.edges:
            if self.metro_topology.edges[edge]["type"] == EdgeType.Intra:
                self.metro_topology.edges[edge]["coupler"].role = CouplerRole.Rail
            elif self.metro_topology.edges[edge]["type"] == EdgeType.Inter:
                self.metro_topology.edges[edge]["coupler"].role = CouplerRole.Interchange
            else:
                raise ValueError("Invalid edge type: {}".format(self.metro_topology.edges[edge]["type"]))

        self.chiplet_topology: nx.Graph = nx.Graph()
        for compute_chiplet in self.compute_chiplets:
            self.chiplet_topology.add_node(str(compute_chiplet), chiplet=compute_chiplet)
        for transfer_chiplet in self.transfer_chiplets:
            self.chiplet_topology.add_node(str(transfer_chiplet), chiplet=transfer_chiplet)
        for gap in self.gap_dict.values():
            self.chiplet_topology.add_edge(str(gap.chiplet_u), str(gap.chiplet_v), gap=gap)

    def divide_compute_modules(self) -> list[ComputeChiplet]:
        """
        Divide the compute areas into manufacturable compute chiplets.

        Returns:
            compute_chiplets (list[ComputeChiplet]): The list of compute chiplets.

        TODO: Set the topology of compute chiplets according to the fabricate model.
        """
        compute_chiplets: list[ComputeChiplet] = []
        for module in self.compute_modules:
            module_x_size = module.right - module.left + 1
            module_y_size = module.bottom - module.top + 1
            chiplet_x_num = (module_x_size - 1) // self.fabricate_model.max_square_size + 1
            chiplet_y_num = (module_y_size - 1) // self.fabricate_model.max_square_size + 1
            chiplet_x_size = module_x_size // chiplet_x_num
            chiplet_y_size = module_y_size // chiplet_y_num
            large_chiplet_x_num = module_x_size - chiplet_x_num * chiplet_x_size
            large_chiplet_y_num = module_y_size - chiplet_y_num * chiplet_y_size
            for i in range(chiplet_x_num):
                for j in range(chiplet_y_num):
                    compute_chiplets.append(ComputeChiplet(
                        module.left + i * chiplet_x_size + min(i, large_chiplet_x_num),
                        module.top + j * chiplet_y_size + min(j, large_chiplet_y_num),
                        module.left + (i + 1) * chiplet_x_size + min(i + 1, large_chiplet_x_num) - 1,
                        module.top + (j + 1) * chiplet_y_size + min(j + 1, large_chiplet_y_num) - 1,
                    ))
        for compute_chiplet in compute_chiplets:
            for node in compute_chiplet.get_nodes():
                self.package_graph.nodes[node]["qubit"].chiplet = compute_chiplet
        return compute_chiplets

    def divide_transfer_lines(self) -> list[TransferChiplet]:
        """
        Divide the transfer lines into manufacturable transfer chiplets.

        Returns:
            transfer_chiplets (list[TransferChiplet]): the list of transfer chiplets
        """
        transfer_chiplets: list[TransferChiplet] = []
        for line in self.transfer_lines:
            line_length = line.end - line.start + 1
            chiplet_num = (line_length - 1) // self.fabricate_model.max_length + 1
            chiplet_length = line_length // chiplet_num
            long_chiplet_num = line_length - chiplet_num * chiplet_length
            for i in range(chiplet_num):
                transfer_chiplets.append(TransferChiplet(
                    line.axis,
                    line.pos,
                    line.start + i * chiplet_length + min(i, long_chiplet_num),
                    line.start + (i + 1) * chiplet_length + min(i + 1, long_chiplet_num) - 1,
                ))
        for transfer_chiplet in transfer_chiplets:
            for node in transfer_chiplet.get_nodes():
                self.package_graph.nodes[node]["qubit"].chiplet = transfer_chiplet
        return transfer_chiplets

    def determine_inter_coupler(self) -> dict[str, Gap]:
        """
        Determine the inter-chiplet coupler in the package.

        Returns:
            gap_dict (dict[str, Gap]): The dictionary of gaps between chiplets.
        """
        gap_dict: dict[str, Gap] = dict()
        for edge in self.package_graph.edges:
            if (
                self.package_graph.nodes[edge[0]]["type"] == NodeType.Empty
                or self.package_graph.nodes[edge[1]]["type"] == NodeType.Empty
            ):
                self.package_graph.edges[edge]["type"] = EdgeType.Empty
                self.package_graph.edges[edge]["coupler"] = None
            elif (
                self.package_graph.nodes[edge[0]]["qubit"].chiplet
                == self.package_graph.nodes[edge[1]]["qubit"].chiplet
            ):
                self.package_graph.edges[edge]["type"] = EdgeType.Intra
                qubit_u: Qubit = self.package_graph.nodes[edge[0]]["qubit"]
                qubit_v: Qubit = self.package_graph.nodes[edge[1]]["qubit"]
                if qubit_u.type == NodeType.Compute:
                    fide = self.fabricate_model.get_intra_fide(qubit_u.chiplet.get_capacity())
                else:
                    fide = self.fabricate_model.enhanced_fide
                self.package_graph.edges[edge]["coupler"] = Coupler(qubit_u, qubit_v, EdgeType.Intra, fide)
            else:
                qubit_u: Qubit = self.package_graph.nodes[edge[0]]["qubit"]
                qubit_v: Qubit = self.package_graph.nodes[edge[1]]["qubit"]
                if qubit_u.coordinate[0] == qubit_v.coordinate[0]:
                    axis = 0
                    pos = qubit_u.coordinate[0]
                else:
                    axis = 1
                    pos = qubit_u.coordinate[1]
                if qubit_u.coordinate[1] < qubit_v.coordinate[1]:
                    chiplet_u = qubit_u.chiplet
                    chiplet_v = qubit_v.chiplet
                else:
                    chiplet_u = qubit_v.chiplet
                    chiplet_v = qubit_u.chiplet
                gap_id = f"Gap_{chiplet_u}_{chiplet_v}"
                if gap_id not in gap_dict:
                    gap_dict[gap_id] = Gap(chiplet_u, chiplet_v, axis, {pos:edge})
                else:
                    gap_dict[gap_id].edges[pos] = edge
                self.package_graph.edges[edge]["type"] = EdgeType.Empty
                self.package_graph.edges[edge]["coupler"] = None
        for gap in gap_dict.values():
            for _, edge in list(sorted(gap.edges.items()))[
                :: self.fabricate_model.inter_sparsity
            ]:
                self.package_graph.edges[edge]["type"] = EdgeType.Inter
                self.package_graph.edges[edge]["coupler"] = Coupler(
                    self.package_graph.nodes[edge[0]]["qubit"],
                    self.package_graph.nodes[edge[1]]["qubit"],
                    EdgeType.Inter,
                    self.fabricate_model.inter_fide,
                )
        return gap_dict

    def random_transfer_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Randomly select a transfer pair from the compute nodes.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: The transfer pair.
        """
        return random.sample(self.compute_nodes, 2)

    def random_transfer_pairs(self, num:int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Randomly select a list of transfer pairs from the compute nodes.

        Args:
            num (int): The number of transfer pairs to select.

        Returns:
            list[tuple[tuple[int, int], tuple[int, int]]]: The list of transfer pairs.
        """
        return [self.random_transfer_pair() for _ in range(num)]

    def edge_fide_weight(self, u:tuple[int,int], v:tuple[int,int], attrs:dict) -> float:
        """
        Calculate the weight of the edge based on the fidelity.

        Args:
            u (tuple[int, int]): The first node.
            v (tuple[int, int]): The second node.
            attrs (dict): The attributes of the edge.
        
        Returns:
            float: The weight of the edge.
        """
        return -math.log(attrs["coupler"].fide)

    def get_shortest_path(self, source:tuple[int,int], target:tuple[int,int]) -> tuple[list[tuple[int, int]], float]:
        """
        Get the shortest path between two physical qubits.

        Args:
            source (tuple[int, int]): The source node of the transfer pair.
            target (tuple[int, int]): The target node of the transfer pair

        Returns:
            path (list[tuple[int, int]]): The path with highest fidelity.
            fidelity (float): The fidelity of the path
        """
        path = nx.shortest_path(
            self.qubit_topology, source, target, weight=self.edge_fide_weight
        )
        fidelity = 1
        for i in range(len(path) - 1):
            fidelity *= self.qubit_topology.edges[(path[i], path[i + 1])]["coupler"].fide
        return path, fidelity

    def estimate_transfer(self, num:int=10) -> float:
        """
        Estimate the transfer fidelity of the package.

        Args:
            num (int): The number of transfer pairs to select.

        Returns:
            float: The average fidelity of the transfer.
        """
        transfer_pairs = self.random_transfer_pairs(num)
        fidelity_list = []
        for transfer_pair in transfer_pairs:
            _, fidelity = self.get_shortest_path(*transfer_pair)
            fidelity_list.append(fidelity)
        return np.mean(fidelity_list)

    def estimate_cost(self) -> float:
        """
        Estimate the cost of the package considering the yield cost.

        Returns:
            float: The estimated cost of the package.
        """
        cost = 0
        for compute_chiplet in self.compute_chiplets:
            cost += self.fabricate_model.get_cost(compute_chiplet.get_capacity())
        for transfer_chiplet in self.transfer_chiplets:
            cost += self.fabricate_model.get_cost(transfer_chiplet.get_capacity())
        return cost

    def estimate_efficiency(self, num:int=10) -> float:
        """
        Estimate the efficiency of package considering the yield cost.

        Args:
            num (int): The number of transfer pairs selected to estimate the package.
        
        Returns:
            float: The estimated efficiency of the package.
        """
        fidelity = self.estimate_transfer(num)
        cost = self.estimate_cost()
        return -np.log(1 - fidelity) / cost

    def get_data_qubit_num(self) -> int:
        """
        Get the number of physical qubits with data role in the package.

        Returns:
            int: The number of data qubits.
        """
        cnt = 0
        for node in self.qubit_topology.nodes:
            if self.qubit_topology.nodes[node]["qubit"].role == QubitRole.Data:
                cnt += 1
        return cnt

    def get_data_qubit_list(self) -> list[tuple[int, int]]:
        """
        Get the list of physical qubits with data role in the package.

        Returns:
            list[tuple[int, int]]: The list of data qubits.
        """
        data_qubit_list = []
        for node in self.qubit_topology.nodes:
            if self.qubit_topology.nodes[node]["qubit"].role == QubitRole.Data:
                data_qubit_list.append(node)
        return data_qubit_list

    def get_physical_qubit_area(self, physical_qubit:tuple[int,int]) -> FuncArea:
        """
        Get the function area of a physical qubit belongs to.

        Args:
            physical_qubit (tuple[int, int]): The coordinate of the physical qubit.

        Returns:
            FuncArea: The function area of the physical qubit.
        """
        for module in self.compute_modules:
            if module.in_area(physical_qubit):
                return module
        for line in self.transfer_lines:
            if line.in_area(physical_qubit):
                return line
        raise ValueError("Invalid physical qubit: {}".format(physical_qubit))

    def get_area_graph(self, area:FuncArea) -> nx.Graph:
        """
        Get the graph of function area in the package.

        Args:
            area (FuncArea): The function area.

        Returns:
            nx.Graph: The subgraph view of qubit topology in the function area.
        """
        return self.qubit_topology.subgraph(area.get_nodes())

    @staticmethod
    def edge_metro_weight(self, u:tuple[int,int], v:tuple[int,int], attrs:dict) -> int:
        """
        Calculate the weight of the edge based on the role.

        Args:
            u (tuple[int, int]): The first node.
            v (tuple[int, int]): The second node.
            attrs (dict): The attributes of the edge.
        
        Returns:
            int: The weight of the edge.
        """
        return attrs["coupler"].role.value

    def get_shortest_metro_path(self, source:tuple[int,int], target:tuple[int,int]) -> tuple[list[tuple[int, int]], int]:
        """
        Get the shortest path in metro topology.

        Args:
            source (tuple[int, int]): The source node of the transfer pair.
            target (tuple[int, int]): The target node of the transfer pair.

        Returns:
            path (list[tuple[int, int]]): The path with least interchanges.
            interchange_num (int): The number of interchanges on the path.
        """
        assert source in self.metro_topology.nodes and target in self.metro_topology.nodes
        path = nx.shortest_path(
            self.metro_topology, source, target, weight=self.edge_metro_weight
        )
        interchange_num = 0
        for i in range(len(path) - 1):
            interchange_num += self.metro_topology.edges[(path[i], path[i + 1])]["coupler"].role.value
        return path, interchange_num

    def draw_qubits(self, 
        marginrate:float=0.2, 
        markersize:float=4, 
        linewidth_t:float=1, 
        linewidth_b:float=2, 
        save_path:str=None
    ):
        """
        Draw the package graph in qubit level.
        """
        plt.figure(figsize=self.package_size)
        for node in self.package_graph.nodes:
            plt.fill(
                node[0] + np.array([-0.5, 0.5, 0.5, -0.5]),
                node[1] + np.array([-0.5, -0.5, 0.5, 0.5]),
                color=color_pool[self.package_graph.nodes[node]["qubit"].type.value],
            )
        plt.fill(
            -0.5 + np.array([-0.5, 0.5, 0.5, -0.5]) * marginrate,
            np.array(
                [-0.5, -0.5, self.package_size[1] - 0.5, self.package_size[1] - 0.5]
            ),
            color="white",
        )
        plt.fill(
            self.package_size[0] - 0.5 + np.array([-0.5, 0.5, 0.5, -0.5]) * marginrate,
            np.array(
                [-0.5, -0.5, self.package_size[1] - 0.5, self.package_size[1] - 0.5]
            ),
            color="white",
        )
        plt.fill(
            np.array(
                [-0.5, self.package_size[0] - 0.5, self.package_size[0] - 0.5, -0.5]
            ),
            -0.5 + np.array([-0.5, -0.5, 0.5, 0.5]) * marginrate,
            color="white",
        )
        plt.fill(
            np.array(
                [-0.5, self.package_size[0] - 0.5, self.package_size[0] - 0.5, -0.5]
            ),
            self.package_size[1] - 0.5 + np.array([-0.5, -0.5, 0.5, 0.5]) * marginrate,
            color="white",
        )
        for edge in self.package_graph.edges:
            if (
                self.package_graph.edges[edge]["type"] == EdgeType.Inter
                or self.package_graph.edges[edge]["type"] == EdgeType.Empty
            ):
                if edge[0][0] == edge[1][0]:
                    plt.fill(
                        edge[0][0] + np.array([-0.5, 0.5, 0.5, -0.5]),
                        (edge[0][1] + edge[1][1]) / 2
                        + np.array([-0.5, -0.5, 0.5, 0.5]) * marginrate,
                        color="white",
                    )
                else:
                    plt.fill(
                        (edge[0][0] + edge[1][0]) / 2
                        + np.array([-0.5, 0.5, 0.5, -0.5]) * marginrate,
                        edge[0][1] + np.array([-0.5, -0.5, 0.5, 0.5]),
                        color="white",
                    )
            if self.package_graph.edges[edge]["type"] == EdgeType.Intra:
                plt.plot(
                    [edge[0][0], edge[1][0]],
                    [edge[0][1], edge[1][1]],
                    "k-",
                    linewidth=linewidth_t,
                )
            elif self.package_graph.edges[edge]["type"] == EdgeType.Inter:
                plt.plot(
                    [edge[0][0], edge[1][0]],
                    [edge[0][1], edge[1][1]],
                    "r-",
                    linewidth=linewidth_b,
                )
        for node in self.package_graph.nodes:
            if self.package_graph.nodes[node]["type"] != NodeType.Empty:
                plt.plot(node[0], node[1], "ko", markersize=markersize)
        plt.axis("off")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

    def draw_chiplets(self, 
        marginrate:float=0.2, 
        markersize:float=4, 
        linewidth_t:float=1, 
        linewidth_b:float=2, 
        save_path:str=None
    ):
        """
        Draw the package graph in chiplet level.
        """
        plt.figure(figsize=self.package_size)
        for node in self.chiplet_topology.nodes:
            chiplet: Chiplet = self.chiplet_topology.nodes[node]["chiplet"]
            plt.plot(*chiplet.get_coordinate(), "ko", markersize=markersize)
        for node_u, node_v in self.chiplet_topology.edges:
            pos_u = self.chiplet_topology.nodes[node_u]["chiplet"].get_coordinate()
            pos_v = self.chiplet_topology.nodes[node_v]["chiplet"].get_coordinate()
            plt.plot(
                [pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], "k-", linewidth=linewidth_t
            )
        plt.axis("off")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
