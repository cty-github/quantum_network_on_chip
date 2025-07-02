import math
import numpy as np
from enum import Enum, auto
from scipy.optimize import curve_fit

from ..utils.utils import log_decay, get_full_space_tuple


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
    Register = 0
    Platform = auto()
    Balise = auto()
    Station = auto()
    Empty = auto()

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
        enhanced_fide (float): The unit fidelity of enhanced qst.
        inter_fide (float): The fidelity of inter-chiplet CZ.
        yield_5 (float): The yield of 5-qubit chiplet.
        yield_decay (float): The decay rate of yield with the qubit number.
    """

    def __init__(self, 
        max_capacity: int = 100, 
        max_length: int = 50, 
        compute_geometry: str = "square", 
        inter_sparsity: int = 3, 
        single_fide: float = 0.9999, 
        intra_fide_dict: dict[int, float] = {27: 0.993, 65: 0.989, 127: 0.988}, 
        enhanced_fide: float = 0.9999, 
        inter_fide: float = 0.99, 
        yield_5: float = 0.99, 
        yield_decay: float = 2, 
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
        self.max_square_size = int(math.sqrt(max_capacity))
        self.max_length = max_length
        self.compute_geometry = compute_geometry
        self.inter_sparsity = inter_sparsity

        self.single_fide = single_fide
        self.intra_fide_params, _ = curve_fit(
            log_decay,
            np.array(list(intra_fide_dict.keys())),
            np.array(list(intra_fide_dict.values())),
            p0=[0.01, 0.01],
        )
        self.enhanced_fide = enhanced_fide
        self.inter_fide = inter_fide

        self.yield_5 = yield_5
        self.yield_decay = yield_decay

    def get_intra_fide(self, qubit_num: int) -> float:
        """
        Get the fidelity of intra-chiplet CZ according to qubit num in the chiplet.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.
        
        Returns:
            float: The fidelity of intra-chiplet CZ.
        """
        return log_decay(qubit_num, *self.intra_fide_params)

    def get_yield(self, qubit_num: int) -> float:
        """
        Get the yield of the chiplet according to the qubit num.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.
        
        Returns:
            float: The yield of the chiplet.
        """
        return (self.yield_5 * (1 - math.tanh(self.yield_decay * (math.log10(qubit_num) - 2))) / 2)

    def get_cost(self, qubit_num: int) -> float:
        """
        Get the cost of the chiplet according to the qubit num.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.

        Returns:
            float: The cost of the chiplet.
        """
        return qubit_num / self.get_yield(qubit_num)

class OperationConfig:
    """
    Represents the configuration of the operation in the package.

    Attributes:
        single_time (float): The time of single gate.
        intra_time (float): The time of intra-chiplet CZ.
        inter_time (float): The time of inter-chiplet CZ.
        qst_time_constant (float): The time constant of the enhanced QST.
        measure_time (float): The time of measurement.
        reset_time (float): The time of reset.
    """
    
    def __init__(self, 
            single_time: float = 0.0001, 
            intra_time: float = 0.0002, 
            inter_time: float = 0.0003, 
            qst_time_constant: float = 0.0004, 
            measure_time: float = 0.0005, 
            reset_time: float = 0.0006
        ):
        """
        Initialize the operation configuration.

        Args:
            single_time (float): The time of single gate.
            intra_time (float): The time of intra-chiplet CZ.
            inter_time (float): The time of inter-chiplet CZ.
            qst_time_constant (float): The time constant of the enhanced QST.
            measure_time (float): The time of measurement.
            reset_time (float): The time of reset.
        """
        self.single_time: float = single_time
        self.intra_time: float = intra_time
        self.inter_time: float = inter_time
        self.qst_time_constant: float = qst_time_constant
        self.measure_time: float = measure_time
        self.reset_time: float = reset_time


class Qubit:
    """
    Represents the physical qubit in the package.

    Attributes:
        coordinate (tuple[int, int]): The coordinate of the qubit.
        type (NodeType): The type of the chiplet that the qubit belongs to.
        fide (float): The single gate fidelity of the qubit.
        role (QubitRole): The role of the qubit.
        chiplet (Chiplet): The chiplet that the qubit belongs to.
        area (FuncArea): The function area that the qubit belongs to.
    """
    
    def __init__(self, 
        coordinate: tuple[int, int], 
        type: NodeType, 
        fide: float, 
        role: QubitRole = None
    ):
        """
        Initialize a physcial qubit.

        Args:
            coordinate (tuple[int, int]): The coordinate of the qubit.
            type (NodeType): The type of the chiplet that the qubit belongs to.
            fide (float): The single gate fidelity of the qubit.
            role (QubitRole): The role of the qubit.
        """
        self.coordinate = coordinate
        self.type = type
        self.fide = fide
        self.role = role
        self.chiplet: Chiplet = None
        self.area: FuncArea = None

    def __repr__(self):
        return f"{self.__class__.__name__}{self.coordinate}"

class Coupler:
    """
    Represents the coupler between two qubits in the package.
    
    Attributes:
        qubit_u (tuple[int, int]): The coordinate of qubit u on one side of the coupler.
        qubit_v (tuple[int, int]): The coordinate of qubit v on the other side of the coupler.
        type (EdgeType): The type of the coupler.
        fide (float): The CZ/EQST fidelity of the coupler.
        role (CouplerRole): The role of the coupler.
    """
    
    def __init__(self, 
        qubit_u: tuple[int, int], 
        qubit_v: tuple[int, int], 
        type: EdgeType, 
        fide: float, 
        role: CouplerRole = None
    ):
        """
        Initialize a coupler.

        Args:
            qubit_u (tuple[int, int]): The coordinate of qubit u on one side of the coupler.
            qubit_v (tuple[int, int]): The coordinate of qubit v on the other side of the coupler.
            type (EdgeType): The type of the coupler.
            fide (float): The CZ/EQST fidelity of the coupler.
            role (CouplerRole): The role of the coupler.
        """
        self.qubit_u = qubit_u
        self.qubit_v = qubit_v
        self.type = type
        self.fide = fide
        self.role = role

    def __repr__(self):
        return f"{self.__class__.__name__}{self.qubit_u}{self.qubit_v}"


class Chiplet:
    """
    Represents the chiplet in the package.

    Attributes:
        intra_fide (float): The fidelity of intra-chiplet CZ.
    """

    def __init__(self, intra_fide: float):
        """
        Initialize a chiplet.

        Args:
            intra_fide (float): The fidelity of intra-chiplet CZ.
        """
        self.intra_fide = intra_fide
    
    def get_coordinate(self) -> tuple[float, float]:
        """
        Get the center coordinate of the chiplet.

        Returns:
            coordinate (tuple[float, float]): the center coordinate of the chiplet
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_nodes(self) -> list[tuple[int, int]]:
        """
        Get the nodes in the chiplet.

        Returns:
            nodes (list[tuple[int, int]]): the list of nodes in the chiplet
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def has_node(self, node: tuple[int, int]) -> bool:
        """
        Check if the node is in the function area.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            bool: True if the node is in the function area, False otherwise.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Get the edges in the chiplet.

        Returns:
            edges (list[tuple[tuple[int, int], tuple[int, int]]]): The list of edges in the chiplet.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def has_edge(self, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        """
        Check if the edge is in the chiplet.

        Args:
            edge (tuple[tuple[int, int], tuple[int, int]]): The edge to check.

        Returns:
            bool: True if the edge is in the chiplet, False otherwise.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        """
        Get the distance between two nodes in the chiplet, which is the number of edges in the shortest path.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            distance (int): The distance between the two nodes, which is the number of edges in the shortest path.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_capacity(self) -> int:
        """
        Get the capacity of the chiplet.

        Returns:
            capacity (int): the capacity of the chiplet
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

class ComputeChiplet(Chiplet):
    """
    Represents the compute chiplet in the package.

    Attributes:
        intra_fide (float): The fidelity of intra-chiplet CZ.
        in_module_coord (tuple[int, int]): The coordinate of the chiplet in the module.
        left (int): The left boundary of the chiplet.
        top (int): The top boundary of the chiplet.
        right (int): The right boundary of the chiplet.
        bottom (int): The bottom boundary of the chiplet.
        geometry (str): The qubit topology in the chiplet, i.e. 'square', 'hexagon', 'heavy square', 'heavy hexagon'.
    """

    def __init__(self, intra_fide: float, in_module_coord: tuple[int, int], left: int, top: int, right: int, bottom: int, geometry: str = "square"):
        """
        Initialize a compute chiplet.

        Args:
            left (int): The left boundary of the chiplet.
            top (int): The top boundary of the chiplet.
            right (int): The right boundary of the chiplet.
            bottom (int): The bottom boundary of the chiplet.
            geometry (str): The qubit topology in the chiplet, i.e. 'square', 'hexagon', 'heavy square', 'heavy hexagon'.
        """
        super().__init__(intra_fide)
        self.in_module_coord = in_module_coord
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.geometry = geometry

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left}_{self.top}_{self.right}_{self.bottom}"
    
    def get_coordinate(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)
    
    def get_nodes(self) -> list[tuple[int, int]]:
        # TODO: Set the topology of compute chiplets according to the geometry.
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
        return self.nodes

    def has_node(self, node: tuple[int, int]) -> bool:
        return node[0] >= self.left and node[0] <= self.right and node[1] >= self.top and node[1] <= self.bottom

    def get_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        # TODO: Set the topology of compute chiplets according to the geometry.
        if not hasattr(self, "edges"):
            self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    if i < self.right:
                        self.edges.append(((i, j), (i + 1, j)))
                    if j < self.bottom:
                        self.edges.append(((i, j), (i, j + 1)))
        return self.edges
    
    def has_edge(self, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        return edge in self.get_edges() or (edge[1], edge[0]) in self.get_edges()
    
    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        # TODO: Find the shortest path considering the topology of compute chiplet.
        assert self.has_node(source), f"source {source} is not in the compute chiplet {self}"
        assert self.has_node(target), f"target {target} is not in the compute chiplet {self}"
        return abs(source[0] - target[0]) + abs(source[1] - target[1])
    
    def get_capacity(self) -> int:
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)

class TransferChiplet(Chiplet):
    """
    Represents the transfer chiplet in the package.
    
    Attributes:
        axis (int): The axis of the chiplet.
        pos (int): The position of the chiplet.
        start (int): The start point of the chiplet.
        end (int): The end point of the chiplet.
    """

    def __init__(self, intra_fide: float, axis: int, pos: int, start: int, end: int):
        """
        Initialize a transfer chiplet.

        Args:
            intra_fide (float): The fidelity of intra-chiplet CZ.
            axis (int): The axis of the chiplet.
            pos (int): The position of the chiplet.
            start (int): The start point of the chiplet.
            end (int): The end point of the chiplet.
        """
        super().__init__(intra_fide)
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start}_{self.end}"

    def get_coordinate(self) -> tuple[float, float]:
        return get_full_space_tuple(self.pos, self.axis, (self.start + self.end) / 2)
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start, self.end + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes

    def has_node(self, node: tuple[int, int]) -> bool:
        return node[1-self.axis] == self.pos and node[self.axis] >= self.start and node[self.axis] <= self.end
    
    def get_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        if not hasattr(self, "edges"):
            self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for i in range(self.start, self.end):
                self.edges.append((get_full_space_tuple(self.pos, self.axis, i), get_full_space_tuple(self.pos, self.axis, i + 1)))
        return self.edges
    
    def has_edge(self, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        return edge in self.get_edges() or (edge[1], edge[0]) in self.get_edges()
    
    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        assert self.has_node(source), f"source {source} is not in the transfer chiplet {self}"
        assert self.has_node(target), f"target {target} is not in the transfer chiplet {self}"
        return abs(source[self.axis] - target[self.axis])
    
    def get_capacity(self) -> int:
        return self.end - self.start + 1


class Gap:
    """
    Represents the gap padded with inter-chiplet couplers in the package.

    Attributes:
        fab_model (FabricateModel): The fabrication model of the package.
        axis (int): The axis of the gap.
        pos (tuple[int, int]): The position of the gap.
        start (int): The start point of the gap.
        end (int): The end point of the gap.
        edges (dict[int, tuple[tuple[int,int], tuple[int,int]]]): The edges in the gap.
    """
    fab_model: FabricateModel = None

    def __init__(self, axis:int, pos: tuple[int, int], start: int, end: int):
        """
        Initialize a gap.

        Args:
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
        """
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end
    
    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start}_{self.end}"
    
    def get_couplers(self) -> dict[int, Coupler]:
        """
        Get the couplers in the gap.

        Returns:
            couplers (dict[int, Coupler]): The dictionary of couplers in the gap.
        """
        if not hasattr(self, "couplers"):
            self.couplers: dict[int, Coupler] = {}
            for i in range(self.start, self.end + 1, Gap.fab_model.inter_sparsity):
                self.couplers[i] = Coupler(
                    get_full_space_tuple(self.pos[0], self.axis, i), 
                    get_full_space_tuple(self.pos[1], self.axis, i), 
                    EdgeType.Inter,
                    Gap.fab_model.inter_fide,
                )
        return self.couplers

class IntraComputeGap(Gap):
    """
    Represents the gap in a compute module.
    """

    def __init__(self, axis:int, pos: tuple[int, int], start: int, end: int):
        """
        Initialize a gap in a function area.

        Args:
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
        """
        super().__init__(axis, pos, start, end)

class ComputeTransferGap(Gap):
    """
    Represents the gap between a compute module and a transfer line.
    """

    def __init__(self, axis:int, pos: tuple[int, int], start: int, end: int):
        """
        Initialize a gap between two function area.

        Args:
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
        """
        super().__init__(axis, pos, start, end)


class FuncArea:
    """
    Represents the function area in the package.

    Attributes:
        fab_model (FabricateModel): The fabrication model of the package.
        nodes (list[tuple[int, int]]): The list of nodes in the function area.
        chiplets (list[Chiplet]): The list of chiplets in the function area.
        intra_edges (list[tuple[tuple[int, int], tuple[int, int]]]): The list of edges inside the chiplets.
    """
    fab_model: FabricateModel = None

    def __init__(self):
        """
        Initialize a function area.
        """
    
    def get_nodes(self) -> list[tuple[int, int]]:
        """
        Get the nodes in the function area.

        Returns:
            nodes (list[tuple[int, int]]): The list of nodes in the function area.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def has_node(self, node: tuple[int, int]) -> bool:
        """
        Check if the node is in the function area.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            bool: True if the node is in the function area, False otherwise.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def determine_chiplets(self):
        """
        Partition the function area into chiplets.
        """
        self.chiplets: list[Chiplet] = []
    
    def get_chiplets(self) -> list[Chiplet]:
        """
        Get the chiplets in the function area.

        Returns:
            chiplets (list[Chiplet]): The list of chiplets in the function area.
        """
        if not hasattr(self, "chiplets"):
            self.determine_chiplets()
        return self.chiplets
    
    def get_intra_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Get the edges inside the chiplets.

        Returns:
            intra_edges (list[tuple[tuple[int, int], tuple[int, int]]]): The list of edges inside the chiplets.
        """
        if not hasattr(self, "intra_edges"):
            self.intra_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for chiplet in self.get_chiplets():
                self.intra_edges.extend(chiplet.get_edges())
        return self.intra_edges

class TransferLine(FuncArea):
    """
    Represents the transfer line in the package.

    Attributes:
        axis (int): The axis of the line.
        pos (int): The position of the line.
        start (tuple[int, Coupler]): The start coordinate and interchange of the line.
        end (tuple[int, Coupler]): The end coordinate and interchange of the line.
    """

    def __init__(self, 
            axis: int, 
            pos: int, 
            start: tuple[int, bool],
            end: tuple[int, bool],
        ):
        """
        Initialize a transfer line.

        Args:
            axis (int): The axis of the line.
            pos (int): The position of the line.
            start (tuple[int, bool]): The start coordinate and boundary flag of the line.
            end (tuple[int, bool]): The end coordinate and boundary flag of the line.
        """
        super().__init__()
        self.axis = axis
        self.pos = pos
        if not start[1]:
            start_interchange = Coupler(
                get_full_space_tuple(pos, axis, start[0]-1),
                get_full_space_tuple(pos, axis, start[0]),
                EdgeType.Inter,
                FuncArea.fab_model.inter_fide,
                CouplerRole.Interchange, 
            )
        else:
            start_interchange = None
        if not end[1]:
            end_interchange = Coupler(
                get_full_space_tuple(pos, axis, end[0]),
                get_full_space_tuple(pos, axis, end[0]+1),
                EdgeType.Inter,
                FuncArea.fab_model.inter_fide,
                CouplerRole.Interchange, 
            )
        else:
            end_interchange = None
        self.start: tuple[int, Coupler] = (start[0], start_interchange)
        self.end: tuple[int, Coupler] = (end[0], end_interchange)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start[0]}_{self.end[0]}"
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start[0], self.end[0] + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes
    
    def has_node(self, node: tuple[int, int]) -> bool:
        return node[1-self.axis] == self.pos and node[self.axis] >= self.start[0] and node[self.axis] <= self.end[0]
    
    def determine_chiplets(self):
        line_length = self.end[0] - self.start[0] + 1
        chiplet_num = (line_length - 1) // FuncArea.fab_model.max_length + 1
        chiplet_length = line_length // chiplet_num
        long_chiplet_num = line_length - chiplet_num * chiplet_length
        self.chiplets: list[TransferChiplet] = []
        for i in range(chiplet_num):
            self.chiplets.append(TransferChiplet(
                FuncArea.fab_model.enhanced_fide,
                self.axis,
                self.pos,
                self.start[0] + i * chiplet_length + min(i, long_chiplet_num),
                self.start[0] + (i + 1) * chiplet_length + min(i + 1, long_chiplet_num) - 1,
            ))
        self.interchanges: set[Coupler] = set()
        if self.start[1]:
            self.interchanges.add(self.start[1])
        if self.end[1]:
            self.interchanges.add(self.end[1])
        for i in range(chiplet_num - 1):
            self.interchanges.add(Coupler(
                get_full_space_tuple(self.pos, self.axis, self.start[0] + (i + 1) * chiplet_length + min(i + 1, long_chiplet_num) - 1),
                get_full_space_tuple(self.pos, self.axis, self.start[0] + (i + 1) * chiplet_length + min(i + 1, long_chiplet_num)),
                EdgeType.Inter,
                FuncArea.fab_model.inter_fide,
                CouplerRole.Interchange,
            ))
    
    def get_chiplets(self) -> list[TransferChiplet]:
        if not hasattr(self, "chiplets"):
            self.determine_chiplets()
        return self.chiplets

    def get_interchanges(self) -> set[Coupler]:
        """
        Get the interchanges in the transfer line.

        Returns:
            interchanges (set[Coupler]): The set of interchanges in the transfer line.
        """
        if not hasattr(self, "interchanges"):
            self.determine_chiplets()
        return self.interchanges

class ComputeModule(FuncArea):
    """
    Represents the compute module in the package.

    Attributes:
        left (tuple[int, TransferLine, ComputeTransferGap]): The left boundary of the module.
        top (tuple[int, TransferLine, ComputeTransferGap]): The top boundary of the module.
        right (tuple[int, TransferLine, ComputeTransferGap]): The right boundary of the module.
        bottom (tuple[int, TransferLine, ComputeTransferGap]): The bottom boundary of the module.
        chiplet_x_num (int): The number of chiplets in the x direction.
        chiplet_y_num (int): The number of chiplets in the y direction.
        chiplets (list[ComputeChiplet]): The list of chiplets in the module.
        chiplet_matrix (dict[tuple[int, int], ComputeChiplet]): The dictionary of chiplets in the module.
        intra_gaps (list[IntraComputeGap]): The list of intra-chiplet gaps in the module.
    """
    
    def __init__(self, 
        left: int | TransferLine, 
        top: int | TransferLine, 
        right: int | TransferLine, 
        bottom: int | TransferLine, 
    ):
        """
        Initialize a compute module.

        Args:
            left (int | Transferline): The left coordinate or transferline of the module.
            top (int | Transferline): The top coordinate or transferline of the module.
            right (int | Transferline): The right coordinate or transferline of the module.
            bottom (int | Transferline): The bottom coordinate or transferline of the module.
        """
        super().__init__()
        if isinstance(left, TransferLine):
            left_coord = left.pos + 1
            left_line = left
        else:
            left_coord = left
            left_line = None
        if isinstance(top, TransferLine):
            top_coord = top.pos + 1
            top_line = top
        else:
            top_coord = top
            top_line = None
        if isinstance(right, TransferLine):
            right_coord = right.pos - 1
            right_line = right
        else:
            right_coord = right
            right_line = None
        if isinstance(bottom, TransferLine):
            bottom_coord = bottom.pos - 1
            bottom_line = bottom
        else:
            bottom_coord = bottom
            bottom_line = None
        left_gap = ComputeTransferGap(
            left_line.axis, 
            (left_line.pos, left_coord), 
            top_coord, 
            bottom_coord
        ) if left_line else None
        top_gap = ComputeTransferGap(
            top_line.axis, 
            (top_line.pos, top_coord), 
            left_coord, 
            right_coord
        ) if top_line else None
        right_gap = ComputeTransferGap(
            right_line.axis, 
            (right_coord, right_line.pos), 
            top_coord, 
            bottom_coord
        ) if right_line else None
        bottom_gap = ComputeTransferGap(
            bottom_line.axis, 
            (bottom_coord, bottom_line.pos), 
            left_coord, 
            right_coord
        ) if bottom_line else None
        self.left: tuple[int, TransferLine, ComputeTransferGap] = (left_coord, left_line, left_gap)
        self.top: tuple[int, TransferLine, ComputeTransferGap] = (top_coord, top_line, top_gap)
        self.right: tuple[int, TransferLine, ComputeTransferGap] = (right_coord, right_line, right_gap)
        self.bottom: tuple[int, TransferLine, ComputeTransferGap] = (bottom_coord, bottom_line, bottom_gap)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left[0]}_{self.top[0]}_{self.right[0]}_{self.bottom[0]}"
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left[0], self.right[0] + 1):
                for j in range(self.top[0], self.bottom[0] + 1):
                    self.nodes.append((i, j))
        return self.nodes
    
    def has_node(self, node: tuple[int, int]) -> bool:
        return node[0] >= self.left[0] and node[0] <= self.right[0] and node[1] >= self.top[0] and node[1] <= self.bottom[0]
    
    def determine_chiplets(self):
        module_x_size = self.right[0] - self.left[0] + 1
        module_y_size = self.bottom[0] - self.top[0] + 1
        if module_x_size < 1 or module_y_size < 1:
            raise ValueError("The size of the module is invalid.")
        self.chiplet_x_num = (module_x_size - 1) // self.fab_model.max_square_size + 1
        self.chiplet_y_num = (module_y_size - 1) // self.fab_model.max_square_size + 1
        chiplet_x_size = module_x_size // self.chiplet_x_num
        chiplet_y_size = module_y_size // self.chiplet_y_num
        large_chiplet_x_num = module_x_size - self.chiplet_x_num * chiplet_x_size
        large_chiplet_y_num = module_y_size - self.chiplet_y_num * chiplet_y_size
        self.chiplets: list[ComputeChiplet] = []
        self.chiplet_matrix: dict[tuple[int, int], ComputeChiplet] = {}
        for i in range(self.chiplet_x_num):
            for j in range(self.chiplet_y_num):
                self.chiplets.append(ComputeChiplet(
                    FuncArea.fab_model.get_intra_fide(
                        (chiplet_x_size + int(i < large_chiplet_x_num)) * (chiplet_y_size + int(j < large_chiplet_y_num))
                    ),
                    (i, j),
                    self.left[0] + i * chiplet_x_size + min(i, large_chiplet_x_num),
                    self.top[0] + j * chiplet_y_size + min(j, large_chiplet_y_num),
                    self.left[0] + (i + 1) * chiplet_x_size + min(i + 1, large_chiplet_x_num) - 1,
                    self.top[0] + (j + 1) * chiplet_y_size + min(j + 1, large_chiplet_y_num) - 1,
                    FuncArea.fab_model.compute_geometry,
                ))
                self.chiplet_matrix[(i, j)] = self.chiplets[-1]
        self.intra_gaps: list[IntraComputeGap] = []
        for i in range(self.chiplet_x_num - 1):
            self.intra_gaps.append(IntraComputeGap(
                1, 
                (
                    self.left[0] + (i + 1) * chiplet_x_size + min(i + 1, large_chiplet_x_num) - 1, 
                    self.left[0] + (i + 1) * chiplet_x_size + min(i + 1, large_chiplet_x_num)
                ), 
                self.top[0], 
                self.bottom[0],
            ))
        for j in range(self.chiplet_y_num - 1):
            self.intra_gaps.append(IntraComputeGap(
                0,
                (
                    self.top[0] + (j + 1) * chiplet_y_size + min(j + 1, large_chiplet_y_num) - 1, 
                    self.top[0] + (j + 1) * chiplet_y_size + min(j + 1, large_chiplet_y_num)
                ),
                self.left[0],
                self.right[0],
            ))
    
    def get_chiplets(self) -> list[ComputeChiplet]:
        if not hasattr(self, "chiplets"):
            self.determine_chiplets()
        return self.chiplets
    
    def get_intra_gaps(self) -> list[IntraComputeGap]:
        """
        Get the gaps in the compute module.

        Returns:
            intra_gaps (list[IntraComputeGap]): The list of gaps in the compute module.
        """
        if not hasattr(self, "intra_gaps"):
            self.determine_chiplets()
        return self.intra_gaps

    def get_compute_transfer_gaps(self) -> list[ComputeTransferGap]:
        """
        Get the compute transfer gaps around the compute module.

        Returns:
            compute_transfer_gaps (list[ComputeTransferGap]): The list of compute transfer gaps in the compute module.
        """
        if not hasattr(self, "compute_transfer_gaps"):
            self.compute_transfer_gaps: list[ComputeTransferGap] = []
            if self.left[2]:
                self.compute_transfer_gaps.append(self.left[2])
            if self.top[2]:
                self.compute_transfer_gaps.append(self.top[2])
            if self.right[2]:
                self.compute_transfer_gaps.append(self.right[2])
            if self.bottom[2]:
                self.compute_transfer_gaps.append(self.bottom[2])
        return self.compute_transfer_gaps

    def get_platform_station_pairs(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Get the platform and station pairs in the compute module.

        Returns:
            platforms_and_stations (list[tuple[tuple[int, int], tuple[int, int]]]): The list of platforms and stations in the compute module.
        """
        if not hasattr(self, "platforms_and_stations"):
            self.platforms_and_stations: list[tuple[tuple[int, int], tuple[int, int]]] = []
            if self.left[2]:
                for coupler in self.left[2].get_couplers().values():
                    self.platforms_and_stations.append((coupler.qubit_v, coupler.qubit_u))
            if self.top[2]:
                for coupler in self.top[2].get_couplers().values():
                    self.platforms_and_stations.append((coupler.qubit_v, coupler.qubit_u))
            if self.right[2]:
                for coupler in self.right[2].get_couplers().values():
                    self.platforms_and_stations.append((coupler.qubit_u, coupler.qubit_v))
            if self.bottom[2]:
                for coupler in self.bottom[2].get_couplers().values():
                    self.platforms_and_stations.append((coupler.qubit_u, coupler.qubit_v))
        return self.platforms_and_stations

    def get_platforms(self) -> set[tuple[int, int]]:
        """
        Get the platforms in the compute module.

        Returns:
            platforms (set[tuple[int, int]]): The set of platforms in the compute module.
        """
        if not hasattr(self, "platforms"):
            self.platforms: set[tuple[int, int]] = set()
            for platform, _ in self.get_platform_station_pairs():
                self.platforms.add(platform)
        return self.platforms

    def get_node_chiplet(self, node: tuple[int, int]) -> ComputeChiplet:
        """
        Get the compute chiplet that contains the node.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            ComputeChiplet: The compute chiplet that contains the node.
        """
        for chiplet in self.get_chiplets():
            if chiplet.has_node(node):
                return chiplet
        raise ValueError(f"Node {node} is not in any chiplet in the compute module {self}")

    def get_chiplet_platforms(self, chiplet: ComputeChiplet) -> set[tuple[int, int]]:
        """
        Get the platforms in the compute chiplet.

        Args:
            chiplet (ComputeChiplet): The compute chiplet.

        Returns:
            platforms (set[tuple[int, int]]): The set of platforms in the compute chiplet.
        """
        assert chiplet in self.get_chiplets(), f"chiplet {chiplet} is not in the compute module {self}"
        platforms: set[tuple[int, int]] = set()
        for platform, _ in self.get_platform_station_pairs():
            if chiplet.has_node(platform):
                platforms.add(platform)
        return platforms

    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        """
        Get the distance between two nodes in the compute module, which is the number of edges in the shortest path.
        # TODO: Find the shortest path considering the topology of chiplet in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            distance (int): The distance between the two nodes, which is the number of edges in the shortest path.
        """
        assert self.has_node(source), f"source {source} is not in the compute module {self}"
        assert self.has_node(target), f"target {target} is not in the compute module {self}"
        return abs(source[0] - target[0]) + abs(source[1] - target[1])
    
    def get_inter_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        """
        Get the number of inter-chiplet coupler in the shortest path between two nodes in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            distance (int): The number of inter-chiplet coupler in the shortest path between the two nodes.
        """
        assert self.has_node(source), f"source {source} is not in the compute module {self}"
        assert self.has_node(target), f"target {target} is not in the compute module {self}"
        source_chiplet = self.get_node_chiplet(source)
        target_chiplet = self.get_node_chiplet(target)
        source_chiplet_coord = source_chiplet.in_module_coord
        target_chiplet_coord = target_chiplet.in_module_coord
        return abs(source_chiplet_coord[0] - target_chiplet_coord[0]) + abs(source_chiplet_coord[1] - target_chiplet_coord[1])

    def get_nearest_platform(self, node: tuple[int, int]) -> tuple[int, int]:
        """
        Get the nearest platform to the node in the compute module.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            tuple[int, int]: The coordinate of the nearest platform.
        """
        assert self.has_node(node), f"node {node} is not in the compute module {self}"
        chiplet = self.get_node_chiplet(node)
        left_inter_distance = chiplet.in_module_coord[0] if self.left[2] else math.inf
        top_inter_distance = chiplet.in_module_coord[1] if self.top[2] else math.inf
        right_inter_distance = self.chiplet_x_num - chiplet.in_module_coord[0] - 1 if self.right[2] else math.inf
        bottom_inter_distance = self.chiplet_y_num - chiplet.in_module_coord[1] - 1 if self.bottom[2] else math.inf
        min_inter_distance = min(left_inter_distance, top_inter_distance, right_inter_distance, bottom_inter_distance)
        nearest_chiplets: list[ComputeChiplet] = []
        if left_inter_distance == min_inter_distance:
            nearest_chiplets.append(self.chiplet_matrix[(0, chiplet.in_module_coord[1])])
        if top_inter_distance == min_inter_distance:
            nearest_chiplets.append(self.chiplet_matrix[(chiplet.in_module_coord[0], 0)])
        if right_inter_distance == min_inter_distance:
            nearest_chiplets.append(self.chiplet_matrix[(self.chiplet_x_num - 1, chiplet.in_module_coord[1])])
        if bottom_inter_distance == min_inter_distance:
            nearest_chiplets.append(self.chiplet_matrix[(chiplet.in_module_coord[0], self.chiplet_y_num - 1)])

        min_distance = math.inf
        nearest_platform = None
        for nearest_chiplet in nearest_chiplets:
            for platform in self.get_chiplet_platforms(nearest_chiplet):
                distance = self.get_distance(node, platform)
                if distance < min_distance:
                    min_distance = distance
                    nearest_platform = platform
        # assert nearest_platform is not None, f"No metro system in the package!!!"
        return nearest_platform

    def get_swap_path(self, source: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Get the swap path between two nodes in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            path (list[tuple[int, int]]): The list of nodes in the swap path.
        """
        # TODO: make sure the path is inside the compute module
        assert self.has_node(source), f"source {source} is not in the compute module {self}"
        assert self.has_node(target), f"target {target} is not in the compute module {self}"
        path = [source]
        append_path = []
        if source == target:
            return path.extend(append_path)
        if source[0] == self.left[0]:
            if source[1] == self.top[0]:
                if (source[0] + 1, source[1]) not in self.get_platforms():
                    source = (source[0] + 1, source[1])
                    path.append(source)
                    source = (source[0], source[1] + 1)
                    path.append(source)
                else:
                    assert (source[0], source[1] + 1) not in self.get_platforms(), f"Data qubit in the top left corner {source} can not get out."
                    source = (source[0], source[1] + 1)
                    path.append(source)
                    source = (source[0] + 1, source[1])
                    path.append(source)
            elif source[1] == self.bottom[0]:
                if (source[0] + 1, source[1]) not in self.get_platforms():
                    source = (source[0] + 1, source[1])
                    path.append(source)
                    source = (source[0], source[1] - 1)
                    path.append(source)
                else:
                    assert (source[0], source[1] - 1) not in self.get_platforms(), f"Data qubit in the bottom left corner {source} can not get out."
                    source = (source[0], source[1] - 1)
                    path.append(source)
                    source = (source[0] + 1, source[1])
                    path.append(source)
            else:
                source = (source[0] + 1, source[1])
                path.append(source)
        elif source[0] == self.right[0]:
            if source[1] == self.top[0]:
                if (source[0] - 1, source[1]) not in self.get_platforms():
                    source = (source[0] - 1, source[1])
                    path.append(source)
                    source = (source[0], source[1] + 1)
                    path.append(source)
                else:
                    assert (source[0], source[1] + 1) not in self.get_platforms(), f"Data qubit in the top right corner {source} can not get out."
                    source = (source[0], source[1] + 1)
                    path.append(source)
                    source = (source[0] - 1, source[1])
                    path.append(source)
            elif source[1] == self.bottom[0]:
                if (source[0] - 1, source[1]) not in self.get_platforms():
                    source = (source[0] - 1, source[1])
                    path.append(source)
                    source = (source[0], source[1] - 1)
                    path.append(source)
                else:
                    assert (source[0], source[1] - 1) not in self.get_platforms(), f"Data qubit in the bottom right corner {source} can not get out."
                    source = (source[0], source[1] - 1)
                    path.append(source)
                    source = (source[0] - 1, source[1])
                    path.append(source)
            else:
                source = (source[0] - 1, source[1])
                path.append(source)
        else:
            if source[1] == self.top[0]:
                source = (source[0], source[1] + 1)
                path.append(source)
            elif source[1] == self.bottom[0]:
                source = (source[0], source[1] - 1)
                path.append(source)
            
        if target[0] == self.left[0]:
            if target[1] == self.top[0]:
                if (target[0] + 1, target[1]) not in self.get_platforms():
                    append_path.insert(0, target)
                    target = (target[0] + 1, target[1])
                    append_path.insert(0, target)
                    target = (target[0], target[1] + 1)
                else:
                    assert (target[0], target[1] + 1) not in self.get_platforms(), f"Data qubit in the top left corner {target} can not get in."
                    append_path.insert(0, target)
                    target = (target[0], target[1] + 1)
                    append_path.insert(0, target)
                    target = (target[0] + 1, target[1])
            elif target[1] == self.bottom[0]:
                if (target[0] + 1, target[1]) not in self.get_platforms():
                    append_path.insert(0, target)
                    target = (target[0] + 1, target[1])
                    append_path.insert(0, target)
                    target = (target[0], target[1] - 1)
                else:
                    assert (target[0], target[1] - 1) not in self.get_platforms(), f"Data qubit in the bottom left corner {target} can not get in."
                    append_path.insert(0, target)
                    target = (target[0], target[1] - 1)
                    append_path.insert(0, target)
                    target = (target[0] + 1, target[1])
            else:
                append_path.insert(0, target)
                target = (target[0] + 1, target[1])
        elif target[0] == self.right[0]:
            if target[1] == self.top[0]:
                if (target[0] - 1, target[1]) not in self.get_platforms():
                    append_path.insert(0, target)
                    target = (target[0] - 1, target[1])
                    append_path.insert(0, target)
                    target = (target[0], target[1] + 1)
                else:
                    assert (target[0], target[1] + 1) not in self.get_platforms(), f"Data qubit in the top right corner {target} can not get in."
                    append_path.insert(0, target)
                    target = (target[0], target[1] + 1)
                    append_path.insert(0, target)
                    target = (target[0] - 1, target[1])
            elif target[1] == self.bottom[0]:
                if (target[0] - 1, target[1]) not in self.get_platforms():
                    append_path.insert(0, target)
                    target = (target[0] - 1, target[1])
                    append_path.insert(0, target)
                    target = (target[0], target[1] - 1)
                else:
                    assert (target[0], target[1] - 1) not in self.get_platforms(), f"Data qubit in the bottom right corner {target} can not get in."
                    append_path.insert(0, target)
                    target = (target[0], target[1] - 1)
                    append_path.insert(0, target)
                    target = (target[0] - 1, target[1])
            else:
                append_path.insert(0, target)
                target = (target[0] - 1, target[1])
        else:
            if target[1] == self.top[0]:
                append_path.insert(0, target)
                target = (target[0], target[1] + 1)
            elif target[1] == self.bottom[0]:
                append_path.insert(0, target)
                target = (target[0], target[1] - 1)
            
        if source == target:
            while len(path) > 1 and len(append_path) > 0 and path[-2] == append_path[0]:
                path.pop()
                append_path.pop(0)
            path.extend(append_path)
            return path

        source_chiplet = self.get_node_chiplet(source)
        target_chiplet = self.get_node_chiplet(target)
        if source_chiplet == target_chiplet:
            if source[0] < target[0]:
                for i in range(source[0] + 1, target[0] + 1):
                    path.append((i, source[1]))
            elif source[0] > target[0]:
                for i in range(source[0] - 1, target[0] - 1, -1):
                    path.append((i, source[1]))
            if source[1] < target[1]:
                for i in range(source[1] + 1, target[1] + 1):
                    path.append((target[0], i))
            elif source[1] > target[1]:
                for i in range(source[1] - 1, target[1] - 1, -1):
                    path.append((target[0], i))
        else:
            # find the x-line index has inter-chiplet coupler
            if source[0] < target[0]:
                inter_coupler_x = self.left[0] + ((source[0] - self.left[0]) // FuncArea.fab_model.inter_sparsity + 1) * FuncArea.fab_model.inter_sparsity
                if inter_coupler_x >= self.right[0]:
                    inter_coupler_x -= FuncArea.fab_model.inter_sparsity
            else:
                inter_coupler_x = self.left[0] + (source[0] - self.left[0]) // FuncArea.fab_model.inter_sparsity * FuncArea.fab_model.inter_sparsity
                if inter_coupler_x == self.left[0]:
                    inter_coupler_x += FuncArea.fab_model.inter_sparsity
            # find the y-line index has inter-chiplet coupler
            if source[1] < target[1]:
                inter_coupler_y = self.top[0] + ((source[1] - self.top[0]) // FuncArea.fab_model.inter_sparsity + 1) * FuncArea.fab_model.inter_sparsity
                if inter_coupler_y >= self.bottom[0]:
                    inter_coupler_y -= FuncArea.fab_model.inter_sparsity
            else:
                inter_coupler_y = self.top[0] + (source[1] - self.top[0]) // FuncArea.fab_model.inter_sparsity * FuncArea.fab_model.inter_sparsity
                if inter_coupler_y == self.top[0]:
                    inter_coupler_y += FuncArea.fab_model.inter_sparsity
                    
            if source[0] < inter_coupler_x:
                for i in range(source[0] + 1, inter_coupler_x + 1):
                    path.append((i, source[1]))
            elif source[0] > inter_coupler_x:
                for i in range(source[0] - 1, inter_coupler_x - 1, -1):
                    path.append((i, source[1]))
            if source[1] < inter_coupler_y:
                for i in range(source[1] + 1, inter_coupler_y + 1):
                    path.append((inter_coupler_x, i))
            elif source[1] > inter_coupler_y:
                for i in range(source[1] - 1, inter_coupler_y - 1, -1):
                    path.append((inter_coupler_x, i))
            if inter_coupler_x < target[0]:
                for i in range(inter_coupler_x + 1, target[0] + 1):
                    path.append((i, inter_coupler_y))
            elif inter_coupler_x > target[0]:
                for i in range(inter_coupler_x - 1, target[0] - 1, -1):
                    path.append((i, inter_coupler_y))

        path.extend(append_path)
        return path

    def get_fidelity(self, source: tuple[int, int], target: tuple[int, int]) -> float:
        """
        Get the fidelity of the shortest path between two nodes in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            fidelity (float): The fidelity of the shortest path between the two nodes.
        """
        assert self.has_node(source), f"source {source} is not in the compute module {self}"
        assert self.has_node(target), f"target {target} is not in the compute module {self}"
        fidelity = 1.0
        left_node, right_node = (source, target) if source[0] < target[0] else (target, source)
        left_chiplet, right_chiplet = self.get_node_chiplet(left_node), self.get_node_chiplet(right_node)
        if left_chiplet.in_module_coord[0] == right_chiplet.in_module_coord[0]:
            fidelity *= left_chiplet.intra_fide ** (right_node[0] - left_node[0])
        else:
            for i in range(left_chiplet.in_module_coord[0], right_chiplet.in_module_coord[0] + 1):
                if i == left_chiplet.in_module_coord[0]:
                    fidelity *= left_chiplet.intra_fide ** (left_chiplet.right - left_node[0])
                elif i == right_chiplet.in_module_coord[0]:
                    curr_chiplet = self.chiplet_matrix[(i, left_chiplet.in_module_coord[1])]
                    fidelity *= curr_chiplet.intra_fide ** (right_node[0] - curr_chiplet.left)
                else:
                    curr_chiplet = self.chiplet_matrix[(i, left_chiplet.in_module_coord[1])]
                    fidelity *= curr_chiplet.intra_fide ** (curr_chiplet.right - curr_chiplet.left)
        moved_left_node = (right_node[0], left_node[1])
        top_node, bottom_node = (moved_left_node, right_node) if moved_left_node[1] < right_node[1] else (right_node, moved_left_node)
        top_chiplet, bottom_chiplet = self.get_node_chiplet(top_node), self.get_node_chiplet(bottom_node)
        if top_chiplet.in_module_coord[1] == bottom_chiplet.in_module_coord[1]:
            fidelity *= top_chiplet.intra_fide ** (bottom_node[1] - top_node[1])
        else:
            for i in range(top_chiplet.in_module_coord[1], bottom_chiplet.in_module_coord[1] + 1):
                if i == top_chiplet.in_module_coord[1]:
                    fidelity *= top_chiplet.intra_fide ** (top_chiplet.bottom - top_node[1])
                elif i == bottom_chiplet.in_module_coord[1]:
                    fidelity *= bottom_chiplet.intra_fide ** (bottom_node[1] - bottom_chiplet.top)
                else:
                    curr_chiplet = self.chiplet_matrix[(top_chiplet.in_module_coord[0], i)]
                    fidelity *= curr_chiplet.intra_fide ** (curr_chiplet.bottom - curr_chiplet.top)
        fidelity *= FuncArea.fab_model.inter_fide ** self.get_inter_distance(source, target)
        return fidelity
