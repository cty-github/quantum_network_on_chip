import math
import numpy as np
from enum import Enum, auto
from scipy.optimize import curve_fit

from ..utils.utils import get_full_space_tuple


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

    def get_intra_fide(self, qubit_num: int) -> float:
        """
        Get the fidelity of intra-chiplet CZ according to qubit num in the chiplet.
        
        Args:
            qubit_num (int): The number of qubits in the chiplet.
        
        Returns:
            float: The fidelity of intra-chiplet CZ.
        """
        return self.intra_fide_log_decay(qubit_num, *self.intra_fide_params)

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


class Qubit:
    """
    Represents the physical qubit in the package.

    Attributes:
        coordinate (tuple[int, int]): The coordinate of the qubit.
        type (NodeType): The type of the chiplet that the qubit belongs to.
        fide (float): The single gate fidelity of the qubit.
        area (FuncArea): The function area that the qubit belongs to.
        role (QubitRole): The role of the qubit.
    """
    
    def __init__(self, coordinate: tuple[int, int], type: NodeType, fide: float):
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
        self.area: FuncArea = None
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
    """
    
    def __init__(self, qubit_u: Qubit, qubit_v: Qubit, type: EdgeType, fide: float):
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
        self.role: CouplerRole = None

    def __repr__(self):
        return f"{self.__class__.__name__}{self.qubit_u.coordinate}{self.qubit_v.coordinate}"


class Chiplet:
    """
    Represents the chiplet in the package.

    Attributes:
        type (NodeType): The type of the chiplet.
    """

    def __init__(self, type: NodeType):
        """
        Initialize a chiplet.

        Args:
            type (NodeType): The type of the chiplet.
        """
        self.type = type
    
    def get_nodes(self) -> list[tuple[int, int]]:
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
    
    def get_coordinate(self) -> tuple[float, float]:
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

    def __init__(self, left: int, top: int, right: int, bottom: int):
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
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
        return self.nodes
    
    def get_capacity(self) -> int:
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)
    
    def get_coordinate(self) -> tuple[float, float]:
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

    def __init__(self, axis: int, pos: int, start: int, end: int):
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
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start, self.end + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes
    
    def get_capacity(self) -> int:
        return self.end - self.start + 1

    def get_coordinate(self) -> tuple[float, float]:
        return get_full_space_tuple(self.pos, self.axis, (self.start + self.end) / 2)


class FuncArea:
    """
    Represents the function area in the package.

    Attributes:
        type (NodeType): The type of the function area.
        fab_model (FabricateModel): The fabrication model of the package.
        nodes (list[tuple[int, int]]): The list of nodes in the function area.
        chiplets (list[Chiplet]): The list of chiplets in the function area.
    """

    def __init__(self, type: NodeType, fab_model: FabricateModel):
        """
        Initialize a function area.

        Args:
            type (NodeType): The type of the function area.
            fab_model (FabricateModel): The fabrication model of the package.
        """
        self.type = type
        self.fab_model = fab_model
    
    def get_nodes(self) -> list[tuple[int, int]]:
        """
        Get the nodes in the function region.

        Returns:
            nodes (list[tuple[int, int]]): The list of nodes in the function region.
        """

    def in_area(self, node: tuple[int, int]) -> bool:
        """
        Check if the node is in the function area.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            bool: True if the node is in the function area, False otherwise.
        """
        return node in self.get_nodes()
    
    def get_chiplets(self) -> list[Chiplet]:
        """
        Get the chiplets in the function region.

        Returns:
            chiplets (list[Chiplet]): The list of chiplets in the function region.
        """

class ComputeModule(FuncArea):
    """
    Represents the compute module in the package.

    Attributes:
        left (int): The left boundary of the module.
        top (int): The top boundary of the module.
        right (int): The right boundary of the module.
        bottom (int): The bottom boundary of the module.
    """
    
    def __init__(self, fab_model: FabricateModel, left: int, top: int, right: int, bottom: int):
        """
        Initialize a compute module.

        Args:
            fab_model (FabricateModel): The fabrication model of the package.
            left (int): The left boundary of the module.
            top (int): The top boundary of the module.
            right (int): The right boundary of the module.
            bottom (int): The bottom boundary of the module.
        """
        super().__init__(NodeType.Compute, fab_model)
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left}_{self.top}_{self.right}_{self.bottom}"
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
        return self.nodes
    
    def get_chiplets(self) -> list[ComputeChiplet]:
        """
        TODO: Set the topology of compute chiplets according to the fabricate model.
        """
        if not hasattr(self, "chiplets"):
            self.chiplets: list[ComputeChiplet] = []
            self.intra_gaps: list[IntraAreaGap] = []
            module_x_size = self.right - self.left + 1
            module_y_size = self.bottom - self.top + 1
            chiplet_x_num = (module_x_size - 1) // self.fab_model.max_square_size + 1
            chiplet_y_num = (module_y_size - 1) // self.fab_model.max_square_size + 1
            chiplet_x_size = module_x_size // chiplet_x_num
            chiplet_y_size = module_y_size // chiplet_y_num
            large_chiplet_x_num = module_x_size - chiplet_x_num * chiplet_x_size
            large_chiplet_y_num = module_y_size - chiplet_y_num * chiplet_y_size
            for i in range(chiplet_x_num):
                for j in range(chiplet_y_num):
                    self.chiplets.append(ComputeChiplet(
                        self.left + i * chiplet_x_size + min(i, large_chiplet_x_num),
                        self.top + j * chiplet_y_size + min(j, large_chiplet_y_num),
                        self.left + (i + 1) * chiplet_x_size + min(i + 1, large_chiplet_x_num) - 1,
                        self.top + (j + 1) * chiplet_y_size + min(j + 1, large_chiplet_y_num) - 1,
                    ))
        return self.chiplets

    def get_platforms(self) -> list[tuple[float, float]]:
        """
        Get the platform qubits in the chiplet.
        """
        if not hasattr(self, "platforms"):
            self.platforms = []
        return self.platforms

class TransferLine(FuncArea):
    """
    Represents the transfer line in the package.

    Attributes:
        axis (int): The axis of the line.
        pos (int): The position of the line.
        start (int): The start point of the line.
        end (int): The end point of the line.
    """

    def __init__(self, fab_model: FabricateModel, axis: int, pos: int, start: int, end: int):
        """
        Initialize a transfer line.

        Args:
            fab_model (FabricateModel): The fabrication model of the package.
            axis (int): The axis of the line.
            pos (int): The position of the line.
            start (int): The start point of the line.
            end (int): The end point of the line.
        """
        super().__init__(NodeType.Transfer, fab_model)
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.axis}_{self.pos}_{self.start}_{self.end}"
    
    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.start, self.end + 1):
                self.nodes.append(get_full_space_tuple(self.pos, self.axis, i))
        return self.nodes
    
    def get_chiplets(self) -> list[TransferChiplet]:
        if not hasattr(self, "chiplets"):
            self.chiplets: list[TransferChiplet] = []
            line_length = self.end - self.start + 1
            chiplet_num = (line_length - 1) // self.fab_model.max_length + 1
            chiplet_length = line_length // chiplet_num
            long_chiplet_num = line_length - chiplet_num * chiplet_length
            for i in range(chiplet_num):
                self.chiplets.append(TransferChiplet(
                    self.axis,
                    self.pos,
                    self.start + i * chiplet_length + min(i, long_chiplet_num),
                    self.start + (i + 1) * chiplet_length + min(i + 1, long_chiplet_num) - 1,
                ))
        return self.chiplets

    def get_stations(self) -> list[tuple[float, float]]:
        """
        Get the station qubits in the chiplet.
        """
        if not hasattr(self, "stations"):
            self.stations = []
        return self.stations


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

    def __init__(self, fab_model: FabricateModel, axis:int, pos: tuple[int, int], start: int, end: int):
        """
        Initialize a gap.

        Args:
            fab_model (FabricateModel): The fabrication model of the package.
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
        """
        self.fab_model = fab_model
        self.axis = axis
        self.pos = pos
        self.start = start
        self.end = end
    
    def get_edges(self) -> tuple[int, tuple[tuple[int,int], tuple[int,int]]]:
        """
        Get the edges in the gap.

        Returns:
            edges (dict[int, tuple[tuple[int,int], tuple[int,int]]]): The edges in the gap.
        """
        if not hasattr(self, "edges"):
            self.edges = {}
            for i in range(self.start, self.end + 1, self.fab_model.inter_sparsity):
                self.edges[i] = (get_full_space_tuple(self.pos[0], self.axis, i), get_full_space_tuple(self.pos[1], self.axis, i))
        return self.edges

class InterAreaGap(Gap):
    """
    Represents the gap between two function area.

    Attributes:
        compute_module (ComputeModule): The compute module on one side of the gap.
        transfer_line (TransferLine): The transfer line on the other side of the gap.
    """

    def __init__(self, fab_model: FabricateModel, axis:int, pos: tuple[int, int], start: int, end: int, compute_module: ComputeModule, transfer_line: TransferLine):
        """
        Initialize a gap between two function area.

        Args:
            fab_model (FabricateModel): The fabrication model of the package.
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
            compute_module (ComputeModule): The compute module on one side of the gap.
            transfer_line (TransferLine): The transfer line on the other side of the gap.
        """
        super().__init__(fab_model, axis, pos, start, end)
        self.compute_module = compute_module
        self.transfer_line = transfer_line

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.compute_module}_{self.transfer_line}"

class IntraAreaGap(Gap):
    """
    Represents the gap in a function area.

    Attributes:
        chiplet_u (Chiplet): The chiplet u on one side of the gap.
        chiplet_v (Chiplet): The chiplet v on the other side of the gap.
    """

    def __init__(self, fab_model: FabricateModel, axis:int, pos: tuple[int, int], start: int, end: int, chiplet_u: Chiplet, chiplet_v: Chiplet):
        """
        Initialize a gap in a function area.

        Args:
            fab_model (FabricateModel): The fabrication model of the package.
            axis (int): The axis of the gap.
            pos (tuple[int, int]): The position of the gap.
            start (int): The start point of the gap.
            end (int): The end point of the gap.
            chiplet_u (Chiplet): The chiplet u on one side of the gap.
            chiplet_v (Chiplet): The chiplet v on the other side of the gap.
        """
        super().__init__(fab_model, axis, pos, start, end)
        self.chiplet_u = chiplet_u
        self.chiplet_v = chiplet_v
    
    def __repr__(self):
        return f"{self.__class__.__name__}_{self.chiplet_u}_{self.chiplet_v}"
