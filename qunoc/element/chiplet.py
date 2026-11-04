import math
from enum import Enum, auto

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from ..utils.utils import get_full_space_tuple, log_decay


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
        compute_geometry (str): The qubit topology in a chiplet, i.e. "square", "hexagon", "heavy square", "heavy hexagon".
        inter_sparsity (int): The sparsity of coupler between chiplets.
        single_fide (float): The fidelity of single gate.
        intra_fide_params (tuple[float, float]): The parameters of the intra_fide function.
        enhanced_fide (float): The unit fidelity of enhanced qst.
        inter_fide (float): The fidelity of inter-chiplet CZ.
        yield_5 (float): The yield of 5-qubit chiplet.
        yield_decay (float): The decay rate of yield with the qubit number.
    """

    def __init__(
        self,
        max_capacity: int = 100,
        max_length: int = 80,
        compute_geometry: str = "square",
        inter_sparsity: int = 3,
        single_fide: float = 0.999,
        intra_fide_dict: dict[int, float] = {27: 0.993, 65: 0.989, 127: 0.988},
        enhanced_fide: float = 0.992**0.2,
        inter_fide: float = 0.983,
        yield_5: float = 0.99,
        yield_decay: float = 2,
    ):
        """
        Initialize a fabrication model.

        Args:
            max_capacity (int): The max number of qubits in a chiplet.
            max_length (int): The max length of a qubit chain in the chiplet.
            compute_geometry (str): The qubit topology in a chiplet, i.e. "square", "hexagon", "heavy square", "heavy hexagon".
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
        return (
            self.yield_5
            * (1 - math.tanh(self.yield_decay * (math.log10(qubit_num) - 2)))
            / 2
        )

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

    def __init__(
        self,
        single_time: float = 34,
        intra_time: float = 25,
        inter_time: float = float(
            np.mean([152, 148, 164, 128, 176, 116, 284, 108, 468, 116])
        ),
        qst_time_constant: float = 125 / 5,
        # qst_time_constant: float = 125 / (5 ** 0.5),
        measure_time: float = 2280,
        reset_time: float = 220,
        eqst_model: str = "legacy",
        eqst_reference_time_ns: float = 125,
        eqst_reference_distance: float = 5,
        eqst_g_max: float | None = 3 * math.pi / 250,
        eqst_max_segment_steps: int = 10,
        eqst_beam_width: int = 128,
        eqst_t1_ns: float = 141500.0,
        eqst_gamma_phi_ns: float | None = None,
        eqst_solver: str = "auto",
        eqst_trajectories: int = 2048,
        eqst_min_trajectories: int = 128,
        eqst_seed: int = 0,
        eqst_atol: float = 0.002,
        eqst_rtol: float = 0.02,
        eqst_confidence: float = 0.95,
        eqst_master_max_dim: int = 128,
        eqst_coupling_policy: str = "preserve-request",
    ):
        """
        功能 / Purpose: Initialize operation timing and dynamical EQST noise / 初始化操作计时与动力学噪声.

        Inputs / 输入:
            single_time (float): The time of single gate.
            intra_time (float): The time of intra-chiplet CZ.
            inter_time (float): The time of inter-chiplet CZ.
            qst_time_constant (float): The time constant of the enhanced QST.
            measure_time (float): The time of measurement.
            reset_time (float): The time of reset.
            eqst_model (str): legacy or Hamiltonian net-map timing / 线性或净映射计时.
            eqst_reference_time_ns (float): Reference transfer time / 参考传输时间(ns).
            eqst_reference_distance (float): Reference edge distance / 参考边数.
            eqst_g_max (float | None): Angular cap in rad/ns, default 6 MHz; None disables / 角耦合上限，None 不限幅.
            eqst_max_segment_steps (int): Planner segment limit / 规划分段上限.
            eqst_beam_width (int): Planner search width / 规划搜索宽度.
            eqst_t1_ns (float): Actual relaxation time; fitted baseline stays 141500 ns / 实际弛豫时间，拟合基准不变.
            eqst_gamma_phi_ns (float | None): Dephasing rate per ns; None uses the frozen reference fit / 每纳秒纯退相干率，None 使用固定参考拟合.
            eqst_solver (str): auto, master, or trajectories / 自动、主方程或轨迹后端.
            eqst_trajectories, eqst_min_trajectories (int): Maximum/minimum MC samples / 最大与最小轨迹数.
            eqst_seed (int): Reproducible sampling seed / 可复现采样种子.
            eqst_atol, eqst_rtol (float): Absolute/relative sampling tolerance / 绝对与相对采样容差.
            eqst_confidence (float): Per-batch confidence level / 单批次置信水平.
            eqst_master_max_dim (int): Density-sector dimension limit (at most 128) / 密度矩阵扇区维数上限.
            eqst_coupling_policy (str): Preserve requested time or run at coupling cap / 保持请求时间或按耦合上限加速.

        Returns / 输出:
            None: Initializes operation timing and EQST settings / 初始化操作计时与 EQST 配置.
        """
        if eqst_model not in {"legacy", "hamiltonian"}:
            raise ValueError("eqst_model must be 'legacy' or 'hamiltonian'")
        self.single_time: float = single_time
        self.intra_time: float = intra_time
        self.inter_time: float = inter_time
        self.qst_time_constant: float = qst_time_constant
        self.measure_time: float = measure_time
        self.reset_time: float = reset_time
        self.eqst_model = eqst_model
        self.eqst_reference_time_ns = eqst_reference_time_ns
        self.eqst_reference_distance = eqst_reference_distance
        self.eqst_g_max = eqst_g_max
        self.eqst_max_segment_steps = eqst_max_segment_steps
        self.eqst_beam_width = eqst_beam_width
        self.eqst_t1_ns = eqst_t1_ns
        self.eqst_gamma_phi_ns = eqst_gamma_phi_ns
        self.eqst_solver = eqst_solver
        self.eqst_trajectories = eqst_trajectories
        self.eqst_min_trajectories = eqst_min_trajectories
        self.eqst_seed = eqst_seed
        self.eqst_atol = eqst_atol
        self.eqst_rtol = eqst_rtol
        self.eqst_confidence = eqst_confidence
        self.eqst_master_max_dim = eqst_master_max_dim
        self.eqst_coupling_policy = eqst_coupling_policy

    def get_eqst_config(self) -> dict:
        """功能 / Purpose: Validate JSON-ready timing/noise settings with old-pickle defaults / 校验可序列化计时与噪声设置，兼容旧文件。

        Inputs / 输入: this operation configuration / 当前操作配置.
        Outputs / 输出: version, planner/noise controls and operation times, without fitting / 模型版本、规划与噪声控制及操作时间，不触发拟合.
        """
        model = getattr(self, "eqst_model", "legacy")
        if model not in {"legacy", "hamiltonian"}:
            raise ValueError("eqst_model must be 'legacy' or 'hamiltonian'")
        settings = {
            "model": model,
            "version": "hamiltonian-nn-lindblad-v3" if model == "hamiltonian" else "legacy-v1",
            "reference_time_ns": getattr(self, "eqst_reference_time_ns", 125),
            "reference_distance": getattr(self, "eqst_reference_distance", 5),
            "g_max": getattr(self, "eqst_g_max", 3 * math.pi / 250),
            "max_segment_steps": getattr(self, "eqst_max_segment_steps", 10),
            "beam_width": getattr(self, "eqst_beam_width", 128),
            "coupling_policy": getattr(self, "eqst_coupling_policy", "preserve-request"),
            "noise_model_version": "hamiltonian-nn-lindblad-v3",
            "t1_ns": getattr(self, "eqst_t1_ns", 141500.0),
            "gamma_phi_ns": getattr(self, "eqst_gamma_phi_ns", None),
            # Hardware sensitivity never changes the reference used to fit gamma.
            # 器件敏感性实验不得随实际 T1 改动参考拟合，避免重新拟合掩盖损耗。
            "calibration_t1_ns": 141500.0,
            "calibration_time_ns": 125.0,
            "calibration_fidelity": 0.992,
            "solver": getattr(self, "eqst_solver", "auto"),
            "trajectories": getattr(self, "eqst_trajectories", 2048),
            "min_trajectories": getattr(self, "eqst_min_trajectories", 128),
            "seed": getattr(self, "eqst_seed", 0),
            "atol": getattr(self, "eqst_atol", 0.002),
            "rtol": getattr(self, "eqst_rtol", 0.02),
            "confidence": getattr(self, "eqst_confidence", 0.95),
            "master_max_dim": getattr(self, "eqst_master_max_dim", 128),
            "ordered": True,
        }
        for name in ("reference_time_ns", "reference_distance", "g_max", "t1_ns", "atol", "confidence"):
            value = settings[name]
            if name == "g_max" and value is None:
                continue
            if isinstance(value, (bool, str)):
                raise ValueError(f"eqst_{name} must be positive and finite")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"eqst_{name} must be positive and finite") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"eqst_{name} must be positive and finite")
            settings[name] = value
        for name in ("gamma_phi_ns", "rtol"):
            value = settings[name]
            if name == "gamma_phi_ns" and value is None:
                continue
            if isinstance(value, (bool, str)):
                raise ValueError(f"eqst_{name} must be nonnegative and finite")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"eqst_{name} must be nonnegative and finite") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"eqst_{name} must be nonnegative and finite")
            settings[name] = value
        for name in ("max_segment_steps", "beam_width", "trajectories", "min_trajectories", "master_max_dim", "seed"):
            value = settings[name]
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"eqst_{name} must be an integer >= {minimum}")
            settings[name] = int(value)
        if not 2 <= settings["min_trajectories"] <= settings["trajectories"]:
            raise ValueError("EQST trajectories must satisfy 2 <= min_trajectories <= trajectories")
        if settings["master_max_dim"] > 128:
            raise ValueError("eqst_master_max_dim must not exceed the resource ceiling 128")
        if settings["confidence"] >= 1:
            raise ValueError("eqst_confidence must lie strictly between 0 and 1")
        if settings["solver"] not in ("auto", "master", "trajectories"):
            raise ValueError("eqst_solver must be 'auto', 'master', or 'trajectories'")
        if settings["coupling_policy"] not in ("preserve-request", "cap-limited-fast"):
            raise ValueError("eqst_coupling_policy must be 'preserve-request' or 'cap-limited-fast'")
        if settings["coupling_policy"] == "cap-limited-fast" and settings["g_max"] is None:
            raise ValueError("cap-limited-fast requires a finite eqst_g_max")
        for name in ("single_time", "intra_time", "inter_time", "qst_time_constant", "measure_time", "reset_time"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
            settings[name] = value
        return settings


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

    def __init__(
        self,
        coordinate: tuple[int, int],
        type: NodeType,
        fide: float,
        role: QubitRole = None,
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

    def __init__(
        self,
        qubit_u: tuple[int, int],
        qubit_v: tuple[int, int],
        type: EdgeType,
        fide: float,
        role: CouplerRole = None,
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
            coordinate (tuple[float, float]): The center coordinate of the chiplet.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_nodes(self) -> list[tuple[int, int]]:
        """
        Get the nodes in the chiplet.

        Returns:
            nodes (list[tuple[int, int]]): The list of nodes in the chiplet.
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
        geometry (str): The qubit topology in the chiplet, i.e. "square", "hexagon", "heavy square", "heavy hexagon".
    """

    def __init__(
        self,
        intra_fide: float,
        in_module_coord: tuple[int, int],
        left: int,
        top: int,
        right: int,
        bottom: int,
        geometry: str = "square",
    ):
        """
        Initialize a compute chiplet.

        Args:
            left (int): The left boundary of the chiplet.
            top (int): The top boundary of the chiplet.
            right (int): The right boundary of the chiplet.
            bottom (int): The bottom boundary of the chiplet.
            geometry (str): The qubit topology in the chiplet, i.e. "square", "hexagon", "heavy square", "heavy hexagon".
        """
        super().__init__(intra_fide)
        self.in_module_coord = in_module_coord
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.geometry = geometry

    def __repr__(self):
        return (
            f"{self.__class__.__name__}_{self.left}_{self.top}_{self.right}_{self.bottom}"
        )

    def get_coordinate(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            for i in range(self.left, self.right + 1):
                for j in range(self.top, self.bottom + 1):
                    self.nodes.append((i, j))
            if self.geometry == "square":
                pass
            elif self.geometry == "hexagon":
                pass
            elif self.geometry == "heavy_square":
                for i in range(self.left + 1, self.right, 2):
                    for j in range(self.top + 1, self.bottom, 2):
                        self.nodes.remove((i, j))
            elif self.geometry == "heavy_hexagon":
                for i in range(self.left + 1, self.right):
                    for j in range(self.top + 1, self.bottom, 2):
                        if (j - self.top) % 4 == 1:
                            if (i - self.left) % 4 != 0:
                                self.nodes.remove((i, j))
                        elif (j - self.top) % 4 == 3:
                            if (i - self.left) % 4 != 2:
                                self.nodes.remove((i, j))
            else:
                raise ValueError(f"Unknown geometry: {self.geometry}")
        return self.nodes

    def has_node(self, node: tuple[int, int]) -> bool:
        if self.geometry == "square":
            return (
                node[0] >= self.left
                and node[0] <= self.right
                and node[1] >= self.top
                and node[1] <= self.bottom
            )
        else:
            return node in self.get_nodes()

    def get_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        if not hasattr(self, "edges"):
            self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for node in self.get_nodes():
                if (node[0] + 1, node[1]) in self.nodes:
                    self.edges.append((node, (node[0] + 1, node[1])))
                if (node[0], node[1] + 1) in self.nodes:
                    self.edges.append((node, (node[0], node[1] + 1)))
            if self.geometry == "square":
                pass
            elif self.geometry == "hexagon":
                for i in range(self.left + 1, self.right):
                    for j in range(self.top, self.bottom):
                        if (j - self.top) % 2 == 0:
                            if (i - self.left) % 2 == 1:
                                self.edges.remove(((i, j), (i, j + 1)))
                        elif (j - self.top) % 2 == 1:
                            if (i - self.left) % 2 == 0:
                                self.edges.remove(((i, j), (i, j + 1)))
            elif self.geometry == "heavy_square":
                pass
            elif self.geometry == "heavy_hexagon":
                pass
        return self.edges

    def has_edge(self, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        return edge in self.get_edges() or (edge[1], edge[0]) in self.get_edges()

    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        assert self.has_node(source), (
            f"source {source} is not in the compute chiplet {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the compute chiplet {self}"
        )

        if self.geometry == "square":
            return abs(source[0] - target[0]) + abs(source[1] - target[1])
        else:
            node_graph = self.get_node_graph()
            if not nx.has_path(node_graph, source, target):
                raise ValueError(
                    f"No path between {source} and {target} in chiplet {self}"
                )
            return nx.shortest_path_length(node_graph, source, target)

    def get_capacity(self) -> int:
        if self.geometry == "square":
            return (self.right - self.left + 1) * (self.bottom - self.top + 1)
        else:
            return len(self.get_nodes())

    def get_node_graph(self) -> nx.Graph:
        """
        Get the node graph of the chiplet.

        Returns:
            graph (nx.Graph): The node graph of the chiplet.
        """
        if not hasattr(self, "node_graph"):
            self.node_graph = nx.Graph()
            for node in self.get_nodes():
                self.node_graph.add_node(node)
            for edge in self.get_edges():
                self.node_graph.add_edge(edge[0], edge[1])
        return self.node_graph


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
        return (
            node[1 - self.axis] == self.pos
            and node[self.axis] >= self.start
            and node[self.axis] <= self.end
        )

    def get_edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        if not hasattr(self, "edges"):
            self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for i in range(self.start, self.end):
                self.edges.append(
                    (
                        get_full_space_tuple(self.pos, self.axis, i),
                        get_full_space_tuple(self.pos, self.axis, i + 1),
                    )
                )
        return self.edges

    def has_edge(self, edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
        return edge in self.get_edges() or (edge[1], edge[0]) in self.get_edges()

    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        assert self.has_node(source), (
            f"source {source} is not in the transfer chiplet {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the transfer chiplet {self}"
        )
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

    def __init__(self, axis: int, pos: tuple[int, int], start: int, end: int):
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

    def __init__(self, axis: int, pos: tuple[int, int], start: int, end: int):
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

    def __init__(self, axis: int, pos: tuple[int, int], start: int, end: int):
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

    def __init__(
        self,
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
                get_full_space_tuple(pos, axis, start[0] - 1),
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
                get_full_space_tuple(pos, axis, end[0] + 1),
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
        return (
            node[1 - self.axis] == self.pos
            and node[self.axis] >= self.start[0]
            and node[self.axis] <= self.end[0]
        )

    def determine_chiplets(self):
        line_length = self.end[0] - self.start[0] + 1
        chiplet_num = (line_length - 1) // FuncArea.fab_model.max_length + 1
        chiplet_length = line_length // chiplet_num
        long_chiplet_num = line_length - chiplet_num * chiplet_length
        self.chiplets: list[TransferChiplet] = []
        for i in range(chiplet_num):
            self.chiplets.append(
                TransferChiplet(
                    FuncArea.fab_model.enhanced_fide,
                    self.axis,
                    self.pos,
                    self.start[0] + i * chiplet_length + min(i, long_chiplet_num),
                    self.start[0]
                    + (i + 1) * chiplet_length
                    + min(i + 1, long_chiplet_num)
                    - 1,
                )
            )
        self.interchanges: set[Coupler] = set()
        if self.start[1]:
            self.interchanges.add(self.start[1])
        if self.end[1]:
            self.interchanges.add(self.end[1])
        for i in range(chiplet_num - 1):
            self.interchanges.add(
                Coupler(
                    get_full_space_tuple(
                        self.pos,
                        self.axis,
                        self.start[0]
                        + (i + 1) * chiplet_length
                        + min(i + 1, long_chiplet_num)
                        - 1,
                    ),
                    get_full_space_tuple(
                        self.pos,
                        self.axis,
                        self.start[0]
                        + (i + 1) * chiplet_length
                        + min(i + 1, long_chiplet_num),
                    ),
                    EdgeType.Inter,
                    FuncArea.fab_model.inter_fide,
                    CouplerRole.Interchange,
                )
            )

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

    def __init__(
        self,
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
        left_gap = (
            ComputeTransferGap(
                left_line.axis, (left_line.pos, left_coord), top_coord, bottom_coord
            )
            if left_line
            else None
        )
        top_gap = (
            ComputeTransferGap(
                top_line.axis, (top_line.pos, top_coord), left_coord, right_coord
            )
            if top_line
            else None
        )
        right_gap = (
            ComputeTransferGap(
                right_line.axis, (right_coord, right_line.pos), top_coord, bottom_coord
            )
            if right_line
            else None
        )
        bottom_gap = (
            ComputeTransferGap(
                bottom_line.axis, (bottom_coord, bottom_line.pos), left_coord, right_coord
            )
            if bottom_line
            else None
        )
        self.left: tuple[int, TransferLine, ComputeTransferGap] = (
            left_coord,
            left_line,
            left_gap,
        )
        self.top: tuple[int, TransferLine, ComputeTransferGap] = (
            top_coord,
            top_line,
            top_gap,
        )
        self.right: tuple[int, TransferLine, ComputeTransferGap] = (
            right_coord,
            right_line,
            right_gap,
        )
        self.bottom: tuple[int, TransferLine, ComputeTransferGap] = (
            bottom_coord,
            bottom_line,
            bottom_gap,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.left[0]}_{self.top[0]}_{self.right[0]}_{self.bottom[0]}"

    def get_nodes(self) -> list[tuple[int, int]]:
        if not hasattr(self, "nodes"):
            self.nodes = []
            if FuncArea.fab_model.compute_geometry == "square":
                for i in range(self.left[0], self.right[0] + 1):
                    for j in range(self.top[0], self.bottom[0] + 1):
                        self.nodes.append((i, j))
            else:
                for chiplet in self.get_chiplets():
                    self.nodes.extend(chiplet.get_nodes())
        return self.nodes

    def has_node(self, node: tuple[int, int]) -> bool:
        if FuncArea.fab_model.compute_geometry == "square":
            return (
                node[0] >= self.left[0]
                and node[0] <= self.right[0]
                and node[1] >= self.top[0]
                and node[1] <= self.bottom[0]
            )
        else:
            return node in self.get_nodes()

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
        self.chiplet_x_size = chiplet_x_size
        self.chiplet_y_size = chiplet_y_size
        self.large_chiplet_x_num = large_chiplet_x_num
        self.large_chiplet_y_num = large_chiplet_y_num
        self.chiplets: list[ComputeChiplet] = []
        self.chiplet_matrix: dict[tuple[int, int], ComputeChiplet] = {}
        for i in range(self.chiplet_x_num):
            for j in range(self.chiplet_y_num):
                self.chiplets.append(
                    ComputeChiplet(
                        FuncArea.fab_model.get_intra_fide(
                            (chiplet_x_size + int(i < large_chiplet_x_num))
                            * (chiplet_y_size + int(j < large_chiplet_y_num))
                        ),
                        (i, j),
                        self.left[0] + i * chiplet_x_size + min(i, large_chiplet_x_num),
                        self.top[0] + j * chiplet_y_size + min(j, large_chiplet_y_num),
                        self.left[0]
                        + (i + 1) * chiplet_x_size
                        + min(i + 1, large_chiplet_x_num)
                        - 1,
                        self.top[0]
                        + (j + 1) * chiplet_y_size
                        + min(j + 1, large_chiplet_y_num)
                        - 1,
                        FuncArea.fab_model.compute_geometry,
                    )
                )
                self.chiplet_matrix[(i, j)] = self.chiplets[-1]
        self.intra_gaps: list[IntraComputeGap] = []
        for i in range(self.chiplet_x_num - 1):
            self.intra_gaps.append(
                IntraComputeGap(
                    1,
                    (
                        self.left[0]
                        + (i + 1) * chiplet_x_size
                        + min(i + 1, large_chiplet_x_num)
                        - 1,
                        self.left[0]
                        + (i + 1) * chiplet_x_size
                        + min(i + 1, large_chiplet_x_num),
                    ),
                    self.top[0],
                    self.bottom[0],
                )
            )
        for j in range(self.chiplet_y_num - 1):
            self.intra_gaps.append(
                IntraComputeGap(
                    0,
                    (
                        self.top[0]
                        + (j + 1) * chiplet_y_size
                        + min(j + 1, large_chiplet_y_num)
                        - 1,
                        self.top[0]
                        + (j + 1) * chiplet_y_size
                        + min(j + 1, large_chiplet_y_num),
                    ),
                    self.left[0],
                    self.right[0],
                )
            )

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
            self.platforms_and_stations: list[
                tuple[tuple[int, int], tuple[int, int]]
            ] = []
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

    @staticmethod
    def _get_chiplet_axis_index(
        coordinate: int,
        module_start: int,
        chiplet_size: int,
        large_chiplet_num: int,
    ) -> int:
        offset = coordinate - module_start
        large_chiplet_size = chiplet_size + 1
        large_chiplet_extent = large_chiplet_num * large_chiplet_size
        if offset < large_chiplet_extent:
            return offset // large_chiplet_size
        return large_chiplet_num + (
            offset - large_chiplet_extent
        ) // chiplet_size

    def _build_node_chiplet_dict(self):
        self.node_chiplet_dict: dict[tuple[int, int], ComputeChiplet] = {}
        for chiplet in self.get_chiplets():
            for chiplet_node in chiplet.get_nodes():
                self.node_chiplet_dict[chiplet_node] = chiplet

    def get_node_chiplet(self, node: tuple[int, int]) -> ComputeChiplet:
        """
        Get the compute chiplet that contains the node.

        Args:
            node (tuple[int, int]): The coordinate of the node.

        Returns:
            ComputeChiplet: The compute chiplet that contains the node.
        """
        if not self.has_node(node):
            raise ValueError(
                f"Node {node} is not in any chiplet in the compute module {self}"
            )

        if FuncArea.fab_model.compute_geometry == "square":
            self.get_chiplets()
            chiplet_x = self._get_chiplet_axis_index(
                node[0],
                self.left[0],
                self.chiplet_x_size,
                self.large_chiplet_x_num,
            )
            chiplet_y = self._get_chiplet_axis_index(
                node[1],
                self.top[0],
                self.chiplet_y_size,
                self.large_chiplet_y_num,
            )
            return self.chiplet_matrix[(chiplet_x, chiplet_y)]

        if not hasattr(self, "node_chiplet_dict"):
            self._build_node_chiplet_dict()
        if node in self.node_chiplet_dict:
            return self.node_chiplet_dict[node]
        raise ValueError(
            f"Node {node} is not in any chiplet in the compute module {self}"
        )

    def get_chiplet_platforms(self, chiplet: ComputeChiplet) -> set[tuple[int, int]]:
        """
        Get the platforms in the compute chiplet.

        Args:
            chiplet (ComputeChiplet): The compute chiplet.

        Returns:
            platforms (set[tuple[int, int]]): The set of platforms in the compute chiplet.
        """
        assert chiplet in self.get_chiplets(), (
            f"chiplet {chiplet} is not in the compute module {self}"
        )
        platforms: set[tuple[int, int]] = set()
        for platform, _ in self.get_platform_station_pairs():
            if chiplet.has_node(platform):
                platforms.add(platform)
        return platforms

    def get_node_graph(self) -> nx.Graph:
        """
        Get the node graph of the compute module.

        Returns:
            graph (nx.Graph): The node graph of the compute module.
        """
        if not hasattr(self, "node_graph"):
            self.node_graph = nx.Graph()
            for chiplet in self.get_chiplets():
                chiplet_node_graph = chiplet.get_node_graph()
                self.node_graph = nx.compose(self.node_graph, chiplet_node_graph)
            for gap in self.get_intra_gaps():
                for coupler in gap.get_couplers().values():
                    self.node_graph.add_edge(coupler.qubit_u, coupler.qubit_v)
        return self.node_graph

    def get_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        """
        Get the distance between two nodes in the compute module, which is the number of edges in the shortest path.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            distance (int): The distance between the two nodes, which is the number of edges in the shortest path.
        """
        assert self.has_node(source), (
            f"source {source} is not in the compute module {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the compute module {self}"
        )

        if FuncArea.fab_model.compute_geometry == "square":
            return abs(source[0] - target[0]) + abs(source[1] - target[1])
        else:
            node_graph = self.get_node_graph()
            if not nx.has_path(node_graph, source, target):
                raise ValueError(
                    f"No path between {source} and {target} in compute module {self}"
                )
            return nx.shortest_path_length(node_graph, source, target)

    def get_inter_distance(self, source: tuple[int, int], target: tuple[int, int]) -> int:
        """
        Get the number of inter-chiplet coupler in the shortest path between two nodes in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            distance (int): The number of inter-chiplet coupler in the shortest path between the two nodes.
        """
        assert self.has_node(source), (
            f"source {source} is not in the compute module {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the compute module {self}"
        )
        source_chiplet = self.get_node_chiplet(source)
        target_chiplet = self.get_node_chiplet(target)
        source_chiplet_coord = source_chiplet.in_module_coord
        target_chiplet_coord = target_chiplet.in_module_coord
        return abs(source_chiplet_coord[0] - target_chiplet_coord[0]) + abs(
            source_chiplet_coord[1] - target_chiplet_coord[1]
        )

    def _get_nearest_platform_by_scan(
        self,
        node: tuple[int, int],
        nearest_chiplets: list[ComputeChiplet],
    ) -> tuple[int, int]:
        min_distance = math.inf
        nearest_platform = None
        visited_chiplets: set[ComputeChiplet] = set()
        for nearest_chiplet in nearest_chiplets:
            if nearest_chiplet in visited_chiplets:
                continue
            visited_chiplets.add(nearest_chiplet)
            for platform in self.get_chiplet_platforms(nearest_chiplet):
                distance = self.get_distance(node, platform)
                if distance < min_distance:
                    min_distance = distance
                    nearest_platform = platform
        assert nearest_platform is not None, "No metro system in the package!!!"
        return nearest_platform

    @staticmethod
    def _get_gap_candidate_indices(
        gap: ComputeTransferGap,
        chiplet: ComputeChiplet,
        target_index: int,
    ) -> list[int]:
        step = FuncArea.fab_model.inter_sparsity
        if step <= 0:
            return []

        if gap.axis == 0:
            span_start = max(gap.start, chiplet.left)
            span_end = min(gap.end, chiplet.right)
        else:
            span_start = max(gap.start, chiplet.top)
            span_end = min(gap.end, chiplet.bottom)
        if span_start > span_end:
            return []

        first_k = max(0, (span_start - gap.start + step - 1) // step)
        last_k = (span_end - gap.start) // step
        if first_k > last_k:
            return []

        base_k = (target_index - gap.start) // step
        candidate_ks = {
            min(max(base_k, first_k), last_k),
            min(max(base_k + 1, first_k), last_k),
        }
        return [gap.start + k * step for k in sorted(candidate_ks)]

    def _get_chiplet_gap_platform_candidates(
        self,
        gap: ComputeTransferGap,
        platform_pos: int,
        chiplet: ComputeChiplet,
        node: tuple[int, int],
    ) -> list[tuple[int, int]]:
        return [
            get_full_space_tuple(platform_pos, gap.axis, index)
            for index in self._get_gap_candidate_indices(
                gap, chiplet, node[gap.axis]
            )
        ]

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
        right_inter_distance = (
            self.chiplet_x_num - chiplet.in_module_coord[0] - 1
            if self.right[2]
            else math.inf
        )
        bottom_inter_distance = (
            self.chiplet_y_num - chiplet.in_module_coord[1] - 1
            if self.bottom[2]
            else math.inf
        )
        min_inter_distance = min(
            left_inter_distance,
            top_inter_distance,
            right_inter_distance,
            bottom_inter_distance,
        )
        boundary_choices: list[
            tuple[ComputeChiplet, ComputeTransferGap, int]
        ] = []
        if self.left[2] and left_inter_distance == min_inter_distance:
            boundary_choices.append(
                (
                    self.chiplet_matrix[(0, chiplet.in_module_coord[1])],
                    self.left[2],
                    self.left[0],
                )
            )
        if self.top[2] and top_inter_distance == min_inter_distance:
            boundary_choices.append(
                (
                    self.chiplet_matrix[(chiplet.in_module_coord[0], 0)],
                    self.top[2],
                    self.top[0],
                )
            )
        if self.right[2] and right_inter_distance == min_inter_distance:
            boundary_choices.append(
                (
                    self.chiplet_matrix[
                        (self.chiplet_x_num - 1, chiplet.in_module_coord[1])
                    ],
                    self.right[2],
                    self.right[0],
                )
            )
        if self.bottom[2] and bottom_inter_distance == min_inter_distance:
            boundary_choices.append(
                (
                    self.chiplet_matrix[
                        (chiplet.in_module_coord[0], self.chiplet_y_num - 1)
                    ],
                    self.bottom[2],
                    self.bottom[0],
                )
            )

        nearest_chiplets = [choice[0] for choice in boundary_choices]
        if FuncArea.fab_model.compute_geometry != "square":
            return self._get_nearest_platform_by_scan(node, nearest_chiplets)

        min_distance = math.inf
        nearest_platform = None
        for nearest_chiplet, gap, platform_pos in boundary_choices:
            for platform in self._get_chiplet_gap_platform_candidates(
                gap, platform_pos, nearest_chiplet, node
            ):
                distance = self.get_distance(node, platform)
                if distance < min_distance:
                    min_distance = distance
                    nearest_platform = platform
        assert nearest_platform is not None, "No metro system in the package!!!"
        return nearest_platform

    def get_swap_path(
        self, source: tuple[int, int], target: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """
        Get the swap path between two nodes in the compute module.

        Args:
            source (tuple[int, int]): The coordinate of the source node.
            target (tuple[int, int]): The coordinate of the target node.

        Returns:
            path (list[tuple[int, int]]): The list of nodes in the swap path.
        """
        assert self.has_node(source), (
            f"source {source} is not in the compute module {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the compute module {self}"
        )

        if FuncArea.fab_model.compute_geometry == "square":
            origin_source = source
            origin_target = target
            path = [source]
            append_path = []
            platforms = self.get_platforms()
            module_x_size = self.right[0] - self.left[0] + 1
            module_y_size = self.bottom[0] - self.top[0] + 1

            def _validate_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
                assert path and path[0] == origin_source, (
                    f"Invalid swap path source in compute module {self}: {path}"
                )
                if origin_target in path:
                    path = path[: path.index(origin_target) + 1]

                normalized_path: list[tuple[int, int]] = []
                node_indices: dict[tuple[int, int], int] = {}
                for node in path:
                    if node in node_indices:
                        duplicate_index = node_indices[node]
                        for removed_node in normalized_path[duplicate_index + 1 :]:
                            node_indices.pop(removed_node, None)
                        normalized_path = normalized_path[: duplicate_index + 1]
                    else:
                        node_indices[node] = len(normalized_path)
                        normalized_path.append(node)
                path = normalized_path

                assert all(self.has_node(node) for node in path), (
                    f"Invalid swap path outside compute module {self}: {path}"
                )

                node_graph = self.get_node_graph()
                if path[-1] != origin_target or any(
                    not node_graph.has_edge(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                ):
                    if not nx.has_path(node_graph, origin_source, origin_target):
                        raise ValueError(
                            f"No path between {origin_source} and {origin_target} in compute module {self}"
                        )
                    return nx.shortest_path(node_graph, origin_source, origin_target)
                return path

            def _step_node(
                node: tuple[int, int], step: tuple[int, int]
            ) -> tuple[int, int]:
                return node[0] + step[0], node[1] + step[1]

            def _prefer_non_platform_steps(
                node: tuple[int, int],
                first_step: tuple[int, int],
                second_step: tuple[int, int],
            ) -> list[tuple[int, int]]:
                first_node = _step_node(node, first_step)
                second_node = _step_node(node, second_step)
                if self.has_node(first_node) and first_node not in platforms:
                    return [first_step, second_step]
                if self.has_node(second_node) and second_node not in platforms:
                    return [second_step, first_step]
                return [first_step, second_step]

            def _boundary_steps(node: tuple[int, int]) -> list[tuple[int, int]]:
                if node[0] == self.left[0]:
                    horizontal_step = (1, 0)
                    if node[1] == self.top[0]:
                        return _prefer_non_platform_steps(
                            node, horizontal_step, (0, 1)
                        )
                    if node[1] == self.bottom[0]:
                        return _prefer_non_platform_steps(
                            node, horizontal_step, (0, -1)
                        )
                    return [horizontal_step]
                if node[0] == self.right[0]:
                    horizontal_step = (-1, 0)
                    if node[1] == self.top[0]:
                        return _prefer_non_platform_steps(
                            node, horizontal_step, (0, 1)
                        )
                    if node[1] == self.bottom[0]:
                        return _prefer_non_platform_steps(
                            node, horizontal_step, (0, -1)
                        )
                    return [horizontal_step]
                if node[1] == self.top[0]:
                    return [(0, 1)]
                if node[1] == self.bottom[0]:
                    return [(0, -1)]
                return []

            if source == target:
                path.extend(append_path)
                return _validate_path(path)

            for step in _boundary_steps(source):
                next_source = _step_node(source, step)
                if self.has_node(next_source):
                    source = next_source
                    path.append(source)

            for step in _boundary_steps(target):
                next_target = _step_node(target, step)
                if self.has_node(next_target):
                    append_path.insert(0, target)
                    target = next_target

            if source == target:
                while (
                    len(path) > 1 and len(append_path) > 0 and path[-2] == append_path[0]
                ):
                    path.pop()
                    append_path.pop(0)
                path.extend(append_path)
                return _validate_path(path)

            if module_x_size == 1 or module_y_size == 1:
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
                path.extend(append_path)
                return _validate_path(path)

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
                def _axis_candidates(
                    axis: int, fixed_coord: int
                ) -> list[int]:
                    step = FuncArea.fab_model.inter_sparsity
                    if step <= 0:
                        return []
                    if axis == 0:
                        start, end = self.left[0], self.right[0]
                        return [
                            coord
                            for coord in range(start, end + 1)
                            if (coord - start) % step == 0
                            and self.has_node((coord, fixed_coord))
                        ]
                    start, end = self.top[0], self.bottom[0]
                    return [
                        coord
                        for coord in range(start, end + 1)
                        if (coord - start) % step == 0
                        and self.has_node((fixed_coord, coord))
                    ]

                def _fallback_shortest_path() -> list[tuple[int, int]]:
                    node_graph = self.get_node_graph()
                    if not nx.has_path(node_graph, source, target):
                        raise ValueError(
                            f"No path between {source} and {target} in compute module {self}"
                        )
                    fallback_path = path.copy()
                    fallback_path.extend(nx.shortest_path(node_graph, source, target)[1:])
                    fallback_path.extend(append_path)
                    return _validate_path(fallback_path)

                # find the x-line index has inter-chiplet coupler
                if source_chiplet.in_module_coord[1] == target_chiplet.in_module_coord[1]:
                    inter_coupler_x = source[0]
                else:
                    x_candidates = _axis_candidates(0, source[1])
                    x_candidates = [
                        coord
                        for coord in x_candidates
                        if self.get_node_chiplet((coord, source[1])).in_module_coord[0]
                        == source_chiplet.in_module_coord[0]
                    ]
                    if not x_candidates:
                        return _fallback_shortest_path()
                    inter_coupler_x = min(
                        x_candidates, key=lambda coord: abs(coord - source[0])
                    )
                # find the y-line index has inter-chiplet coupler
                if source_chiplet.in_module_coord[0] == target_chiplet.in_module_coord[0]:
                    inter_coupler_y = target[1]
                else:
                    y_candidates = _axis_candidates(1, target[0])
                    y_candidates = [
                        coord
                        for coord in y_candidates
                        if self.get_node_chiplet((target[0], coord)).in_module_coord[1]
                        == target_chiplet.in_module_coord[1]
                    ]
                    if not y_candidates:
                        return _fallback_shortest_path()
                    inter_coupler_y = min(
                        y_candidates, key=lambda coord: abs(coord - target[1])
                    )

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
                if inter_coupler_y < target[1]:
                    for i in range(inter_coupler_y + 1, target[1] + 1):
                        path.append((target[0], i))
                elif inter_coupler_y > target[1]:
                    for i in range(inter_coupler_y - 1, target[1] - 1, -1):
                        path.append((target[0], i))

            path.extend(append_path)
            return _validate_path(path)
        else:
            node_graph = self.get_node_graph()
            if not nx.has_path(node_graph, source, target):
                raise ValueError(
                    f"No path between {source} and {target} in compute module {self}"
                )
            path = nx.shortest_path(node_graph, source, target)
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
        assert self.has_node(source), (
            f"source {source} is not in the compute module {self}"
        )
        assert self.has_node(target), (
            f"target {target} is not in the compute module {self}"
        )

        fidelity = 1.0
        if FuncArea.fab_model.compute_geometry == "square":
            left_node, right_node = (
                (source, target) if source[0] < target[0] else (target, source)
            )
            left_chiplet, right_chiplet = (
                self.get_node_chiplet(left_node),
                self.get_node_chiplet(right_node),
            )
            if left_chiplet.in_module_coord[0] == right_chiplet.in_module_coord[0]:
                fidelity *= (left_chiplet.intra_fide**3) ** (right_node[0] - left_node[0])
            else:
                for i in range(
                    left_chiplet.in_module_coord[0], right_chiplet.in_module_coord[0] + 1
                ):
                    if i == left_chiplet.in_module_coord[0]:
                        fidelity *= (left_chiplet.intra_fide**3) ** (
                            left_chiplet.right - left_node[0]
                        )
                    elif i == right_chiplet.in_module_coord[0]:
                        curr_chiplet = self.chiplet_matrix[
                            (i, left_chiplet.in_module_coord[1])
                        ]
                        fidelity *= (curr_chiplet.intra_fide**3) ** (
                            right_node[0] - curr_chiplet.left
                        )
                    else:
                        curr_chiplet = self.chiplet_matrix[
                            (i, left_chiplet.in_module_coord[1])
                        ]
                        fidelity *= (curr_chiplet.intra_fide**3) ** (
                            curr_chiplet.right - curr_chiplet.left
                        )
            moved_left_node = (right_node[0], left_node[1])
            top_node, bottom_node = (
                (moved_left_node, right_node)
                if moved_left_node[1] < right_node[1]
                else (right_node, moved_left_node)
            )
            top_chiplet, bottom_chiplet = (
                self.get_node_chiplet(top_node),
                self.get_node_chiplet(bottom_node),
            )
            if top_chiplet.in_module_coord[1] == bottom_chiplet.in_module_coord[1]:
                fidelity *= (top_chiplet.intra_fide**3) ** (bottom_node[1] - top_node[1])
            else:
                for i in range(
                    top_chiplet.in_module_coord[1], bottom_chiplet.in_module_coord[1] + 1
                ):
                    if i == top_chiplet.in_module_coord[1]:
                        fidelity *= (top_chiplet.intra_fide**3) ** (
                            top_chiplet.bottom - top_node[1]
                        )
                    elif i == bottom_chiplet.in_module_coord[1]:
                        fidelity *= (bottom_chiplet.intra_fide**3) ** (
                            bottom_node[1] - bottom_chiplet.top
                        )
                    else:
                        curr_chiplet = self.chiplet_matrix[
                            (top_chiplet.in_module_coord[0], i)
                        ]
                        fidelity *= (curr_chiplet.intra_fide**3) ** (
                            curr_chiplet.bottom - curr_chiplet.top
                        )
            fidelity *= (FuncArea.fab_model.inter_fide**3) ** self.get_inter_distance(
                source, target
            )
        else:
            path = self.get_swap_path(source, target)
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i + 1]
                source_chiplet = self.get_node_chiplet(source_node)
                target_chiplet = self.get_node_chiplet(target_node)
                if source_chiplet == target_chiplet:
                    fidelity *= source_chiplet.intra_fide**3
                else:
                    fidelity *= FuncArea.fab_model.inter_fide**3
        return fidelity
        return fidelity
