import os
import random
from itertools import product
from collections import defaultdict

import networkx as nx
from anytree import Node
import matplotlib.pyplot as plt

from ..utils.utils import color_pool, get_unique_file_save_path, make_gif
from .chiplet import *


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


class Road:
    """
    Represents the road between two adjacent nodes in a path.

    Attributes:
        source (tuple[int, int]): The source node of the road.
        target (tuple[int, int]): The target node of the road.
    """

    def __init__(self, source: tuple[int, int], target: tuple[int, int]):
        """
        Initialize a road.
        
        Args:
            source (tuple[int, int]): The source node of the road.
            target (tuple[int, int]): The target node of the road.
        """
        self.source = source
        self.target = target

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.source}_{self.target}"
    
    def get_fidelity(self) -> float:
        """
        Calculate the fidelity of the road.

        Returns:
            float: The fidelity of the road.
        """

class IntraComputeRoad(Road):
    """
    Represents the road between two adjacent nodes in the same compute module.
    """

    def __init__(self, source: tuple[int, int], target: tuple[int, int], module: ComputeModule):
        """
        Initialize a road inside a compute module.

        Args:
            source (tuple[int, int]): The source node of the road.
            target (tuple[int, int]): The target node of the road.
            module (ComputeModule): The compute module to which the two nodes belong.
        """
        super().__init__(source, target)
        self.module = module

    def get_fidelity(self) -> float:
        distance = self.module.get_distance(self.source, self.target)
        inter_distance = self.module.get_inter_distance(self.source, self.target)
        return self.module.fab_model ** (distance - inter_distance) * self.module.inter_fide ** inter_distance

class IntraTransferRoad(Road):
    """
    Represents the road between two adjacent nodes in the same transfer chiplet.

    Attributes:
        chiplet (TransferChiplet): The transfer chiplet to which the two nodes belong.
    """

    def __init__(self, source: tuple[int, int], target: tuple[int, int], chiplet: TransferChiplet):
        """
        Initialize a road inside a transfer chiplet.

        Args:
            source (tuple[int, int]): The source node of the road.
            target (tuple[int, int]): The target node of the road.
            chiplet (TransferChiplet): The transfer chiplet to which the two nodes belong.
        """
        super().__init__(source, target)
        self.chiplet = chiplet

    def get_fidelity(self) -> float:
        return self.chiplet.intra_fide ** self.chiplet.get_distance(self.source, self.target)

class InterChipletRoad(Road):
    """
    Represents the road between two adjacent nodes in different chiplets.

    Attributes:
        source_chiplet (Chiplet): The source chiplet of the road.
        target_chiplet (Chiplet): The target chiplet of the road.
    """

    def __init__(self, source: tuple[int, int], target: tuple[int, int], source_chiplet: Chiplet, target_chiplet: Chiplet, fide: float):
        """
        Initialize a road between two chiplets.

        Args:
            source (tuple[int, int]): The source node of the road.
            target (tuple[int, int]): The target node of the road.
            source_chiplet (Chiplet): The source chiplet of the road.
            target_chiplet (Chiplet): The target chiplet of the road.
            fide (float): The fidelity of the road.
        """
        super().__init__(source, target)
        self.source_chiplet = source_chiplet
        self.target_chiplet = target_chiplet
        self.fide = fide
    
    def get_fidelity(self) -> float:
        return self.fide

class Path:
    """
    Represents the path on the quantum chiplet architecture (quantum network-on-chip architecture).

    Attributes:
        roads (list[Road]): The list of roads in the path.
    """

    def __init__(self, roads: list[Road]):
        """
        Initialize a path.
        
        Args:
            roads (list[Road]): The list of roads in the path.
        """
        for i in range(len(roads) - 1):
            assert roads[i].target == roads[i + 1].source, f"Invalid path: {roads[i]} -> {roads[i + 1]}"
        self.roads = roads
    
    def get_fidelity(self) -> float:
        """
        Calculate the fidelity of the path.

        Returns:
            float: The fidelity of the path.
        """
        fidelity = 1.0
        for road in self.roads:
            fidelity *= road.get_fidelity()
        return fidelity


class Package:
    """
    Represents the package of the quantum chiplet architecture (quantum network-on-chip architecture).

    Attributes:
        package_size (tuple[int, int]): The size of the package.
        node_num (int): The number of nodes in the package.
        fab_model (FabricateModel): The fabrication model of the package.
        design_scheme (str): The design scheme of the package, i.e. "top_down", "bottom_up".
        design_strategy (str): The design strategy of the package, i.e. "heuristic", "greedy", "rl".
        top_down_max_step (int): The max number of steps for top down design.
        result_dir (str): The directory to save the design results.

        package_graph (nx.Graph): The graph recording the grid structure of the package.
            node: tuple[int, int], {"type": NodeType, "qubit": Qubit}
            edge: tuple[node, node], {"type": EdgeType, "coupler": Coupler}
        compute_nodes (set[tuple[int, int]]): The set of nodes which are on the compute chiplets.
        transfer_nodes (set[tuple[int, int]]): The set of nodes which are on the transfer chiplets.

        design_tree_root (Node): The root node of the design tree.
        design_tree_dict (dict[str, Node]): The tree structure of the design process.
        compute_modules (list[ComputeModule]): The list of modules which are padded with compute chiplets.
        transfer_lines (list[TransferLine]): The list of lines which are padded with transfer chiplets.

        compute_chiplets (list[ComputeChiplet]): The list of chiplets, which are used for compute.
        transfer_chiplets (list[TransferChiplet]): The list of chiplets, which are used for transfer.
        stations (list[tuple[int, int]]): The list of stations in the package.
        platform_stations_dict (dict[tuple[int, int], list[tuple[int, int]]]): The dictionary recording neighbor stations of platforms in the package.
        interchange (set[Coupler]): The set of couplers which are used for interchange.

        metro_topology (nx.Graph): The subgraph view of package_graph recording the topology of transfer qubits in the package.
        interchange_dict (dict[TransferChiplet, dict[TransferChiplet, list[TransferChiplet]]]): The dictionary recording the interchange path between transfer chiplets.
    """

    def __init__(self, 
        package_size: tuple[int, int] = (100, 100), 
        fab_model: FabricateModel = FabricateModel(), 
        design_scheme: str = "top_down", 
        design_strategy: str = "heuristic", 
        top_down_max_step: int = 10, 
        bottom_up_init_type: np.ndarray = None, 
        result_dir: str = None,
    ):
        """
        Initialize a package.

        Args:
            package_size (tuple[int, int]): The size of the package.
            fab_model (FabricateModel): The fabrication model of the package.
            design_scheme (str): The design scheme of the package, i.e. "top_down", "bottom_up".
            design_strategy (str): The design strategy of the package, i.e. "heuristic", "greedy", "rl".
            top_down_max_step (int): The max number of steps for top down design.
            bottom_up_init_type (np.ndarray): The initial type of nodes used for bottom up design.
            result_dir (str): The directory to save the design results.
        """
        self.package_size = package_size
        self.node_num = package_size[0] * package_size[1]
        self.fab_model = fab_model
        FuncArea.fab_model = fab_model
        Gap.fab_model = fab_model
        self.design_scheme = design_scheme
        self.design_strategy = design_strategy
        self.top_down_max_step = top_down_max_step
        self.result_dir = result_dir

        self.package_graph: nx.Graph = nx.grid_2d_graph(*self.package_size)
        for node in self.package_graph.nodes:
            self.package_graph.nodes[node]["type"] = NodeType.Empty
            self.package_graph.nodes[node]["qubit"] = Qubit(node, NodeType.Empty, self.fab_model.single_fide)
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
                    self.package_graph.nodes[node]["qubit"].type = self.package_graph.nodes[node]["type"]
                    if bottom_up_init_type[node[0], node[1]] == NodeType.Compute.value:
                        self.compute_nodes.add(node)
                    elif bottom_up_init_type[node[0], node[1]] == NodeType.Transfer.value:
                        self.transfer_nodes.add(node)
                else:
                    self.package_graph.nodes[node]["qubit"] = None
            self.design_qunoc_bottom_up()
            self.compute_modules, self.transfer_lines = self.check_func_area()
            self.complete_package()
        elif self.design_scheme == "top_down":
            self.compute_modules = [ComputeModule(0, 0, package_size[0] - 1, package_size[1] - 1)]
            self.transfer_lines: list[TransferLine] = []
            self.design_tree_root = Node(self.compute_modules[0])
            self.design_tree_dict = {str(self.compute_modules[0]): self.design_tree_root}
            self.complete_area(self.compute_modules[0])
            if self.design_strategy == "heuristic":
                old_compute_module = self.compute_modules[0]
                length_width_ratio = max(self.package_size) / min(self.package_size)
                if length_width_ratio > 1:
                    if self.package_size[0] > self.package_size[1]:
                        self.design_module_top_down(
                            old_compute_module,
                            DesignTemplate.Strip,
                            DesignOrientation.Y,
                            min(self.package_size[1] - 1, (self.package_size[0] + 1) // 2 - 1),
                        )
                    else:
                        self.design_module_top_down(
                            old_compute_module,
                            DesignTemplate.Strip,
                            DesignOrientation.X,
                            min(self.package_size[0] - 1, (self.package_size[1] + 1) // 2 - 1),
                        )
                    old_compute_module = self.compute_modules[len(self.compute_modules) // 2]
                grid_round = len(self.compute_modules) - 1
                radial_round = (
                    min(self.package_size) - 2 * self.fab_model.max_square_size
                ) // (4 * self.fab_model.max_square_size)
                for i in range(radial_round):
                    self.design_module_top_down(
                        old_compute_module,
                        DesignTemplate.Radial,
                        DesignOrientation.X,
                        2 * self.fab_model.max_square_size - 1,
                    )
                    old_compute_module = self.compute_modules[-1]
                for i in range(grid_round):
                    old_compute_module = self.compute_modules[0]
                    if self.package_size[0] > self.package_size[1]:
                        self.design_module_top_down(
                            old_compute_module,
                            DesignTemplate.Grid,
                            DesignOrientation.X,
                            min(
                                3 * self.fab_model.max_square_size - 1,
                                (min(self.package_size[1], old_compute_module.right - old_compute_module.left + 1) + 1) // 2 - 1,
                            ),
                        )
                    else:
                        self.design_module_top_down(
                            old_compute_module,
                            DesignTemplate.Grid,
                            DesignOrientation.Y,
                            min(
                                3 * self.fab_model.max_square_size - 1,
                                (min(self.package_size[0], old_compute_module.bottom - old_compute_module.top + 1) + 1) // 2 - 1,
                            ),
                        )
                self.complete_package()
            elif self.design_strategy == "greedy":
                os.makedirs(self.result_dir, exist_ok=True)
                self.complete_package()
                best_score = self.estimate_efficiency()
                print("Initial score: ", best_score)
                self.draw_qubits(show=False, save_path=get_unique_file_save_path(self.result_dir, "greedy_process_", ".jpg"))
                while True:
                    best_action: tuple = None
                    candidate_modules: list[ComputeModule] = [area.name for area in list(self.design_tree_root.leaves) if area.name.type == NodeType.Compute]
                    for action in product(
                        candidate_modules, 
                        list(DesignTemplate), 
                        list(DesignOrientation), 
                        range(self.fab_model.max_square_size, max(self.package_size)//2, self.fab_model.max_square_size)
                    ):  
                        score = self.try_design_module(*action)
                        if score > best_score:
                            best_action = action
                            best_score = score
                    if best_action is None:
                        break
                    self.design_module_top_down(*best_action)
                    self.complete_package()
                    print("New action: ", best_action)
                    print("New score: ", best_score)
                    self.draw_qubits(show=False, save_path=get_unique_file_save_path(self.result_dir, "greedy_process_", ".jpg"))
                make_gif(self.result_dir, "greedy_process_", ".jpg", "greedy_process.gif")
            elif self.design_strategy == "rl":
                self.complete_package()
                # The rest rl design process is implemented outside in the LayoutAgent.
        else:
            raise ValueError("Invalid design scheme: {}".format(design_scheme))

    # Functions for bottom up design
    def update_node(self, node: tuple[int, int], new_type: NodeType):
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

    def update_all_nodes(self, new_type_matrix: np.ndarray):
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

    # Functions for top down design
    @staticmethod
    def is_valid_to_design(
        module: ComputeModule, 
        template: DesignTemplate, 
        orient: DesignOrientation, 
        interval: int, 
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
        elif template == DesignTemplate.Grid:
            if module.bottom - module.top + 1 < 2 * interval + 1:
                return False
            if module.right - module.left + 1 < 2 * interval + 1:
                return False
        return True

    def complete_area(self, func_area: FuncArea):
        """
        Complete the node information of the package graph for nodes in function area.
        Note: called after the change of function area.

        Args:
            func_area (FuncArea): The function area to be completed.
        """
        for node in func_area.get_nodes():
            self.package_graph.nodes[node]["qubit"].area = func_area
            if isinstance(func_area, ComputeModule):
                self.package_graph.nodes[node]["type"] = NodeType.Compute
                self.package_graph.nodes[node]["qubit"].type = NodeType.Compute
                self.compute_nodes.add(node)
                self.transfer_nodes.discard(node)
            elif isinstance(func_area, TransferLine):
                self.package_graph.nodes[node]["type"] = NodeType.Transfer
                self.package_graph.nodes[node]["qubit"].type = NodeType.Transfer
                self.transfer_nodes.add(node)
                self.compute_nodes.discard(node)

    def design_module_top_down(self, 
        old_module: ComputeModule, 
        template: DesignTemplate, 
        orient: DesignOrientation, 
        interval: int, 
    ):
        """
        Design the module with the given template and arguments.
        
        Args:
            module (ComputeModule): The module to be designed.
            template (DesignTemplate): The template to be used.
            orient (DesignOrientation): The orientation of the template.
            interval (int): The interval between the compute modules.
        """
        assert Package.is_valid_to_design(old_module, template, orient, interval), (
            f"Invalid design: module_{old_module}, template_{template}, orient_{orient}, interval_{interval}"
        )
        old_left_coord, old_left_line, _ = old_module.left
        old_left_arg = old_left_line if old_left_line else old_left_coord
        old_top_coord, old_top_line, _ = old_module.top
        old_top_arg = old_top_line if old_top_line else old_top_coord
        old_right_coord, old_right_line, _ = old_module.right
        old_right_arg = old_right_line if old_right_line else old_right_coord
        old_bottom_coord, old_bottom_line, _ = old_module.bottom
        old_bottom_arg = old_bottom_line if old_bottom_line else old_bottom_coord
        new_compute_modules = []
        new_transfer_lines = []
        if template == DesignTemplate.Strip:
            if orient == DesignOrientation.X:
                n_blocks = (old_bottom_coord - old_top_coord + 1) // (interval + 1)
                for i in range(n_blocks - 1):
                    new_transfer_lines.append(TransferLine(
                        0, 
                        old_top_coord + (i + 1) * (interval + 1) - 1, 
                        (old_left_coord, old_left_coord == 0), 
                        (old_right_coord, old_right_coord == self.package_size[0] - 1), 
                    ))
                for i in range(n_blocks):
                    new_top_arg = new_transfer_lines[i - 1]
                    new_bottom_arg = new_transfer_lines[i]
                    if i == 0:
                        new_top_arg = old_top_arg
                    elif i == n_blocks - 1:
                        new_bottom_arg = old_bottom_arg
                    new_compute_modules.append(ComputeModule(
                        old_left_arg, 
                        new_top_arg,
                        old_right_arg, 
                        new_bottom_arg,
                    ))
            elif orient == DesignOrientation.Y:
                n_blocks = (old_right_coord - old_left_coord + 1) // (interval + 1)
                for i in range(n_blocks - 1):
                    new_transfer_lines.append(TransferLine(
                        1, 
                        old_left_coord + (i + 1) * (interval + 1) - 1, 
                        (old_top_coord, old_top_coord == 0), 
                        (old_bottom_coord, old_bottom_coord == self.package_size[1] - 1), 
                    ))
                for i in range(n_blocks):
                    new_left_arg = new_transfer_lines[i - 1]
                    new_right_arg = new_transfer_lines[i]
                    if i == 0:
                        new_left_arg = old_left_arg
                    elif i == n_blocks - 1:
                        new_right_arg = old_right_arg
                    new_compute_modules.append(ComputeModule(
                        new_left_arg,
                        old_top_arg, 
                        new_right_arg,
                        old_bottom_arg, 
                    ))
        elif template == DesignTemplate.Radial:
            if orient == DesignOrientation.X:
                new_transfer_lines.append(TransferLine(
                    0, 
                    old_top_coord + interval, 
                    (old_left_coord, old_left_coord == 0), 
                    (old_right_coord - interval - 1, False), 
                ))
                new_transfer_lines.append(TransferLine(
                    1,
                    old_right_coord - interval,
                    (old_top_coord, old_top_coord == 0),
                    (old_bottom_coord - interval - 1, False),
                ))
                new_transfer_lines.append(TransferLine(
                    0,
                    old_bottom_coord - interval,
                    (old_left_coord + interval + 1, False),
                    (old_right_coord, old_right_coord == self.package_size[0] - 1),
                ))
                new_transfer_lines.append(TransferLine(
                    1,
                    old_left_coord + interval,
                    (old_top_coord + interval + 1, False),
                    (old_bottom_coord, old_bottom_coord == self.package_size[1] - 1),
                ))
                new_compute_modules.append(ComputeModule(
                    old_left_arg, 
                    old_top_arg,
                    new_transfer_lines[1],
                    new_transfer_lines[0],
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[1],
                    old_top_arg,
                    old_right_arg,
                    new_transfer_lines[2], 
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[3],
                    new_transfer_lines[2],
                    old_right_arg,
                    old_bottom_arg,
                ))
                new_compute_modules.append(ComputeModule(
                    old_left_arg,
                    new_transfer_lines[0],
                    new_transfer_lines[3],
                    old_bottom_arg,
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[3],
                    new_transfer_lines[0],
                    new_transfer_lines[1],
                    new_transfer_lines[2],
                ))
            elif orient == DesignOrientation.Y:
                new_transfer_lines.append(TransferLine(
                    1,
                    old_left_coord + interval,
                    (old_top_coord, old_top_coord == 0),
                    (old_bottom_coord - interval - 1, False),
                ))
                new_transfer_lines.append(TransferLine(
                    0,
                    old_bottom_coord - interval,
                    (old_left_coord, old_left_coord == 0),
                    (old_right_coord - interval - 1, False),
                ))
                new_transfer_lines.append(TransferLine(
                    1,
                    old_right_coord - interval,
                    (old_top_coord + interval + 1, False),
                    (old_bottom_coord, old_bottom_coord == self.package_size[1] - 1),
                ))
                new_transfer_lines.append(TransferLine(
                    0,
                    old_top_coord + interval,
                    (old_left_coord + interval + 1, False),
                    (old_right_coord, old_right_coord == self.package_size[0] - 1),
                ))
                new_compute_modules.append(ComputeModule(
                    old_left_arg, 
                    old_top_arg,
                    new_transfer_lines[0],
                    new_transfer_lines[1],
                ))
                new_compute_modules.append(ComputeModule(
                    old_left_arg,
                    new_transfer_lines[1],
                    new_transfer_lines[2],
                    old_bottom_arg,
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[2],
                    new_transfer_lines[3],
                    old_right_arg,
                    old_bottom_arg,
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[0],
                    old_top_arg,
                    old_right_arg,
                    new_transfer_lines[3],
                ))
                new_compute_modules.append(ComputeModule(
                    new_transfer_lines[0],
                    new_transfer_lines[3],
                    new_transfer_lines[2],
                    new_transfer_lines[1],
                ))
        elif template == DesignTemplate.Grid:
            if orient == DesignOrientation.X:
                n_blocks = (old_bottom_coord - old_top_coord + 1) // (interval + 1)
                n_units = (old_right_coord - old_left_coord + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units - 1):
                        new_pos = old_left_coord + (j + 1) * (interval + 1) - 1
                        new_start_coord = old_top_coord + i * (interval + 1)
                        if i == n_blocks - 1:
                            new_end_coord = old_bottom_coord
                        else:
                            new_end_coord = old_top_coord + i * (interval + 1) + interval - 1
                        new_transfer_lines.append(TransferLine(
                            1, 
                            new_pos, 
                            (new_start_coord, new_start_coord == 0), 
                            (new_end_coord, new_end_coord == self.package_size[1] - 1),
                        ))
                    if i < n_blocks - 1:
                        new_transfer_lines.append(TransferLine(
                            0, 
                            old_top_coord + (i + 1) * (interval + 1) - 1, 
                            old_left_coord, 
                            old_right_coord,
                        ))
                for i in range(n_blocks):
                    for j in range(n_units):
                        new_top_arg = new_transfer_lines[i * (n_units - 1)]
                        new_bottom_arg = new_transfer_lines[(i + 1) * (n_units - 1)]
                        new_left_arg = new_transfer_lines[i * (n_units - 1) + j - 1]
                        new_right_arg = new_transfer_lines[i * (n_units - 1) + j]
                        if i == 0:
                            new_top_arg = old_top_arg
                        elif i == n_blocks - 1:
                            new_bottom_arg = old_bottom_arg
                        if j == 0:
                            new_left_arg = old_left_arg
                        elif j == n_units - 1:
                            new_right_arg = old_right_arg
                        new_compute_modules.append(ComputeModule(
                            new_left_arg,
                            new_top_arg,
                            new_right_arg,
                            new_bottom_arg,
                        ))
            elif orient == DesignOrientation.Y:
                n_blocks = (old_right_coord - old_left_coord + 1) // (interval + 1)
                n_units = (old_bottom_coord - old_top_coord + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units - 1):
                        new_pos = old_top_coord + (j + 1) * (interval + 1) - 1
                        new_start_coord = old_left_coord + i * (interval + 1)
                        if i == n_blocks - 1:
                            new_end_coord = old_right_coord
                        else:
                            new_end_coord = old_left_coord + i * (interval + 1) + interval - 1
                        new_transfer_lines.append(TransferLine(
                            0, 
                            new_pos, 
                            (new_start_coord, new_start_coord == 0), 
                            (new_end_coord, new_end_coord == self.package_size[0] - 1),
                        ))
                    if i < n_blocks - 1:
                        new_transfer_lines.append(TransferLine(
                            1, 
                            old_left_coord + (i + 1) * (interval + 1) - 1, 
                            old_top_coord, 
                            old_bottom_coord,
                        ))
                for i in range(n_blocks):
                    for j in range(n_units):
                        new_left_arg = new_transfer_lines[i * (n_units - 1)]
                        new_right_arg = new_transfer_lines[(i + 1) * (n_units - 1)]
                        new_top_arg = new_transfer_lines[i * (n_units - 1) + j - 1]
                        new_bottom_arg = new_transfer_lines[i * (n_units - 1) + j]
                        if i == 0:
                            new_left_arg = old_left_arg
                        elif i == n_blocks - 1:
                            new_right_arg = old_right_arg
                        if j == 0:
                            new_top_arg = old_top_arg
                        elif j == n_units - 1:
                            new_bottom_arg = old_bottom_arg
                        new_compute_modules.append(ComputeModule(
                            new_left_arg,
                            new_top_arg,
                            new_right_arg,
                            new_bottom_arg,
                        ))
        self.compute_modules.remove(old_module)
        self.compute_modules.extend(new_compute_modules)
        self.transfer_lines.extend(new_transfer_lines)
        for func_area in new_compute_modules + new_transfer_lines:
            self.design_tree_dict[str(func_area)] = Node(func_area, parent=self.design_tree_dict[str(old_module)])
            self.complete_area(func_area)

    def try_design_module(self, 
        old_module: ComputeModule, 
        template: DesignTemplate, 
        orient: DesignOrientation, 
        interval: int, 
    ) -> float:
        """
        Try to design the module with the given template and arguments. This function does not actually change the package and only returns the score.

        Args:
            module (ComputeModule): The module to be designed.
            template (DesignTemplate): The template to be used.
            orient (DesignOrientation): The orientation of the template.
            interval (int): The interval between the compute modules.

        Returns:
            score (float): The score of the design.
        """
        if not Package.is_valid_to_design(old_module, template, orient, interval):
            return -math.inf
        self.design_module_top_down(old_module, template, orient, interval)
        self.complete_package()
        score = self.estimate_efficiency()

        # restore the package
        for child in list(self.design_tree_dict[str(old_module)].children):
            if child.name.type == NodeType.Compute:
                self.compute_modules.remove(child.name)
            elif child.name.type == NodeType.Transfer:
                self.transfer_lines.remove(child.name)
            child.parent = None
            self.design_tree_dict.pop(str(child.name))
        self.compute_modules.append(old_module)
        self.complete_area(old_module)
        return score

    def update_chiplets_and_gaps(self):
        """
        Update the chiplet and gap information from function areas.
        """
        self.compute_chiplets: list[ComputeChiplet] = []
        self.stations: set[tuple[int, int]] = set()
        self.platform_stations_dict: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for compute_module in self.compute_modules:
            self.compute_chiplets.extend(compute_module.get_chiplets())
            platform_station_pairs = compute_module.get_platform_station_pairs()
            for platform, station in platform_station_pairs:
                self.stations.add(station)
                self.platform_stations_dict[platform].append(station)
                self.package_graph.nodes[platform]["qubit"].role = QubitRole.Platform
                self.package_graph.nodes[station]["qubit"].role = QubitRole.Station
        self.transfer_chiplets: list[TransferChiplet] = []
        self.interchanges: set[Coupler] = set()
        for transfer_line in self.transfer_lines:
            self.transfer_chiplets.extend(transfer_line.get_chiplets())
            self.interchanges.update(transfer_line.get_interchanges())

    def analysis_metro(self):
        """
        Update and analyze the metro topology.
        """
        self.metro_topology: nx.Graph = nx.Graph()
        for transfer_chiplet in self.transfer_chiplets:
            self.metro_topology.add_node(transfer_chiplet)
        for interchange in self.interchanges:
            transfer_chiplet_u = self.get_node_chiplet(interchange.qubit_u)
            transfer_chiplet_v = self.get_node_chiplet(interchange.qubit_v)
            self.metro_topology.add_edge(transfer_chiplet_u, transfer_chiplet_v, coupler=interchange)
        least_transfer_chiplets_dict: dict[TransferChiplet, dict[TransferChiplet, list[TransferChiplet]]] = nx.shortest_path(self.metro_topology)
        self.interchange_dict: dict[TransferChiplet, dict[TransferChiplet, Path]] = {}
        for transfer_chiplet_u in self.transfer_chiplets:
            self.interchange_dict[transfer_chiplet_u] = {}
            for transfer_chiplet_v in self.transfer_chiplets:
                least_transfer_chiplets = least_transfer_chiplets_dict[transfer_chiplet_u][transfer_chiplet_v]
                nodes: list[tuple[int, int]] = []
                for i in range(len(least_transfer_chiplets) - 1):
                    interchange: Coupler = self.metro_topology.edges[(least_transfer_chiplets[i], least_transfer_chiplets[i + 1])]["coupler"]
                    if least_transfer_chiplets[i].has_node(interchange.qubit_u):
                        nodes.extend([interchange.qubit_u, interchange.qubit_v])
                    elif least_transfer_chiplets[i].has_node(interchange.qubit_v):
                        nodes.extend([interchange.qubit_v, interchange.qubit_u])
                roads: list[Road] = []
                for i in range(len(nodes) - 1):
                    if i % 2 == 0:
                        road = InterChipletRoad(nodes[i], nodes[i + 1], least_transfer_chiplets[i // 2], least_transfer_chiplets[i // 2 + 1], self.fab_model.inter_fide)
                    else:
                        road = IntraTransferRoad(nodes[i], nodes[i + 1], least_transfer_chiplets[i // 2 + 1])
                    roads.append(road)
                self.interchange_dict[transfer_chiplet_u][transfer_chiplet_v] = Path(roads)

    def complete_package(self):
        """
        Complete the package graph after design the function area.
        Note: called before estimation or after design.

        1. Update the chiplet and gap information from function areas.
        2. Analyze the metro topology.
        """
        self.update_chiplets_and_gaps()
        self.analysis_metro()

        # self.chiplet_topology: nx.Graph = nx.Graph()
        # for compute_chiplet in self.compute_chiplets:
        #     self.chiplet_topology.add_node(str(compute_chiplet), chiplet=compute_chiplet)
        # for transfer_chiplet in self.transfer_chiplets:
        #     self.chiplet_topology.add_node(str(transfer_chiplet), chiplet=transfer_chiplet)
        # for gap in self.inter_gaps.values():
        #     self.chiplet_topology.add_edge(str(gap.area_u), str(gap.area_v), gap=gap)

    # Functions for package operation and estimation
    def random_transfer_pair(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Randomly select a transfer pair from the compute nodes.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: The transfer pair.
        """
        return random.sample(self.compute_nodes, 2)

    def random_transfer_pairs(self, num: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """
        Randomly select a list of transfer pairs from the compute nodes.

        Args:
            num (int): The number of transfer pairs to select.

        Returns:
            list[tuple[tuple[int, int], tuple[int, int]]]: The list of transfer pairs.
        """
        return [self.random_transfer_pair() for _ in range(num)]

    # @staticmethod
    # def edge_fide_weight(u: tuple[int, int], v: tuple[int, int], attrs: dict) -> float:
    #     """
    #     Calculate the weight of the edge based on the fidelity.

    #     Args:
    #         u (tuple[int, int]): The first node.
    #         v (tuple[int, int]): The second node.
    #         attrs (dict): The attributes of the edge.
        
    #     Returns:
    #         float: The weight of the edge.
    #     """
    #     return -math.log(attrs["coupler"].fide)

    def get_shortest_path(self, source: tuple[int, int], target: tuple[int, int]) -> tuple[Path, float]:
        """
        Get the shortest path between two physical qubits.

        Args:
            source (tuple[int, int]): The source node of the transfer pair.
            target (tuple[int, int]): The target node of the transfer pair

        Returns:
            path (Path): The path with highest fidelity.
            fidelity (float): The fidelity of the path
        """
        source_area = self.get_node_area(source)
        if isinstance(source_area, ComputeModule):
            source_platform = source_area.get_nearest_platform(source)
            source_station = self.platform_stations_dict[source_platform][0]
            source_transfer_node = source_station
            source_transfer_chiplet = self.get_node_chiplet(source_station)
        else:
            source_transfer_node = source
            source_transfer_chiplet = self.get_node_chiplet(source)
        target_area = self.get_node_area(target)
        if isinstance(target_area, ComputeModule):
            target_platform = target_area.get_nearest_platform(target)
            target_station = self.platform_stations_dict[target_platform][0]
            target_transfer_node = target_station
            target_transfer_chiplet = self.get_node_chiplet(target_station)
        else:
            target_transfer_node = target
            target_transfer_chiplet = self.get_node_chiplet(target)

        roads: list[Road] = []
        if isinstance(source_area, ComputeModule):
            roads.append(IntraComputeRoad(source, source_platform, source_area))
            roads.append(InterChipletRoad(source_platform, source_station, self.get_node_chiplet(source_platform), source_transfer_chiplet, self.fab_model.inter_fide))
        interchange_path = self.interchange_dict[source_transfer_chiplet][target_transfer_chiplet]
        if len(interchange_path.roads) == 0:
            roads.append(IntraTransferRoad(source_transfer_node, target_transfer_node, source_transfer_chiplet))
        else:
            roads.append(IntraTransferRoad(source_transfer_node, interchange_path.roads[0].source, source_transfer_chiplet))
            roads.extend(interchange_path.roads)
            roads.append(IntraTransferRoad(interchange_path.roads[-1].target, target_transfer_node, target_transfer_chiplet))
        if isinstance(target_area, ComputeModule):
            roads.append(InterChipletRoad(target_station, target_platform, target_transfer_chiplet, self.get_node_chiplet(target_platform), self.fab_model.inter_fide))
            roads.append(IntraComputeRoad(target_platform, target, target_area))
        
        path = Path(roads)
        fidelity = path.get_fidelity()
        if isinstance(source_area, ComputeModule) and source_area == target_area:
            no_transfer_path = Path([IntraComputeRoad(source, target, source_area)])
            no_transfer_fidelity = no_transfer_path.get_fidelity()
            if no_transfer_fidelity > fidelity:
                path = no_transfer_path
                fidelity = no_transfer_fidelity
        return path, fidelity

    def estimate_transfer(self, num: int = None) -> tuple[float, float]:
        """
        Estimate the transfer fidelity of the package.
        Note: call complete_package() before this function.

        Args:
            num (int): The number of transfer pairs to select.

        Returns:
            tuple[float, float]: The mean and standard deviation of the transfer fidelity.
        """
        if num is None:
            num = int(math.sqrt(self.node_num))
        transfer_pairs = self.random_transfer_pairs(num)
        fidelity_list = []
        for transfer_pair in transfer_pairs:
            _, fidelity = self.get_shortest_path(*transfer_pair)
            fidelity_list.append(fidelity)
        return np.mean(fidelity_list), np.std(fidelity_list)

    def estimate_cost(self) -> float:
        """
        Estimate the cost of the package considering the yield cost.
        Note: call complete_package() before this function.

        Returns:
            float: The estimated cost of the package.
        """
        cost = 0
        for compute_chiplet in self.compute_chiplets:
            cost += self.fab_model.get_cost(compute_chiplet.get_capacity())
        for transfer_chiplet in self.transfer_chiplets:
            cost += self.fab_model.get_cost(transfer_chiplet.get_capacity())
        return cost

    def estimate_efficiency(self, num: int = None) -> float:
        """
        Estimate the efficiency of package considering the yield cost.
        Note: call complete_package() before this function.

        Args:
            num (int): The number of transfer pairs selected to estimate the package.
        
        Returns:
            float: The estimated efficiency of the package.
        """
        if num is None:
            num = int(math.sqrt(self.node_num))
        fide_avg, fide_std = self.estimate_transfer(num)
        return -math.log(1 - fide_avg) / fide_std * len(self.compute_nodes) / (self.node_num)
        cost = self.estimate_cost()
        return -math.log(1 - fide_avg) / cost

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

    def get_node_chiplet(self, node: tuple[int, int]) -> Chiplet:
        """
        Get the chiplet that contains the node.

        Args:
            node (tuple[int, int]): The coordinate of the node to locate.

        Returns:
            Chiplet: The chiplet that contains the node.
        """
        return self.package_graph.nodes[node]["qubit"].chiplet
        for chiplet in self.compute_chiplets:
            if chiplet.in_area(node):
                return chiplet
        for chiplet in self.transfer_chiplets:
            if chiplet.in_area(node):
                return chiplet
        raise ValueError("Invalid physical qubit: {}".format(node))

    def get_node_area(self, node: tuple[int, int]) -> FuncArea:
        """
        Get the function area that contains the node.

        Args:
            node (tuple[int, int]): The coordinate of the node to locate.

        Returns:
            FuncArea: The function area that contains the node.
        """
        return self.package_graph.nodes[node]["qubit"].area
        for module in self.compute_modules:
            if module.in_area(node):
                return module
        for line in self.transfer_lines:
            if line.in_area(node):
                return line
        raise ValueError("Invalid physical qubit: {}".format(node))

    def get_area_graph(self, area: FuncArea) -> nx.Graph:
        """
        Get the graph of function area in the package.

        Args:
            area (FuncArea): The function area.

        Returns:
            nx.Graph: The subgraph view of qubit topology in the function area.
        """
        return self.qubit_topology.subgraph(area.get_nodes())

    # @staticmethod
    # def edge_metro_weight(u: tuple[int, int], v: tuple[int, int], attrs: dict) -> int:
    #     """
    #     Calculate the weight of the edge based on the role.

    #     Args:
    #         u (tuple[int, int]): The first node.
    #         v (tuple[int, int]): The second node.
    #         attrs (dict): The attributes of the edge.
        
    #     Returns:
    #         int: The weight of the edge.
    #     """
    #     return attrs["coupler"].role.value

    # def get_shortest_metro_path(self, source: tuple[int, int], target: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    #     """
    #     Get the shortest path in metro topology.

    #     Args:
    #         source (tuple[int, int]): The source node of the transfer pair.
    #         target (tuple[int, int]): The target node of the transfer pair.

    #     Returns:
    #         path (list[tuple[int, int]]): The path with least interchanges.
    #         interchange_num (int): The number of interchanges on the path.
    #     """
    #     assert source in self.metro_topology.nodes and target in self.metro_topology.nodes
    #     path = nx.shortest_path(
    #         self.metro_topology, source, target, weight=Package.edge_metro_weight
    #     )
    #     interchange_num = 0
    #     for i in range(len(path) - 1):
    #         interchange_num += self.metro_topology.edges[(path[i], path[i + 1])]["coupler"].role.value
    #     return path, interchange_num

    def draw_qubits(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
        show: bool = True, 
        save_path: str = None, 
    ):
        """
        Draw the package graph in qubit level.

        Args:
            marginrate (float): The margin rate of the package.
            markersize (float): The size of the markers.
            linewidth_t (float): The linewidth of the transfer lines.
            linewidth_b (float): The linewidth of the compute lines.
            show (bool): Whether to show the figure.
            save_path (str): The path to save the figure.
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
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # def draw_chiplets(self, 
    #     marginrate: float = 0.2, 
    #     markersize: float = 4, 
    #     linewidth_t: float = 1, 
    #     linewidth_b: float = 2, 
    #     show: bool = True, 
    #     save_path: str = None, 
    # ):
    #     """
    #     Draw the package graph in chiplet level.

    #     Args:
    #         marginrate (float): The margin rate of the package.
    #         markersize (float): The size of the markers.
    #         linewidth_t (float): The linewidth of the transfer lines.
    #         linewidth_b (float): The linewidth of the compute lines.
    #         show (bool): Whether to show the figure.
    #         save_path (str): The path to save the figure.
    #     """
    #     plt.figure(figsize=self.package_size)
    #     for node in self.chiplet_topology.nodes:
    #         chiplet: Chiplet = self.chiplet_topology.nodes[node]["chiplet"]
    #         plt.plot(*chiplet.get_coordinate(), "ko", markersize=markersize)
    #     for node_u, node_v in self.chiplet_topology.edges:
    #         pos_u = self.chiplet_topology.nodes[node_u]["chiplet"].get_coordinate()
    #         pos_v = self.chiplet_topology.nodes[node_v]["chiplet"].get_coordinate()
    #         plt.plot(
    #             [pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], "k-", linewidth=linewidth_t
    #         )
    #     plt.axis("off")
    #     plt.gca().spines["top"].set_visible(False)
    #     plt.gca().spines["right"].set_visible(False)
    #     plt.gca().spines["bottom"].set_visible(False)
    #     plt.gca().spines["left"].set_visible(False)
    #     if show:
    #         plt.show()
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches="tight")
    #     plt.close()
