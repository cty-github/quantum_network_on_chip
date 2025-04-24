import os
import random
from itertools import product

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
        gap_dict (dict[str, Gap]): The gaps between chiplets in the package.

        qubit_topology (nx.Graph): The subgraph view of package_graph recording the topology of physical qubits in the package.
        metro_topology (nx.Graph): The subgraph view of package_graph recording the topology of transfer qubits in the package.
        chiplet_topology (nx.Graph): The graph recording chiplet topology in the package.
            node: str, {"chiplet": Chiplet}
            edge: tuple[str, str], {"gap": Gap}
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
        self.design_scheme = design_scheme
        self.design_strategy = design_strategy
        self.top_down_max_step = top_down_max_step
        self.result_dir = result_dir

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
                    self.package_graph.nodes[node]["qubit"] = Qubit(node, self.package_graph.nodes[node]["type"], self.fab_model.single_fide)
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
            self.compute_modules = [ComputeModule(fab_model, 0, 0, package_size[0] - 1, package_size[1] - 1)]
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
            self.package_graph.nodes[node]["type"] = func_area.type
            if "qubit" not in self.package_graph.nodes[node]:
                self.package_graph.nodes[node]["qubit"] = Qubit(node, func_area.type, self.fab_model.single_fide)
            else:
                self.package_graph.nodes[node]["qubit"].type = func_area.type
            if func_area.type == NodeType.Compute:
                self.compute_nodes.add(node)
                self.transfer_nodes.discard(node)
            elif func_area.type == NodeType.Transfer:
                self.transfer_nodes.add(node)
                self.compute_nodes.discard(node)

    def design_module_top_down(self, 
        module: ComputeModule, 
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
        
        Returns:
            tuple[list[ComputeModule], list[TransferLine]]: The new compute modules and transfer lines.
        """
        assert Package.is_valid_to_design(module, template, orient, interval), (
            f"Invalid design: module_{module}, template_{template}, orient_{orient}, interval_{interval}"
        )
        new_compute_modules = []
        new_transfer_lines = []
        if template == DesignTemplate.Strip:
            if orient == DesignOrientation.X:
                n_blocks = (module.bottom - module.top + 1) // (interval + 1)
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.top + i * (interval + 1), module.right, module.bottom))
                    else:
                        new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.top + i * (interval + 1), module.right, module.top + i * (interval + 1) + interval - 1))
                        new_transfer_lines.append(TransferLine(self.fab_model, 0, module.top + (i + 1) * (interval + 1) - 1, module.left, module.right))
            elif orient == DesignOrientation.Y:
                n_blocks = (module.right - module.left + 1) // (interval + 1)
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_modules.append(ComputeModule(self.fab_model, module.left + i * (interval + 1), module.top, module.right, module.bottom))
                    else:
                        new_compute_modules.append(ComputeModule(self.fab_model, module.left + i * (interval + 1), module.top, module.left + i * (interval + 1) + interval - 1, module.bottom))
                        new_transfer_lines.append(TransferLine(self.fab_model, 1, module.left + (i + 1) * (interval + 1) - 1, module.top, module.bottom))
        elif template == DesignTemplate.Radial:
            if orient == DesignOrientation.X:
                new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.top, module.right - interval - 1, module.top + interval - 1))
                new_transfer_lines.append(TransferLine(self.fab_model, 0, module.top + interval, module.left, module.right - interval - 1))
                new_compute_modules.append(ComputeModule(self.fab_model, module.right - interval + 1, module.top, module.right, module.bottom - interval - 1))
                new_transfer_lines.append(TransferLine(self.fab_model, 1, module.right - interval, module.top, module.bottom - interval - 1))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left + interval + 1, module.bottom - interval + 1, module.right, module.bottom))
                new_transfer_lines.append(TransferLine(self.fab_model, 0, module.bottom - interval, module.left + interval + 1, module.right))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.top + interval + 1, module.left + interval - 1, module.bottom))
                new_transfer_lines.append(TransferLine(self.fab_model, 1, module.left + interval, module.top + interval + 1, module.bottom))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left + interval + 1, module.top + interval + 1, module.right - interval - 1, module.bottom - interval - 1))
            elif orient == DesignOrientation.Y:
                new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.top, module.left + interval - 1, module.bottom - interval - 1))
                new_transfer_lines.append(TransferLine(self.fab_model, 1, module.left + interval, module.top, module.bottom - interval - 1))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left + interval + 1, module.top, module.right, module.top + interval - 1))
                new_transfer_lines.append(TransferLine(self.fab_model, 0, module.top + interval, module.left + interval + 1, module.right))
                new_compute_modules.append(ComputeModule(self.fab_model, module.right - interval + 1, module.top + interval + 1, module.right, module.bottom))
                new_transfer_lines.append(TransferLine(self.fab_model, 1, module.right - interval, module.top + interval + 1, module.bottom))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left, module.bottom - interval + 1, module.right - interval - 1, module.bottom))
                new_transfer_lines.append(TransferLine(self.fab_model, 0, module.bottom - interval, module.left, module.right - interval - 1))
                new_compute_modules.append(ComputeModule(self.fab_model, module.left + interval + 1, module.top + interval + 1, module.right - interval - 1, module.bottom - interval - 1))
        elif template == DesignTemplate.Grid:
            if orient == DesignOrientation.X:
                n_blocks = (module.bottom - module.top + 1) // (interval + 1)
                n_units = (module.right - module.left + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units):
                        if i == n_blocks - 1:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.right,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.left + j * (interval + 1) + interval - 1,
                                    module.bottom,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    1, 
                                    module.left + (j + 1) * (interval + 1) - 1,
                                    module.top + i * (interval + 1),
                                    module.bottom,
                                ))
                        else:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.right,
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    0,
                                    module.top + (i + 1) * (interval + 1) - 1,
                                    module.left,
                                    module.right,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + j * (interval + 1),
                                    module.top + i * (interval + 1),
                                    module.left + j * (interval + 1) + interval - 1,
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    1,
                                    module.left + (j + 1) * (interval + 1) - 1,
                                    module.top + i * (interval + 1),
                                    module.top + i * (interval + 1) + interval - 1,
                                ))
            elif orient == DesignOrientation.Y:
                n_blocks = (module.right - module.left + 1) // (interval + 1)
                n_units = (module.bottom - module.top + 1) // (interval + 1)
                for i in range(n_blocks):
                    for j in range(n_units):
                        if i == n_blocks - 1:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.right,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.right,
                                    module.top + j * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    0,
                                    module.top + (j + 1) * (interval + 1) - 1,
                                    module.left + i * (interval + 1),
                                    module.right,
                                ))
                        else:
                            if j == n_units - 1:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                    module.bottom,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    1,
                                    module.left + (i + 1) * (interval + 1) - 1,
                                    module.top,
                                    module.bottom,
                                ))
                            else:
                                new_compute_modules.append(ComputeModule(
                                    self.fab_model, 
                                    module.left + i * (interval + 1),
                                    module.top + j * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                    module.top + j * (interval + 1) + interval - 1,
                                ))
                                new_transfer_lines.append(TransferLine(
                                    self.fab_model, 
                                    0,
                                    module.top + (j + 1) * (interval + 1) - 1,
                                    module.left + i * (interval + 1),
                                    module.left + i * (interval + 1) + interval - 1,
                                ))
        self.compute_modules.remove(module)
        self.compute_modules.extend(new_compute_modules)
        self.transfer_lines.extend(new_transfer_lines)
        for func_area in new_compute_modules + new_transfer_lines:
            self.design_tree_dict[str(func_area)] = Node(func_area, parent=self.design_tree_dict[str(module)])
            self.complete_area(func_area)

    def try_design_module(self, 
        module: ComputeModule, 
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
        if not Package.is_valid_to_design(module, template, orient, interval):
            return -math.inf
        self.design_module_top_down(module, template, orient, interval)
        self.complete_package()
        score = self.estimate_efficiency()

        # restore the package
        for child in list(self.design_tree_dict[str(module)].children):
            if child.name.type == NodeType.Compute:
                self.compute_modules.remove(child.name)
            elif child.name.type == NodeType.Transfer:
                self.transfer_lines.remove(child.name)
            child.parent = None
            self.design_tree_dict.pop(str(child.name))
        self.compute_modules.append(module)
        self.complete_area(module)
        return score

    def collect_compute_chiplets(self) -> list[ComputeChiplet]:
        """
        Collect the compute chiplets from the compute modules and mark the nodes.

        Returns:
            compute_chiplets (list[ComputeChiplet]): The list of compute chiplets.
        """
        compute_chiplets: list[ComputeChiplet] = []
        for compute_module in self.compute_modules:
            compute_chiplets.extend(compute_module.get_chiplets())
        for compute_chiplet in compute_chiplets:
            for node in compute_chiplet.get_nodes():
                self.package_graph.nodes[node]["qubit"].chiplet = compute_chiplet
        return compute_chiplets

    def collect_transfer_chiplets(self) -> list[TransferChiplet]:
        """
        Collect the transfer chiplets from the transfer lines and mark the nodes.

        Returns:
            transfer_chiplets (list[TransferChiplet]): the list of transfer chiplets
        """
        transfer_chiplets: list[TransferChiplet] = []
        for transfer_line in self.transfer_lines:
            transfer_chiplets.extend(transfer_line.get_chiplets())
        for transfer_chiplet in transfer_chiplets:
            for node in transfer_chiplet.get_nodes():
                self.package_graph.nodes[node]["qubit"].chiplet = transfer_chiplet
        return transfer_chiplets

    def determine_inter_gap(self) -> list[InterAreaGap]:
        """
        Determine the inter-area gaps in the package.

        Returns:
            inter_gaps (list[InterAreaGap]): The list of inter-area gaps.
        """
        inter_gaps: list[InterAreaGap] = []
        for compute_module in self.compute_modules:
            if compute_module.left > 0:
                inter_gaps.append(InterAreaGap(
                    self.fab_model, 1, 
                    (compute_module.left-1, compute_module.left), 
                    compute_module.top, 
                    compute_module.bottom, 
                    compute_module, 
                    self.get_physical_qubit_area((compute_module.left-1, compute_module.top))
                ))
            if compute_module.top > 0:
                inter_gaps.append(InterAreaGap(
                    self.fab_model, 0, 
                    (compute_module.top-1, compute_module.top), 
                    compute_module.left, 
                    compute_module.right, 
                    compute_module, 
                    self.get_physical_qubit_area((compute_module.left, compute_module.top-1))
                ))
            if compute_module.right < self.package_size[0] - 1:
                inter_gaps.append(InterAreaGap(
                    self.fab_model, 1, 
                    (compute_module.right, compute_module.right+1), 
                    compute_module.top, 
                    compute_module.bottom, 
                    compute_module, 
                    self.get_physical_qubit_area((compute_module.right+1, compute_module.top))
                ))
            if compute_module.bottom < self.package_size[1] - 1:
                inter_gaps.append(InterAreaGap(
                    self.fab_model, 0, 
                    (compute_module.bottom, compute_module.bottom+1), 
                    compute_module.left, 
                    compute_module.right, 
                    compute_module, 
                    self.get_physical_qubit_area((compute_module.left, compute_module.bottom+1))
                ))
        return inter_gaps

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
                    fide = self.fab_model.get_intra_fide(qubit_u.chiplet.get_capacity())
                else:
                    fide = self.fab_model.enhanced_fide
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
                if gap_id not in inter_gaps:
                    inter_gaps[gap_id] = InterAreaGap(chiplet_u, chiplet_v, axis, {pos:edge})
                else:
                    inter_gaps[gap_id].edges[pos] = edge
                self.package_graph.edges[edge]["type"] = EdgeType.Empty
                self.package_graph.edges[edge]["coupler"] = None
        for gap in inter_gaps.values():
            for _, edge in list(sorted(gap.edges.items()))[
                :: self.fab_model.inter_sparsity
            ]:
                self.package_graph.edges[edge]["type"] = EdgeType.Inter
                self.package_graph.edges[edge]["coupler"] = Coupler(
                    self.package_graph.nodes[edge[0]]["qubit"],
                    self.package_graph.nodes[edge[1]]["qubit"],
                    EdgeType.Inter,
                    self.fab_model.inter_fide,
                )
        return inter_gaps

    def analysis_metro(self):
        self.metro_topology: nx.Graph = self.qubit_topology.subgraph(self.transfer_nodes)
        for edge in self.metro_topology.edges:
            if self.metro_topology.edges[edge]["type"] == EdgeType.Intra:
                self.metro_topology.edges[edge]["coupler"].role = CouplerRole.Rail
            elif self.metro_topology.edges[edge]["type"] == EdgeType.Inter:
                self.metro_topology.edges[edge]["coupler"].role = CouplerRole.Interchange
            else:
                raise ValueError("Invalid edge type: {}".format(self.metro_topology.edges[edge]["type"]))
        if not hasattr(self, "station_list"):
            for transfer_chiplet in self.transfer_chiplets:
                if transfer_chiplet.axis == 0:
                    self.station_list.append((transfer_chiplet.pos, transfer_chiplet.start))
                    self.station_list.append((transfer_chiplet.pos, transfer_chiplet.end))
                elif transfer_chiplet.axis == 1:
                    self.station_list.append((transfer_chiplet.start, transfer_chiplet.pos))
                    self.station_list.append((transfer_chiplet.end, transfer_chiplet.pos))
        self.balise_station_distance_dict = {balise: {station: {"interchange": 0, "path": None} for station in self.station_list} for balise in self.metro_topology.nodes}
        for balise in self.metro_topology.nodes:
            for station in self.station_list:
                shortest_path, interchange_num = self.get_shortest_metro_path(balise, station)
                self.balise_station_distance_dict[balise][station]["interchange"] = interchange_num
                self.balise_station_distance_dict[balise][station]["path"] = shortest_path

    def complete_package(self):
        """
        Complete the package graph after design the function area.
        Note: called before estimation or after design.

        1. Divide the compute modules into chiplets.
        2. Divide the transfer lines into chiplets.
        3. Determine the gaps between chiplets.
        """
        self.compute_chiplets = self.collect_compute_chiplets()
        self.transfer_chiplets = self.collect_transfer_chiplets()
        self.inter_gaps = self.determine_inter_gap()
        # self.analysis_metro()
        
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

        self.chiplet_topology: nx.Graph = nx.Graph()
        for compute_chiplet in self.compute_chiplets:
            self.chiplet_topology.add_node(str(compute_chiplet), chiplet=compute_chiplet)
        for transfer_chiplet in self.transfer_chiplets:
            self.chiplet_topology.add_node(str(transfer_chiplet), chiplet=transfer_chiplet)
        for gap in self.inter_gaps.values():
            self.chiplet_topology.add_edge(str(gap.area_u), str(gap.area_v), gap=gap)

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

    @staticmethod
    def edge_fide_weight(u: tuple[int, int], v: tuple[int, int], attrs: dict) -> float:
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

    def get_shortest_path(self, source: tuple[int, int], target: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
        """
        Get the shortest path between two physical qubits.
        TODO: Replace dijkstra with cascade greedy.

        Args:
            source (tuple[int, int]): The source node of the transfer pair.
            target (tuple[int, int]): The target node of the transfer pair

        Returns:
            path (list[tuple[int, int]]): The path with highest fidelity.
            fidelity (float): The fidelity of the path
        """
        path = nx.shortest_path(
            self.qubit_topology, source, target, weight=Package.edge_fide_weight
        )
        fidelity = 1
        for i in range(len(path) - 1):
            fidelity *= self.qubit_topology.edges[(path[i], path[i + 1])]["coupler"].fide
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

    def get_physical_qubit_area(self, physical_qubit: tuple[int, int]) -> FuncArea:
        """
        Get the function area of a physical qubit belongs to.

        Args:
            physical_qubit (tuple[int, int]): The coordinate of the physical qubit.

        Returns:
            FuncArea: The function area of the physical qubit.
        """
        # return self.qubit_topology.nodes[physical_qubit]["qubit"].area
        for module in self.compute_modules:
            if module.in_area(physical_qubit):
                return module
        for line in self.transfer_lines:
            if line.in_area(physical_qubit):
                return line
        raise ValueError("Invalid physical qubit: {}".format(physical_qubit))

    def get_area_graph(self, area: FuncArea) -> nx.Graph:
        """
        Get the graph of function area in the package.

        Args:
            area (FuncArea): The function area.

        Returns:
            nx.Graph: The subgraph view of qubit topology in the function area.
        """
        return self.qubit_topology.subgraph(area.get_nodes())

    @staticmethod
    def edge_metro_weight(u: tuple[int, int], v: tuple[int, int], attrs: dict) -> int:
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

    def get_shortest_metro_path(self, source: tuple[int, int], target: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
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
            self.metro_topology, source, target, weight=Package.edge_metro_weight
        )
        interchange_num = 0
        for i in range(len(path) - 1):
            interchange_num += self.metro_topology.edges[(path[i], path[i + 1])]["coupler"].role.value
        return path, interchange_num

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

    def draw_chiplets(self, 
        marginrate: float = 0.2, 
        markersize: float = 4, 
        linewidth_t: float = 1, 
        linewidth_b: float = 2, 
        show: bool = True, 
        save_path: str = None, 
    ):
        """
        Draw the package graph in chiplet level.

        Args:
            marginrate (float): The margin rate of the package.
            markersize (float): The size of the markers.
            linewidth_t (float): The linewidth of the transfer lines.
            linewidth_b (float): The linewidth of the compute lines.
            show (bool): Whether to show the figure.
            save_path (str): The path to save the figure.
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
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
