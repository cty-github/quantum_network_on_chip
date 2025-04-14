import math
import random
from copy import deepcopy
from enum import Enum, auto

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from ..utils.utils import color_pool, get_full_space_tuple

design_templates = ["strip", "radial", "grid"]
design_orientations = ["x", "y"]


class CellType(Enum):
    Empty = 0
    Compute = auto()
    Transfer = auto()


class CouplerType(Enum):
    Empty = 0
    IntraDie = auto()
    InterDie = auto()


class RoleType(Enum):
    Empty = 0
    Data = auto()
    Platform = auto()
    Balise = auto()
    Station = auto()
    Rail = auto()
    Interchange = auto()


class FabricateModel:
    def __init__(
        self,
        max_die_size=(5, 5),
        max_chain_length=20,
        compute_geometry="square",
        inter_die_sparsity=2,
        single_gate_fide=0.9999,
        intra_die_fide_dict={27: 0.993, 65: 0.989, 127: 0.988},
        enhanced_fide=0.9999,
        inter_die_fide=0.99,
        wafer_size=100,
        yield_5=0.99,
        yield_decay=2,
    ):
        """
        The model for fabrication parameters.
        max_die_size: (int, int), the maximum size of a die
        max_chain_length: int, the maximum length of a transfer chain
        compute_geometry: str, 'square', 'hexagon', 'heavy square', 'heavy hexagon'
        inter_die_sparsity: int, the sparsity of coupler between two dies
        single_gate_fide: float, the fidelity of single gate
        intra_die_fide_dict: dict, the fidelity of intra-die coupler according to the number of qubits
        enhanced_fide: float, the unit fidelity of enhanced transfer
        inter_die_fide: float, the fidelity of inter-die coupler
        wafer_size: int, the size of wafer
        yield_rate: float, the yield rate of fabrication
        """

        self.max_die_size = max_die_size
        self.max_chain_length = max_chain_length

        self.compute_geometry = compute_geometry
        self.inter_die_sparsity = inter_die_sparsity

        self.single_gate_fide = single_gate_fide
        self.intra_die_fide_dict = intra_die_fide_dict

        def log_decay(x, a, b):
            return 1 - a * np.log(1 + b * x)

        params, _ = curve_fit(
            log_decay,
            np.array(list(intra_die_fide_dict.keys())),
            np.array(list(intra_die_fide_dict.values())),
            p0=[0.01, 0.01],
        )
        self.intra_die_fide_a, self.intra_die_fide_b = params
        self.intra_die_fide_log_decay = log_decay

        self.enhanced_fide = enhanced_fide
        self.inter_die_fide = inter_die_fide

        self.wafer_size = wafer_size
        self.yield_5 = yield_5
        self.yield_decay = yield_decay

    def get_intra_die_fide(self, die_size):
        """
        Get the fidelity of intra-die coupler according to the die size.
        die_size: (int, int), the size of die
        """

        qubit_num = die_size[0] * die_size[1]
        return self.intra_die_fide_log_decay(
            qubit_num, self.intra_die_fide_a, self.intra_die_fide_b
        )

    def get_die_yield(self, die_size):
        """
        Get the yield of a die according to the die size.
        die_size: (int, int), the size of die
        """

        qubit_num = die_size[0] * die_size[1]
        return (
            self.yield_5 * (1 - np.tanh(self.yield_decay * (np.log10(qubit_num) - 2))) / 2
        )

    def get_die_cost(self, die_size):
        """
        Get the cost of a die according to the die size.
        die_size: (int, int), the size of die
        """

        return die_size[0] * die_size[1] / self.get_die_yield(die_size)


class Package:
    def __init__(
        self,
        package_size=(30, 10),
        fabricate_model=FabricateModel(),
        design_scheme="top_down",
        design_steps=10,
        design_strategy="heuristic",
        type_matrix=None,
        template="string",
    ):
        """
        package_size: (int, int), the size of package
        fabricate_model: FabricateModel, the model of fabrication
        design_scheme: str, 'top_down' or 'bottom_up', the design scheme of package
        design_steps: int, the number of steps for top down design
        design_strategy: str, 'heuristic' or 'greedy' or 'rl', the strategy of top down design
        type_matrix: np.ndarray, the matrix of cell types
        template: str, 'string' or 'random', the template of partition for bottom up design

        cell_graph: nx.Graph, the graph of cells in the package
                    node: (x, y), {'type': CellType, 'fide': float, 'die': str}
                    edge: ((x1, y1), (x2, y2)), {'type': CouplerType, 'fide': float}
        compute_cells: list, the list of compute cells
        transfer_cells: list, the list of transfer cells
        compute_areas: list, the list of compute areas
                    area: {'type': CellType, 'left': int, 'top': int, 'right': int, 'bottom': int}
        transfer_lines: list, the list of transfer lines
                    line: {'type': CellType, 'axis': int, 'pos': int, 'start': int, 'end': int}
        compute_dies: list, the list of compute dies
        transfer_dies: list, the list of transfer dies
        qubit_topology: nx.Graph, the graph of qubit topology
                    node: (x, y), {'type': CellType, 'fide': float, 'die': str, 'role': RoleType}
                    edge: ((x1, y1), (x2, y2)), {'type': CouplerType, 'fide': float}
        die_topology: nx.Graph, the graph of die topology
                    node: id, {'type': CellType, 'die': str, 'pos': (x, y)}
                    edge: (id1, id2), {'gap': gap}
        """

        self.package_size = package_size
        self.cell_num = package_size[0] * package_size[1]
        self.fabricate_model = fabricate_model
        self.design_scheme = design_scheme
        self.design_steps = design_steps
        self.design_strategy = design_strategy
        self.type_matrix = type_matrix
        self.template = template

        # determine compute area and transfer line
        if self.design_scheme == "top_down":
            self.compute_areas = [
                {
                    "type": CellType.Compute,
                    "left": 0,
                    "top": 0,
                    "right": self.package_size[0] - 1,
                    "bottom": self.package_size[1] - 1,
                }
            ]
            self.transfer_lines = []
            if self.design_strategy == "heuristic":
                area2design = self.compute_areas[0]
                length_width_ratio = max(self.package_size) / min(self.package_size)
                if length_width_ratio > 1:
                    if self.package_size[0] > self.package_size[1]:
                        print(
                            "Interval: ",
                            min(
                                self.package_size[1] - 1,
                                (self.package_size[0] + 1) // 2 - 1,
                            ),
                        )
                        new_compute_areas, new_transfer_lines = self.design_area(
                            area2design,
                            "strip",
                            min(
                                self.package_size[1] - 1,
                                (self.package_size[0] + 1) // 2 - 1,
                            ),
                            "y",
                        )
                    else:
                        new_compute_areas, new_transfer_lines = self.design_area(
                            area2design,
                            "strip",
                            min(
                                self.package_size[0] - 1,
                                (self.package_size[1] + 1) // 2 - 1,
                            ),
                            "x",
                        )
                    self.compute_areas.remove(area2design)
                    self.compute_areas.extend(new_compute_areas)
                    self.transfer_lines.extend(new_transfer_lines)
                    area2design = self.compute_areas[len(self.compute_areas) // 2]
                grid_round = len(self.compute_areas) - 1
                radial_round = (
                    min(self.package_size) - 2 * max(self.fabricate_model.max_die_size)
                ) // (4 * max(self.fabricate_model.max_die_size))
                for i in range(radial_round):
                    new_compute_areas, new_transfer_lines = self.design_area(
                        area2design,
                        "radial",
                        2 * max(self.fabricate_model.max_die_size) - 1,
                        "x",
                    )
                    self.compute_areas.remove(area2design)
                    self.compute_areas.extend(new_compute_areas)
                    self.transfer_lines.extend(new_transfer_lines)
                    area2design = self.compute_areas[-1]
                for i in range(grid_round):
                    area2design = self.compute_areas[0]
                    if self.package_size[0] > self.package_size[1]:
                        new_compute_areas, new_transfer_lines = self.design_area(
                            area2design,
                            "grid",
                            min(
                                3 * max(self.fabricate_model.max_die_size) - 1,
                                (
                                    min(
                                        self.package_size[1],
                                        area2design["right"] - area2design["left"] + 1,
                                    )
                                    + 1
                                )
                                // 2
                                - 1,
                            ),
                            "x",
                        )
                    else:
                        new_compute_areas, new_transfer_lines = self.design_area(
                            area2design,
                            "grid",
                            min(
                                3 * max(self.fabricate_model.max_die_size) - 1,
                                (
                                    min(
                                        self.package_size[0],
                                        area2design["bottom"] - area2design["top"] + 1,
                                    )
                                    + 1
                                )
                                // 2
                                - 1,
                            ),
                            "y",
                        )
                    self.compute_areas.remove(area2design)
                    self.compute_areas.extend(new_compute_areas)
                    self.transfer_lines.extend(new_transfer_lines)
            elif self.design_strategy == "greedy":
                # TODO: Implement greedy strategy
                pass
            elif self.design_strategy == "rl":
                # TODO: Implement reinforcement learning strategy
                pass
            self.cell_graph: nx.Graph = nx.grid_2d_graph(*self.package_size)
            self.type_matrix = np.zeros(self.package_size, dtype=np.int8)
            self.compute_cells = []
            self.transfer_cells = []
            for area in self.compute_areas:
                for i in range(area["left"], area["right"] + 1):
                    for j in range(area["top"], area["bottom"] + 1):
                        self.cell_graph.nodes[(i, j)]["fide"] = (
                            self.fabricate_model.single_gate_fide
                        )
                        self.cell_graph.nodes[(i, j)]["type"] = area["type"]
                        self.type_matrix[i, j] = area["type"].value
                        self.compute_cells.append((i, j))
            for line in self.transfer_lines:
                for i in range(line["start"], line["end"] + 1):
                    node = get_full_space_tuple(line["pos"], line["axis"], i)
                    self.cell_graph.nodes[node]["fide"] = (
                        self.fabricate_model.single_gate_fide
                    )
                    self.cell_graph.nodes[node]["type"] = line["type"]
                    self.type_matrix[node] = line["type"].value
                    self.transfer_cells.append(node)
        elif self.design_scheme == "bottom_up":
            if self.type_matrix:
                assert self.type_matrix.shape == self.package_size
                self.cell_graph: nx.Graph = nx.grid_2d_graph(*self.package_size)
                self.compute_cells = []
                self.transfer_cells = []
                for node in self.cell_graph.nodes:
                    self.cell_graph.nodes[node]["fide"] = (
                        self.fabricate_model.single_gate_fide
                    )
                    self.cell_graph.nodes[node]["type"] = CellType(
                        self.type_matrix[node[0], node[1]]
                    )
                    if self.type_matrix[node[0], node[1]] == CellType.Compute.value:
                        self.compute_cells.append(node)
                    elif self.type_matrix[node[0], node[1]] == CellType.Transfer.value:
                        self.transfer_cells.append(node)
            else:
                self.cell_graph: nx.Graph = nx.grid_2d_graph(
                    self.package_size[0], self.package_size[1]
                )
                self.determine_transfer_cell()
                self.type_matrix = np.zeros(self.package_size, dtype=np.int8)
                self.compute_cells = []
                self.transfer_cells = []
                for node in self.cell_graph.nodes:
                    self.cell_graph.nodes[node]["fide"] = (
                        self.fabricate_model.single_gate_fide
                    )
                    self.type_matrix[node[0], node[1]] = self.cell_graph.nodes[node][
                        "type"
                    ].value
                    if self.cell_graph.nodes[node]["type"] == CellType.Compute:
                        self.compute_cells.append(node)
                    elif self.cell_graph.nodes[node]["type"] == CellType.Transfer:
                        self.transfer_cells.append(node)
                self.compute_areas, self.transfer_lines = self.check_consecutive_area()
        else:
            raise ValueError("Invalid design scheme: {}".format(design_scheme))

        self.complete_package()

    def determine_transfer_cell(self):
        """
        Determine the transfer chain to partition the package.
        template: str, 'string' or 'random', the template of partition
        """

        if self.template == "string":
            for node in self.cell_graph.nodes:
                if node[1] == 0 or node[1] == self.package_size[1] - 1:
                    self.cell_graph.nodes[node]["type"] = CellType.Transfer
                else:
                    self.cell_graph.nodes[node]["type"] = CellType.Compute
        elif self.template == "random":
            # TODO: Implement random partition
            pass
        else:
            raise ValueError("Invalid partition mode: {}".format(self.template))

    def check_consecutive_area(self):
        """
        Find all the consecutive areas in the package greedily.

        Return:
        compute_areas: list, the list of compute areas
        transfer_areas: list, the list of transfer areas
        area: {'type': CellType, 'left': int, 'top': int, 'right': int, 'bottom': int}
        """

        compute_areas = []
        transfer_areas = []
        check_flags = np.zeros(self.package_size, dtype=bool)
        for node in self.cell_graph.nodes:
            if not check_flags[node[0], node[1]]:
                if self.cell_graph.nodes[node]["type"] == CellType.Empty:
                    check_flags[node[0], node[1]] = True
                    continue
                area = {}
                area["type"] = self.cell_graph.nodes[node]["type"]
                area["left"] = node[0]
                area["top"] = node[1]
                area["right"] = node[0]
                area["bottom"] = node[1]
                check_flags[node[0], node[1]] = True
                for i in range(node[0] + 1, self.package_size[0]):
                    if self.cell_graph.nodes[(i, node[1])]["type"] == area["type"]:
                        area["right"] = i
                        check_flags[i, node[1]] = True
                    else:
                        break
                for j in range(node[1] + 1, self.package_size[1]):
                    consec_flag = True
                    for i in range(area["left"], area["right"] + 1):
                        if self.cell_graph.nodes[(i, j)]["type"] != area["type"]:
                            consec_flag = False
                            break
                    if consec_flag:
                        area["bottom"] = j
                        for i in range(area["left"], area["right"] + 1):
                            check_flags[i, j] = True
                    else:
                        break
                if area["type"] == CellType.Compute:
                    compute_areas.append(area)
                elif area["type"] == CellType.Transfer:
                    assert area["right"] == area["left"] or area["bottom"] == area["top"]
                    transfer_areas.append(area)
        return compute_areas, self.transfer_area_to_line(transfer_areas)

    def transfer_area_to_line(self, transfer_areas):
        transfer_lines = []
        for area in transfer_areas:
            if area["bottom"] == area["top"]:
                transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 0,
                        "pos": area["top"],
                        "start": area["left"],
                        "end": area["right"],
                    }
                )
            elif area["right"] == area["left"]:
                transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 1,
                        "pos": area["left"],
                        "start": area["top"],
                        "end": area["bottom"],
                    }
                )
            else:
                raise ValueError("Invalid transfer area: {}".format(area))
        return transfer_lines

    @staticmethod
    def is_valid_to_design(area, template, interval, orient):
        if area["type"] != CellType.Compute:
            return False
        if template not in design_templates:
            return False
        if orient not in design_orientations:
            return False
        if template == "strip":
            if orient == "x":
                if area["bottom"] - area["top"] + 1 < 2 * interval + 1:
                    return False
            elif orient == "y":
                if area["right"] - area["left"] + 1 < 2 * interval + 1:
                    return False
        elif template == "radial":
            if area["bottom"] - area["top"] + 1 < 3 * interval + 2:
                return False
            if area["right"] - area["left"] + 1 < 3 * interval + 2:
                return False
        elif template == "grid":
            if area["bottom"] - area["top"] + 1 < 2 * interval + 1:
                return False
            if area["right"] - area["left"] + 1 < 2 * interval + 1:
                return False
        return True

    @staticmethod
    def design_area(area, template, interval, orient, check_valid=True):
        if check_valid:
            assert Package.is_valid_to_design(area, template, interval, orient), (
                f"Invalid area to design:\narea={area}\ntemplate={template}\ninterval={interval}\norient={orient}"
            )
        new_compute_areas = []
        new_transfer_lines = []
        if template == "strip":
            if orient == "x":
                n_blocks = (area["bottom"] - area["top"] + 1) // (interval + 1)
                if n_blocks == 0:
                    new_compute_areas.append(
                        {
                            "type": CellType.Compute,
                            "left": area["left"],
                            "top": area["top"],
                            "right": area["right"],
                            "bottom": area["bottom"],
                        }
                    )
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_areas.append(
                            {
                                "type": CellType.Compute,
                                "left": area["left"],
                                "top": area["top"] + i * (interval + 1),
                                "right": area["right"],
                                "bottom": area["bottom"],
                            }
                        )
                    else:
                        new_compute_areas.append(
                            {
                                "type": CellType.Compute,
                                "left": area["left"],
                                "top": area["top"] + i * (interval + 1),
                                "right": area["right"],
                                "bottom": area["top"] + i * (interval + 1) + interval - 1,
                            }
                        )
                        new_transfer_lines.append(
                            {
                                "type": CellType.Transfer,
                                "axis": 0,
                                "pos": area["top"] + (i + 1) * (interval + 1) - 1,
                                "start": area["left"],
                                "end": area["right"],
                            }
                        )
            elif orient == "y":
                n_blocks = (area["right"] - area["left"] + 1) // (interval + 1)
                if n_blocks == 0:
                    new_compute_areas.append(
                        {
                            "type": CellType.Compute,
                            "left": area["left"],
                            "top": area["top"],
                            "right": area["right"],
                            "bottom": area["bottom"],
                        }
                    )
                for i in range(n_blocks):
                    if i == n_blocks - 1:
                        new_compute_areas.append(
                            {
                                "type": CellType.Compute,
                                "left": area["left"] + i * (interval + 1),
                                "top": area["top"],
                                "right": area["right"],
                                "bottom": area["bottom"],
                            }
                        )
                    else:
                        new_compute_areas.append(
                            {
                                "type": CellType.Compute,
                                "left": area["left"] + i * (interval + 1),
                                "top": area["top"],
                                "right": area["left"] + i * (interval + 1) + interval - 1,
                                "bottom": area["bottom"],
                            }
                        )
                        new_transfer_lines.append(
                            {
                                "type": CellType.Transfer,
                                "axis": 1,
                                "pos": area["left"] + (i + 1) * (interval + 1) - 1,
                                "start": area["top"],
                                "end": area["bottom"],
                            }
                        )
            else:
                raise ValueError("Invalid design orientation: {}".format(orient))
        elif template == "radial":
            assert area["bottom"] - area["top"] + 1 > 4
            assert area["right"] - area["left"] + 1 > 4
            if orient == "x":
                if area["bottom"] - area["top"] + 1 <= 3 * interval + 2:
                    interval = (area["bottom"] - area["top"] - 1) // 3
                if area["right"] - area["left"] + 1 <= 3 * interval + 2:
                    interval = (area["right"] - area["left"] - 1) // 3
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"],
                        "top": area["top"],
                        "right": area["right"] - interval - 1,
                        "bottom": area["top"] + interval - 1,
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 0,
                        "pos": area["top"] + interval,
                        "start": area["left"],
                        "end": area["right"] - interval - 1,
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["right"] - interval + 1,
                        "top": area["top"],
                        "right": area["right"],
                        "bottom": area["bottom"] - interval - 1,
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 1,
                        "pos": area["right"] - interval,
                        "start": area["top"],
                        "end": area["bottom"] - interval - 1,
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"] + interval + 1,
                        "top": area["bottom"] - interval + 1,
                        "right": area["right"],
                        "bottom": area["bottom"],
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 0,
                        "pos": area["bottom"] - interval,
                        "start": area["left"] + interval + 1,
                        "end": area["right"],
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"],
                        "top": area["top"] + interval + 1,
                        "right": area["left"] + interval - 1,
                        "bottom": area["bottom"],
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 1,
                        "pos": area["left"] + interval,
                        "start": area["top"] + interval + 1,
                        "end": area["bottom"],
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"] + interval + 1,
                        "top": area["top"] + interval + 1,
                        "right": area["right"] - interval - 1,
                        "bottom": area["bottom"] - interval - 1,
                    }
                )
            elif orient == "y":
                if area["bottom"] - area["top"] + 1 <= 3 * interval + 2:
                    interval = (area["bottom"] - area["top"] - 1) // 3
                if area["right"] - area["left"] + 1 <= 3 * interval + 2:
                    interval = (area["right"] - area["left"] - 1) // 3
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"],
                        "top": area["top"],
                        "right": area["left"] + interval - 1,
                        "bottom": area["bottom"] - interval - 1,
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 1,
                        "pos": area["left"] + interval,
                        "start": area["top"],
                        "end": area["bottom"] - interval - 1,
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"] + interval + 1,
                        "top": area["top"],
                        "right": area["right"],
                        "bottom": area["top"] + interval - 1,
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 0,
                        "pos": area["top"] + interval,
                        "start": area["left"] + interval + 1,
                        "end": area["right"],
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["right"] - interval + 1,
                        "top": area["top"] + interval + 1,
                        "right": area["right"],
                        "bottom": area["bottom"],
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 1,
                        "pos": area["right"] - interval,
                        "start": area["top"] + interval + 1,
                        "end": area["bottom"],
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"],
                        "top": area["bottom"] - interval + 1,
                        "right": area["right"] - interval - 1,
                        "bottom": area["bottom"],
                    }
                )
                new_transfer_lines.append(
                    {
                        "type": CellType.Transfer,
                        "axis": 0,
                        "pos": area["bottom"] - interval,
                        "start": area["left"],
                        "end": area["right"] - interval - 1,
                    }
                )
                new_compute_areas.append(
                    {
                        "type": CellType.Compute,
                        "left": area["left"] + interval + 1,
                        "top": area["top"] + interval + 1,
                        "right": area["right"] - interval - 1,
                        "bottom": area["bottom"] - interval - 1,
                    }
                )
            else:
                raise ValueError("Invalid design orientation: {}".format(orient))
        elif template == "grid":
            if orient == "x":
                n_blocks = (area["bottom"] - area["top"] + 1) // (interval + 1)
                n_units = (area["right"] - area["left"] + 1) // (interval + 1)
                if n_blocks == 0 or n_units == 0:
                    new_compute_areas.append(
                        {
                            "type": CellType.Compute,
                            "left": area["left"],
                            "top": area["top"],
                            "right": area["right"],
                            "bottom": area["bottom"],
                        }
                    )
                else:
                    for i in range(n_blocks):
                        for j in range(n_units):
                            if i == n_blocks - 1:
                                if j == n_units - 1:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + j * (interval + 1),
                                            "top": area["top"] + i * (interval + 1),
                                            "right": area["right"],
                                            "bottom": area["bottom"],
                                        }
                                    )
                                else:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + j * (interval + 1),
                                            "top": area["top"] + i * (interval + 1),
                                            "right": area["left"]
                                            + j * (interval + 1)
                                            + interval
                                            - 1,
                                            "bottom": area["bottom"],
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 1,
                                            "pos": area["left"]
                                            + (j + 1) * (interval + 1)
                                            - 1,
                                            "start": area["top"] + i * (interval + 1),
                                            "end": area["bottom"],
                                        }
                                    )
                            else:
                                if j == n_units - 1:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + j * (interval + 1),
                                            "top": area["top"] + i * (interval + 1),
                                            "right": area["right"],
                                            "bottom": area["top"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 0,
                                            "pos": area["top"]
                                            + (i + 1) * (interval + 1)
                                            - 1,
                                            "start": area["left"],
                                            "end": area["right"],
                                        }
                                    )
                                else:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + j * (interval + 1),
                                            "top": area["top"] + i * (interval + 1),
                                            "right": area["left"]
                                            + j * (interval + 1)
                                            + interval
                                            - 1,
                                            "bottom": area["top"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 1,
                                            "pos": area["left"]
                                            + (j + 1) * (interval + 1)
                                            - 1,
                                            "start": area["top"] + i * (interval + 1),
                                            "end": area["top"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
            elif orient == "y":
                n_blocks = (area["right"] - area["left"] + 1) // (interval + 1)
                n_units = (area["bottom"] - area["top"] + 1) // (interval + 1)
                if n_blocks == 0 or n_units == 0:
                    new_compute_areas.append(
                        {
                            "type": CellType.Compute,
                            "left": area["left"],
                            "top": area["top"],
                            "right": area["right"],
                            "bottom": area["bottom"],
                        }
                    )
                else:
                    for i in range(n_blocks):
                        for j in range(n_units):
                            if i == n_blocks - 1:
                                if j == n_units - 1:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + i * (interval + 1),
                                            "top": area["top"] + j * (interval + 1),
                                            "right": area["right"],
                                            "bottom": area["bottom"],
                                        }
                                    )
                                else:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + i * (interval + 1),
                                            "top": area["top"] + j * (interval + 1),
                                            "right": area["right"],
                                            "bottom": area["top"]
                                            + j * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 0,
                                            "pos": area["top"]
                                            + (j + 1) * (interval + 1)
                                            - 1,
                                            "start": area["left"] + i * (interval + 1),
                                            "end": area["right"],
                                        }
                                    )
                            else:
                                if j == n_units - 1:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + i * (interval + 1),
                                            "top": area["top"] + j * (interval + 1),
                                            "right": area["left"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                            "bottom": area["bottom"],
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 1,
                                            "pos": area["left"]
                                            + (i + 1) * (interval + 1)
                                            - 1,
                                            "start": area["top"],
                                            "end": area["bottom"],
                                        }
                                    )
                                else:
                                    new_compute_areas.append(
                                        {
                                            "type": CellType.Compute,
                                            "left": area["left"] + i * (interval + 1),
                                            "top": area["top"] + j * (interval + 1),
                                            "right": area["left"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                            "bottom": area["top"]
                                            + j * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
                                    new_transfer_lines.append(
                                        {
                                            "type": CellType.Transfer,
                                            "axis": 0,
                                            "pos": area["top"]
                                            + (j + 1) * (interval + 1)
                                            - 1,
                                            "start": area["left"] + i * (interval + 1),
                                            "end": area["left"]
                                            + i * (interval + 1)
                                            + interval
                                            - 1,
                                        }
                                    )
            else:
                raise ValueError("Invalid design orientation: {}".format(orient))
        else:
            raise ValueError("Invalid design template: {}".format(template))
        return new_compute_areas, new_transfer_lines

    def divide_compute_areas(self):
        """
        Divide the compute areas into manufacturable compute dies.

        Return:
        compute_dies: list, the list of compute dies
        die: {'type': CellType, 'left': int, 'top': int, 'right': int, 'bottom': int}
        """

        # TODO: set the topology of compute dies according to the fabricate model
        compute_dies = []
        for area in self.compute_areas:
            area_x_size = area["right"] - area["left"] + 1
            area_y_size = area["bottom"] - area["top"] + 1
            die_x_num = (area_x_size - 1) // self.fabricate_model.max_die_size[0] + 1
            die_y_num = (area_y_size - 1) // self.fabricate_model.max_die_size[1] + 1
            die_x_size = area_x_size // die_x_num
            die_y_size = area_y_size // die_y_num
            large_die_x_num = area_x_size - die_x_num * die_x_size
            large_die_y_num = area_y_size - die_y_num * die_y_size
            for i in range(die_x_num):
                for j in range(die_y_num):
                    compute_dies.append(
                        {
                            "type": CellType.Compute,
                            "left": area["left"]
                            + i * die_x_size
                            + min(i, large_die_x_num),
                            "top": area["top"] + j * die_y_size + min(j, large_die_y_num),
                            "right": area["left"]
                            + (i + 1) * die_x_size
                            + min(i + 1, large_die_x_num)
                            - 1,
                            "bottom": area["top"]
                            + (j + 1) * die_y_size
                            + min(j + 1, large_die_y_num)
                            - 1,
                        }
                    )
                    compute_dies[-1]["id"] = (
                        "c_"
                        + str(compute_dies[-1]["left"])
                        + "_"
                        + str(compute_dies[-1]["top"])
                        + "_"
                        + str(compute_dies[-1]["right"])
                        + "_"
                        + str(compute_dies[-1]["bottom"])
                    )
        return compute_dies

    def divide_transfer_lines(self):
        """
        Divide the transfer lines into manufacturable transfer dies.

        Return:
        transfer_dies: list, the list of transfer dies
        die: {'type': CellType, 'axis': int, 'pos': int, 'start': int, 'end': int}
        """

        transfer_dies = []
        for line in self.transfer_lines:
            area_length = line["end"] - line["start"] + 1
            chain_num = (area_length - 1) // self.fabricate_model.max_chain_length + 1
            chain_length = area_length // chain_num
            long_chain_num = area_length - chain_num * chain_length
            for i in range(chain_num):
                transfer_dies.append(
                    {
                        "type": CellType.Transfer,
                        "axis": line["axis"],
                        "pos": line["pos"],
                        "start": line["start"]
                        + i * chain_length
                        + min(i, long_chain_num),
                        "end": line["start"]
                        + (i + 1) * chain_length
                        + min(i + 1, long_chain_num)
                        - 1,
                    }
                )
                transfer_dies[-1]["id"] = (
                    "t_"
                    + str(transfer_dies[-1]["axis"])
                    + "_"
                    + str(transfer_dies[-1]["pos"])
                    + "_"
                    + str(transfer_dies[-1]["start"])
                    + "_"
                    + str(transfer_dies[-1]["end"])
                )
        return transfer_dies

    def determine_inter_die_coupler(self):
        """
        Determine the inter-die coupler in the package.

        Return:
        die_gaps: dict, the dictionary of inter-die gaps
        gap: {'id': str, 'axis': int, 'edges': dict}
        """

        for compute_die in self.compute_dies:
            for i in range(compute_die["left"], compute_die["right"] + 1):
                for j in range(compute_die["top"], compute_die["bottom"] + 1):
                    self.cell_graph.nodes[(i, j)]["die"] = compute_die["id"]
        for transfer_die in self.transfer_dies:
            for i in range(transfer_die["start"], transfer_die["end"] + 1):
                node = get_full_space_tuple(transfer_die["pos"], transfer_die["axis"], i)
                self.cell_graph.nodes[node]["die"] = transfer_die["id"]

        die_gaps = dict()
        for edge in self.cell_graph.edges:
            if (
                self.cell_graph.nodes[edge[0]]["type"] == CellType.Empty
                or self.cell_graph.nodes[edge[1]]["type"] == CellType.Empty
            ):
                self.cell_graph.edges[edge]["type"] = CouplerType.Empty
                self.cell_graph.edges[edge]["fide"] = None
            elif (
                self.cell_graph.nodes[edge[0]]["die"]
                == self.cell_graph.nodes[edge[1]]["die"]
            ):
                self.cell_graph.edges[edge]["type"] = CouplerType.IntraDie
                if self.cell_graph.nodes[edge[0]]["type"] == CellType.Compute:
                    die_id = self.cell_graph.nodes[edge[0]]["die"]
                    die_left = int(die_id.split("_")[1])
                    die_top = int(die_id.split("_")[2])
                    die_right = int(die_id.split("_")[3])
                    die_bottom = int(die_id.split("_")[4])
                    die_size = (die_right - die_left + 1, die_bottom - die_top + 1)
                    self.cell_graph.edges[edge]["fide"] = (
                        self.fabricate_model.get_intra_die_fide(die_size)
                    )
                else:
                    self.cell_graph.edges[edge]["fide"] = (
                        self.fabricate_model.enhanced_fide
                    )
            else:
                if edge[0][0] == edge[1][0]:
                    gap_axis = 0
                    if edge[0][1] < edge[1][1]:
                        gap_name = (
                            self.cell_graph.nodes[edge[0]]["die"]
                            + ", "
                            + self.cell_graph.nodes[edge[1]]["die"]
                        )
                    else:
                        gap_name = (
                            self.cell_graph.nodes[edge[1]]["die"]
                            + ", "
                            + self.cell_graph.nodes[edge[0]]["die"]
                        )
                else:
                    gap_axis = 1
                    if edge[0][0] < edge[1][0]:
                        gap_name = (
                            self.cell_graph.nodes[edge[0]]["die"]
                            + ", "
                            + self.cell_graph.nodes[edge[1]]["die"]
                        )
                    else:
                        gap_name = (
                            self.cell_graph.nodes[edge[1]]["die"]
                            + ", "
                            + self.cell_graph.nodes[edge[0]]["die"]
                        )
                if gap_name not in die_gaps:
                    die_gaps[gap_name] = {
                        "id": gap_name,
                        "axis": gap_axis,
                        "edges": {edge[0][gap_axis]: edge},
                    }
                else:
                    die_gaps[gap_name]["edges"][edge[0][gap_axis]] = edge
                self.cell_graph.edges[edge]["type"] = CouplerType.Empty
                self.cell_graph.edges[edge]["fide"] = None
        for die_gap in die_gaps.values():
            for _, edge in list(sorted(die_gap["edges"].items()))[
                :: self.fabricate_model.inter_die_sparsity
            ]:
                self.cell_graph.edges[edge]["type"] = CouplerType.InterDie
                self.cell_graph.edges[edge]["fide"] = self.fabricate_model.inter_die_fide
        return die_gaps

    def complete_package(self):
        """
        Complete the package graph.
        """

        self.compute_dies = self.divide_compute_areas()
        self.transfer_dies = self.divide_transfer_lines()
        self.die_gaps = self.determine_inter_die_coupler()

        self.metro_graph = nx.Graph()
        self.platform_list = []
        self.station_list = []
        self.interchange_list = []

        self.qubit_topology: nx.Graph = deepcopy(self.cell_graph)
        for node in self.qubit_topology.nodes:
            if self.qubit_topology.nodes[node]["type"] == CellType.Compute:
                self.qubit_topology.nodes[node]["role"] = RoleType.Data
            elif self.qubit_topology.nodes[node]["type"] == CellType.Transfer:
                self.qubit_topology.nodes[node]["role"] = RoleType.Balise
                self.metro_graph.add_node(node, role=RoleType.Balise)
            elif self.qubit_topology.nodes[node]["type"] == CellType.Empty:
                self.qubit_topology.remove_node(node)
        for edge in self.qubit_topology.edges:
            if self.qubit_topology.edges[edge]["type"] == CouplerType.Empty:
                self.qubit_topology.remove_edge(*edge)
            if (
                self.qubit_topology.nodes[edge[0]]["type"] == CellType.Transfer
                and self.qubit_topology.nodes[edge[1]]["type"] == CellType.Transfer
            ):
                self.metro_graph.add_edge(edge[0], edge[1], role=RoleType.Rail)
        for transfer_die in self.transfer_dies:
            for i in range(transfer_die["start"], transfer_die["end"] + 1):
                node = get_full_space_tuple(transfer_die["pos"], transfer_die["axis"], i)
                neighbor_nodes = self.qubit_topology.neighbors(node)
                for neighbor_node in neighbor_nodes:
                    if (
                        self.qubit_topology.edges[(node, neighbor_node)]["type"]
                        == CouplerType.InterDie
                    ):
                        if (
                            self.qubit_topology.nodes[neighbor_node]["type"]
                            == CellType.Compute
                        ):
                            if (
                                self.qubit_topology.nodes[node]["role"]
                                != RoleType.Interchange
                            ):
                                self.qubit_topology.nodes[node]["role"] = RoleType.Station
                        elif (
                            self.qubit_topology.nodes[neighbor_node]["type"]
                            == CellType.Transfer
                        ):
                            self.qubit_topology.nodes[node]["role"] = RoleType.Interchange
                if transfer_die["pos"] > 0:
                    prev_node = get_full_space_tuple(
                        transfer_die["pos"] - 1, transfer_die["axis"], i
                    )
                    self.qubit_topology.nodes[prev_node]["role"] = RoleType.Platform
                if transfer_die["pos"] < self.package_size[1 - transfer_die["axis"]] - 1:
                    next_node = get_full_space_tuple(
                        transfer_die["pos"] + 1, transfer_die["axis"], i
                    )
                    self.qubit_topology.nodes[next_node]["role"] = RoleType.Platform

        self.die_topology = nx.Graph()
        for compute_die in self.compute_dies:
            self.die_topology.add_node(
                compute_die["id"],
                die=compute_die,
                type=CellType.Compute,
                pos=(
                    (compute_die["left"] + compute_die["right"]) / 2,
                    (compute_die["top"] + compute_die["bottom"]) / 2,
                ),
            )
        for transfer_die in self.transfer_dies:
            pos = get_full_space_tuple(
                transfer_die["pos"],
                transfer_die["axis"],
                (transfer_die["start"] + transfer_die["end"]) / 2,
            )
            self.die_topology.add_node(
                transfer_die["id"],
                die=transfer_die,
                type=CellType.Transfer,
                pos=pos,
            )
        for die_gap in self.die_gaps.values():
            self.die_topology.add_edge(
                die_gap["id"].split(", ")[0], die_gap["id"].split(", ")[1], gap=die_gap
            )

    def update_cell(self, x, y, cell_type):
        """
        Update the cell type in the package.
        """

        if self.cell_graph.nodes[(x, y)]["type"] == cell_type:
            return

        if (x, y) in self.compute_cells:
            self.compute_cells.remove((x, y))
        elif (x, y) in self.transfer_cells:
            self.transfer_cells.remove((x, y))

        self.cell_graph.nodes[(x, y)]["type"] = cell_type
        self.type_matrix[x, y] = cell_type.value
        if cell_type == CellType.Compute:
            assert (x, y) not in self.compute_cells
            self.compute_cells.append((x, y))
        elif cell_type == CellType.Transfer:
            assert (x, y) not in self.transfer_cells
            self.transfer_cells.append((x, y))

    def update_all_cells(self, type_matrix):
        """
        Update all the cell types in the package.
        """

        self.type_matrix = type_matrix
        self.compute_cells = []
        self.transfer_cells = []
        for node in self.cell_graph.nodes:
            self.cell_graph.nodes[node]["type"] = CellType(
                self.type_matrix[node[0], node[1]]
            )
            if self.type_matrix[node[0], node[1]] == CellType.Compute.value:
                self.compute_cells.append(node)
            elif self.type_matrix[node[0], node[1]] == CellType.Transfer.value:
                self.transfer_cells.append(node)

    def random_transfer_pair(self):
        return random.sample(self.compute_cells, 2)

    def random_transfer_pairs(self, num):
        return [self.random_transfer_pair() for _ in range(num)]

    def edge_fide_weight(self, u, v, attrs):
        return -math.log(attrs["fide"])

    def get_shortest_path(self, transfer_pair):
        source, target = transfer_pair
        path = nx.shortest_path(
            self.qubit_topology, source, target, weight=self.edge_fide_weight
        )
        fidelity = 1
        for i in range(len(path) - 1):
            fidelity *= self.qubit_topology.edges[(path[i], path[i + 1])]["fide"]
        return path, fidelity

    def estimate_transfer(self, transfer_pairs=None):
        """
        Estimate the fidelity of transfer pairs.
        """

        if transfer_pairs is None:
            transfer_pairs = self.random_transfer_pairs(10)

        path_list = []
        fidelity_list = []
        for transfer_pair in transfer_pairs:
            path, fidelity = self.get_shortest_path(transfer_pair)
            path_list.append(path)
            fidelity_list.append(fidelity)

        return np.mean(fidelity_list), path_list, fidelity_list

    def estimate_efficiency(self, transfer_pairs=None):
        """
        Estimate the efficiency of package considering the yield cost.
        """

        if transfer_pairs is None:
            transfer_pairs = self.random_transfer_pairs(10)
        fidelity, _, _ = self.estimate_transfer(transfer_pairs)

        cost = 0
        for compute_die in self.compute_dies:
            die_size = (
                compute_die["right"] - compute_die["left"] + 1,
                compute_die["bottom"] - compute_die["top"] + 1,
            )
            cost += self.fabricate_model.get_die_cost(die_size)
        for transfer_die in self.transfer_dies:
            die_size = get_full_space_tuple(
                1, transfer_die["axis"], transfer_die["end"] - transfer_die["start"] + 1
            )
            cost += self.fabricate_model.get_die_cost(die_size)

        return -np.log(1 - fidelity) / cost

    def get_routing_topology(self):
        """
        Get the routing topology of the package.

        Return:
        qubit_topology, the graph of all the qubits
        die_topology, the graph of all the dies
        """

        return deepcopy(self.qubit_topology), deepcopy(self.die_topology)

    def get_data_qubit_num(self):
        """
        Get the number of physical qubits with data role in the package.
        """

        cnt = 0
        for node in self.qubit_topology.nodes:
            if self.qubit_topology.nodes[node]["role"] == RoleType.Data:
                cnt += 1
        return cnt

    def get_data_qubit_list(self):
        """
        Get the list of physical qubits with data role in the package.
        """

        data_qubit_list = []
        for node in self.qubit_topology.nodes:
            if self.qubit_topology.nodes[node]["role"] == RoleType.Data:
                data_qubit_list.append(node)
        return data_qubit_list

    def get_physical_qubit_area(self, physical_qubit: tuple[int, int]):
        for area in self.compute_areas:
            if (
                area["left"] <= physical_qubit[0] <= area["right"]
                and area["top"] <= physical_qubit[1] <= area["bottom"]
            ):
                return area
        for line in self.transfer_lines:
            if line["axis"] == 0:
                if (
                    line["pos"] == physical_qubit[1]
                    and line["start"] <= physical_qubit[0] <= line["end"]
                ):
                    return line
            elif line["axis"] == 1:
                if (
                    line["pos"] == physical_qubit[0]
                    and line["start"] <= physical_qubit[1] <= line["end"]
                ):
                    return line
            else:
                pass

    def get_die_capacity(self, die_id: str):
        die_info = die_id.split("_")
        if die_info[0] != "c":
            return 0
        die_left = int(die_info[1])
        die_top = int(die_info[2])
        die_right = int(die_info[3])
        die_bottom = int(die_info[4])
        capacity = 0
        for i in range(die_left, die_right + 1):
            for j in range(die_top, die_bottom + 1):
                if self.qubit_topology.nodes[(i, j)]["role"] == RoleType.Data:
                    capacity += 1
        return capacity

    def get_platform_list(self):
        return deepcopy(self.platform_list)

    def get_station_list(self):
        return deepcopy(self.station_list)

    def get_interchange_list(self):
        return deepcopy(self.interchange_list)

    def get_area_graph(self, area) -> nx.Graph:
        """
        Get the graph of some area in the package.
        """

        sub_nodes = []
        for i in range(area["left"], area["right"] + 1):
            for j in range(area["top"], area["bottom"] + 1):
                sub_nodes.append((i, j))

        schedule_graph = self.qubit_topology.subgraph(sub_nodes)
        return schedule_graph

    @staticmethod
    def edge_metro_weight(self, u, v, attrs):
        if attrs["role"] == RoleType.Rail:
            return 0
        elif attrs["role"] == RoleType.Interchange:
            return 1

    def get_shortest_metro_path(self, source, target):
        """
        Get the shortest path in metro graph.
        """

        assert source in self.metro_graph.nodes and target in self.metro_graph.nodes
        path = nx.shortest_path(
            self.metro_graph, source, target, weight=self.edge_metro_weight
        )
        interchange_num = 0
        for i in range(len(path) - 1):
            if (
                self.metro_graph.edges[(path[i], path[i + 1])]["role"]
                == RoleType.Interchange
            ):
                interchange_num += 1
        return path, interchange_num

    def draw_qubits(
        self, marginrate=0.2, markersize=4, linewidth_t=1, linewidth_b=2, save_path=None
    ):
        """
        Draw the package graph in qubit level.
        """

        plt.figure(figsize=self.package_size)
        for node in self.cell_graph.nodes:
            plt.fill(
                node[0] + np.array([-0.5, 0.5, 0.5, -0.5]),
                node[1] + np.array([-0.5, -0.5, 0.5, 0.5]),
                color=color_pool[self.type_matrix[node[0], node[1]]],
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
        for edge in self.cell_graph.edges:
            if (
                self.cell_graph.edges[edge]["type"] == CouplerType.InterDie
                or self.cell_graph.edges[edge]["type"] == CouplerType.Empty
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
            if self.cell_graph.edges[edge]["type"] == CouplerType.IntraDie:
                plt.plot(
                    [edge[0][0], edge[1][0]],
                    [edge[0][1], edge[1][1]],
                    "k-",
                    linewidth=linewidth_t,
                )
            elif self.cell_graph.edges[edge]["type"] == CouplerType.InterDie:
                plt.plot(
                    [edge[0][0], edge[1][0]],
                    [edge[0][1], edge[1][1]],
                    "r-",
                    linewidth=linewidth_b,
                )
        for node in self.cell_graph.nodes:
            if self.cell_graph.nodes[node]["type"] != CellType.Empty:
                plt.plot(node[0], node[1], "ko", markersize=markersize)
        plt.axis("off")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

    def draw_dies(
        self, marginrate=0.2, markersize=4, linewidth_t=1, linewidth_b=2, save_path=None
    ):
        """
        Draw the package graph in die level.
        """

        plt.figure(figsize=self.package_size)
        for die_id in self.die_topology.nodes:
            die = self.die_topology.nodes[die_id]
            plt.plot(*die["pos"], "ko", markersize=markersize)
        for u_id, v_id in self.die_topology.edges:
            u_pos = self.die_topology.nodes[u_id]["pos"]
            v_pos = self.die_topology.nodes[v_id]["pos"]
            plt.plot(
                [u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]], "k-", linewidth=linewidth_t
            )
        plt.axis("off")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

    def draw_path(
        self,
        path,
        figsize=(10, 5),
        marginrate=0.2,
        markersize=4,
        linewidth_t=1,
        linewidth_b=2,
        save_path=None,
    ):
        """
        Draw the package graph with a path for transfer pair.
        """

        # TODO: Implement draw_path
        pass
