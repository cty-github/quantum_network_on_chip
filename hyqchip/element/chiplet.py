import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from ..utils.utils import *


def gen_square_lattice(x_size, y_size):
    return nx.grid_2d_graph(x_size, y_size)

def gen_hexagon_lattice(x_size, y_size):
    lattice = nx.grid_2d_graph(x_size, y_size)
    for node in lattice.nodes:
        if node[1] < y_size-1 and (node[0] + node[1])%2 == 1:
            lattice.remove_edge(node, (node[0], node[1]+1))
    return lattice

def gen_heavy_square_lattice(x_size, y_size):
    lattice = nx.grid_2d_graph(x_size, y_size)
    nodes_to_remove = []
    for node in lattice.nodes:
        if node[0]%2==1 and node[1]%2==1:
           nodes_to_remove.append(node)
    lattice.remove_nodes_from(nodes_to_remove)
    return lattice

def gen_heavy_hexagon_lattice(x_size, y_size):
    lattice = gen_heavy_square_lattice(x_size, y_size)
    nodes_to_remove = []
    for x,y in lattice.nodes:
        if y<y_size-1 and x%2==0 and y%2==0 and (x+y)%4==2:
            nodes_to_remove.append((x,y+1))
    lattice.remove_nodes_from(nodes_to_remove)
    return lattice

def gen_linear_lattice(size):
    return nx.path_graph(size)


class ComputeChiplet(nx.Graph):
    def __init__(self, geometry, chiplet_x_size, chiplet_y_size, array_coordinate, cross_link_sparsity=None, cross_link_row_sparsity=None, cross_link_col_sparsity=None, auto=True):
        self.structure = geometry
        self.chiplet_x_size = chiplet_x_size
        self.chiplet_y_size = chiplet_y_size
        self.array_coordinate = array_coordinate

        if geometry == 'square':
            super().__init__(gen_square_lattice(chiplet_x_size, chiplet_y_size))
            self.cross_link_rows = list(range(chiplet_y_size))
            self.cross_link_cols = list(range(chiplet_x_size))
        if geometry == 'hexagon':
            if auto:
                if chiplet_x_size % 2 == 1:
                    print('WARNING: hexagon chiplets should have even columns')
                    chiplet_x_size -= 1
            super().__init__(gen_hexagon_lattice(chiplet_x_size, chiplet_y_size))
            self.cross_link_rows = list(range(chiplet_y_size))
            self.cross_link_cols = list(range((chiplet_y_size+1)%2,chiplet_x_size,2))
        if geometry == 'heavy_square':
            if auto:
                if chiplet_x_size % 2 == 1:
                    print('WARNING: heavy square chiplets should have even columns')
                    chiplet_x_size -= 1
                if chiplet_y_size % 2 == 1:
                    print('WARNING: heavy square chiplets should have even rows')
                    chiplet_y_size -= 1
            super().__init__(gen_heavy_square_lattice(chiplet_x_size, chiplet_y_size))
            self.cross_link_rows = list(range(0,chiplet_y_size,2))
            self.cross_link_cols = list(range(0,chiplet_x_size,2))
        if geometry == 'heavy_hexagon':
            if auto:
                if chiplet_x_size % 4 != 0:
                    print('WARNING: heavy hexagon chiplets should have 4n columns')
                    chiplet_x_size = chiplet_x_size // 4 * 4
                if chiplet_y_size % 4 != 0:
                    print('WARNING: heavy hexagon chiplets should have 4n rows')
                    chiplet_y_size = chiplet_y_size // 4 * 4
            super().__init__(gen_heavy_hexagon_lattice(chiplet_x_size, chiplet_y_size))
            self.cross_link_rows = list(range(0,chiplet_y_size,2))
            self.cross_link_cols = list(range(2,chiplet_x_size,4))
        
        if cross_link_row_sparsity is None and cross_link_col_sparsity is None and cross_link_sparsity is not None:
            cross_link_row_sparsity = cross_link_col_sparsity = cross_link_sparsity
        if cross_link_row_sparsity:
            self.cross_link_rows = pick_n_links(self.chiplet_y_size, sorted(self.cross_link_rows), cross_link_row_sparsity)
        if cross_link_col_sparsity:
            self.cross_link_cols = pick_n_links(self.chiplet_x_size, sorted(self.cross_link_cols), cross_link_col_sparsity)
        
        for node in self.nodes:
            self.nodes[node]['type'] = 'data'
    
    def draw(self, with_labels=False, node_size=100, node_color='black'):
        nx.draw(self, pos={node: node for node in self.nodes}, with_labels=with_labels, node_size=node_size, node_color=node_color)


class TransferChiplet(nx.Graph):
    def __init__(self, chiplet_size, array_coordinate, chiplet_axis=0, auto=True):
        self.chiplet_size = chiplet_size
        self.chiplet_axis = chiplet_axis
        self.array_coordinate = array_coordinate
        
        super().__init__(gen_linear_lattice(chiplet_size))

    def draw(self, with_labels=False, node_size=100, node_color='black'):
        if self.chiplet_axis == 0:
            pos = {node: (node, 0) for node in self.nodes}
        else:
            pos = {node: (0, node) for node in self.nodes}
        nx.draw(self, pos=pos, with_labels=with_labels, node_size=node_size, node_color=node_color)


if __name__ == '__main__':
    geometries = ['square', 'hexagon', 'heavy_square', 'heavy_hexagon']
    plt.figure(figsize=(8,8))
    for geometry in geometries:
        compute_chiplet = ComputeChiplet(geometry, 8, 8, (0,0))
        print(compute_chiplet.number_of_nodes())
        print(compute_chiplet.nodes)
        plt.subplot(221+geometries.index(geometry))
        compute_chiplet.draw()
        plt.title(geometry)
    plt.show()

    transfer_chiplet = TransferChiplet(8, (0,0))
    print(transfer_chiplet.number_of_nodes())
    plt.figure(figsize=(8,8))
    transfer_chiplet.draw()
    plt.title('transfer')
    plt.show()
