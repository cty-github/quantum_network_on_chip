import networkx as nx


def gen_linear_lattice(size):
    return nx.path_graph(size)

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
