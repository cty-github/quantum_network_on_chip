import os
import numpy as np
from PIL import Image
from datetime import datetime
import networkx as nx

color_pool = ['#D9D9D9', '#C9E0E5', '#BED2C6', '#FFD19D', '#FEB29B', '#ED8687']

def log_decay(x, a, b):
    return 1 - a * np.log(1 + b * x)

def get_full_space_tuple(pos: int, axis: int, index: int) -> tuple[int, int]:
    """
    Get the full space tuple for a given position, axis, and index.

    Args:
        pos (int): The position in the specified axis.
        axis (int): The axis to insert the index into (0 or 1).
        index (int): The index to insert at the specified axis.

    Returns:
        tuple[int, int]: A tuple representing the full space coordinates.
    """
    coordinate_list = [pos]
    coordinate_list.insert(axis, index)
    return tuple(coordinate_list)

def get_timestamp() -> str:
    """
    Get current time as unique stamp.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")

def get_unique_file_save_path(dir: str, prefix: str, ext: str) -> str:
    """
    Get the unique name for file with the same prefix in the director.

    Args:
        dir (str): The director to save file.
        prefix (str): The prefix of the file.
        ext (str): The ext of the file.
    
    Returns:
        str: The save path with unique file name.
    """
    index = len([f for f in os.listdir(dir) if f.startswith(prefix) and f.endswith(ext)])
    return f"{dir}{prefix}{index}{ext}"

def make_gif(dir: str, prefix, ext: str, filename: str, duration: int = 1000):
    """
    Make gif with figures in the director with given prefix.

    Args:
        dir (str): The director saving the figures.
        prefix (str): The prefix of the figures.
        ext (str): The ext of the figures.
        filename (str): The filename of gif.
        duration (int): The duration of each frame in the gif.
    """

    files = sorted([f for f in os.listdir(dir) if f.startswith(prefix) and f.endswith(ext)])
    figures = [Image.open(os.path.join(dir, f)) for f in files]
    figures[0].save(
        f"{dir}{filename}",
        save_all=True,
        append_images=figures[1:],
        duration=duration,
        loop=0
    )

def find_longest_topo_distance(G: nx.DiGraph) -> int:
    """
    Find the length of the longest path in a directed acyclic graph (DAG).

    Args:
        G (nx.DiGraph): The directed acyclic graph.

    Returns:
        int: The length of the longest path in the graph.
    """
    topo_order = list(nx.topological_sort(G))
    dist = {node: 0 for node in G.nodes()}
    for u in topo_order:
        for v in G.successors(u):
            if dist[v] < dist[u] + 1:
                dist[v] = dist[u] + 1
    farthest_node = max(dist, key=dist.get)
    return dist[farthest_node]
