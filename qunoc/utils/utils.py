import os
import numpy as np
from PIL import Image
from datetime import datetime

color_pool = ['#D9D9D9', '#C9E0E5', '#BED2C6', '#FFD19D', '#FEB29B', '#ED8687']

def pick_n_elements(lst, n):
    m = len(lst) // n
    r = len(lst) % n
    idx = [np.floor(r/2) + np.ceil((m-1)/2) + i for i in range(len(lst)) if i % m == 0][:n]
    return [lst[int(i)] for i in idx]

def pick_n_links(size, lst, n):
    if n > len(lst):
        print('AT MOST {} LINKS'.format(len(lst)))
        return lst
    links_1 = pick_n_elements(lst, n)
    mid_link_1 = links_1[int(np.floor(len(links_1) / 2))]
    new_lst = list(lst[::-1])
    links_2 = pick_n_elements(new_lst, n)
    mid_link_2 = links_2[int(np.floor(len(links_2) / 2))]
    mid_link = (size+1) / 2 - 1
    if abs(mid_link_2 - mid_link) < abs(mid_link_1 - mid_link):
        return links_2
    else:
        return links_1

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
