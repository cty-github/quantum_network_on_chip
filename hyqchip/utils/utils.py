import numpy as np

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
