import numpy as np
from matplotlib.colors import hsv_to_rgb

def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    """
    Returns a different color RGB for every element in the array_color
    """
    if array_colors is not None:
        number_of_colors = len(array_colors)

    h = np.linspace(0.1, 0.8, number_of_colors)
    np.random.shuffle(h)
    # values = np.linspace(0, np.pi, number_of_colors)
    colors = np.ones((3, number_of_colors))

    colors[0, :] = h
    
    return hsv_to_rgb(colors.T).T
    # colors[1, :] *= abs(np.cos(values * (number_of_colors*2) * np.pi))
    # # colors[2, :] *= abs(np.sin(values * (number_of_colors*1000) * np.pi))
    # colors[2, :] = 1-colors[0, :]
    # if return_list:
    #     return [c for c in colors.T]
    # return colors
