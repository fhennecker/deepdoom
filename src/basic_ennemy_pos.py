import numpy as np


def basic_ennemy_x(state):
    """
    Return a float between 0 (left) and 1 (right) representing the relative
    position of the ennemy in the player field of view in the basic scenario.
    """
    # Player horizon
    horiz = state.depth_buffer[65].astype(np.int32)
    # How is the distance varying along the horizon
    line = np.diff(horiz)
    line[np.abs(line) > 15] = 0
    imin = line.argmin()
    line[:imin] = 0
    imax = line.argmax()
    return (imin + imax) / 2 / len(horiz)
