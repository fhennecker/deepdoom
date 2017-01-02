import numpy as np

N_FEATURES = 2


def basic_ennemy_x(state):
    """
    Return a float between 0 (left) and 1 (right) representing the relative
    position of the ennemy in the player field of view in the basic scenario.
    """
    # Player horizon
    horiz = state.depth_buffer[65].astype(np.int32)
    # How is the distance varying along the horizon
    line = np.diff(horiz)
    # Ignore high distance diff (player gun)
    line[np.abs(line) > 20] = 0
    # Return the relative position
    return (line.argmin() + line.argmax()) / 2 / len(horiz)


def basic_ennemy_pos_features(state):
    ennemy_pos = basic_ennemy_x(state)
    return [
        ennemy_pos < 0.5,            # Ennemy on the left
        0.45 <= ennemy_pos <= 0.55,  # Ennemy in the middle
    ]
