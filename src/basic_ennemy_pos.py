import numpy as np
import vizdoom as vd
import map_parser

N_FEATURES = 1


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
        ennemy_pos# < 0.5,            # Ennemy on the left
        #0.45 <= ennemy_pos <= 0.55,  # Ennemy in the middle
    ]

def create_game():
    game = vd.DoomGame()
    game.load_config("basic.cfg")

    # Ennemy detection
    walls = map_parser.parse("maps/deathmatch.txt")
    game.clear_available_game_variables()
    game.add_available_game_variable(vd.GameVariable.POSITION_X)  # 0
    game.add_available_game_variable(vd.GameVariable.POSITION_Y)  # 1
    game.add_available_game_variable(vd.GameVariable.POSITION_Z)  # 2

    game.add_available_game_variable(vd.GameVariable.KILLCOUNT)   # 3
    game.add_available_game_variable(vd.GameVariable.DEATHCOUNT)  # 4
    game.add_available_game_variable(vd.GameVariable.ITEMCOUNT)   # 5

    game.set_labels_buffer_enabled(True)

    game.init()
    return game, walls