import numpy as np
import vizdoom as vd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from time import sleep


def where_is_the_ennemy_x(state):
    """Return a tuple of x positions within which the ennemy should be"""
    line = np.diff(state.depth_buffer[65].astype(np.int32))
    return line.argmin(), line.argmax()


game = vd.DoomGame()
game.load_config("basic.cfg")
game.init()

actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

for i in tqdm(range(20)):
    game.new_episode()

    while not game.is_episode_finished():
        print(where_is_the_ennemy_x(game.get_state()))
        reward = game.make_action(random.choice(actions))
        sleep(0.01)
    sleep(1)
game.close()

