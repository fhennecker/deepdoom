import numpy as np
import vizdoom as vd
import random
from tqdm import tqdm
from time import sleep


game = vd.DoomGame()
game.load_config("basic.cfg")
game.init()

actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

for i in tqdm(range(20)):
    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        misc = state.game_variables
        reward = game.make_action(random.choice(actions))
        sleep(0.01)
    sleep(1)
game.close()

