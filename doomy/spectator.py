#!/usr/bin/env python

from __future__ import print_function

from time import sleep
from vizdoom import * # NOQA
import ennemies
import map_parser

game = DoomGame()
game.set_labels_buffer_enabled(True)
game.set_depth_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

game.load_config("scenarios/deathmatch.cfg")
game.add_game_args("+freelook 1")

game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)

game.clear_available_game_variables()
game.add_available_game_variable(GameVariable.POSITION_X)
game.add_available_game_variable(GameVariable.POSITION_Y)
game.add_available_game_variable(GameVariable.POSITION_Z)


game.init()

episodes = 10

walls = map_parser.parse("maps/deathmatch.txt")

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()
    j = 0
    while not game.is_episode_finished():
        state = game.get_state()
        j += 1
        if j % 15 == 0:
            print([x.object_name for x in ennemies.get_visible_ennemies(state, walls)])

        game.advance_action()
        last_action = game.get_last_action()
        reward = game.get_last_reward()

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
