#!/usr/bin/env python
import random
import numpy as np
import vizdoom as vd
import scipy.ndimage as Simg
import ennemies
import map_parser
from tqdm import tqdm # NOQA

from network import tf, DRQN
from memory import ReplayMemory
from size_config import MIN_MEM_SIZE, MAX_MEM_SIZE, TRAINING_STEPS


fake_dataset_size = 100
batch_size = 25
sequence_length = 8
im_w = 108
im_h = 60
k = 1
n_actions = 2
ACTION_SET = np.eye(n_actions, dtype=np.uint32).tolist()


def create_game():
    game = vd.DoomGame()
    game.load_config("basic.cfg")

    # Ennemy detection
    walls = map_parser.parse("maps/basic.txt")
    game.clear_available_game_variables()
    game.add_available_game_variable(vd.GameVariable.POSITION_X)
    game.add_available_game_variable(vd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vd.GameVariable.POSITION_Z)
    game.set_labels_buffer_enabled(True)

    game.init()
    return game, walls


def play_episode(game, walls, verbose=False):
    epsilon = 1
    game.new_episode()
    dump = []
    while not game.is_episode_finished():
        # Get screen buf
        state = game.get_state()
        S = state.screen_buffer # NOQA

        # Resample to our network size
        h, w = S.shape[:2]
        S = Simg.zoom(S, [1.*im_h/h, 1.*im_w/w, 1]) # NOQA

        enn = len(ennemies.get_visible_ennemies(state, walls)) > 0
        game_features = [enn]

        # Epsilon-Greedy strat
        if np.random.rand() < epsilon:
            action = random.choice(ACTION_SET)
        else:
            action_no = sess.run(main.choice, feed_dict={
                main.images: [[S]],
            })
            action = ACTION_SET[action_no[0][0]]

        if verbose:
            print(action)
        reward = game.make_action(action)
        dump.append((S, action, reward, game_features))
    return dump


def wrap_play_episode(i):
    game, walls = create_game()
    res = play_episode(game, walls)
    game.close()
    return res


if __name__ == '__main__':
    print('Building main DRQN')
    main = DRQN(im_h, im_w, k, n_actions, 'main')
    # print('Building target DRQN')
    # target = DRQN(im_h, im_w, k, n_actions, 'target')
    # TODO target = main

    # fake states
    Xtr = np.ones((fake_dataset_size, sequence_length, im_h, im_w, 3))

    # initial LSTM state
    state = (np.zeros([batch_size, main.h_size]),
             np.zeros([batch_size, main.h_size]))

    game, walls = create_game()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Training vars:", [v.name for v in tf.trainable_variables()])

        # 1 / Bootstrap memory
        mem = ReplayMemory(min_size=MIN_MEM_SIZE, max_size=MAX_MEM_SIZE)

        from multiprocessing import Pool, cpu_count
        while not mem.initialized:
            for episode in Pool(cpu_count()).map(wrap_play_episode, range(cpu_count())):
                mem.add(episode)
            print(sum(map(len, mem.episodes)))

        # 2 / Replay all date shitte
        print("Replay ~o~ !!!")
        game, walls = create_game()
        for i in tqdm(range(TRAINING_STEPS)):
            mem.add(play_episode(game, walls))
            samples = mem.sample(batch_size, sequence_length)
            screens, actions, rewards, game_features = map(np.array, zip(*samples))
            loss = sess.run(main.features_loss, feed_dict={
                main.batch_size: batch_size,
                main.sequence_length: sequence_length,
                main.images: screens,
                main.game_features_in: game_features
            })

        print(loss)
