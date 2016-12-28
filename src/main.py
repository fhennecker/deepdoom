import random
import numpy as np
import vizdoom as vd
import scipy.ndimage as Simg

from network import tf, DRQN
from memory import ReplayMemory


fake_dataset_size = 100
batch_size = 10
sequence_length = 8
im_w = 108
im_h = 60
k = 1
n_actions = 6


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

    game = vd.DoomGame()
    game.load_config("basic.cfg")
    game.init()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Training vars:", [v.name for v in tf.trainable_variables()])

        actions = np.eye(n_actions, dtype=np.uint32).tolist()

        def play_episode(epsilon):
            game.new_episode()
            dump = []
            while not game.is_episode_finished():
                # Get screen buf
                state = game.get_state()
                S = state.screen_buffer

                # Resample to our network size
                h, w = S.shape[:2]
                S = Simg.zoom(S, [1.*im_h/h, 1.*im_w/w, 1])

                # Epsilon-Greedy strat
                if np.random.rand() < epsilon:
                    action = random.choice(actions)
                else:
                    action_no = sess.run(main.choice, feed_dict={
                        main.images: [[S]],
                    })
                    action = actions[action_no[0][0]]
                reward = game.make_action(action)
                dump.append((S, action, reward))
            return dump

        # 1 / Bootstrap memory
        mem = ReplayMemory(min_size=1000, max_size=10000)
        while not mem.initialized:
            mem.add(play_episode(epsilon=1))
            print(sum(map(len, mem.episodes)))

        # 2 / Replay all date shitte
        print("Replay ~o~ !!!")
        for i in range(1000):
            samples = mem.sample(batch_size, sequence_length)
            for screens, actions, rewards in samples:
                sess.run(main.choice, feed_dict={
                    main.batch_size: 1,
                    main.images: [screens]
                })
            break
