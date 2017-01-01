import random
import vizdoom as vd
import numpy as np
import scipy.ndimage as Simg

import map_parser
import ennemies
from network import tf, DRQN
from memory import ReplayMemory
from config import (
    N_ACTIONS, LEARNING_RATE, MIN_MEM_SIZE, MAX_MEM_SIZE,
    MAX_CPUS, TRAINING_STEPS, BATCH_SIZE, SEQUENCE_LENGTH,
    QLEARNING_STEPS
)

# Config variables
im_w, im_h = 108, 60
N_FEATURES = 3
ACTION_SET = np.eye(N_ACTIONS, dtype=np.uint32).tolist()
SECTION_SEPARATOR = "------------"

# Neural nets and tools
print('Building main DRQN')
main = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'main', LEARNING_RATE)
print('Building target DRQN')
target = DRQN(im_h, im_w, N_FEATURES, N_ACTIONS, 'target', LEARNING_RATE)
saver = tf.train.Saver()
mem = ReplayMemory(MIN_MEM_SIZE, MAX_MEM_SIZE)


def csv_output(*columns):
    def wrapper(func):
        def inner(*args, **kwargs):
            print("---------")
            print("::", func.__name__, "::")
            print(",".join(columns))
            return func(*args, **kwargs)
        return inner
    return wrapper


def create_game():
    game = vd.DoomGame()
    game.load_config("basic.cfg")

    # Ennemy detection
    walls = map_parser.parse("maps/deathmatch.txt")
    game.clear_available_game_variables()
    game.add_available_game_variable(vd.GameVariable.POSITION_X)
    game.add_available_game_variable(vd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vd.GameVariable.POSITION_Z)
    game.set_labels_buffer_enabled(True)

    game.init()
    return game, walls


def play_random_episode(game, walls, verbose=False, skip=1):
    game.new_episode()
    dump = []
    zoomed = np.zeros((500, im_h, im_w, 3), dtype=np.uint8)
    action = ACTION_SET[0]
    while not game.is_episode_finished():
        # Get screen buf
        state = game.get_state()
        S = state.screen_buffer  # NOQA

        # Resample to our network size
        h, w = S.shape[:2]
        Simg.zoom(S, [1. * im_h / h, 1. * im_w / w, 1],
                  output=zoomed[len(dump)], order=0)
        S = zoomed[len(dump)]  # NOQA

        # Get game features an action
        game_features = ennemies.has_visible_entities(state, walls)
        action = random.choice(ACTION_SET)
        reward = game.make_action(action, skip)
        dump.append((S, action, reward, game_features))
    return dump


def wrap_play_random_episode(i=0):
    game, walls = create_game()
    res = play_random_episode(game, walls, skip=4)
    game.close()
    return res

# Need to be imported and created after wrap_play_random_episode
from multiprocessing import Pool, cpu_count
N_CORES = min(cpu_count(), MAX_CPUS)


def multiplay():
    if N_CORES == 1:
        return [wrap_play_random_episode()]
    else:
        workers = Pool(N_CORES)
        return workers.map(wrap_play_random_episode, range(N_CORES))


def update_target(sess):
    """Transfer learned parameters from main to target NN"""
    v = tf.trainable_variables()
    main_vars = filter(lambda x: x.name.startswith('main'), v)
    target_vars = filter(lambda x: x.name.startswith('target'), v)
    for t, m in zip(target_vars, main_vars):
        sess.run(t.assign(m.value()))


@csv_output("mem_size", "n_games")
def init_phase(sess):
    """
    Attempt to restore a model, or initialize all variables.
    Then fills replay memory with random-action games
    """
    try:
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print("Successfully loaded model")
    except:
        import traceback
        traceback.print_exc()
        init = tf.global_variables_initializer()
        sess.run(init)
        print("=== Recreate new model ! ===")

    while not mem.initialized:
        for episode in multiplay():
            mem.add(episode)
        print("{},{}".format(len(mem), len(mem.episodes)))


@csv_output("training_step", "loss_training", "loss_test")
def training_phase(sess):
    """Supervised learning for game features"""
    for i in range(TRAINING_STEPS // N_CORES):
        # Play and add new episodes to memory
        for episode in multiplay():
            mem.add(episode)

        for j in range(N_CORES):
            # Sample a batch and ingest into the NN
            samples = mem.sample(BATCH_SIZE, SEQUENCE_LENGTH)
            # screens, actions, rewards, game_features
            S, A, R, F = map(np.array, zip(*samples))
            # Supervised learning for game features
            main.learn_game_features(S, F)
            # Feed recurrent part
            main.reset_hidden_state(batch_size=BATCH_SIZE)
            main.feed_lstm(sess, S, A, R)
        training_loss = main.current_game_features_loss(S, F)

        # Sample a batch and ingest into the NN
        samples = mem.sample(BATCH_SIZE, SEQUENCE_LENGTH)
        # screens, actions, rewards, game_features
        S, A, R, F = map(np.array, zip(*samples))
        test_loss = main.current_game_features_loss(S, F)

        print("{},{},{}".format(i*N_CORES, training_loss, test_loss))

        if i > 0 and i % 100 == 0:
            saver.save(sess, "./model.ckpt")


@csv_output("qlearning_step", "epsilon", "reward", "steps")
def learning_phase(sess):
    """Reinforcement learning for Qvalues"""
    game, walls = create_game()

    # From now on, we don't use game features, but we provide an empty
    # numpy array so that the ReplayMemory is still zippable
    game_features = np.zeros(N_FEATURES)
    screenbuf = np.zeros((im_h, im_w, 3), dtype=np.uint8)
    for i in range(QLEARNING_STEPS):
        # Linearly decreasing epsilon
        epsilon = 1 - (0.9 * i / QLEARNING_STEPS)
        game.new_episode()
        episode = []

        # Initialize new hidden state
        main.reset_hidden_state(batch_size=1)
        while not game.is_episode_finished():
            # Get and resize screen buffer
            state = game.get_state()
            h, w, d = state.screen_buffer.shape
            Simg.zoom(state.screen_buffer,
                      [1. * im_h / h, 1. * im_w / w, 1],
                      output=screenbuf, order=0)

            # Choose action with e-greedy network
            action_no = main.choose(sess, epsilon, screenbuf)
            action = ACTION_SET[action_no]
            reward = game.make_action(action, 4)
            episode.append((screenbuf, action, reward, game_features))
        mem.add(episode)
        tot_reward = sum(r for (s, a, r, f) in episode)
        print("{},{},{},{}".format(i, epsilon, tot_reward, len(episode)))

        # Adapt target every 10 runs
        if i > 0 and i % 10 == 0:
            for i in range(10):
                main.reset_hidden_state(batch_size=BATCH_SIZE)
                # Sample a batch and ingest into the NN
                samples = mem.sample(BATCH_SIZE, SEQUENCE_LENGTH)
                # screens, actions, rewards, game_features
                S, A, R, F = map(np.array, zip(*samples))

                target_q = sess.run(target.Q, feed_dict={
                    target.batch_size: BATCH_SIZE,
                    target.sequence_length: SEQUENCE_LENGTH,
                    target.images: S,
                })

                sess.run(main.train_step, feed_dict={
                    main.batch_size: BATCH_SIZE,
                    main.sequence_length: SEQUENCE_LENGTH,
                    main.images: S,
                    main.target_q: target_q,
                    main.gamma: 0.99,
                    main.rewards: R,
                })
                # main.train_q(S, A, R, target_q)

        # Save the model periodically
        if i > 0 and i % 100 == 0:
            saver.save(sess, "./model.ckpt")

    game.close()
