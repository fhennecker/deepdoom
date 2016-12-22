import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import vizdoom as vd
import random
import scipy.ndimage as Simg
from tqdm import tqdm
from time import sleep

class ReplayMemory():
    def __init__(self, min_size, max_size):
        self.episodes = []
        self.min_size, self.max_size = min_size, max_size

    @property
    def full(self):
        return len(self.episodes) >= self.max_size

    def add(self, episode):
        if self.full:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size, sequence_length):
        batch = []
        for b in range(batch_size):
            episode = random.choice(self.episodes)
            start = random.randint(0, len(episode)-sequence_length)
            batch.append(episode[start:start+sequence_length])
        return batch


class DRQN():
    def __init__(self, im_h, im_w, k, n_actions, scope):
        self.im_h, self.im_w, self.k = im_h, im_w, k
        self.scope, self.n_actions = scope, n_actions
        
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        self.images = tf.placeholder(tf.float32, name='images',
                shape=[None, None, im_h, im_w, 3])
        # we'll merge all sequences in one single batch for treatment
        # but all outputs will be reshaped to [batch_size, length, ...]
        self.all_images = tf.reshape(self.images,
                [self.batch_size*self.sequence_length, im_h, im_w, 3])

        self._init_conv_layers()
        self._init_game_features_output()
        self._init_recurrent_part()
        self._define_loss()

    def _init_conv_layers(self):
        self.conv1 = slim.conv2d(
                self.all_images, 32, [8, 8], [4, 4], 'VALID',
                biases_initializer=None, scope=self.scope+'_conv1')
        self.conv2 = slim.conv2d(
                self.conv1, 64, [4, 4], [2, 2], 'VALID',
                biases_initializer=None, scope=self.scope+'_conv2')

    def _init_game_features_output(self):
        self.layer4 = slim.fully_connected(
                slim.flatten(self.conv2), 512, scope=self.scope+'_l4')
        self.flat_game_features = slim.fully_connected(
                self.layer4, k, scope=self.scope+'_l4.5')
        self.game_features = tf.reshape(
                self.flat_game_features,
                [self.batch_size, self.sequence_length, k])

    def _init_recurrent_part(self):
        self.layer3 = tf.reshape(slim.flatten(self.conv2),
                                 [self.batch_size, self.sequence_length, 4608])
        self.h_size = 4608

        self.cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        self.rnn_output, self.state_out = tf.nn.dynamic_rnn(
                self.cell,
                self.layer3,
                initial_state=self.state_in,
                dtype=tf.float32,
                scope=self.scope+'_RNN/')

        self.rnn_output = tf.reshape(self.rnn_output, [-1, self.h_size])
        self.Q = slim.fully_connected(
            self.rnn_output, self.n_actions, scope=self.scope+'_actions',
            activation_fn=None)
        self.Q = tf.reshape(self.Q,
                [self.batch_size, self.sequence_length, self.n_actions])
        self.choice = tf.argmax(self.Q, 2)

    def _define_loss(self):
        self.gamma = tf.placeholder(tf.float32, name='gamma')
        self.target_q = tf.placeholder(tf.float32, name='target_q',
                shape=[None, None, self.n_actions])
        self.rewards = tf.placeholder(tf.float32, name='rewards',
                shape=[None, None])
        y = self.rewards + self.gamma * tf.reduce_sum(
                tf.one_hot(self.choice, self.n_actions) * self.target_q, 2)
        Qas = tf.reduce_sum(tf.one_hot(self.choice, self.n_actions) * self.Q, 2)
        self.loss = tf.reduce_mean(tf.square(y-Qas))
        self.train_step = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

if __name__ == '__main__':
    fake_dataset_size = 100
    batch_size = 1
    sequence_length = 1
    im_w = 108
    im_h = 60
    k = 1
    n_actions = 3

    print('Building main DRQN')
    main = DRQN(im_h, im_w, k, n_actions, 'main')
    print('Building target DRQN')
    target = DRQN(im_h, im_w, k, n_actions, 'target')
    # TODO target = main

    # fake states
    Xtr = np.ones((fake_dataset_size, sequence_length, im_h, im_w, 3))

    # initial LSTM state
    state = (np.zeros([batch_size, main.h_size]),
             np.zeros([batch_size, main.h_size]))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Training vars:", [v.name for v in tf.trainable_variables()])

        game = vd.DoomGame()
        game.load_config("basic.cfg")
        game.init()

        actions = np.eye(3, dtype=np.uint32).tolist()

        def play_episode(epsilon):
            game.new_episode()
            dump = []
            while not game.is_episode_finished():
                state = game.get_state()
                S = state.screen_buffer
                h, w = S.shape[:2]
                S = Simg.zoom(S, [1.*im_h/h, 1.*im_w/w, 1])
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
        mem = ReplayMemory(min_size=100, max_size=1000)
        while not mem.full:
            mem.add(play_episode(epsilon=1))
            print(len(mem.episodes))

