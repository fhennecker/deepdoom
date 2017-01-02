import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DRQN():
    def __init__(self, im_h, im_w, k, n_actions, scope, learning_rate, test=False):
        self.learning_rate = learning_rate
        self.im_h, self.im_w, self.k = im_h, im_w, k
        self.scope, self.n_actions = scope, n_actions
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')

        # Dropout probability
        self.dropout_p = tf.placeholder(tf.float32, name='dropout_p')

        self.images = tf.placeholder(tf.float32, name='images',
                                     shape=[None, None, im_h, im_w, 3])
        # we'll merge all sequences in one single batch for treatment
        # but all outputs will be reshaped to [batch_size, length, ...]
        self.all_images = tf.reshape(self.images,
                                     [self.batch_size*self.sequence_length,
                                      im_h, im_w, 3])

        self._init_conv_layers()
        self._init_game_features_output()
        self._init_recurrent_part()
        if not test:
            self._define_loss()

    def _init_conv_layers(self):
        # First convolution from screen buffer
        self.conv1 = slim.conv2d(
            self.all_images, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None, scope=self.scope+'_conv1'
        )

        # Second convolution layer
        self.conv2 = slim.conv2d(
            self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None, scope=self.scope+'_conv2'
        )

    def _init_game_features_output(self):
        self.layer4 = tf.nn.dropout(
            slim.fully_connected(slim.flatten(self.conv2), 512,
                                 scope=self.scope+'_l4'),
            self.dropout_p,
        )
        self.flat_game_features = slim.fully_connected(self.layer4, 2*self.k,
                                                       scope=self.scope+'_l4.5')
        reshaped = tf.reshape(self.flat_game_features,
                              [self.batch_size, self.sequence_length,
                               self.k, 2])

        # Output layer
        self.game_features = tf.nn.softmax(reshaped)
        # Observed game features
        self.game_features_in = tf.placeholder(tf.float32,
                                               name='game_features_in',
                                               shape=[None, None, self.k, 2])

        # Difference between observed and predicted game features
        delta = self.game_features - self.game_features_in
        # delta = tf.Print(delta, [delta], summarize=10, name="dFeatures")

        # Optimize on RMS of this difference
        self.features_loss = tf.reduce_mean(tf.square(delta))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.features_train_step = optimizer.minimize(self.features_loss)

    def _init_recurrent_part(self):
        # Flat fully connected layer (Layer3' in the paper)
        self.h_size = 300
        self.reset_hidden_state()
        self.layer3 = tf.nn.dropout(
            tf.reshape(slim.flatten(self.conv2),
                       [self.batch_size, self.sequence_length, self.h_size]),
            self.dropout_p,
        )

        # LSTM cell
        self.cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)

        # Recurrence
        rnn_output, self.state_out = tf.nn.dynamic_rnn(
                self.cell,
                self.layer3,
                initial_state=self.state_in,
                dtype=tf.float32,
                scope=self.scope+'_RNN/')

        self.rnn_output = tf.reshape(rnn_output, [-1, self.h_size])

        # Q-estimator for actions
        Q = slim.fully_connected(
            self.rnn_output, self.n_actions, scope=self.scope+'_actions',
            activation_fn=None)
        self.Q = tf.reshape(Q, [self.batch_size, self.sequence_length,
                                self.n_actions])
        self.choice = tf.argmax(self.Q, 2)
        self.max_Q = tf.reduce_max(self.Q, 2)

    def _define_loss(self):
        self.gamma = tf.placeholder(tf.float32, name='gamma')
        self.target_q = tf.placeholder(tf.float32, name='target_q',
                                       shape=[None, None])
        self.rewards = tf.placeholder(tf.float32, name='rewards',
                                      shape=[None, None])
        self.actions = tf.placeholder(tf.float32, name='actions',
                                      shape=[None, None, self.n_actions])
        y = self.rewards + self.gamma * self.target_q
        Qas = tf.reduce_sum(tf.one_hot(tf.argmax(self.actions, 2), 
                                       self.n_actions) * self.Q, 2)

        self.ignore_up_to = tf.placeholder(tf.int32, name='ignore_up_to')
        y = tf.slice(y, [0, self.ignore_up_to], [-1, -1])
        Qas = tf.slice(Qas, [0, self.ignore_up_to], [-1, -1])

        self.loss = tf.reduce_mean(tf.square(y-Qas))
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _game_features_learning(self, func, screens, features):
        assert screens.shape[:2] == features.shape[:2]
        batch_size, sequence_length = features.shape[:2]
        F = np.zeros((batch_size, sequence_length, self.k, 2))  # NOQA
        F[:, :, :, 0] = features
        F[:, :, :, 1] = ~features
        return func(feed_dict={
            self.batch_size: batch_size,
            self.sequence_length: sequence_length,
            self.images: screens,
            self.game_features_in: F,
        })

    def learn_game_features(self, screens, features):
        return self._game_features_learning(self.features_train_step.run,
                                            screens, features)

    def current_game_features_loss(self, screens, features):
        return self._game_features_learning(self.features_loss.eval,
                                            screens, features)

    def reset_hidden_state(self, batch_size=1):
        shape = batch_size, self.h_size
        self.rnn_state = np.zeros(shape), np.zeros(shape)

    def feed_lstm(self, sess, screens, actions, rewards):
        assert screens.shape[:2] == actions.shape[:2]
        assert screens.shape[:2] == rewards.shape[:2]
        batch_size, sequence_length = screens.shape[:2]

        actions, state = sess.run([self.choice, self.state_out], feed_dict={
            self.batch_size: batch_size,
            self.sequence_length: sequence_length,
            self.images: screens,
            self.state_in: self.rnn_state,
        })

        self.last_state = sess.run(self.state_out, feed_dict={
            self.batch_size: batch_size,
            self.sequence_length: sequence_length,
            self.images: screens,
            self.state_in: self.rnn_state,
        })

    def choose(self, sess, epsilon, screenbuf, dropout_p):
        """Choose an action based on the current screen buffer"""
        is_random = np.random.rand() <= epsilon
        to_get = [self.rnn_output]
        if not is_random:
            to_get += [self.choice]
        r = sess.run(to_get, feed_dict={
            self.batch_size: 1,
            self.sequence_length: 1,
            self.images: [[screenbuf]],
            self.dropout_p: dropout_p,
        })
        return np.random.randint(self.n_actions) if is_random else r[1][0][0]
