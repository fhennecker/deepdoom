import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class DRQN():
    def __init__(self, im_h, im_w, k, n_actions, scope, learning_rate, 
            test=False, use_game_features=False, learn_q=True, recurrent=True,
            softmax_features=False):
        self.learning_rate = learning_rate
        self.im_h, self.im_w, self.k = im_h, im_w, k
        self.scope, self.n_actions = scope, n_actions
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')
        self.use_game_features = use_game_features
        self.learn_q = learn_q
        self.recurrent = recurrent
        self.softmax_features = softmax_features

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
        if recurrent:
            self._init_recurrent_part()
        else:
            self._init_dqn_output()
        if not test:
            self._define_loss()

    def _init_conv_layers(self):
        # First convolution from screen buffer
        self.conv1 = slim.conv2d(
            self.all_images, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            scope=self.scope+'_conv1'
        )

        # Second convolution layer
        self.conv2 = slim.conv2d(
            self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            scope=self.scope+'_conv2'
        )

    def _init_game_features_output(self):
        self.layer4 = tf.nn.dropout(
            slim.fully_connected(slim.flatten(self.conv2), 512,
                                 scope=self.scope+'_l4'),
            self.dropout_p,
        )

        # Observed game features
        self.game_features_in = tf.placeholder(tf.float32,
                                               name='game_features_in',
                                               shape=[None, None, self.k])

        if self.softmax_features:
            self.flat_game_features = slim.fully_connected(self.layer4, self.k,
                                                           scope=self.scope+'_l4.5',
                                                           activation_fn=None)

            # Output layer
            self.game_features = tf.reshape(self.flat_game_features,
                                            shape=[self.batch_size,
                                                   self.sequence_length,
                                                   self.k])
            cross = tf.nn.sigmoid_cross_entropy_with_logits(self.game_features,
                                                            self.game_features_in)
            self.features_loss = tf.reduce_mean(cross)
        else:
            self.flat_game_features = slim.fully_connected(self.layer4, self.k,
                                                           scope=self.scope+'_l4.5',
                                                           activation_fn=None)

            # Output layer
            self.game_features = tf.reshape(self.flat_game_features,
                                            shape=[self.batch_size, self.sequence_length, self.k])

            # Difference between observed and predicted game features
            delta = self.game_features - self.game_features_in
            # delta = tf.Print(delta, [delta], summarize=10, name="dFeatures")

            # Optimize on RMS of this difference
            self.features_loss = tf.reduce_mean(tf.square(delta))

    def _init_dqn_output(self):
        self.layer3 = tf.nn.dropout(
            tf.reshape(slim.flatten(self.conv2),
                       [self.batch_size, self.sequence_length, 4608]),
            self.dropout_p,
        )
        self.layer3_5 = slim.fully_connected(self.layer3, 512,
                scope=self.scope+"_layer3_5")
        Q = slim.fully_connected(
            self.layer3_5, self.n_actions, scope=self.scope+'_actions',
            activation_fn=None)
        self.Q = tf.reshape(Q, [self.batch_size, self.sequence_length,
                                self.n_actions])
        self.choice = tf.argmax(self.Q, 2)
        self.max_Q = tf.reduce_max(self.Q, 2)

    def _init_recurrent_part(self):
        # Flat fully connected layer (Layer3' in the paper)
        self.h_size = 300
        self.layer3 = tf.nn.dropout(
            tf.reshape(slim.flatten(self.conv2),
                       [self.batch_size, self.sequence_length, 4608]),
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
        self.q_loss = tf.reduce_mean(tf.square(y-Qas))

        if self.use_game_features:
            if self.learn_q:
                print("Learn Q and Game Features")
                self.loss = self.q_loss + self.features_loss
            else:
                print("Learn Game Features only")
                self.loss = self.features_loss
        elif self.learn_q:
            print("Learn Q only")
            self.loss = self.q_loss
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def feed_lstm(self, sess, screens, actions, rewards):
        assert screens.shape[:2] == actions.shape[:2]
        assert screens.shape[:2] == rewards.shape[:2]
        batch_size, sequence_length = screens.shape[:2]

        actions, state = sess.run([self.choice, self.state_out], feed_dict={
            self.batch_size: batch_size,
            self.sequence_length: sequence_length,
            self.images: screens,
            self.state_in: self.rnn_state,
            self.dropout_p: 0.75,
        })

        self.last_state = sess.run(self.state_out, feed_dict={
            self.batch_size: batch_size,
            self.sequence_length: sequence_length,
            self.images: screens,
            self.state_in: self.rnn_state,
            self.dropout_p: 0.75,
        })

    def choose(self, sess, epsilon, screenbuf, dropout_p, state_in):
        """Choose an action based on the current screen buffer"""
        is_random = np.random.rand() <= epsilon
        to_get = [self.Q] if not self.recurrent else [self.state_out]
        if not is_random:
            to_get += [self.choice]
        feed_dict={
            self.batch_size: 1,
            self.sequence_length: 1,
            self.images: [[screenbuf]],
            self.dropout_p: dropout_p,
        }
        if self.recurrent:
            feed_dict[self.state_in] = state_in
        r = sess.run(to_get, feed_dict)
        res = (np.random.randint(self.n_actions), r[0])
        if not is_random:
            res = (r[1][0][0], r[0])
        return res
