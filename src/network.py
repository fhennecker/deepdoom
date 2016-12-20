import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class DRQN():
    def __init__(self, im_h, im_w, k, batch_size, sequence_length, n_actions, scope):
        self.im_h, self.im_w, self.k = im_h, im_w, k
        self.batch_size, self.n_actions = batch_size, n_actions
        self.sequence_length = sequence_length
        self.scope = scope

        self.images = tf.placeholder(tf.float32, 
                shape=[batch_size, sequence_length, im_h, im_w, 3])
        # we'll merge all sequences in one single batch for treatment
        # but all outputs will be reshaped to [batch_size, length, ...]
        self.all_images = tf.reshape(self.images, 
                [batch_size*sequence_length, im_h, im_w, 3])

        self._init_conv_layers()
        self._init_game_features_output()
        self._init_recurrent_part()
        
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
                                 [self.batch_size, self.sequence_length, -1])
        self.h_size = int(self.layer3.get_shape()[2])

        self.cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
        self.state_in = self.cell.zero_state(batch_size, tf.float32)
        self.rnn_output, self.state_out = tf.nn.dynamic_rnn(
                self.cell,
                self.layer3,
                initial_state=self.state_in,
                dtype=tf.float32,
                scope=self.scope+'_RNN/')

        self.rnn_output = tf.reshape(self.rnn_output, [-1, self.h_size])
        self.actions = slim.fully_connected(
            self.rnn_output, self.n_actions, scope=self.scope+'_actions',
            activation_fn=None)
        self.actions = tf.reshape(self.actions,
                [self.batch_size, self.sequence_length, self.n_actions])

    #  def loss(targets, predictions):
        #  return tf.reduce_mean(tf.reduce_mean(tf.square(targets-predictions), 1))

if __name__ == '__main__':
    fake_dataset_size = 100
    batch_size = 10
    sequence_length = 8
    im_w = 108
    im_h = 60
    k = 1
    n_actions = 3

    main = DRQN(im_h, im_w, k, batch_size, sequence_length, n_actions, 'main')
    target = DRQN(im_h, im_w, k, batch_size, sequence_length, n_actions, 'target')

    # fake replay memory
    Xtr = np.ones((fake_dataset_size, sequence_length, im_h, im_w, 3))
    
    # initial LSTM state
    state = (np.zeros([batch_size, main.h_size]), 
             np.zeros([batch_size, main.h_size]))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(list(map(lambda v:v.name, tf.trainable_variables())))

        for episode in range(2):
            # TODO reset env
            max_episode_length = 2
            t = 0

            while t < max_episode_length: # TODO or other reasons to stop episode
                t += 1

                # # PLAYING BLOCK : 
                # if epsilon-greedy:
                #     run main network and save hidden state
                #     select action at random 
                # else:
                #     run main net, save hidden state and chosen action
                # state, reward = env(action)

                # # TRAINING BLOCK : 
                # if finished_pretraining (only random actions)
                #     reduce epsilon
                #     
                #     if time to update main network (every 5t or so):
                #         reset hidden state
                #         sample batch from replay memory
                #         predict Qm from main and Qt from target for this batch
                #         run backprop minimizing Qm-Qt
                # 
                #     if time to update target network (every 5000t or so):
                #         update_target_network() (save weights from main to target)
                #

            # get actions and hidden_state from network (no backprop):
            actions, state = sess.run(
                    [main.actions, main.state_out], 
                    feed_dict={
                        main.images : Xtr[0:batch_size],
                        main.state_in : state,
                    }
            )

            # only get hidden state from network (stil no backprop):
            state = sess.run(
                    [main.state_out], 
                    feed_dict={
                        main.images : Xtr[0:batch_size],
                        main.state_in : state,
                    }
            )
            # basic structure of a sess.run : 
            # output1, output2, ... = sess.run(
            #     [outputnode1, outputnode2, ...],
            #     feed_dict={
            #         inputnodeA : somevalue,
            #         inputnodeB : othervalue,
            #     }
            # )

