import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv(x, shape, strides):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def linear(x, n_output_cells, batch_size):
    unstacked = tf.unstack(x, num=batch_size)
    batch = []
    W = tf.Variable(tf.truncated_normal(
        [int(tf.reshape(unstacked[0], [1,-1]).get_shape()[1]), n_output_cells], 
        stddev=0.1
    ))
    b = tf.Variable(tf.constant(0.1, shape=[n_output_cells]))
    for individual_x in tf.unstack(x, num=batch_size):
        individual_x = tf.reshape(individual_x, [1, -1])  # flatten
        batch.append(tf.matmul(individual_x, W) + b)
    return tf.pack(batch)

class DRQN():
    def __init__(self, im_h, im_w, k, batch_size, sequence_length, n_actions, scope):
        self.im_h, self.im_w, self.k = im_h, im_w, k
        self.batch_size, self.n_actions = batch_size, n_actions
        self.sequence_length = sequence_length
        self.scope = scope

        self.images = tf.placeholder(tf.float32, 
                shape=[batch_size, sequence_length, im_h, im_w, 3])
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
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(
                self.cell,
                self.layer3,
                initial_state=initial_state,
                dtype=tf.float32)

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
    drqn = DRQN(im_h, im_w, k, batch_size, sequence_length, n_actions, 'drqn')
    Xtr = np.ones((fake_dataset_size, sequence_length, im_h, im_w, 3))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(tf.trainable_variables())
        action = sess.run([drqn.actions], feed_dict={drqn.images:Xtr[0:batch_size]})

    #  x, y, output = build_graph(125, 200, 1, batch_size)
    #  l = loss(y, output)
    #  loss_summary = tf.summary.scalar('loss', l)

    #  train_step = tf.train.RMSPropOptimizer(0.001).minimize(l)

    #  with tf.Session() as sess:
        #  init = tf.global_variables_initializer()
        #  sess.run(init)

        #  train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

        #  for i in range(100000):
            #  x_batch = Xtr[0:batch_size]
            #  y_batch = ytr[0:batch_size]
            #  loss_value, _, summary = sess.run([l, train_step, loss_summary], 
                    #  feed_dict={x: x_batch, y: y_batch})

            #  if i % 100 == 0:
                #  print('Step %d : %d' % (i, loss_value))
                #  train_writer.add_summary(summary, i)

