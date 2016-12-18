import tensorflow as tf
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

def build_graph(h, w, k, batch_size):
    x = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    y = tf.placeholder(tf.float32, shape=[None, k])
    
    # shape is [kernel_h, kernel_w, n_input_channels, n_output_channels]
    conv1 = conv(x, [8, 8, 3, 32], [1, 4, 4, 1])
    conv2 = conv(conv1, [4, 4, 32, 64], [1, 2, 2, 1])
    layer4 = tf.tanh(linear(conv2, 512, batch_size))
    game_features = linear(layer4, k, batch_size)

    return x, y, game_features

def loss(targets, predictions):
    return tf.reduce_mean(tf.reduce_mean(tf.square(targets-predictions), 1))

if __name__ == '__main__':
    Xtr = np.ones((100, 125, 200, 3))
    ytr = np.ones((100, 1))
    batch_size = 3

    x, y, output = build_graph(125, 200, 1, batch_size)
    l = loss(y, output)
    loss_summary = tf.summary.scalar('loss', l)

    train_step = tf.train.RMSPropOptimizer(0.001).minimize(l)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)

        for i in range(100000):
            x_batch = Xtr[0:batch_size]
            y_batch = ytr[0:batch_size]
            loss_value, _, summary = sess.run([l, train_step, loss_summary], 
                    feed_dict={x: x_batch, y: y_batch})

            if i % 100 == 0:
                print('Step %d : %d' % (i, loss_value))
                train_writer.add_summary(summary, i)

