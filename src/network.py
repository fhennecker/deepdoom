import tensorflow as tf

def conv(x, shape, strides):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
    return tf.nn.conv2d([x], W, strides=strides, padding='SAME')

def linear(x, n_output_cells):
    x = tf.reshape(x, [1, -1])  # flatten
    W = tf.Variable(tf.truncated_normal(
        [int(x.get_shape()[1]), n_output_cells], 
        stddev=0.1
    ))
    b = tf.Variable(tf.constant(0.1, shape=[n_output_cells]))
    return tf.matmul(x, W) + b

def build_graph(h, w, k):
    x = tf.placeholder(tf.float32, shape=[h, h, 3])
    y = tf.placeholder(tf.float32, shape=[k])
    
    # shape is [kernel_h, kernel_w, n_input_channels, n_output_channels]
    conv1 = conv(x, [8, 8, 3, 32], [1, 4, 4, 1])
    conv2 = conv(conv1[0], [4, 4, 32, 64], [1, 2, 2, 1])
    layer4 = tf.tanh(linear(conv2, 512))
    game_features = linear(layer4, k)

    return game_features

if __name__ == '__main__':
    build_graph(125, 200, 3)

