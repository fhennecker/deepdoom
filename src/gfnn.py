import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.ndimage as Simg


class ReplayMem:
    def __init__(self, capacity, shapes, dtypes):
        self.capacity = capacity
        self.frames = [
            np.zeros((capacity,)+s, dtype=d)
            for s, d in zip(shapes, dtypes)
        ]
        self.size = 0
        self.next_insert = 0

    def add(self, *frame):
        i = self.next_insert
        for s, f in zip(self.frames, frame):
            s[i] = f
        self.next_insert = (i + 1) % self.capacity
        self.size = max(self.next_insert, self.size)

    def batch(self, size):
        positions = np.random.randint(0, self.size, size)
        return [f[positions] for f in self.frames]


class GFNN:
    """Game Features supervised learning network"""
    W, H = 108, 60
    IMAGE_SHAPE = (H, W, 3)

    def my(self, name):
        return self.scope + '.' + name

    def __init__(self, scope, n_features, learning_rate=0.001, batch_size=32):
        self.learning_rate = learning_rate
        self.scope = scope
        self.batch_size = batch_size
        self.n_features = n_features

        # Dropout probability
        self.dropout_p = tf.placeholder(tf.float32, name='dropout_keep')

        # Input images
        self.images = tf.placeholder(tf.float32, name='images',
                                     shape=[batch_size, self.H, self.W, 3])

        # Input features
        self.features = tf.placeholder(tf.float32, name='features',
                                       shape=[batch_size, n_features])

        # First convolution from screen buffer
        self.conv1 = slim.conv2d(
            self.images, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            scope=self.my("conv1")
        )

        # Second convolution layer
        self.conv2 = slim.conv2d(
            self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            scope=self.my("conv2")
        )

        # Fully connected flat output with dropout
        self.cnn_output = tf.nn.dropout(
            slim.fully_connected(slim.flatten(self.conv2), 512,
                                 scope=self.my('conv_flat')),
            self.dropout_p,
        )

        # Output layer
        self.prediction = slim.fully_connected(self.cnn_output, n_features,
                                               scope=self.my("output"),
                                               activation_fn=None)

        # Backprop
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(self.prediction,
        #                                                     self.features)
        # self.loss = tf.reduce_mean(tf.square(cross_ent))
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.features))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

    def train(self, sess, images, features, dropout_p=0.8):
        assert 0 <= dropout_p <= 1
        assert images.shape == (self.batch_size, self.H, self.W, 3)
        assert features.shape == (self.batch_size, self.n_features)

        train, loss = sess.run([self.train_step, self.loss], feed_dict={
            self.dropout_p: dropout_p,
            self.images: images,
            self.features: features,
        })
        return loss

    def predict(self, sess, images):
        assert images.shape == (self.batch_size, self.H, self.W, 3)

        return sess.run(self.prediction, feed_dict={
            self.dropout_p: 1,
            self.images: images,
        })

if __name__ == "__main__":
    import random
    # import matplotlib.pyplot as plt
    from basic_ennemy_pos import (
        create_game, basic_ennemy_pos_features, N_FEATURES,
        basic_ennemy_x
    )

    nn = GFNN('gfnn', N_FEATURES, batch_size=32)
    mem = ReplayMem(100000,
                    [nn.IMAGE_SHAPE, (N_FEATURES,)],
                    [np.uint8, np.float32])

    game, walls = create_game()
    screenbuf = np.zeros((nn.H, nn.W, 3), dtype=np.uint8)
    ACTION_SET = np.eye(3, dtype=np.int32).tolist()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        try:
            saver = tf.train.import_meta_graph('gfnn.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
        except:
            print("CREATE NEW MODEL")
            init = tf.global_variables_initializer()
            sess.run(init)


        i = 0
        while i < 1e6:
            # Linearly decreasing epsilon
            epsilon = 0.1
            game.new_episode()

            # Initialize new hidden state
            total_reward = 0
            while not game.is_episode_finished():
                if i > 0 and i % 1000 == 0:
                    saver.save(sess, "gfnn.ckpt")

                i += 1
                if i % 1e3 == 0:
                    # Get and resize screen buffer
                    state = game.get_state()
                    h, w, d = state.screen_buffer.shape
                    Simg.zoom(state.screen_buffer,
                              [1. * nn.H / h, 1. * nn.W / w, 1],
                              output=screenbuf, order=0)

                    features = basic_ennemy_pos_features(state)
                    mem.add(screenbuf, features)
                    action = random.choice(ACTION_SET)
                    game.make_action(action, 4)

                    preds = nn.predict(sess, np.array([screenbuf]*nn.batch_size))
                    p = preds[0]
                    d = p - features
                    print("Test: ", d)

                    # plt.imshow(screenbuf)
                    # plt.axvline(p[0]*screenbuf.shape[1], c='g')
                    # plt.axvline(features[0]*screenbuf.shape[1], c='r')
                    # plt.show()

                elif mem.size > 1000 and i % 100 == 0:
                    loss = 0
                    for j in range(10):
                        images, features = mem.batch(nn.batch_size)
                        loss += nn.train(sess, images, features)
                    print("\rLoss %5d:" % i, loss/10)
                else:
                    # Get and resize screen buffer
                    state = game.get_state()
                    h, w, d = state.screen_buffer.shape
                    Simg.zoom(state.screen_buffer,
                              [1. * nn.H / h, 1. * nn.W / w, 1],
                              output=screenbuf, order=0)

                    features = basic_ennemy_pos_features(state)
                    mem.add(screenbuf, features)
                    action = random.choice(ACTION_SET)
                    game.make_action(action, 4)
