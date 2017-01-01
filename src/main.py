#!/usr/bin/env python
from agent import tf, init_phase, training_phase, learning_phase, update_target

if __name__ == '__main__':
    from os import system
    system('git show | head -1')

    with tf.Session() as sess:
        init_phase(sess)

        training_phase(sess)
        update_target(sess)

        learning_phase(sess)
