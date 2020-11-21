import tensorflow as tf
# import gym
# from gym import spaces
# from gym.utils import seeding
# import numpy as np
# import itertools
# import logging
# from six import StringIO
# import sys

# State is determined by a 4x4x16 tensor. A -1 in all channels of
# a cell means the cell is empty.
FALSE_COL = tf.reshape(tf.constant([False] * 4 * 16), (1, 4, 1, 16))
FALSE_ROW = tf.reshape(tf.constant([False] * 4 * 16), (1, 1, 4, 16))


def convert_board_to_tensor(x):
    """
    Args:
      a (tf.Tensor): Shape 4x4 with integer components (powers of 2).
    """
    assert x.shape[-3:] == (4, 4)
    x = tf.cast(x, tf.float32)
    x = tf.math.log(x) / tf.math.log(2.)
    x = tf.maximum(x, -1.)
    x += 1.
    x = tf.one_hot(tf.cast(x, tf.int32), 17)
    return x


def convert_tensor_to_board(x):
    assert x.shape[-3:] == (4, 4, 17)
    x = tf.argmax(x, axis=2)
    x = tf.cast(x, tf.float32)
    x -= 1.
    x = tf.exp(x * tf.math.log(2.))
    x = tf.floor(x)
    return x


def rotate_right(x):
    return tf.concat([x[:, :, 3:4, :], x[:, :, :3, :]], axis=2)


def slide_right(x):
    """
    Args:
      x (tf.Tensor): NHWC
    """
    # cond = tf.math.equal(x[:, :, 1:, :], -1)
    cond = tf.argmax(x[:, :, 1:, :], axis=3) == 0
    print('cond shape', cond.shape)
    FALSE_COL = tf.reshape(tf.constant([False] * 4), (1, 4, 1))
    print('false shape', FALSE_COL.shape)
    cond = tf.concat([FALSE_COL, cond], axis=2)
    print('cond shape', cond.shape)
    print(cond)

    cond = tf.expand_dims(cond, 3)
    cond = tf.repeat(cond, [17], axis=3)
    # print('rotate', rotate_right(x))
    # print('cond', cond)
    x = tf.where(cond, x, rotate_right(x))
    return x


if __name__ == '__main__':
    board = tf.constant([
        [0., 0., 0., 0.],
        [2., 0., 0., 0.],
        [0., 4., 0., 2.],
        [2., 2., 0., 0.],
    ])

    x = convert_board_to_tensor(board)
    x = tf.expand_dims(x, 0)

    x = slide_right(x)

    b = convert_tensor_to_board(x[0])
    print(b)
