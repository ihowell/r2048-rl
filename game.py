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

FALSE_COL = tf.reshape(tf.constant([False]*4), (1, 4, 1))

EMPTY_CELL = tf.constant([1.] + [0.]*16)
EMPTY_CELL = tf.reshape(EMPTY_CELL, (1, 1, 1, 17))
EMPTY_COL = tf.repeat(EMPTY_CELL, [4], axis=1)
EMPTY_ROW = tf.repeat(EMPTY_CELL, [4], axis=2)

EMPTY_BOARD = tf.repeat(EMPTY_COL, [4], axis=2)


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


def shift_right(x):
    return tf.concat([EMPTY_COL, x[:, :, :3, :]], axis=2)


def shift_left(x):
    return tf.concat([x[:, :, 1:, :], EMPTY_COL], axis=2)


def _slide_right(x):
    """
    Args:
      x (tf.Tensor): NHWC
    """
    cond = tf.argmax(x[:, :, 1:, :], axis=3) == 0
    cond = tf.concat([cond[:, :, :1], cond], axis=2)

    for _ in range(3):
        cond = tf.math.logical_or(cond, tf.concat([cond[:, :, 1:], FALSE_COL], 2))

    cond = tf.expand_dims(cond, 3)
    cond = tf.repeat(cond, [17], axis=3)

    sr = shift_right(x)

    x = tf.where(cond, sr, x)
    return x


def _slide_h(x, d):
    if d == 'l':
        cond = tf.argmax(x[:, :, :-1, :], axis=3) == 0
        cond = tf.concat([cond, cond[:, :, -1:]], axis=2)
    elif d == 'r':
        cond = tf.argmax(x[:, :, 1:, :], axis=3) == 0
        cond = tf.concat([cond[:, :, :1], cond], axis=2)

    for _ in range(3):
        if d == 'l':
            cond = tf.math.logical_or(cond, tf.concat([FALSE_COL, cond[:, :, :-1]], 2))
        elif d == 'r':
            cond = tf.math.logical_or(cond, tf.concat([cond[:, :, 1:], FALSE_COL], 2))

    cond = tf.expand_dims(cond, 3)
    cond = tf.repeat(cond, [17], axis=3)

    if d == 'l':
        shift = shift_left(x)
    if d == 'r':
        shift = shift_right(x)

    x = tf.where(cond, shift, x)
    return x


def slide_h(x, d):
    for _ in range(3):
        x = _slide_h(x, d)
    return x


def merge_right(x):
    sr = shift_right(x)
    sl = shift_left(x)

    x = tf.argmax(x, axis=3)
    sr_max = tf.argmax(sr, axis=3)
    sl_max = tf.argmax(sl, axis=3)

    inc_cond = tf.math.logical_and(x != 0, x == sr_max)
    zero_cond = tf.math.logical_and(x != 0, x == sl_max)

    all_equal = tf.reduce_all(inc_cond[:, :, 1:], axis=2)

    x = tf.where(inc_cond, x + 1, x)
    x = tf.where(zero_cond, 0, x)

    empty_col = tf.reshape(tf.constant([0] * 4, dtype=tf.int64), (1, 4, 1))
    sll = tf.concat([x[:, :, 1:], empty_col], axis=2)

    all_equal = tf.expand_dims(all_equal, 2)

    merge_2_cond = tf.concat([FALSE_COL, FALSE_COL, all_equal, FALSE_COL], axis=2)

    x = tf.where(merge_2_cond, sll, x)

    x = tf.one_hot(tf.cast(x, tf.int32), 17)

    x = _slide_right(x)
    return x


def merge_h(x, d):
    sr = shift_right(x)
    sl = shift_left(x)

    x = tf.argmax(x, axis=3)
    sr_max = tf.argmax(sr, axis=3)
    sl_max = tf.argmax(sl, axis=3)

    if d == 'l':
        inc_cond = tf.math.logical_and(x != 0, x == sl_max)
        zero_cond = tf.math.logical_and(x != 0, x == sr_max)
        all_equal = tf.reduce_all(inc_cond[:, :, :-1], axis=2)
    elif d == 'r':
        inc_cond = tf.math.logical_and(x != 0, x == sr_max)
        zero_cond = tf.math.logical_and(x != 0, x == sl_max)
        all_equal = tf.reduce_all(inc_cond[:, :, 1:], axis=2)

    x = tf.where(inc_cond, x + 1, x)
    x = tf.where(zero_cond, 0, x)
    all_equal = tf.expand_dims(all_equal, 2)

    empty_col = tf.reshape(tf.constant([0] * 4, dtype=tf.int64), (1, 4, 1))
    if d == 'l':
        shift = tf.concat([empty_col, x[:, :, :-1]], axis=2)
        merge_2_cond = tf.concat([FALSE_COL, all_equal, FALSE_COL, FALSE_COL], axis=2)
    if d == 'r':
        shift = tf.concat([x[:, :, 1:], empty_col], axis=2)
        merge_2_cond = tf.concat([FALSE_COL, FALSE_COL, all_equal, FALSE_COL], axis=2)

    x = tf.where(merge_2_cond, shift, x)
    x = tf.one_hot(tf.cast(x, tf.int32), 17)
    x = _slide_h(x, d)
    return x


if __name__ == '__main__':
    board = tf.constant([
        [4., 4., 4., 4.],
        [2., 2., 4., 4.],
        [0., 4., 0., 2.],
        [2., 2., 0., 0.],
    ])

    print(board)

    x = convert_board_to_tensor(board)
    x = tf.expand_dims(x, 0)

    # x = slide_h(x, 'r')

    # x = merge_h(x, 'r')


    x = slide_h(x, 'l')
    x = merge_h(x, 'l')
    print(convert_tensor_to_board(x[0]))
    # x = merge_right(x)

    # x = slide_left(x)

    # print(convert_tensor_to_board(x[0]))
