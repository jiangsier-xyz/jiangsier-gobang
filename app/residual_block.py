#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import *


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self._filter_num = filter_num
        self._stride = stride
        self._conv1 = None
        self._bn1 = None
        self._conv2 = None
        self._bn2 = None
        self._down_sample = None

    def build(self, input_shape):
        self._conv1 = make_conv2d_layer(filters=self._filter_num,
                                        kernel_size=(3, 3),
                                        strides=self._stride,
                                        padding='same')
        self._bn1 = layers.BatchNormalization()
        self._conv2 = make_conv2d_layer(filters=self._filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same')
        self._bn2 = layers.BatchNormalization()
        if self._stride != 1:
            self._down_sample = Sequential()
            self._down_sample.add(make_conv2d_layer(filters=self._filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=self._stride))
            self._down_sample.add(layers.BatchNormalization())
        else:
            self._down_sample = lambda x: x

        super(BasicBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        residual = self._down_sample(inputs)

        x = self._conv1(inputs)
        x = self._bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self._conv2(x)
        x = self._bn2(x, training=training)

        output = tf.nn.relu(layers.add([residual, x]))

        return output

    def get_config(self):
        config = super(BasicBlock, self).get_config()
        config.update({'filter_num': self._filter_num, 'stride': self._stride})
        return config


class BottleNeck(layers.Layer):
    def __init__(self, filter_num, stride=1, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self._filter_num = filter_num
        self._stride = stride
        self._conv1 = None
        self._bn1 = None
        self._conv2 = None
        self._bn2 = None
        self._conv3 = None
        self._bn3 = None
        self._down_sample = None

    def build(self, input_shape):
        self._conv1 = make_conv2d_layer(filters=self._filter_num,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding='same')
        self._bn1 = layers.BatchNormalization()
        self._conv2 = make_conv2d_layer(filters=self._filter_num,
                                        kernel_size=(3, 3),
                                        strides=self._stride,
                                        padding='same')
        self._bn2 = layers.BatchNormalization()
        self._conv3 = make_conv2d_layer(filters=self._filter_num * 4,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding='same')
        self._bn3 = layers.BatchNormalization()

        self._down_sample = Sequential()
        self._down_sample.add(make_conv2d_layer(filters=self._filter_num * 4,
                                                kernel_size=(1, 1),
                                                strides=self._stride))
        self._down_sample.add(layers.BatchNormalization())

        super(BottleNeck, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        residual = self._down_sample(inputs)

        x = self._conv1(inputs)
        x = self._bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self._conv2(x)
        x = self._bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self._conv3(x)
        x = self._bn3(x, training=training)

        output = tf.nn.relu(layers.add([residual, x]))

        return output

    def get_config(self):
        config = super(BottleNeck, self).get_config()
        config.update({'filter_num': self._filter_num, 'stride': self._stride})
        return config


class FcBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(FcBlock, self).__init__(**kwargs)
        self._dense = None
        self._bn = None

    def build(self, input_shape):
        self._dense = layers.Dense(units=input_shape[-1])
        self._bn = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self._dense(inputs)
        x = self._bn(x, training=training)
        output = tf.nn.relu(layers.add([inputs, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_layer = Sequential()
    res_layer.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_layer.add(BasicBlock(filter_num, stride=1))

    return res_layer


def make_bottle_neck_layer(filter_num, blocks, stride=1):
    res_layer = Sequential()
    res_layer.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_layer.add(BottleNeck(filter_num, stride=1))

    return res_layer


def make_fc_block_layer(blocks):
    res_layer = Sequential()

    for _ in range(blocks):
        res_layer.add(FcBlock())

    return res_layer


def make_conv2d_layer(filters, kernel_size, strides=(1, 1), padding='same', activation=None):
    return layers.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=activation,
                         data_format='channels_last')
