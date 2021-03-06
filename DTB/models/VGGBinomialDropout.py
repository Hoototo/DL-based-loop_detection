#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Builds the VGG-like network with binomial dropout layers
applyed after avery layer of neurons"""

import tensorflow as tf
from . import utils
from .Classifier import Classifier


class VGGBinomialDropout(Classifier):
    """Builds the VGG-like network with binomial dropout layers
    applyed after avery layer of neurons"""

    def _inference(self,
                   images,
                   num_classes,
                   is_training_,
                   train_phase=False,
                   l2_penalty=0.0):
        """Build the VGG-like model.

        Args:
          images: Images returned from distorted_inputs() or inputs().
          num_classes: Number of classes to predict
          is_training_: enable/disable training ops at run time
          train_phase: Boolean to enable/disable train ops at build time
          l2_penalty: float value, weight decay (l2) penalty

        Returns:
          Logits.
        """

        with tf.variable_scope(self.__class__.__name__):
            with tf.variable_scope('64'):
                with tf.variable_scope('conv1'):
                    conv1 = utils.conv_layer(
                        images, [3, 3, 3, 64],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)
                    if train_phase:
                        conv1 = utils.binomial_dropout(conv1, 0.7)

                with tf.variable_scope('conv2'):
                    conv2 = utils.conv_layer(
                        conv1, [3, 3, 64, 64],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv2 = utils.binomial_dropout(conv2, 0.6)

            with tf.variable_scope('pool1'):
                pool1 = tf.nn.max_pool(
                    conv2,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

            with tf.variable_scope('128'):
                with tf.variable_scope('conv3'):
                    conv3 = utils.conv_layer(
                        pool1, [3, 3, 64, 128],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv3 = utils.binomial_dropout(conv3, 0.6)

                with tf.variable_scope('conv4'):
                    conv4 = utils.conv_layer(
                        conv3, [3, 3, 128, 128],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv4 = utils.binomial_dropout(conv4, 0.6)

            with tf.variable_scope('pool2'):
                pool2 = tf.nn.max_pool(
                    conv4,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

            with tf.variable_scope('256'):
                with tf.variable_scope('conv5'):
                    conv5 = utils.conv_layer(
                        pool2, [3, 3, 128, 256],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv5 = utils.binomial_dropout(conv5, 0.6)

                with tf.variable_scope('conv6'):
                    conv6 = utils.conv_layer(
                        conv5, [3, 3, 256, 256],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv6 = utils.binomial_dropout(conv6, 0.6)

                with tf.variable_scope('conv7'):
                    conv7 = utils.conv_layer(
                        conv6, [3, 3, 256, 256],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv7 = utils.binomial_dropout(conv7, 0.6)

            with tf.variable_scope('pool3'):
                pool3 = tf.nn.max_pool(
                    conv7,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

            with tf.variable_scope('512'):
                with tf.variable_scope('conv8'):
                    conv8 = utils.conv_layer(
                        pool3, [3, 3, 256, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv8 = utils.binomial_dropout(conv8, 0.6)

                with tf.variable_scope('conv9'):
                    conv9 = utils.conv_layer(
                        conv8, [3, 3, 512, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv9 = utils.binomial_dropout(conv9, 0.6)

                with tf.variable_scope('conv10'):
                    conv10 = utils.conv_layer(
                        conv9, [3, 3, 512, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv10 = utils.binomial_dropout(conv10, 0.6)

            with tf.variable_scope('pool4'):
                pool4 = tf.nn.max_pool(
                    conv10,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

            with tf.variable_scope('512b2'):
                with tf.variable_scope('conv11'):
                    conv11 = utils.conv_layer(
                        pool4, [3, 3, 512, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv11 = utils.binomial_dropout(conv11, 0.6)

                with tf.variable_scope('conv12'):
                    conv12 = utils.conv_layer(
                        conv11, [3, 3, 512, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv12 = utils.binomial_dropout(conv12, 0.6)

                with tf.variable_scope('conv13'):
                    conv13 = utils.conv_layer(
                        conv12, [3, 3, 512, 512],
                        1,
                        'SAME',
                        activation=tf.nn.relu,
                        wd=l2_penalty)

                    if train_phase:
                        conv13 = utils.binomial_dropout(conv13, 0.6)

            with tf.variable_scope('pool5'):
                pool5 = tf.nn.max_pool(
                    conv13,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID')

                pool5 = tf.reshape(pool5, [-1, 512])

            with tf.variable_scope('fc'):
                fc1 = utils.fc_layer(
                    pool5, [512, 512], activation=tf.nn.relu, wd=l2_penalty)

                if train_phase:
                    fc1 = utils.binomial_dropout(fc1, 0.5)

            with tf.variable_scope('softmax_linear'):
                logits = utils.fc_layer(fc1, [512, num_classes])
        return logits

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]

        Returns:
          Loss tensor of type float.
        """
        with tf.variable_scope('loss'):
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, labels, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # The total loss is defined as the cross entropy loss plus all of the weight
            # decay terms (L2 loss).
            error = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return error

    def get(self, images, num_classes, train_phase=False, l2_penalty=0.0):
        """ define the model with its inputs.
        Use this function to define the model in training and when exporting the model
        in the protobuf format.

        Args:
            images: model input
            num_classes: number of classes to predict
            train_phase: set it to True when defining the model, during train
            l2_penalty: float value, weight decay (l2) penalty

        Return:
            is_training_: enable/disable training ops at run time
            logits: the model output
        """
        is_training_ = tf.placeholder(tf.bool, shape=(), name="is_training_")
        # build a graph that computes the logits predictions from the images
        logits = self._inference(images, num_classes, is_training_, train_phase,
                                 l2_penalty)

        return is_training_, logits
