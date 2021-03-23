# BSD 3-Clause License
#
# Copyright (c) 2021 BBC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors may
#  be used to endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import os
import tensorflow as tf
import time
import sys


class BaseCNN(object):
    """
    Class for the base CNN for all models that contains generalised parameters and functions
    """
    def __init__(self, sess, cfg):
        """
        Initialise the BaseCNN model
        :param sess: TensorFlow session
        :param cfg: model details taken from the config file
        """
        self.sess = sess
        self.cfg = cfg

        self.inputs = tf.placeholder(tf.float32, [None, None, None, 1], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')

        self.weights = None
        self.loss = None
        self.pred = None
        self.saver = None
        self.train_op = None

        self.minimum = sys.maxsize
        self.counter = 0

    def initialize_graph(self, graphs_subdir):
        """
        Initialize TensorBoard summaries
        :param graphs_subdir: subdirectory of graphs storage location
        :return: FileWriter and merged summaries
        """
        tf.global_variables_initializer().run()
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning rate', self.cfg.learning_rate)
        tf.summary.histogram('w1', self.weights['w1'])
        tf.summary.histogram('w2', self.weights['w2'])
        tf.summary.histogram('w3', self.weights['w3'])
        tf.summary.image('inputs', self.inputs, max_outputs=2)
        tf.summary.image('labels', self.labels, max_outputs=2)
        tf.summary.image('prediction', self.pred[:, :, :, 0:1], max_outputs=2)

        # save summaries in the graphs directory
        sub_dir = os.path.join(self.cfg.graphs_dir, graphs_subdir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        writer = tf.summary.FileWriter(sub_dir, self.sess.graph)
        merged = tf.summary.merge_all()

        return writer, merged

    def prepare_feed_dict(self, inputs, labels, i):
        batch_inputs = inputs[i * self.cfg.batch_size: (i + 1) * self.cfg.batch_size]
        batch_labels = labels[i * self.cfg.batch_size: (i + 1) * self.cfg.batch_size]
        feed_dict = {self.inputs: batch_inputs, self.labels: batch_labels}

        return feed_dict

    def save_epoch(self, current_epoch, current_step, start_time, train_error, val_error, model_subdir):
        """
        Function that runs all necessary steps to finish a epoch:
        print status, save model if loss has decreased, return counter towards early stopping
        :param current_epoch: current epoch of trained model
        :param current_step: current step of trained model
        :param start_time: timestamp when training started
        :param train_error: training error of current epoch
        :param val_error: validation error of current epoch
        :param model_subdir: model subdirectory organisation
        :return: early stopping counter
        """
        print("Epoch: [%2d], time: [%4.4f], training loss: [%.8f], validation loss: [%.8f]"
              % ((current_epoch + 1), time.time() - start_time, train_error, val_error))

        # save model if validation loss has decreased, else increase counter towards early stopping
        if self.minimum > val_error:
            self.minimum = val_error
            self.counter = 0
            self.save(current_step, model_subdir)
        else:
            self.counter += 1

        return self.counter

    def save(self, step, checkpoint_subdir):
        """
        Save model to checkpoint
        :param step: current step of training
        :param checkpoint_subdir: subdirectory of checkpoint to be saved
        """
        model_name = self.cfg.model_name.upper() + ".model"
        checkpoint_loc = os.path.join(self.cfg.checkpoint_dir, checkpoint_subdir)

        os.makedirs(checkpoint_loc, exist_ok=True)

        self.saver.save(self.sess, os.path.join(checkpoint_loc, model_name), global_step=step)

    def load(self, checkpoint_subdir):
        """
        Load model from latest checkpoint
        :param checkpoint_subdir: subdirectory of checkpoint to be loaded
        :return: the latest training step
        """
        print(" [*] Reading checkpoints...")
        checkpoint_loc = os.path.join(self.cfg.checkpoint_dir, checkpoint_subdir)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_loc)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            global_step = int(checkpoint_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_loc, checkpoint_name))
            print(" [*] Load SUCCESS")
            return global_step
        else:
            print(" [!] Load failed")
            return 0

    @staticmethod
    def conv_layer(layer_name, layer_input, layer_weights, conv_padding, conv_name):
        with tf.variable_scope(layer_name):
            conv = tf.nn.conv2d(layer_input, layer_weights, strides=[1, 1, 1, 1], padding=conv_padding, name=conv_name)
            tf.summary.image(conv_name, tf.expand_dims(conv[:, :, :, 0], 3), max_outputs=2)
        return conv

    @staticmethod
    def bias_op(layer_name, layer_input, layer_weights, bias_name):
        with tf.variable_scope(layer_name):
            bias = tf.add(layer_input, layer_weights, name=bias_name)
        return bias

    @staticmethod
    def relu_op(layer_name, layer_input, relu_name):
        with tf.variable_scope(layer_name):
            relu = tf.nn.relu(layer_input, name=relu_name)
        return relu

    def linear_model(self):
        """
        Function that builds a 3-layer linear model
        :return: output of the final convolutional layer
        """
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')
        conv2 = self.conv_layer('layer2', conv1, self.weights['w2'], 'VALID', 'conv2')
        conv3 = self.conv_layer('layer3', conv2, self.weights['w3'], 'VALID', 'conv3')

        return conv3

    def loss_functions(self, loss_choice, outputs=None):
        """
        Method that contains different options for computing losses
        :param loss_choice: Sum of Absolute Differences (SAD), Mean Squared Error (MSE) or complex
        :param outputs:  number of outputs of the last convolution layer, only used with "complex" loss choice
        :return: batch loss
        """
        if loss_choice == "SAD":
            return tf.reduce_mean(tf.math.abs(self.labels - self.pred), name="loss")
        elif loss_choice == "MSE":
            return tf.reduce_mean(tf.square(self.labels - self.pred), name="loss")
        elif loss_choice == "complex":
            tile_axis = tf.constant([1, 1, 1, outputs])
            current_inputs = tf.tile(self.inputs, tile_axis)
            current_labels = tf.tile(self.labels, tile_axis)

            current_pred = (current_inputs[:, 6:-6, 6:-6, :] + self.pred)

            return tf.reduce_mean(tf.math.abs(current_labels - current_pred), axis=(1, 2))
