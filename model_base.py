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

        self.half_kernel = 0

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
        for w in self.weights:
            tf.summary.histogram(w, self.weights[w])
        tf.summary.image('inputs', self.inputs, max_outputs=2)
        tf.summary.image('labels', self.labels, max_outputs=2)
        tf.summary.image('prediction', self.pred[:, :, :, 0:1], max_outputs=2)

        # save summaries in the graphs directory
        sub_dir = os.path.join(self.cfg.graphs_dir, graphs_subdir)
        os.makedirs(sub_dir, exist_ok=True)

        writer = tf.summary.FileWriter(sub_dir, self.sess.graph)
        merged = tf.summary.merge_all()

        return writer, merged

    def prepare_feed_dict(self, inputs, labels, i):
        """
        Method that prepares a batch of inputs / labels to be fed into the model
        :param inputs: input data
        :param labels: label data
        :param i: index pointing to the current position within the data
        :return a batch-sized dictionary of inputs / labels
        """
        batch_inputs = inputs[i * self.cfg.batch_size: (i + 1) * self.cfg.batch_size]
        batch_labels = labels[i * self.cfg.batch_size: (i + 1) * self.cfg.batch_size]
        feed_dict = {self.inputs: batch_inputs, self.labels: batch_labels}

        return feed_dict

    def save_epoch(self, current_epoch, current_step, start_time, train_error, val_error, model_subdir):
        """
        Run all necessary steps to finish a epoch:
        print status, save model if loss has decreased, return counter towards early stopping
        :param current_epoch: current epoch of trained model
        :param current_step: current step of trained model
        :param start_time: timestamp when training started
        :param train_error: training error of current epoch
        :param val_error: validation error of current epoch
        :param model_subdir: model subdirectory organisation
        :return: early stopping counter
        """
        print(f"Epoch: [{current_epoch + 1}], time: [{time.time() - start_time:.4f}], "
              f"training loss: [{train_error:.8f}], validation loss: [{val_error:.8f}]")

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
        model_name = f"{self.cfg.model_name.upper()}.model"
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

    def calculate_half_kernel_size(self):
        """
        Calculate half kernel size for convolutional neural networks
        Set half kernel parameter, as it's needed for slicing the input in residual learning networks
        """
        for w in self.weights:
            self.half_kernel += self.weights[w].get_shape()[0].value//2

    @staticmethod
    def conv_layer(layer_name, layer_input, layer_weights, conv_padding, conv_name):
        """
        Create a convolutional operation for a named layer in a NN model
        :param layer_name: name of current neural network layer
        :param layer_input: input to the convolutional operation
        :param layer_weights: trainable convolutional kernels of the convolutional operation
        :param conv_padding: padding of the convolutional operation (SAME or VALID)
        :param conv_name: name of the convolutional operation
        :return output of the convolution
        """
        with tf.variable_scope(layer_name):
            conv = tf.nn.conv2d(layer_input, layer_weights, strides=[1, 1, 1, 1], padding=conv_padding, name=conv_name)
            tf.summary.image(conv_name, tf.expand_dims(conv[:, :, :, 0], 3), max_outputs=2)
        return conv

    @staticmethod
    def bias_op(layer_name, layer_input, layer_weights, bias_name):
        """
        Add a bias variable to the specified input
        :param layer_name: name of current neural network layer
        :param layer_input: input to the addition operation
        :param layer_weights: trainable bias coefficients
        :param bias_name: name of the addition operation
        :return output of the bias addition operation
        """
        with tf.variable_scope(layer_name):
            bias = tf.add(layer_input, layer_weights, name=bias_name)
        return bias

    @staticmethod
    def relu_op(layer_name, layer_input, relu_name):
        """
        Apply a Rectified Linear Unit (ReLU) operation on the specified input
        :param layer_name: name of current neural network layer
        :param layer_input: input to the ReLU operation
        :param relu_name: name of the ReLU operation
        :return output of the ReLU operation
        """
        with tf.variable_scope(layer_name):
            relu = tf.nn.relu(layer_input, name=relu_name)
        return relu

    def linear_model(self):
        """
        Build a 3-layer linear model
        :return: output of the final convolutional layer
        """
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')
        conv2 = self.conv_layer('layer2', conv1, self.weights['w2'], 'VALID', 'conv2')
        conv3 = self.conv_layer('layer3', conv2, self.weights['w3'], 'VALID', 'conv3')

        return conv3

    @staticmethod
    def loss_functions(loss_choice, labels, prediction, axis=(0, 1, 2, 3)):
        """
        Method that contains different options for computing losses
        :param loss_choice: Sum of Absolute Differences (SAD) or Mean Squared Error (MSE)
        :param labels: batch of ground truth labels
        :param prediction: batch of NN prediction outputs
        :param axis: axis along which the mean operation will be performed
        :return: batch loss
        """
        if loss_choice == "SAD":
            return tf.reduce_mean(tf.math.abs(labels - prediction), axis=axis, name="loss")
        elif loss_choice == "MSE":
            return tf.reduce_mean(tf.square(labels - prediction), axis=axis, name="loss")
        else:
            print("Invalid loss function selected!")

    def complex_loss(self, loss_choice, outputs):
        """
        :param loss_choice: choice of loss for underlying loss_functions method
        :param outputs: number of outputs of the last convolution layer
        """
        tile_axis = tf.constant([1, 1, 1, outputs])
        current_inputs = tf.tile(self.inputs, tile_axis)
        current_labels = tf.tile(self.labels, tile_axis)

        current_pred = current_inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] \
            + self.pred

        return self.loss_functions(loss_choice, current_labels, current_pred, (1, 2))
