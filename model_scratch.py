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

from utils import read_data, read_testdata, calculate_batch_number, calculate_test_error, save_results
from model_base import BaseCNN
import time
import os
import tensorflow as tf
import numpy as np
import math
from abc import ABCMeta, abstractmethod


class ScratchBaseCNN(BaseCNN, metaclass=ABCMeta):
    """
    Class for the base CNN for scratch models that contains generalised parameters and functions
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

    @abstractmethod
    def model(self):
        pass

    def train(self):
        """
        Training procedure for the CNN model: read dataset, initialize graph, load model checkpoint if possible,
                                              train and validate model on different block sizes,
                                              save model if it performs better than in the previous epoch,
                                              stop training after all epochs or if model doesn't improve for x epochs
        """

        # get training and validation inputs/labels
        train_data, train_label, val_data, val_label = read_data(self.cfg.dataset_dir, self.cfg.batch_size,
                                                                 self.cfg.qp, self.cfg.fractional_pixel,
                                                                 self.half_kernel, self.cfg.model_name)

        # initialize logging
        writer, merged = self.initialize_graph(self.subdirectory())

        # load model if possible
        global_step = self.load(self.subdirectory())

        # calculate number of training / validation batches
        batch_train, batch_val = calculate_batch_number(train_data, val_data, self.cfg.batch_size)

        start_epoch = global_step // sum(batch_train)
        print("Training %s network (%s), QP=%d, from epoch %d" % (self.cfg.model_name.upper(),
                                                                  self.cfg.fractional_pixel, self.cfg.qp, start_epoch))

        start_time = time.time()
        err_train = None
        for ep in range(start_epoch, self.cfg.epoch):
            # Run on batches of training inputs, per block size
            for index, block in enumerate(train_data):
                for idx in range(batch_train[index]):
                    feed_dict = self.prepare_feed_dict(train_data[block], train_label[block], idx)
                    _, err_train, summary = self.sess.run([self.train_op, self.loss, merged], feed_dict=feed_dict)
                    global_step += 1
                    writer.add_summary(summary, global_step)

            # Run on batches of validation inputs, per block size
            error_val_list = []
            for index, block in enumerate(val_data):
                for idx in range(batch_val[index]):
                    feed_dict = self.prepare_feed_dict(val_data[block], val_label[block], idx)
                    err_valid = self.sess.run([self.loss], feed_dict=feed_dict)
                    error_val_list.append(err_valid[0])

            err_val = sum(error_val_list) / len(error_val_list)

            # save model if better than previously, check early stopping condition
            counter = self.save_epoch(ep, global_step, start_time, err_train, err_val, self.subdirectory())
            if counter == self.cfg.early_stopping - 1:
                break

    def test(self):
        """
        Testing procedure for the CNN model: read dataset, initialize variables, load model checkpoint,
                                              test model on different block sizes,
                                              calculate SAD loss and compare to VVC,
                                              save results to specified directory
        """
        test_data, test_label, test_sad = read_testdata(self.cfg.test_dataset_dir, self.cfg.fractional_pixel,
                                                        self.half_kernel, self.cfg.model_name)

        tf.global_variables_initializer().run()

        # load model
        global_step = self.load(self.subdirectory())
        if not global_step:
            raise SystemError("Failed to load a trained model!")

        print("Testing %s network (%s), QP=%d" % (self.cfg.model_name.upper(), self.cfg.fractional_pixel, self.cfg.qp))

        # Run test, per block size
        error_pred, error_vvc, error_switch = ([] for _ in range(3))
        for block in test_data:
            batch_test = math.ceil(len(test_data[block]) / self.cfg.batch_size)
            result = np.array([])

            for idx in range(batch_test):
                feed_dict = self.prepare_feed_dict(test_data[block], test_label[block], idx)
                res = self.sess.run([self.pred], feed_dict=feed_dict)
                result = np.vstack([result, res[0]]) if result.size else res[0]

            # calculate NN loss and compare it to VVC loss
            nn_cost, vvc_cost, switch_cost = calculate_test_error(result, test_label[block], test_sad[block])
            error_pred.append(nn_cost)
            error_vvc.append(vvc_cost)
            error_switch.append(switch_cost)

        save_results(self.cfg.results_dir, self.cfg.model_name, self.subdirectory(),
                     error_pred, error_vvc, error_switch)

    def subdirectory(self):
        """
        Model subdirectory details
        """
        return os.path.join(self.cfg.model_name, self.cfg.dataset_dir.split("/")[1],
                            f"{self.cfg.fractional_pixel}-{self.cfg.qp}")


class ScratchCNN(ScratchBaseCNN):
    """
    Class ScratchCNN, a 3-layer model with no activation functions nor biases,
    uses valid padding, residual learning and SAD loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv3 = self.linear_model()
        return self.inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] + conv3


class ScratchActCNN(ScratchBaseCNN):
    """
    Class ScratchActCNN, a 3-layer model with only activation functions,
    uses valid padding, residual learning and SAD loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')
        relu1 = self.relu_op('layer1', conv1, 'relu1')
        conv2 = self.conv_layer('layer2', relu1, self.weights['w2'], 'VALID', 'conv2')
        relu2 = self.relu_op('layer2', conv2, 'relu2')
        conv3 = self.conv_layer('layer3', relu2, self.weights['w3'], 'VALID', 'conv3')

        return self.inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] + conv3


class ScratchBiasCNN(ScratchBaseCNN):
    """
    Class ScratchBiasCNN, a 3-layer model with only biases,
    uses valid padding, residual learning and SAD loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.biases = {
            'b1': tf.get_variable('b1', initializer=tf.zeros([64])),
            'b2': tf.get_variable('b2', initializer=tf.zeros([32])),
            'b3': tf.get_variable('b3', initializer=tf.zeros([1]))
        }

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')
        bias1 = self.bias_op('layer1', conv1, self.biases['b1'], 'bias1')
        conv2 = self.conv_layer('layer2', bias1, self.weights['w2'], 'VALID', 'conv2')
        bias2 = self.bias_op('layer2', conv2, self.biases['b2'], 'bias2')
        conv3 = self.conv_layer('layer3', bias2, self.weights['w3'], 'VALID', 'conv3')
        bias3 = self.bias_op('layer3', conv3, self.biases['b3'], 'bias3')

        return self.inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] + bias3


class ScratchAllCNN(ScratchBaseCNN):
    """
    Class ScratchAllCNN, a 3-layer model with activation functions and biases,
    uses valid padding, residual learning and SAD loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.biases = {
            'b1': tf.get_variable('b1', initializer=tf.zeros([64])),
            'b2': tf.get_variable('b2', initializer=tf.zeros([32])),
            'b3': tf.get_variable('b3', initializer=tf.zeros([1]))
        }

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')
        bias1 = self.bias_op('layer1', conv1, self.biases['b1'], 'bias1')
        relu1 = self.relu_op('layer1', bias1, 'relu1')
        conv2 = self.conv_layer('layer2', relu1, self.weights['w2'], 'VALID', 'conv2')
        bias2 = self.bias_op('layer2', conv2, self.biases['b2'], 'bias2')
        relu2 = self.relu_op('layer2', bias2, 'relu2')
        conv3 = self.conv_layer('layer3', relu2, self.weights['w3'], 'VALID', 'conv3')
        bias3 = self.bias_op('layer3', conv3, self.biases['b3'], 'bias3')

        return self.inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] + bias3


class ScratchOneCNN(ScratchBaseCNN):
    """
    Class ScratchOneCNN, a 1-layer model with no activation functions nor biases,
    uses valid padding, residual learning and SAD loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[13, 13, 1, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'VALID', 'conv1')

        return self.inputs[:, self.half_kernel:-self.half_kernel, self.half_kernel:-self.half_kernel, :] + conv1


class SRCNN(ScratchBaseCNN):
    """
    Class SRCNN, a 3-layer model with activation functions and biases,
    uses same padding, residual learning and MSE loss
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 1],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.biases = {
            'b1': tf.get_variable('b1', initializer=tf.zeros([64])),
            'b2': tf.get_variable('b2', initializer=tf.zeros([32])),
            'b3': tf.get_variable('b3', initializer=tf.zeros([1]))
        }

        self.pred = self.model()

        self.loss = self.loss_functions(self.cfg.loss, self.labels, self.pred)

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = self.conv_layer('layer1', self.inputs, self.weights['w1'], 'SAME', 'conv1')
        bias1 = self.bias_op('layer1', conv1, self.biases['b1'], 'bias1')
        relu1 = self.relu_op('layer1', bias1, 'relu1')
        conv2 = self.conv_layer('layer2', relu1, self.weights['w2'], 'SAME', 'conv2')
        bias2 = self.bias_op('layer2', conv2, self.biases['b2'], 'bias2')
        relu2 = self.relu_op('layer2', bias2, 'relu2')
        conv3 = self.conv_layer('layer3', relu2, self.weights['w3'], 'SAME', 'conv3')
        bias3 = self.bias_op('layer3', conv3, self.biases['b3'], 'bias3')

        return self.inputs + bias3
