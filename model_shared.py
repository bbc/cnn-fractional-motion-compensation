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

from utils import read_shared_data, read_shared_testdata, calculate_batch_number, \
    calculate_test_error, save_results, frac_positions
from model_base import BaseCNN
import time
import os
import tensorflow as tf
import numpy as np
import math


class SharedBaseCNN(BaseCNN):
    """
    Class for the base CNN for shared models that contains generalised parameters and functions
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.subset = tf.placeholder(tf.int32, name='subset')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

    def train(self):
        """
        Training procedure for the CNN model: read dataset, initialize graph, load model checkpoint if possible,
                                              train and validate model on different block sizes,
                                              save model if it performs better than in the previous epoch,
                                              stop training after all epochs or if model doesn't improve for x epochs
        """

        # get training and validation inputs/labels
        train_data, train_label, _, val_data, val_label, _ = read_shared_data(self.cfg.dataset_dir, self.cfg.batch_size)

        # initialize logging
        writer, merged = self.initialize_graph(self.subdirectory())

        # load model if possible
        global_step = self.load(self.subdirectory())

        # calculate number of training / validation batches for each block size per fractional position
        batch_train, batch_val = calculate_batch_number(train_data, val_data, self.cfg.batch_size, nested=True)

        start_epoch = global_step // sum(batch_train*15)
        print("Training %s network, from epoch %d" % (self.cfg.model_name.upper(), start_epoch))

        start_time = time.time()
        err_train = None
        for ep in range(start_epoch, self.cfg.epoch):
            # Run on batches of training inputs, per fractional position and block size
            for i, frac in enumerate(frac_positions()):
                for index, block in enumerate(train_data):
                    for idx in range(batch_train[index]):
                        feed_dict = self.shared_feed_dict(train_data[block][frac], train_label[block][frac], idx, i)
                        _, err_train, summary = self.sess.run([self.train_op, self.loss, merged], feed_dict=feed_dict)
                        global_step += 1
                        writer.add_summary(summary, global_step)

            # Run on batches of validation inputs, per fractional position and block size
            error_val_list = []
            for i, frac in enumerate(frac_positions()):
                for index, block in enumerate(val_data):
                    for idx in range(batch_val[index]):
                        feed_dict = self.shared_feed_dict(val_data[block][frac], val_label[block][frac], idx, i)
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
        test_data, test_label, test_sad = read_shared_testdata(self.cfg.test_dataset_dir)

        tf.global_variables_initializer().run()

        # load model
        global_step = self.load(self.subdirectory())
        if not global_step:
            raise SystemError("Failed to load a trained model!")

        print("Testing %s network" % self.cfg.model_name.upper())

        # Run test, per block size and fractional position
        error_pred, error_vvc, error_switch = ([] for _ in range(3))
        for i, frac in enumerate(frac_positions()):
            for block in test_data:
                batch_test = math.ceil(len(test_data[block][frac]) / self.cfg.batch_size)
                result = np.array([])

                for idx in range(batch_test):
                    feed_dict = self.shared_feed_dict(test_data[block][frac], test_label[block][frac], idx, i)
                    res = self.sess.run([self.pred], feed_dict=feed_dict)
                    cropped_input = feed_dict[self.inputs][:, self.half_kernel:-self.half_kernel,
                                                           self.half_kernel:-self.half_kernel, :]
                    result = np.vstack([result, res[0] + cropped_input]) if result.size else res[0] + cropped_input

                # calculate SAD NN loss and compare it to VVC loss
                nn_cost, vvc_cost, switch_cost = calculate_test_error(result,
                                                                      test_label[block][frac], test_sad[block][frac])
                error_pred.append(nn_cost)
                error_vvc.append(vvc_cost)
                error_switch.append(switch_cost)

        save_results(self.cfg.results_dir, self.cfg.model_name, self.subdirectory(),
                     error_pred, error_vvc, error_switch)

    def subdirectory(self):
        """
        Model subdirectory details
        """
        return os.path.join(self.cfg.model_name, self.cfg.dataset_dir.split("/")[-1])

    def shared_feed_dict(self, inputs, labels, i, subset):
        """
        Method that prepares a batch of inputs / labels to be fed into the shared model
        :param inputs: input data
        :param labels: label data
        :param i: index pointing to the current position within the data
        :param subset: index indicating which branch of the output layer to update
        :return a batch-sized dictionary of inputs / labels / subset / batch_size
        """
        feed_dict = self.prepare_feed_dict(inputs, labels, i)
        feed_dict.update({self.subset: subset, self.batch_size: len(feed_dict[self.inputs])})
        return feed_dict


class SharedCNN(SharedBaseCNN):
    """
    Class SharedCNN, a 3-layer model with 15 outputs,
    each branch output is updated with its corresponding data subset (per fractional position)
    """
    def __init__(self, sess, cfg):
        super().__init__(sess, cfg)

        self.weights = {
            'w1': tf.get_variable('w1', shape=[9, 9, 1, 64],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('w2', shape=[1, 1, 64, 32],
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('w3', shape=[5, 5, 32, 15],
                                  initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        # parameter half_kernel needed for residual learning
        self.calculate_half_kernel_size()

        self.pred = self.linear_model()

        self.loss = self.calculate_loss()

        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def calculate_loss(self):
        cost = self.complex_loss(self.cfg.loss, self.weights[list(self.weights.keys())[-1]].get_shape()[-1].value)

        # Update the branch of the subset
        cost = tf.slice(cost, [0, self.subset], [self.batch_size, 1])

        return tf.reduce_mean(cost, name="loss")
