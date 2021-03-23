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

import numpy as np
import os
import random
import sys
import math
from itertools import product
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from silx.io.dictdump import h5todict


class VideoYUV:
    """
    Class for reading YUV files up to 16-bit depth
    """
    def __init__(self, filename, width, height, bit_depth):
        """
        Initialize class
        :param filename: YUV file path
        :param width: width of YUV file
        :param height: height of YUV file
        :param bit_depth: bit depth of YUV file
        """
        self.width = width
        self.height = height
        self.f = open(filename, 'rb')
        self.bit_depth = bit_depth

    def read(self, output_depth):
        """
        Extract luma values from bitstream
        :param output_depth: bit depth of output frame
        :return: True/False if a frame was successfully read, array of luma values at specified bit depth
        """
        if self.bit_depth <= 8:
            raw = self.f.read(self.width*self.height)
            try:
                luma = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width)
            except Exception as e:
                print(str(e) + " - for file " + self.f.name)
                return False, None
            self.f.read(self.width*self.height//2)
            return True, np.uint8(np.round(luma * 2**(output_depth - self.bit_depth)))
        elif 8 < self.bit_depth <= 16:
            raw = self.f.read(2*self.width*self.height)
            try:
                luma = np.frombuffer(raw, dtype=np.uint16).reshape(self.height, self.width)
            except Exception as e:
                print(str(e) + " - for file " + self.f.name)
                return False, None
            self.f.read(self.width*self.height)
            return True, np.uint16(np.round(luma * 2**(output_depth - self.bit_depth)))
        else:
            print("Invalid bit depth specified!")
            return False, None

    def skip(self, num):
        """
        Skip a number of frames in the YUV file
        :param num: Frame number where the YUV file should point to next
        """
        if self.bit_depth <= 8:
            self.f.seek(num*self.width*self.height*3//2)
        elif 8 < self.bit_depth <= 16:
            self.f.seek(num*self.width*self.height*3)
        else:
            print("Invalid bit depth specified!")

    def close(self):
        """
        Close the YUV file
        """
        self.f.close()


def import_path(path):
    """
    Function for reading config files
    :param path: path to the config file
    :return: module that contains all the parameters as specified in the config file
    """
    module_name = os.path.basename(path)
    spec = spec_from_loader(
        module_name,
        SourceFileLoader(module_name, path)
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def nested_dict():
    """
    Function for creating generic nested dictionaries
    :return: empty nested dictionary
    """
    return defaultdict(nested_dict)


def clip_round(value):
    """
    Function used for rounding and clipping interpolated values to 10-bit range
    :param value: float input
    :return: rounded and clipped value
    """
    value = np.round(value/64)
    value = max(0, min(value, 1023))
    return value


def interp_filtering(input_block, kernel_size, x_frac, y_frac):
    """
    Function that interpolates rectangular blocks using DCT filter coefficients based on the selected fractional inputs
    :param input_block: rectangular block to be interpolated
    :param kernel_size: combined size of convolutional kernels, affects the label shape
    :param x_frac: fractional position on the x-axis
    :param y_frac: fractional position on the y-axis
    :return: label, as an interpolated rectangular block
    """
    input_block = input_block.astype(np.float)
    label = np.zeros((input_block.shape[0] - kernel_size + 1, input_block.shape[1] - kernel_size + 1, 1))

    #  only horizontal filtering
    if x_frac != 0 and y_frac == 0:
        filter_x = filter_coefficients(x_frac)
        for i, j in product(range(label.shape[0]), range(label.shape[1])):
            label[i, j, :] = sum(val * input_block[i + 6, j + ind + 3, :] for ind, val in enumerate(filter_x))
            label[i, j, :] = clip_round(label[i, j, :])
    # only vertical filtering
    elif x_frac == 0 and y_frac != 0:
        filter_y = filter_coefficients(y_frac)
        for i, j in product(range(label.shape[0]), range(label.shape[1])):
            label[i, j, :] = sum(val * input_block[i + ind + 3, j + 6, :] for ind, val in enumerate(filter_y))
            label[i, j, :] = clip_round(label[i, j, :])
    # horizontal and vertical filtering
    elif x_frac != 0 and y_frac != 0:
        temp = np.zeros((label.shape[0] + 7, label.shape[1], label.shape[2]))
        filter_x = filter_coefficients(x_frac)
        for i, j in product(range(temp.shape[0]), range(temp.shape[1])):
            temp[i, j, :] = sum(val * input_block[i + 3, j + ind + 3, :] for ind, val in enumerate(filter_x))
            temp[i, j, :] = clip_round(temp[i, j, :])
        filter_y = filter_coefficients(y_frac)
        for i, j in product(range(label.shape[0]), range(label.shape[1])):
            label[i, j, :] = sum(val * temp[i + ind, j, :] for ind, val in enumerate(filter_y))
            label[i, j, :] = clip_round(label[i, j, :])

    return label.astype(np.int16)


def frac_positions():
    """
    Function that returns all quarter-pixel fractional positions in a 2D block (horizontal/vertical)
    :return: string list of "x-axis,y-axis" fractions, in increments of 4
    """
    return [f"{x},{y}" for x in range(0, 15, 4) for y in range(0, 15, 4) if x != 0 or y != 0]


def block_sizes(max_size):
    """
    Function that returns all possible block sizes of an inter-predicted frame (min size is 4x8 or 8x4)
    :param max_size: maximum exponent of the power of 2 specifying width/height (f.e. 6 = 32x32), max value is 8
    :return: string list of "width x height" sizes
    """
    if max_size > 8:
        raise ValueError("Invalid max_size value specified!")
    else:
        return [f"{2**x}x{2**y}" for x in range(2, max_size) for y in range(2, max_size) if x != 2 or y != 2]


def filter_coefficients(position):
    """
    15 quarter-pixel interpolation filters, with coefficients taken directly from VVC implementation
    :param position: 1 indicates (0,4) fractional shift, ..., 4 is (4,0), ... 9 is (8,4), ...
    :return: 8 filter coefficients for the specified fractional shift
    """
    return {
        1: [0, 1, -3, 63, 4, -2, 1, 0],
        2: [-1, 2, -5, 62, 8, -3, 1, 0],
        3: [-1, 3, -8, 60, 13, -4, 1, 0],
        4: [-1, 4, -10, 58, 17, -5, 1, 0],
        5: [-1, 4, -11, 52, 26, -8, 3, -1],
        6: [-1, 3, -9, 47, 31, -10, 4, -1],
        7: [-1, 4, -11, 45, 34, -10, 4, -1],
        8: [-1, 4, -11, 40, 40, -11, 4, -1],
        9: [-1, 4, -10, 34, 45, -11, 4, -1],
        10: [-1, 4, -10, 31, 47, -9, 3, -1],
        11: [-1, 3, -8, 26, 52, -11, 4, -1],
        12: [0, 1, -5, 17, 58, -10, 4, -1],
        13: [0, 1, -4, 13, 60, -8, 3, -1],
        14: [0, 1, -3, 8, 62, -5, 2, -1],
        15: [0, 1, -2, 4, 63, -3, 1, 0]
    }.get(position, 'Invalid fractional pixel position!')


def concatenate_dictionary_keys(dictionary):
    """
    Function that concatenates all values under dictionary keys into one value
    :param dictionary: dictionary to be concatenated
    :return: concatenated dictionary
    """
    dictionary = np.concatenate(list(dictionary[i] for i in dictionary))
    return dictionary


def get_test_dict(path):
    """
    Function that loads testing dictionaries from input path
    :param path: path to test H5py dataset
    :return: dictionaries of inputs, labels and SAD losses
    """
    filename = [file for file in os.listdir(path) if file.endswith('test.hdf5')][0]
    f = h5todict(os.path.join(path, filename))

    inputs_dict = f.get('inputs')
    labels_dict = f.get('labels')
    sad_dict = f.get('sad_loss')
    return inputs_dict, labels_dict, sad_dict


def read_testdata(path, frac, model):
    """
    Load H5py test dataset for Scratch model
    :param path: file path of dataset
    :param frac: fractional position pair for which the dataset has been generated
    :param model: name of the model to be tested
    :return: arrays of inputs, labels and SAD losses
    """
    # load dataset and dictionaries of inputs, labels, SAD (Sum of Absolute Differences) loss
    inputs_dict, labels_dict, sad_dict = get_test_dict(path)

    # create testing dictionaries
    block_keys = [k for k in inputs_dict]
    test_inputs_dict, test_labels_dict, sad_loss_dict = (dict() for _ in range(3))

    # get inputs / labels / sad_loss for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block][frac]

        if len(inputs) == 0:
            continue

        # if SRCNN, use same input & label size
        inputs = inputs[:, 6:-6, 6:-6, :] if model == "srcnn" else inputs

        labels = labels_dict[block][frac]
        sad_loss = sad_dict[block][frac]

        test_inputs_dict[block] = np.array(inputs).astype(float)
        test_labels_dict[block] = np.array(labels).astype(float)
        sad_loss_dict[block] = np.array(sad_loss).astype(float)

    return test_inputs_dict, test_labels_dict, sad_loss_dict


def read_shared_testdata(path):
    """
    Load H5py test dataset for Shared model
    :param path: file path of dataset
    :return: arrays of inputs, labels and SAD losses
    """
    # load dataset and dictionaries of inputs, labels, SAD (Sum of Absolute Differences) loss
    inputs_dict, labels_dict, sad_dict = get_test_dict(path)

    # create testing dictionaries
    block_keys = [k for k in inputs_dict]
    test_inputs_dict, test_labels_dict, sad_loss_dict = (nested_dict() for _ in range(3))

    # get inputs / labels / sad_loss for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block]
        if min(list(len(inputs[frac]) for frac in inputs)) == 0:
            continue

        for frac in inputs_dict[block]:
            inputs = inputs_dict[block][frac]
            labels = labels_dict[block][frac]
            sad_loss = sad_dict[block][frac]

            test_inputs_dict[block][frac] = np.array(inputs).astype(float)
            test_labels_dict[block][frac] = np.array(labels).astype(float)
            sad_loss_dict[block][frac] = np.array(sad_loss).astype(float)

    return test_inputs_dict, test_labels_dict, sad_loss_dict


def read_combined_testdata(path):
    """
    Load H5py test dataset for Competition models
    :param path: file path of dataset
    :return: arrays of inputs, labels and SAD losses
    """
    # combine all subsets into one and shuffle
    test_inputs_sub, test_labels_sub, sad_loss_sub = read_shared_testdata(path)

    test_inputs_all, test_labels_all, test_sad_all = (dict() for _ in range(3))
    for block in test_inputs_sub:
        test_inputs_all[block] = concatenate_dictionary_keys(test_inputs_all[block])
        test_labels_all[block] = concatenate_dictionary_keys(test_labels_all[block])
        test_sad_all[block] = concatenate_dictionary_keys(test_sad_all[block])

    return test_inputs_all, test_labels_all, test_sad_all


def split_data(data, train_percentage):
    """
    Function that splits data into training and validation sets according to the specified percentage
    :param data: data to be split
    :param train_percentage: percentage of training data in the entire data
    :return: training & validation data
    """
    return data[0:int(train_percentage * len(data))],  data[int(train_percentage * len(data)):]


def array_triple_shuffle(first, second, third):
    """
    Shuffle a triple of arrays
    :param first: first array out of three
    :param second: second array out of three
    :param third: third array out of three
    :return: shuffled triple of arrays
    """
    c = list(zip(first, second, third))
    random.Random(42).shuffle(c)
    first, second, third = zip(*c)

    first = np.array(first).astype(float)
    second = np.array(second).astype(float)
    third = np.array(third).astype(float)
    return first, second, third


def read_data(path, batch_size, qp, frac, model):
    """
    Load H5py dataset for Scratch models
    :param path: file path of dataset
    :param batch_size: size of batch
    :param qp: Quantization Parameter (QP) of the dataset
    :param frac: fractional position pair for which the dataset has been generated
    :param model: name of the model to be trained
    :return: arrays of inputs and labels, separated into training and validation sets
    """
    # load h5 file and get dictionaries
    filename = [file for file in os.listdir(path) if file.endswith('_%d.hdf5' % qp)][0]
    f = h5todict(os.path.join(path, filename))

    inputs_dict = f.get('inputs')
    labels_dict = f.get('labels')

    # create training / validation dictionaries
    block_keys = [k for k in inputs_dict]
    train_inputs_dict, train_labels_dict, val_inputs_dict, val_labels_dict = (dict() for _ in range(4))

    # get inputs / labels for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block][frac]

        # only use inputs that can be split 80 / 20 train / validation and fill out a batch
        if len(inputs) < 1.25*batch_size:
            continue

        # if SRCNN, use same input & label size
        inputs = inputs[:, 6:-6, 6:-6, :] if model == "srcnn" else inputs

        labels = labels_dict[block][frac]

        # shuffle the pairs
        c = list(zip(inputs, labels))
        random.Random(42).shuffle(c)
        inputs, labels = zip(*c)

        inputs = np.array(inputs).astype(float)
        labels = np.array(labels).astype(float)

        # split 80 / 20
        train_inputs, val_inputs = split_data(inputs, 0.8)
        train_labels, val_labels = split_data(labels, 0.8)

        # ensure the training size is a multiple of batch size
        train_mod = len(train_inputs) % batch_size
        train_inputs_dict[block] = train_inputs[:-train_mod] if train_mod else train_inputs
        train_labels_dict[block] = train_labels[:-train_mod] if train_mod else train_labels
        val_inputs_dict[block] = np.vstack((val_inputs, train_inputs[-train_mod:])) if train_mod else val_inputs
        val_labels_dict[block] = np.vstack((val_labels, train_labels[-train_mod:])) if train_mod else val_labels

    return train_inputs_dict, train_labels_dict, val_inputs_dict, val_labels_dict


def read_shared_data(path, batch_size):
    """
    Load H5py dataset for Shared models
    :param path: file path of dataset
    :param batch_size: size of batch
    :return: arrays of inputs, labels and SAD losses, separated into training and validation sets
    """
    # load h5 file and get dictionaries
    filename = [file for file in os.listdir(path) if file.endswith('_27.hdf5')][0]
    f = h5todict(os.path.join(path, filename))

    inputs_dict = f.get('inputs')
    labels_dict = f.get('labels')
    sad_dict = f.get('sad_loss')

    # create training / validation dictionaries
    block_keys = [k for k in inputs_dict]
    train_inputs_dict, train_labels_dict, train_sad_dict, val_inputs_dict, val_labels_dict, val_sad_dict = \
        (nested_dict() for _ in range(6))

    # get inputs / labels / SAD losses for block, don't separate by frac position
    for block in block_keys:
        inputs = inputs_dict[block]

        # find minimum input size across all fractional positions for that block size
        balanced_size = min(list(len(inputs[frac]) for frac in inputs))

        # only use inputs that can be split 80 / 20 train / validation and fill out a batch
        if balanced_size < 1.25*batch_size:
            continue

        labels = labels_dict[block]
        sad_loss = sad_dict[block]

        train_inputs, train_labels, train_sad, val_inputs, val_labels, val_sad = (dict() for _ in range(6))

        # shuffle datasets per fractional position and reduce each subset to the same size, then split
        for frac in inputs:
            c = list(zip(inputs[frac], labels[frac], sad_loss[frac]))
            random.Random(42).shuffle(c)
            inputs[frac], labels[frac], sad_loss[frac] = zip(*c)

            inputs[frac] = np.array(inputs[frac][:balanced_size]).astype(float)
            labels[frac] = np.array(labels[frac][:balanced_size]).astype(float)
            sad_loss[frac] = np.array(sad_loss[frac][:balanced_size]).astype(float)

            # split 80 / 20
            train_inputs[frac], val_inputs[frac] = split_data(inputs[frac], 0.8)
            train_labels[frac], val_labels[frac] = split_data(labels[frac], 0.8)
            train_sad[frac], val_sad[frac] = split_data(sad_loss[frac], 0.8)

            # ensure the training size is a multiple of batch size
            train_mod = len(train_inputs[frac]) % batch_size
            train_inputs_dict[block][frac] = train_inputs[frac][:-train_mod] if train_mod else train_inputs[frac]
            train_labels_dict[block][frac] = train_labels[frac][:-train_mod] if train_mod else train_labels[frac]
            train_sad_dict[block][frac] = train_sad[frac][:-train_mod] if train_mod else train_sad[frac]
            val_inputs_dict[block][frac] = np.vstack((val_inputs[frac], train_inputs[frac][-train_mod:])) \
                if train_mod else val_inputs[frac]
            val_labels_dict[block][frac] = np.vstack((val_labels[frac], train_labels[frac][-train_mod:])) \
                if train_mod else val_labels[frac]
            val_sad_dict[block][frac] = np.hstack((val_sad[frac], train_sad[frac][-train_mod:])) \
                if train_mod else val_sad[frac]

    return train_inputs_dict, train_labels_dict, train_sad_dict, val_inputs_dict, val_labels_dict, val_sad_dict


def read_combined_data(train_inputs, train_labels, train_sad, val_inputs, val_labels, val_sad):
    """
    Load H5py datasets for Competition models
    :param train_inputs: dictionary of training inputs, separated by block sizes and fractional positions
    :param train_labels: dictionary of training labels, separated by block sizes and fractional positions
    :param train_sad: dictionary of training SAD losses, separated by block sizes and fractional positions
    :param val_inputs: dictionary of validation inputs, separated by block sizes and fractional positions
    :param val_labels: dictionary of validation labels, separated by block sizes and fractional positions
    :param val_sad: dictionary of validation SAD losses, separated by block sizes and fractional positions
    :return: arrays of inputs, labels and SAD losses, separated into training and validation sets
    """
    # combine all subsets into one and shuffle
    train_inputs_all, train_labels_all, val_inputs_all, val_labels_all, train_sad_all, val_sad_all = \
        (dict() for _ in range(6))
    for block in train_inputs:
        train_inputs_all[block] = concatenate_dictionary_keys(train_inputs[block])
        train_labels_all[block] = concatenate_dictionary_keys(train_labels[block])
        train_sad_all[block] = concatenate_dictionary_keys(train_sad[block])
        val_inputs_all[block] = concatenate_dictionary_keys(val_inputs[block])
        val_labels_all[block] = concatenate_dictionary_keys(val_labels[block])
        val_sad_all[block] = concatenate_dictionary_keys(val_sad[block])

        # shuffle the triples
        train_inputs_all[block], train_labels_all[block], train_sad_all[block] = \
            array_triple_shuffle(train_inputs_all[block], train_labels_all[block], train_sad_all[block])
        val_inputs_all[block], val_labels_all[block], val_sad_all[block] = \
            array_triple_shuffle(val_inputs_all[block], val_labels_all[block], val_sad_all[block])

    return train_inputs_all, train_labels_all, train_sad_all, val_inputs_all, val_labels_all, val_sad_all


def calculate_batch_number(train_data, val_data, batch_size):
    """
    Function that calculates number of training / validation batches according to the specified batch size
    :param train_data: training dictionary
    :param val_data: validation dictionary
    :param batch_size: batch size of model to be trained
    :return: number of training and validation batches per key in dictionary
    """
    batch_train, batch_val = ([] for _ in range(2))
    for key in train_data:
        batch_train.append(int(len(train_data[key]) / batch_size))
        batch_val.append(math.ceil(len(val_data[key]) / batch_size))

    return batch_train, batch_val


def calculate_batch_number_nested(train_data, val_data, batch_size):
    """
    Function that calculates number of training / validation batches of a nested dictionary
    according to the specified batch size
    :param train_data: nested training dictionary
    :param val_data: nested validation dictionary
    :param batch_size: batch size of model to be trained
    :return: number of training and validation batches per key in top dictionary
    """
    batch_train, batch_val = ([] for _ in range(2))
    for key in train_data:
        batch_train.append(int(len(train_data[key][next(iter(train_data[key]))]) / batch_size))
        batch_val.append(math.ceil(len(val_data[key][next(iter(val_data[key]))]) / batch_size))

    return batch_train, batch_val


def calculate_test_error(result, test_label, test_sad):
    """
    Method that calculates prediction error of the Neural Network (NN), VVC and their combination batch
    to the original label
    :param result: NN prediction
    :param test_label: ground truth
    :param test_sad: SAD loss of VVC and the ground truth
    :return: a batch summation of the NN, VVC and switchable (combined / minimum) SAD losses
    """
    result = np.round(result).astype(int)
    nn_cost = np.mean(np.abs(test_label - result), axis=(1, 2, 3))
    vvc_cost = test_sad

    # calculate switchable filter loss
    switch_cost = np.stack([nn_cost, vvc_cost])
    switch_cost = np.min(switch_cost, axis=0)

    return np.mean(nn_cost), np.mean(vvc_cost), np.mean(switch_cost)


def save_results(results_dir, model_name, model_subdir, error_pred, error_vvc, error_switch):
    """
    Function that saves results of model testing into a .txt file in the results directory
    :param results_dir: results directory
    :param model_name: name of trained model
    :param model_subdir: model subdirectory organisation
    :param error_pred: NN model SAD loss
    :param error_vvc: VVC SAD loss
    :param error_switch: Switchable (combined / minimum) SAD loss
    """
    os.makedirs(os.path.join(results_dir, model_subdir), exist_ok=True)
    with open(os.path.join(results_dir, model_subdir, f"test_sad_loss.txt"), 'a') as out_file:
        out_file.write(f"NN SAD loss : {np.mean(error_pred):.4f}")
        out_file.write(f"VVC SAD loss : {np.mean(error_vvc):.4f}")
        out_file.write(f"Switchable {model_name.upper()} SAD loss: {np.mean(error_switch):.4f}")
