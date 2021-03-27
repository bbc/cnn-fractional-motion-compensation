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
    Read config files
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
    Create generic nested dictionary
    :return: empty nested dictionary
    """
    return defaultdict(nested_dict)


def clip_round(value):
    """
    Round and clip interpolated values to 10-bit range
    :param value: float input
    :return: rounded and clipped value
    """
    return max(0, min(np.round(value/64), 1023))


def interp_filtering(input_block, kernel_size, x_frac, y_frac):
    """
    Interpolate rectangular blocks using DCT filter coefficients based on the selected fractional inputs
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


def vvc_filters_2d(kernel_size):
    """
    Compute 2D version of VVC filters and store them in a kernel-sized grid
    :param kernel_size: size of kernel for 2D VVC filter representation (min value is 8)
    :return a list of 15 2D VVC filters
    """
    vvc_filters = []
    half_kernel = (kernel_size - 8) // 2
    for frac_pos in frac_positions():
        filter_x = filter_coefficients(int(frac_pos.split(",")[0]))
        filter_y = filter_coefficients(int(frac_pos.split(",")[1]))

        filter_vvc = np.tile(filter_x, 8).reshape((8, 8))
        for index in range(len(filter_y)):
            filter_vvc[index, :] *= filter_y[index]
        filter_vvc = filter_vvc / (64 * 64)

        vvc_filters.append(np.pad(filter_vvc, ((half_kernel + 1, half_kernel), (half_kernel + 1, half_kernel)),
                                  'constant', constant_values=0))
    return vvc_filters


def zncc(template, image):
    """
    Zero-normalized cross-correlation, a 2D version of Pearson product-moment correlation coefficient
    :param
    :param
    :return cross-correlation coefficient
    """
    std_template = np.std(template)
    if std_template == 0:
        return 0

    std_image = np.std(image)
    if std_image == 0:
        return 0

    template_t = template - np.mean(template)
    image_t = image - np.mean(image)

    cross_correlation = np.mean(np.multiply(template_t, image_t)) / (std_template * std_image)
    return cross_correlation


def frac_positions():
    """
    Return all quarter-pixel fractional positions in a 2D block (horizontal/vertical)
    :return: string list of "x-axis,y-axis" fractions, in increments of 4
    """
    return [f"{x},{y}" for x in range(0, 15, 4) for y in range(0, 15, 4) if x != 0 or y != 0]


def block_sizes(max_size):
    """
    Return all possible block sizes of an inter-predicted frame (min size is 4x8 or 8x4)
    :param max_size: maximum exponent of the power of 2 specifying width/height (e.g. 6 = 32x32), max value is 8
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
        0: [0, 0, 0, 64, 0, 0, 0, 0],
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


def concatenate_dictionary_keys(*dictionaries):
    """
    Concatenate all values under dictionary keys into one value
    :param dictionaries: one or more dictionaries to be concatenated
    :return: one or more concatenated dictionaries
    """
    concat_dict = []
    for entry in dictionaries:
        concat_dict.append(np.concatenate(list(entry[i] for i in entry)))
    return concat_dict


def get_dataset_dict(path, identifier):
    """
    Load dataset dictionaries from input path
    :param path: path to H5py dataset
    :param identifier: dataset filename identifier, either QP value or 'test'
    :return: dictionaries of inputs, labels and SAD losses
    """
    filename = [file for file in os.listdir(path) if file.endswith(f"{identifier}.hdf5")][0]
    f = h5todict(os.path.join(path, filename))

    return f.get('inputs'), f.get('labels'), f.get('sad_loss')


def read_testdata(path, frac, kernel, model):
    """
    Load H5py test dataset for Scratch model
    :param path: file path of dataset
    :param frac: fractional position pair for which the dataset has been generated
    :param kernel: half size of the kernel, needed for slicing the input
    :param model: name of the model to be tested
    :return: arrays of inputs, labels and SAD losses
    """
    # load dataset and dictionaries of inputs, labels, SAD (Sum of Absolute Differences) loss
    inputs_dict, labels_dict, sad_dict = get_dataset_dict(path, "test")

    # create testing dictionaries
    block_keys = [k for k in inputs_dict]
    test_inputs_dict, test_labels_dict, sad_loss_dict = (dict() for _ in range(3))

    # get inputs / labels / sad_loss for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block][frac]

        if len(inputs) == 0:
            continue

        # if model contains non-linear activations, use same input & label size
        inputs = inputs[:, kernel:-kernel, kernel:-kernel, :] if "scratch" not in model else inputs

        labels = labels_dict[block][frac]
        sad_loss = sad_dict[block][frac]

        test_inputs_dict[block] = inputs.astype(float)
        test_labels_dict[block] = labels.astype(float)
        sad_loss_dict[block] = sad_loss.astype(float)

    return test_inputs_dict, test_labels_dict, sad_loss_dict


def read_shared_testdata(path):
    """
    Load H5py test dataset for Shared model
    :param path: file path of dataset
    :return: arrays of inputs, labels and SAD losses
    """
    # load dataset and dictionaries of inputs, labels, SAD (Sum of Absolute Differences) loss
    inputs_dict, labels_dict, sad_dict = get_dataset_dict(path, "test")

    # create testing dictionaries
    block_keys = [k for k in inputs_dict]
    test_inputs_dict, test_labels_dict, sad_loss_dict = (nested_dict() for _ in range(3))

    # get inputs / labels / sad_loss for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block]
        if min(list(len(inputs[frac]) for frac in inputs)) == 0:
            continue

        for frac in inputs_dict[block]:
            test_inputs_dict[block][frac] = inputs_dict[block][frac].astype(float)
            test_labels_dict[block][frac] = labels_dict[block][frac].astype(float)
            sad_loss_dict[block][frac] = sad_dict[block][frac].astype(float)

    return test_inputs_dict, test_labels_dict, sad_loss_dict


def read_combined_testdata(path):
    """
    Load H5py test dataset for Competition models
    :param path: file path of dataset
    :return: arrays of inputs, labels and SAD losses
    """
    # combine all subsets into one and shuffle
    test_inputs_sub, test_labels_sub, test_sad_sub = read_shared_testdata(path)

    test_inputs_all, test_labels_all, test_sad_all = (dict() for _ in range(3))
    for block in test_inputs_sub:
        test_inputs_all[block], test_labels_all[block], test_sad_all[block] = \
            concatenate_dictionary_keys(test_inputs_sub[block], test_labels_sub[block], test_sad_sub[block])

    return test_inputs_all, test_labels_all, test_sad_all


def split_data(train_percentage, *data):
    """
    Split data into training and validation sets according to the specified percentage
    :param train_percentage: percentage of training data in the entire data
    :param data: data to be split
    :return: training & validation data
    """
    train = [entry[0:int(train_percentage * len(entry))] for entry in data]
    val = [entry[int(train_percentage * len(entry)):] for entry in data]
    return train, val


def array_shuffle(length, *arrays):
    """
    Shuffle arrays by permuting their indices
    :param length: length of output arrays
    :param arrays: variable number of arrays
    :return: shuffled arrays
    """
    p = np.random.RandomState(42).permutation(length)
    return [entry[p] for entry in arrays]


def read_data(path, batch_size, qp, frac, kernel, model):
    """
    Load H5py dataset for Scratch models
    :param path: file path of dataset
    :param batch_size: size of batch
    :param qp: Quantization Parameter (QP) of the dataset
    :param frac: fractional position pair for which the dataset has been generated
    :param kernel: half size of the kernel, needed for slicing the input
    :param model: name of the model to be tested
    :return: arrays of inputs and labels, separated into training and validation sets
    """
    # load h5 file and get dictionaries
    inputs_dict, labels_dict, _ = get_dataset_dict(path, qp)

    # create training / validation dictionaries
    block_keys = [k for k in inputs_dict]
    train_inputs_dict, train_labels_dict, val_inputs_dict, val_labels_dict = (dict() for _ in range(4))

    # get inputs / labels for block & frac position
    for block in block_keys:
        inputs = inputs_dict[block][frac]

        # only use inputs that can be split 80 / 20 train / validation and fill out a batch
        split_percentage = 4/5
        if len(inputs) < batch_size / split_percentage:
            continue

        # if model contains non-linear activations, use same input & label size
        inputs = inputs[:, kernel:-kernel, kernel:-kernel, :] if "scratch" not in model else inputs

        labels = labels_dict[block][frac]

        # shuffle the pairs
        inputs, labels = array_shuffle(len(inputs), inputs, labels)

        # split 80 / 20
        (train_inputs, train_labels), (val_inputs, val_labels) = split_data(split_percentage, inputs, labels)

        # put into correct dictionary entry
        train_inputs_dict[block] = train_inputs
        train_labels_dict[block] = train_labels
        val_inputs_dict[block] = val_inputs
        val_labels_dict[block] = val_labels

    return train_inputs_dict, train_labels_dict, val_inputs_dict, val_labels_dict


def read_shared_data(path, batch_size):
    """
    Load H5py dataset for Shared models
    :param path: file path of dataset
    :param batch_size: size of batch
    :return: arrays of inputs, labels and SAD losses, separated into training and validation sets
    """
    # load h5 file and get dictionaries
    inputs_dict, labels_dict, sad_dict = get_dataset_dict(path, "27")

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
        split_percentage = 4/5
        if balanced_size < batch_size / split_percentage:
            continue

        labels = labels_dict[block]
        sad_loss = sad_dict[block]

        train_inputs, train_labels, train_sad, val_inputs, val_labels, val_sad = (dict() for _ in range(6))

        # shuffle datasets per fractional position and reduce each subset to the same size, then split
        for frac in inputs:
            inputs[frac], labels[frac], sad_loss[frac] = array_shuffle(balanced_size,
                                                                       inputs[frac], labels[frac], sad_loss[frac])

            # split 80 / 20
            (train_inputs[frac], train_labels[frac], train_sad[frac]), \
                (val_inputs[frac], val_labels[frac], val_sad[frac]) = \
                split_data(split_percentage, inputs[frac], labels[frac], sad_loss[frac])

            # put into correct dictionary entry
            train_inputs_dict[block][frac] = train_inputs[frac]
            train_labels_dict[block][frac] = train_labels[frac]
            train_sad_dict[block][frac] = train_sad[frac]
            val_inputs_dict[block][frac] = val_inputs[frac]
            val_labels_dict[block][frac] = val_labels[frac]
            val_sad_dict[block][frac] = val_sad[frac]

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
        train_inputs_all[block], train_labels_all[block], train_sad_all[block] = \
            concatenate_dictionary_keys(train_inputs[block], train_labels[block], train_sad[block])
        val_inputs_all[block], val_labels_all[block], val_sad_all[block] = \
            concatenate_dictionary_keys(val_inputs[block], val_labels[block], val_sad[block])

        # shuffle the triples
        train_inputs_all[block], train_labels_all[block], train_sad_all[block] = \
            array_shuffle(len(train_sad_all[block]),
                          train_inputs_all[block], train_labels_all[block], train_sad_all[block])
        val_inputs_all[block], val_labels_all[block], val_sad_all[block] = \
            array_shuffle(len(val_sad_all[block]), val_inputs_all[block], val_labels_all[block], val_sad_all[block])

    return train_inputs_all, train_labels_all, train_sad_all, val_inputs_all, val_labels_all, val_sad_all


def calculate_batch_number(train_data, val_data, batch_size, nested=False):
    """
    Calculate number of training / validation batches according to the specified batch size
    :param train_data: training dictionary
    :param val_data: validation dictionary
    :param batch_size: batch size of model to be trained
    :param nested: indicates whether it's a nested dictionary; if yes, it needs to be of balanced size
    :return: number of training and validation batches per key in dictionary
    """
    batch_train, batch_val = ([] for _ in range(2))
    for key in train_data:
        if nested:
            batch_train.append(int(len(train_data[key][next(iter(train_data[key]))]) / batch_size))
            batch_val.append(math.ceil(len(val_data[key][next(iter(val_data[key]))]) / batch_size))
        else:
            batch_train.append(int(len(train_data[key]) / batch_size))
            batch_val.append(math.ceil(len(val_data[key]) / batch_size))

    return batch_train, batch_val


def calculate_test_error(result, test_label, test_sad):
    """
    Calculate prediction error of the Neural Network, VVC and their combination batch compared to the original label
    :param result: NN prediction
    :param test_label: ground truth
    :param test_sad: SAD loss of VVC and the ground truth
    :return: a batch summation of the NN, VVC and switchable (combined / minimum) SAD losses
    """
    result = np.round(result).astype(int)
    nn_cost = np.mean(np.abs(test_label - result), axis=(1, 2, 3))

    # calculate switchable filter loss
    switch_cost = np.stack([nn_cost, test_sad])
    switch_cost = np.min(switch_cost, axis=0)

    return np.mean(nn_cost), np.mean(test_sad), np.mean(switch_cost)


def save_results(results_dir, model_name, model_subdir, error_pred, error_vvc, error_switch):
    """
    Save results of model testing into a .txt file in the results directory
    :param results_dir: results directory
    :param model_name: name of trained model
    :param model_subdir: model subdirectory organisation
    :param error_pred: NN model SAD loss
    :param error_vvc: VVC SAD loss
    :param error_switch: Switchable (combined / minimum) SAD loss
    """
    os.makedirs(os.path.join(results_dir, "models", model_subdir), exist_ok=True)
    with open(os.path.join(results_dir, "models", model_subdir, f"test_sad_loss.txt"), 'a') as out_file:
        out_file.write(f"NN SAD loss : {np.mean(error_pred):.4f}\n")
        out_file.write(f"VVC SAD loss : {np.mean(error_vvc):.4f}\n")
        out_file.write(f"Switchable {model_name.upper()} SAD loss: {np.mean(error_switch):.4f}\n")
