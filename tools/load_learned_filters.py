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
import tensorflow as tf
import os
import pandas as pd
import argparse
from itertools import product
from model_scratch import ScratchCNN, ScratchOneCNN
from model_shared import SharedCNN
from model_competition import CompetitionCNN
from utils import import_path, frac_positions, nested_dict, vvc_filters_2d, zncc


def write_to_txt(file, qp_range, filter_dict):
    """
    Write to file that will contain the C++ array with filter coefficients
    :param file: .txt file to be written to
    :param qp_range: list of QPs for which the network has been trained
    :param filter_dict: dictionary of learned filter, separated by fractional shift and, optionally, QP
    """
    single_qp = 1 if len(qp_range) == 1 else 0

    file.write("const double InterpolationFilter::m_AltNNFilters[15][NTAPS_NN][NTAPS_NN] =\n{\n") if combined else \
        file.write("const double InterpolationFilter::m_AltNNFilters[4][15][NTAPS_NN][NTAPS_NN] =\n{\n")

    spaces = 2
    for current_qp in qp_range:
        # write all coefficients into the .txt file, per fractional shift and per QP
        if not single_qp:
            w_file.write(" " * spaces + "{\n")
            spaces += 2
        for current_key in filter_dict[current_qp]:
            current_filter = filter_dict[current_qp][current_key]

            file.write(" " * spaces + "{\n")
            spaces += 2
            for r in range(current_filter.shape[0]):
                file.write(" " * spaces + "{")
                for c in range(current_filter.shape[1]):
                    file.write(f" {current_filter[r][c]},") if c != current_filter.shape[1] - 1 else \
                        file.write(f" {current_filter[r][c]}")
                file.write(" },\n") if r != current_filter.shape[0] - 1 else file.write(" }\n")

            spaces -= 2
            file.write(" " * spaces + "},\n") if current_key != len(frac_positions()) - 1 else \
                file.write(" " * spaces + "}\n")

        if not single_qp:
            spaces -= 2
            file.write(" " * spaces + "},\n") if current_qp != qp_list[-1] else file.write(" " * spaces + "}\n")

    file.write("};")
    file.close()


def write_to_xlsx(df_input, write, name):
    """
    Write an input to a .xlsx file
    :param df_input: input for DataFrame
    :param write: Excel writer
    :param name: name of the sheet
    """
    df = pd.DataFrame(df_input)
    df.to_excel(write, sheet_name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create worksheet with coefficients of a CNN interpolation filter set')
    parser.add_argument('-m', '--model', type=str, default=None, required=True, help='Model name')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='CUDA device number')
    parser.add_argument('-gf', '--gpu_fraction', type=float, default=1.0, help='Fraction of GPU usage')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # import model config
    model_cfg = import_path(f"../model-configs/{args.model}.py")
    model_cfg.results_dir = "../" + model_cfg.results_dir
    model_cfg.checkpoint_dir = "../" + model_cfg.checkpoint_dir

    # loop through all fractional shifts and QPs the network has been trained for
    combined = 1 if model_cfg.model_name == "sharedcnn" or model_cfg.model_name == "competitioncnn" else 0
    frac_xy = ["all"] if combined else frac_positions()
    qp_list = [27] if combined else [22, 27, 32, 37]

    # prepare dictionaries that will store the corresponding filters
    learned_filters = nested_dict()
    max_correlation = {k: [-1] * len(frac_positions()) for k in qp_list}

    for frac, qp in product(frac_xy, qp_list):
        model_cfg.fractional_pixel = frac
        model_cfg.qp = qp

        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # initialise the specified cnn model
            if model_cfg.model_name == "scratchcnn":
                cnn_model = ScratchCNN(sess, model_cfg)
            elif model_cfg.model_name == "scratchcnn_onelayer":
                cnn_model = ScratchOneCNN(sess, model_cfg)
            elif model_cfg.model_name == "sharedcnn":
                cnn_model = SharedCNN(sess, model_cfg)
            elif model_cfg.model_name == "competitioncnn":
                cnn_model = CompetitionCNN(sess, model_cfg)
            else:
                raise ValueError("Invalid model name specified!")

            # load the trained model
            tf.global_variables_initializer().run()
            _ = cnn_model.load(cnn_model.subdirectory())

            # compute filter coefficients from trained weights of the CNN model
            # first, get shape and values of kernels from each layer, and the entire kernel size
            weight_shapes, weight_values = ([] for _ in range(2))
            kernel = 0
            for w in cnn_model.weights:
                w_shape = cnn_model.weights[w].get_shape().as_list()
                weight_shapes.append(w_shape)
                weight_values.append(np.reshape(cnn_model.sess.run(cnn_model.weights[w]),
                                                (w_shape[0] * w_shape[1] * w_shape[2], w_shape[3])))
                kernel += w_shape[0] - 1 if w_shape[0] > 1 and len(cnn_model.weights) > 1 else w_shape[0]

            # with a multi-layer CNN
            if len(cnn_model.weights) > 1:
                # transpose each layer before last and reshape the final layer vector into a 3D matrix
                for i in range(len(weight_values)-1):
                    weight_values[i] = np.transpose(weight_values[i])
                weight_values[-1] = np.reshape(weight_values[-1], (w_shape[0] * w_shape[1], w_shape[2], w_shape[3]))

                # get the master weight matrix that contains all matrix multiplication values
                l_master = weight_values[-1]
                for values in reversed(weight_values[:-1]):
                    l_master = np.einsum("ijk,jl->ilk", l_master, values)
                l_master = np.reshape(l_master,
                                      (l_master.shape[0], weight_shapes[0][0], weight_shapes[0][1], l_master.shape[-1]))

                # create the final kernel matrix by adding coefficients from the master weight matrix
                # to overlapping patch positions within the kernel
                all_coefficients = np.zeros((kernel, kernel, weight_shapes[-1][-1]))
                for i in range(all_coefficients.shape[-1]):
                    for j in range(l_master.shape[0]):
                        all_coefficients[j // 5:j // 5 + 9, j % 5:j % 5 + 9, i] += l_master[j, :, :, i]

            # with a single-layer CNN, directly use the coefficients
            else:
                all_coefficients = np.reshape(weight_values[0], (kernel, kernel, weight_shapes[-1][-1]))

            # add 1 to the central coefficient because the network learns residuals
            for k in range(weight_shapes[-1][-1]):
                all_coefficients[all_coefficients.shape[0] // 2, all_coefficients.shape[1] // 2, k] += 1

                # scale coefficients so their sum equals 1
                all_coefficients[:, :, k] *= 1 / np.sum(all_coefficients[:, :, k])

        # calculate correlation of coefficients to existing vvc filters and store the best filters into a dict
        # per fractional shift and per QP
        existing_filters = vvc_filters_2d(kernel)
        for k in range(all_coefficients.shape[2]):
            nn_filter = all_coefficients[:, :, k]
            for i, vvc_filter in enumerate(existing_filters):
                cc = zncc(vvc_filter, nn_filter)
                if cc > max_correlation[qp][i]:
                    max_correlation[qp][i] = cc
                    learned_filters[qp][i] = nn_filter

    # create directories if needed
    results_subdir = os.path.join("models", model_cfg.model_name, model_cfg.dataset_dir.split('/')[1])
    os.makedirs(os.path.join(model_cfg.results_dir, results_subdir), exist_ok=True)

    # write all learned filters to .txt file
    w_file = open(os.path.join(model_cfg.results_dir, results_subdir, "nn_filters_for_vtm.txt"), "w+")
    write_to_txt(w_file, qp_list, learned_filters)

    for qp in qp_list:
        # open the .xlsx file where all final filter coefficients per QP will be displayed
        writer = pd.ExcelWriter(os.path.join(model_cfg.results_dir, results_subdir, f"filters_{qp}.xlsx"),
                                engine='xlsxwriter')
        for key in learned_filters[qp]:
            # write the filter coefficients into a separate worksheet within the .xlsx file
            write_to_xlsx(learned_filters[qp][key], writer, f"NN Filter {key}")
        writer.save()
        writer.close()
