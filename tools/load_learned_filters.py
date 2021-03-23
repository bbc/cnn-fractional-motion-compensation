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
from model_scratch import ScratchCNN, ScratchActCNN, ScratchBiasCNN, ScratchAllCNN, ScratchOneCNN
from model_shared import SharedCNN
from model_competition import CompetitionCNN
from utils import import_path


def write_to_xlsx(df_input, write, name):
    df = pd.DataFrame(df_input)
    df.to_excel(write, sheet_name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create worksheet with coefficients of a CNN interpolation filter set')
    parser.add_argument('-m', '--model', type=str, default=None, required=True, help='Model name')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='CUDA device number')
    parser.add_argument('-gf', '--gpu_fraction', type=float, default=1.0, help='Fraction of GPU usage')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # import model config
        model_cfg = import_path(f"../model-configs/{args.model_name}.py")

        # initialise the specified cnn model
        if model_cfg.model_name == "scratchcnn":
            cnn_model = ScratchCNN(sess, model_cfg)
        elif model_cfg.model_name == "scratchcnn_activation":
            cnn_model = ScratchActCNN(sess, model_cfg)
        elif model_cfg.model_name == "scratchcnn_bias":
            cnn_model = ScratchBiasCNN(sess, model_cfg)
        elif model_cfg.model_name == "scratchcnn_all":
            cnn_model = ScratchAllCNN(sess, model_cfg)
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
        global_step = cnn_model.load(cnn_model.subdirectory())

        # compute filter coefficients from trained weights of the CNN model
        # get shape of kernels from each layer
        w1_shape = cnn_model.weights['w1'].get_shape()
        w2_shape = cnn_model.weights['w2'].get_shape()
        w3_shape = cnn_model.weights['w3'].get_shape()

        # create matrix of minimum input size to the neural network, with labels for each position in the 13x13 grid
        # extract patches from the matrix revealing where each input value was copied to before the 1st convolution
        increment = np.reshape(np.arange(1, 13 ** 2 + 1), (13, 13))
        increment_patches = tf.extract_image_patches(
            images=np.reshape(increment, (1, 13, 13, 1)),
            ksizes=[1, w1_shape[0], w1_shape[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
            padding='VALID').eval()
        # final input matrix that contains info for all positions
        # that each coefficient from the master weight matrix will be applied to
        increment_patches = np.reshape(increment_patches, (increment_patches.shape[1] * increment_patches.shape[2],
                                                           increment_patches.shape[3]))

        # reshape layer weights into 2D matrices
        w1_rshp = np.reshape(cnn_model.sess.run(cnn_model.weights['w1']),
                             (w1_shape[0] * w1_shape[1], w1_shape[3]))
        w2_rshp = np.reshape(cnn_model.sess.run(cnn_model.weights['w2']), (w2_shape[2], w2_shape[3]))

        for l3_filter in range(w3_shape[3]):
            w3_rshp = np.reshape(cnn_model.sess.run(cnn_model.weights['w3'])[:, :, :, l3_filter],
                                 (w3_shape[0] * w3_shape[1] * w3_shape[2], 1))

            # transform the final layer vector into a 2D matrix
            l3_trans = np.zeros((w3_shape[2], w3_shape[0] * w3_shape[1]))
            for i, j in product(range(l3_trans.shape[0]), range(l3_trans.shape[1])):
                l3_trans[i, j] = w3_rshp[j * l3_trans.shape[0] + i]

            # get the master weight matrix that contains all matrix multiplication values
            l_master = np.dot(np.dot(w1_rshp, w2_rshp), l3_trans)

            # create a dictionary where keys are positions in the final 13x13 matrix
            # and values are the final coefficients,
            # obtained by summing the correct values from the master weight matrix
            # (the info on which values contribute to each position is stored in the increment_patches matrix)
            keys = np.arange(1, 13 ** 2 + 1)
            values = np.zeros(13 ** 2)
            finalmatrix_dict = dict(zip(keys, values))
            for i, j in product(range(increment_patches.shape[0]), range(increment_patches.shape[1])):
                finalmatrix_dict[increment_patches[i, j]] += l_master[j, i]

            # create list of final coefficients
            all_val = list(finalmatrix_dict.values())

            # reshape the list into a 13x13 matrix and add 1 to the central coefficient
            # because the network learns residuals
            all_shapes = np.reshape(all_val, (13, 13))
            all_shapes[int(all_shapes.shape[0] / 2), int(all_shapes.shape[1] / 2)] += 1

            # scale coefficients so their sum equals 1
            all_shapes *= 1/np.sum(all_shapes)

            # create directories if needed
            results_subdir = f"{model_cfg.model_name}-{model_cfg.dataset_dir.split('/')[1]}_{str(model_cfg.qp)}"
            os.makedirs(os.path.join(model_cfg.results_dir, results_subdir), exist_ok=True)

            writer = pd.ExcelWriter(os.path.join(model_cfg.results_dir, results_subdir, "filters.xlsx"),
                                    engine='xlsxwriter')

            if model_cfg == "sharedcnn" or "competitioncnn":
                filter_list = [0, 5, 1, 3, 9, 4, 2, 8, 11, 7, 10, 12, 6, 13, 14]
            else:
                filter_list = list(range(15))

            w_file = open(os.path.join(model_cfg.results_dir, results_subdir, "nn_filter_for_vtm.txt"), "w+")
            w_file.write("const double InterpolationFilter::m_AltNNFilters[15][NTAPS_NN][NTAPS_NN] =\n{\n")

            write_to_xlsx(all_shapes, writer, f"NN Filter {l3_filter}")

            w_file.write("  {")
            for r in range(all_shapes.shape[0]):
                w_file.write("    {") if r != 0 else w_file.write(" {")
                for c in range(all_shapes.shape[1]):
                    w_file.write(" " + str(all_shapes[r][c]) + ",") if c != all_shapes.shape[1] - 1 else \
                        w_file.write(" " + str(all_shapes[r][c]))
                w_file.write(" },\n") if r != all_shapes.shape[0] - 1 else w_file.write(" }\n")

            w_file.write("  },\n") if l3_filter != filter_list[-1] else w_file.write("  }\n")

    w_file.write("};")
    w_file.close()

    writer.save()
    writer.close()
