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

from model_scratch import ScratchCNN, ScratchActCNN, ScratchBiasCNN, ScratchAllCNN, ScratchOneCNN, SRCNN
from model_shared import SharedCNN
from model_competition import CompetitionCNN
from utils import import_path
import tensorflow as tf
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Subpixel Motion Interpolation')
    parser.add_argument('-c', '--config', type=str, default=None, required=True, help='Config filepath')
    parser.add_argument('-a', '--action', type=str, default=None, required=True, help='train or test')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='CUDA device number')
    parser.add_argument('-gf', '--gpu_fraction', type=float, default=1.0, help='Fraction of GPU usage')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # import config file details
    model_cfg = import_path(args.config)

    # create directories where necessary
    os.makedirs(model_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(model_cfg.results_dir, exist_ok=True)
    os.makedirs(model_cfg.graphs_dir, exist_ok=True)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

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
        elif model_cfg.model_name == "srcnn":
            cnn_model = SRCNN(sess, model_cfg)
        elif model_cfg.model_name == "sharedcnn":
            cnn_model = SharedCNN(sess, model_cfg)
        elif model_cfg.model_name == "competitioncnn":
            cnn_model = CompetitionCNN(sess, model_cfg)
        else:
            raise ValueError("Invalid model name specified!")

        # train or test the model
        if args.action == "train":
            if not os.path.exists(model_cfg.dataset_dir):
                raise ValueError("Invalid training dataset directory specified!")
            cnn_model.train()
        elif args.action == "test":
            if not os.path.exists(model_cfg.test_dataset_dir):
                raise ValueError("Invalid testing dataset directory specified!")
            cnn_model.test()
        else:
            raise ValueError("Invalid action specified!")
