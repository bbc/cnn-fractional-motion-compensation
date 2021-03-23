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

checkpoint_dir = "checkpoint"                       # Name of checkpoint directory
dataset_dir = "dataset/blowingbubbles-ld_P_main10"  # Name of dataset directory
test_dataset_dir = "dataset/classD-test"            # Name of testing dataset directory
results_dir = "results"                             # Name of results directory
graphs_dir = "graphs"                               # Name of graphs directory

model_name = "srcnn"                                # Name of model

fractional_pixel = "0,4"                            # x,y pair of the interpolated fractional pixel [0,4 , ..., 12,12]
qp = 22                                             # Quantization Parameter of produced dataset [22, 27, 32, 37]

epoch = 1000                                        # Number of epochs
early_stopping = 50                                 # Threshold value for early stopping of algorithm
batch_size = 32                                     # The size of batch inputs
learning_rate = 1e-4                                # The learning rate of the optimising algorithm
