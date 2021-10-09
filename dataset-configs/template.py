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

experiment_path = "experiments"      # Path to where the YUV files and log files are stored
encoder_cfg = "test"                 # VVC encoding configuration (for train: ld_P_main10, ra_main10; for test: test)
orig_bitdepth = 8                    # Bit depth of the original YUV sequence
deco_bitdepth = 10                   # Bit depth of the decoded YUV sequence
qp_list = [22, 27, 32, 37]           # List of Quantization Parameters (QPs) for which the sequence was coded

sequence = "dummy"                   # Name of the video sequence
size = (416, 240)                    # (width, height) of the YUV sequence

max_block_exp = 5                    # Maximum rectangular block size exponent (e.g. 5 = max block size of 32x32)
kernel_size = 13                     # Combined size of convolutional kernels (e.g. 9x9, 1x1, 5x5 = 13x13)

dataset_path = "dataset"             # Directory where the dataset will be saved
