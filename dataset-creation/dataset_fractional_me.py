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
import numpy as np
from silx.io.dictdump import dicttoh5
from copy import deepcopy
import argparse
from utils import VideoYUV, import_path, frac_positions, block_sizes, interp_filtering


def write_dataset(path, seq, seq_qp, seq_inputs, seq_labels, seq_sad):
    """
    Write dictionaries of inputs, labels, SAD loss values into an h5py dataset
    :param path: dataset directory
    :param seq: name of the video sequence used for creating the dataset
    :param seq_qp: Quantization Parameter (QP) of the compressed video sequence
    :param seq_inputs: dictionary of rectangular reference (coded) input blocks,
                       grouped by fractional positions and block sizes
    :param seq_labels: dictionary of rectangular current (to-be-coded) input blocks,
                       grouped by fractional positions and block sizes
    :param seq_sad: dictionary of Sum of Absolute Differences (SAD) losses between VVC coded blocks and to-be-coded
                    blocks, grouped by fractional positions and block sizes
    """
    os.makedirs(path, exist_ok=True)
    filename = f"{seq}_{seq_qp}.hdf5"

    create_ds_args = {'compression': "gzip", 'shuffle': True, 'fletcher32': True}
    dicttoh5(seq_inputs, os.path.join(path, filename), "inputs", create_dataset_args=create_ds_args)
    dicttoh5(seq_labels, os.path.join(path, filename), "labels", "a", create_dataset_args=create_ds_args)
    dicttoh5(seq_sad, os.path.join(path, filename), "sad_loss", "a", create_dataset_args=create_ds_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating a Dataset for CNN Subpixel Motion Interpolation')
    parser.add_argument("-c", "--config", type=str, default=None, help="Config filepath")
    args = parser.parse_args()

    # import config file details
    data_cfg = import_path(args.config)

    # open original (uncompressed) YUV video sequence
    path_to_sequence = os.path.join(data_cfg.experiment_path, data_cfg.sequence)
    orig = VideoYUV(os.path.join(path_to_sequence, "original.yuv"), data_cfg.size[0], data_cfg.size[1],
                    data_cfg.orig_bitdepth)

    # create datasets per specified QPs
    for qp in data_cfg.qp_list:
        # create dictionaries grouped by block sizes (up to 2^max_block_exp x 2^max_block_exp)
        # and quarter-pixel fractional positions
        frac_dict = {key: [] for key in frac_positions()}
        block_key = block_sizes(data_cfg.max_block_exp)
        inputs, labels, sad_loss = ({key: deepcopy(frac_dict) for key in block_key} for i in range(3))

        # open the decoded YUV video sequence and filter the encoder log file for details on fractional blocks
        deco_buffer, orig_buffer = ([] for i in range(2))
        deco = VideoYUV(os.path.join(path_to_sequence, data_cfg.encoder_cfg, f"decoded_{qp}.yuv"),
                        data_cfg.size[0], data_cfg.size[1], data_cfg.deco_bitdepth)

        encoder_log = os.path.join(path_to_sequence, data_cfg.encoder_cfg, f"encoder_{qp}.log")
        encoder_lines = list(filter(lambda l: l[:1].isdigit(), [line.rstrip('\n') for line in open(encoder_log)]))

        while 1:
            # put all frames in buffer arrays
            deco_ret, deco_frame = deco.read(data_cfg.deco_bitdepth)
            orig_ret, orig_frame = orig.read(data_cfg.deco_bitdepth)

            orig_buffer.append(orig_frame)

            # repetitive padding on edges of decoded frames, as per VVC conventions
            half = int(data_cfg.kernel_size / 2)
            ref = np.pad(deco_frame, (half, half), 'edge')
            deco_buffer.append(ref)

            # after the whole sequence is read
            if not deco_ret and orig_ret:
                deco.close()
                orig.close()

                # read log file lines that are organised as:
                # current POC, top-left coordinates, block dimensions, reference POC, integer MVs, fractional MVs
                for lines in encoder_lines:
                    top_left, block, mv, frac_mv = ([None] * 2 for i in range(4))
                    poc, top_left[0], top_left[1], block[0], block[1], ref_poc, mv[0], mv[1], frac_mv[0], frac_mv[1] = \
                        [int(x) for x in lines.split(" ")]

                    block_string = f"{block[1]}x{block[0]}"
                    frac_string = f"{frac_mv[0]},{frac_mv[1]}"

                    if block_string not in block_key or frac_string not in frac_dict:
                        continue

                    original = orig_buffer[poc]
                    decoded = deco_buffer[ref_poc]

                    # find the reference block in the decoded frame and add it to the input dictionary
                    x = top_left[0] + mv[0]
                    if x < 0 or x > (data_cfg.size[0] - (block[0] + 1)):
                        continue
                    y = top_left[1] + mv[1]
                    if y < 0 or y > (data_cfg.size[1] - (block[1] + 1)):
                        continue
                    # pad the block by specified kernel size
                    crop_padded = decoded[y:y + block[1] + data_cfg.kernel_size - 1,
                                          x:x + block[0] + data_cfg.kernel_size - 1]
                    crop_padded = crop_padded[:, :, np.newaxis].astype(np.int16)

                    inputs[block_string][frac_string].append(crop_padded)

                    # find the current block in the original frame and add it to the label dictionary
                    x, y = top_left[0], top_left[1]
                    crop_orig = original[y:y + block[1], x:x + block[0]]
                    crop_orig = crop_orig[:, :, np.newaxis].astype(np.int16)

                    labels[block_string][frac_string].append(crop_orig)

                    # interpolate the original block with VVC filters and compute the SAD loss
                    vvc_label = interp_filtering(crop_padded, data_cfg.kernel_size, frac_mv[0], frac_mv[1])

                    sad_loss[block_string][frac_string].append(np.mean(np.abs(crop_orig - vvc_label)))

                # clear the buffers and exit the while loop
                deco_buffer.clear()
                orig_buffer.clear()
                break

        # open and fill the hdf5 file with inputs, labels, sad_losses
        dataset_dir = os.path.join(data_cfg.dataset_path, f"{data_cfg.sequence}-{data_cfg.encoder_cfg}")
        write_dataset(dataset_dir, data_cfg.sequence, qp, inputs, labels, sad_loss)
