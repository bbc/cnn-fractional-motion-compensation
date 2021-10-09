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

import argparse
import os
import numpy as np
from imageio import imwrite
from utils import import_path, frac_positions, VideoYUV


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze Decoder Statistics of the Switchable Interpolation Filter Set')
    parser.add_argument("-c", "--config", type=str, default=None, help="Config filepath")
    args = parser.parse_args()

    # import config file details
    stats_cfg = import_path(args.config)

    # open decoded YUV file with switchable filter implementation
    path_to_sequence = os.path.join(stats_cfg.experiments_dir, stats_cfg.sequence)
    deco = VideoYUV(os.path.join(path_to_sequence, stats_cfg.encoder_cfg, f"decoded-switchable_{stats_cfg.qp}.yuv"),
                    stats_cfg.size[0], stats_cfg.size[1], stats_cfg.deco_bitdepth)

    # put all frames in buffer array
    # convert grayscale to rgb and store values as 8-bit pixels
    deco_buffer = []
    while 1:
        deco_ret, deco_frame = deco.read(8)
        if not deco_ret:
            deco.close()
            break
        deco_buffer.append(np.repeat(deco_frame[:, :, np.newaxis], 3, axis=2))

    hit_mc = 0
    blue = np.array([0, 0, 255])
    red = np.array([255, 0, 0])

    # read log file and extract decoder statistics
    decoder_log = os.path.join(path_to_sequence, stats_cfg.encoder_cfg, f"decoder-switchable_{stats_cfg.qp}.log")
    hit_lines = list(filter(lambda l: l[:1].isdigit(), [line.rstrip('\n') for line in open(decoder_log)]))

    for line in hit_lines:
        frac_mv = [None] * 2
        poc, lx, ly, width, height, frac_mv[0], frac_mv[1], hit_status = [int(x) for x in line.split(" ")]
        frac_string = f"{frac_mv[0]},{frac_mv[1]}"

        if frac_string not in frac_positions():
            continue

        hit_mc += hit_status

        # draw red lines around NN filtered blocks and blue lines around VVC filtered blocks
        deco_buffer[poc][ly, lx:lx + width, :] = red if hit_status else blue
        deco_buffer[poc][ly + height - 1, lx:lx + width, :] = red if hit_status else blue
        deco_buffer[poc][ly:ly + height, lx, :] = red if hit_status else blue
        deco_buffer[poc][ly:ly + height, lx + width - 1, :] = red if hit_status else blue

    # write frames and hit ratio statistics to results directory
    results_subdir = os.path.join("decoder-stats", f"{stats_cfg.sequence}-{stats_cfg.encoder_cfg}")
    os.makedirs(os.path.join(stats_cfg.results_dir, results_subdir, "frames"), exist_ok=True)

    for i, frame in enumerate(deco_buffer):
        imwrite(os.path.join(stats_cfg.results_dir, results_subdir, "frames", f"{i}.png"), deco_buffer[i])
    with open(os.path.join(stats_cfg.results_dir, results_subdir, "hit_ratio.txt"), 'a') as out_file:
        out_file.write(f"The overall hit ratio is: {hit_mc/len(hit_lines)*100:.2f} %\n")
