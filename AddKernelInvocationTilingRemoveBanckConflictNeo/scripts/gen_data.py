#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data_simple():
    vector_len = 48 * 4096
    input_x = np.random.uniform(0, 0.5, vector_len).astype(np.float16)
    input_y = np.random.uniform(0, 0.5, vector_len).astype(np.float16)

    # product = input_x * input_y
    total_len = vector_len
    block_num = 48
    tile_num = 1
    block_len = total_len // block_num
    tiles_in_block = tile_num * 2
    tile_len = block_len // tiles_in_block
    tiles_all = block_num * tiles_in_block

    # product_reshape = product.reshape(tiles_all, tile_len)
    # tile_sum = product_reshape.sum(axis=1)
    # golden = np.zeros_like(product_reshape, dtype=np.float16)
    # golden[:, 0] = tile_sum

    # product = input_x * input_y
    # golden = np.zeros_like(input_x, dtype=np.float16)
    # golden = golden[0:16]
    # golden[0] = product.sum()
    golden = (input_x + input_y).astype(np.float16)
    
    #uint32_t totalLength;
    # uint32_t tileNum;
    # // uint32_t ma_method;
    # uint32_t element_size;
    # uint32_t gm_sync_size;
    element_size = 2
    # gm_sync_size = (block_num*32*block_num + block_num*32 + 32)*2
    # , element_size, 48*4
    tiling = np.array([vector_len, tile_num], dtype=np.uint32)
    tiling.tofile("./input/input_tiling.bin")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
