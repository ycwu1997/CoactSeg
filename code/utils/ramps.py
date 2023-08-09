# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np
import torch

def norm_image(image, ep = 1e-8):
    range = torch.max(image)-torch.min(image)
    if range == 0:
        norm_image = image
    else:
        norm_image = (image-torch.min(image))/(range+ep)
    return norm_image

def get_imgs(p_first, p_second, p_new, volume_batch, label_batch, sample_index):
    ins_width = 2
    num_colums = 7
    B,C,H,W,D = p_first.size()
    snapshot_img = torch.zeros(size = (D, 3, num_colums * H + num_colums * ins_width, W + ins_width), dtype = torch.float32)

    for icol in range(1, num_colums+1):
        snapshot_img[:,:, icol*(H+ins_width)-ins_width:icol*(H+ins_width),:] = 1
    snapshot_img[:,:, :,W:W+ins_width] = 1

    seg_out_1 = p_first[sample_index,0].permute(2,0,1).cpu().data
    seg_out_2 = p_second[sample_index,0].permute(2,0,1).cpu().data
    seg_out_3 = p_new[sample_index,0].permute(2,0,1).cpu().data

    target =  label_batch[sample_index].permute(2,0,1).cpu().data

    train_img_1 = volume_batch[sample_index,0].permute(2,0,1).cpu().data
    train_img_2 = volume_batch[sample_index,1].permute(2,0,1).cpu().data
    
    for i_rgb in range(3):
        snapshot_img[:, i_rgb,:H,:W] = norm_image(train_img_1)
        snapshot_img[:, i_rgb, H+ ins_width:2*H+ ins_width,:W] = norm_image(train_img_2)
        snapshot_img[:, i_rgb, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = norm_image(train_img_2-train_img_1)
        snapshot_img[:, i_rgb, 3*H+ 3*ins_width:4*H+ 3*ins_width,:W] = seg_out_1
        snapshot_img[:, i_rgb, 4*H+ 4*ins_width:5*H+ 4*ins_width,:W] = seg_out_2
        snapshot_img[:, i_rgb, 5*H+ 5*ins_width:6*H+ 5*ins_width,:W] = seg_out_3
        snapshot_img[:, i_rgb, 6*H+ 6*ins_width:7*H+ 6*ins_width,:W] = target

    return snapshot_img

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
