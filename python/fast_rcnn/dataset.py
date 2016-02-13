#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np
from sklearn.datasets.base import Bunch


this_dir = osp.dirname(osp.abspath(__file__))


def get_data_dir():
    return osp.realpath(osp.join(this_dir, '../../data'))


def mask_to_roi(mask):
    where = np.argwhere(mask)
    (ymin, xmin), (yend, xend) = where.min(0), where.max(0) + 1
    return ymin, xmin, yend, xend


def im_to_blob(im):
    """Convert image to blob.
    @param im: its shape is (height, width, channel)
    @type im: numpy.ndarray
    """
    blob = im.transpose((2, 0, 1))
    blob = blob.astype(np.float32)
    blob /= 255.
    return blob


def blob_to_im(blob):
    """Convert blob to image.
    @param blob: its shape is (channel, height, width)
    @type blob: numpy.ndarray
    """
    im = blob * 255.
    im = im.transpose((1, 2, 0))
    im = im.astype(np.uint8)
    return im


def load_APC2015berkeley():
    data_dir = osp.join(get_data_dir(), 'APC2015berkeley')
    target_names = os.listdir(data_dir)
    for n in target_names:
        # skip 'masks' dir
        obj_data_dir = os.path.join(data_dir, n)
        im_files = [os.path.join(obj_data_dir, f)
                    for f in os.listdir(obj_data_dir) if f != 'masks']
        N = len(im_files)
        H = 4272 // 4
        W = 2848 // 4
        blobs = np.zeros((N, 3, H, W), dtype=np.float32)
        rois = np.zeros((N, 4), dtype=np.int32)
        for i, f in enumerate(im_files):
            # generate blob
            im = cv2.imread(f)
            im = cv2.resize(im, (W, H))
            blob = im_to_blob(im)
            blobs[i] = blob
            # generate roi
            dirname, basefile = osp.split(f)
            base, _ = osp.splitext(basefile)
            mask_file = osp.join(dirname, 'masks', base + '_mask.jpg')
            mask = cv2.imread(mask_file, 0)
            roi = mask_to_roi(mask)
            rois[i] = roi
    dataset = Bunch(
        target_names=target_names,
        blobs=blobs,
        rois=rois,
    )
    return dataset
