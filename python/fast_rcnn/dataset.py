#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os
import os.path as osp
import cPickle as pickle

import cv2
import dlib
import numpy as np
from sklearn.datasets import load_files


this_dir = osp.dirname(osp.abspath(__file__))


def get_data_dir():
    return osp.realpath(osp.join(this_dir, '../../data'))


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


def mask_to_roi(mask, im_scale):
    where = np.argwhere(mask)
    (ymin, xmin), (yend, xend) = where.min(0), where.max(0) + 1
    roi = np.array([xmin, ymin, xend, yend], dtype=np.float32)
    roi *= im_scale
    return roi


def img_preprocessing(orig_img, pixel_means=None, max_size=1000, scale=600):
    if pixel_means is None:
        pixel_means = np.array([102.9801, 115.9465, 122.7717],
                               dtype=np.float32)

    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.rint(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def get_bboxes(orig_img, im_scale, min_size=500, dedup_boxes=1. / 16):
    img = cv2.resize(orig_img, None, None, fx=im_scale, fy=im_scale,
                     interpolation=cv2.INTER_LINEAR)
    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=min_size)
    rects = [[0, d.left(), d.top(), d.right(), d.bottom()] for d in rects]
    rects = np.asarray(rects, dtype=np.float32)
    # bbox pre-processing
    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    hashes = np.round(rects * dedup_boxes).dot(v)
    _, index, inv_index = np.unique(hashes, return_index=True,
                                    return_inverse=True)
    rects = rects[index, :]
    return rects


def get_region_targets(roi, bboxes, label, bg_label, overlap_thresh=.5):
    from fast_rcnn.utils.cython_bbox import bbox_overlaps
    overlaps = bbox_overlaps(roi[np.newaxis, :], bboxes)
    N = len(overlaps[0])
    labels = np.zeros(N).astype(np.int32)
    roi_deltas = np.zeros((N, 4)).astype(np.float32)
    for i, lap in enumerate(overlaps[0]):
        labels[i] = bg_label if lap < overlap_thresh else label

        def bbox_stat(bbox):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + 0.5 * width
            center_y = y1 + 0.5 * height
            return width, height, center_x, center_y

        roi_width, roi_height, roi_center_x, roi_center_y = bbox_stat(roi)

        bbox = bboxes[i]
        bbox_width, bbox_height, bbox_center_x, bbox_center_y = bbox_stat(bbox)

        dx = (bbox_center_x - roi_center_x) / roi_width
        dy = (bbox_center_y - roi_center_y) / roi_height
        dw = np.log(bbox_width / roi_width)
        dh = np.log(bbox_height / roi_height)
        roi_deltas[i] = [dx, dy, dw, dh]
    return labels, roi_deltas


def load_batch_APC2015berkeley(labels, bg_label, fnames):
    """Get blob and bboxes from image, roi from mask."""
    blob_batch, bbox_batch, label_batch, roi_delta_batch = [], [], [], []
    i_batch = 0
    for label, fname in zip(labels, fnames):
        path_to, im_file_base = osp.split(fname)

        # load cache file if exists
        label_name = osp.split(path_to)[1]
        cache_fname = osp.join(
            get_data_dir(), 'APC2015berkeley_cache',
            label_name, osp.splitext(im_file_base)[0] + '.pkl')
        if osp.exists(cache_fname):
            # load cache
            with open(cache_fname, 'rb') as f:
                blob, bboxes, labels, roi_deltas = pickle.load(f)
        else:
            orig_im = cv2.imread(fname)
            if im_file_base.startswith('NP'):  # resize NPXXX file
                _, w = orig_im.shape[:2]
                h = int(w / 1.5)
                orig_im = orig_im[:h, :]
            blob, im_scale = img_preprocessing(orig_im)
            # get bboxes
            bboxes = get_bboxes(orig_im, im_scale)
            bboxes[:, 0].fill(i_batch)
            # get rois
            mask_file = osp.join(
                path_to, 'masks', osp.splitext(im_file_base)[0] + '_mask.jpg')
            mask = cv2.imread(mask_file, 0)
            if im_file_base.startswith('NP'):  # resize NPXXX file
                _, w = mask.shape
                h = int(w / 1.5)
                mask = mask[:h, :]
            roi = mask_to_roi(mask, im_scale)
            # get each region labels and roi_deltas
            labels, roi_deltas = get_region_targets(
                roi.astype(np.float), bboxes[:, 1:].astype(np.float),
                label, bg_label)
            # save cache
            if not osp.exists(osp.dirname(cache_fname)):
                os.makedirs(osp.dirname(cache_fname))
            with open(cache_fname, 'wb') as f:
                pickle.dump((blob, bboxes, labels, roi_deltas), f)

        blob_batch.append(blob)
        bbox_batch.append(bboxes)
        label_batch.extend(labels.tolist())
        roi_delta_batch.append(roi_deltas)
        i_batch += 1
    blob_batch = np.array(blob_batch)
    bbox_batch = np.vstack(bbox_batch)
    label_batch = np.array(label_batch)
    roi_delta_batch = np.vstack(roi_delta_batch)
    return blob_batch, bbox_batch, label_batch, roi_delta_batch


def load_APC2015berkeley():
    data_dir = osp.join(get_data_dir(), 'APC2015berkeley')
    dataset = load_files(data_dir, description='APC2015berkeley',
                         load_content=False)
    dataset.target_names.append('__background__')
    return dataset
