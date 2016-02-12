#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer import Variable


class VGG_CNN_M_1024(chainer.Chain):

    def __init__(self):
        super(VGG_CNN_M_1024, self).__init__(
            conv1=F.Convolution2D(3, 96, ksize=7, stride=2),
            conv2=F.Convolution2D(96, 256, ksize=5, stride=2, pad=1),
            conv3=F.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            conv4=F.Convolution2D(512, 512, ksize=3, stride=1, pad=1),
            conv5=F.Convolution2D(512, 512, ksize=3, stride=1, pad=1),
            fc6=F.Linear(4608, 4096),
            fc7=F.Linear(4096, 1024),
            cls_score=F.Linear(1024, 21),
            bbox_pred=F.Linear(1024, 84)
        )

    def __call__(self, x, rois, t=None):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.local_response_normalization(h, n=5, k=2, alpha=5e-4, beta=.75)
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = self.conv2(h)
        h = F.relu(h)
        h = F.local_response_normalization(h, n=5, k=2, alpha=5e-4, beta=.75)
        h = F.max_pooling_2d(h, ksize=3, stride=2)

        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.conv4(h)
        h = F.relu(h)

        h = self.conv5(h)
        h = F.relu(h)

        h = F.roi_pooling_2d(h, rois, outh=6, outw=6, spatial_scale=0.0625)

        h = self.fc6(h)
        h = F.relu(h)
        h = F.dropout(h, train=self.train, ratio=.5)

        h = self.fc7(h)
        h = F.relu(h)
        h = F.dropout(h, train=self.train, ratio=.5)

        h_cls_score = self.cls_score(h)
        cls_score = F.softmax(h_cls_score)
        bbox_pred = self.bbox_pred(h)

        if t is None:
            return cls_score, bbox_pred

        t_cls, t_bbox = t
        self.cls_loss = F.softmax_cross_entropy(h_cls_score, t_cls)
        self.bbox_loss = F.smooth_l1_loss(bbox_pred, t_bbox)
        return self.cls_loss, self.bbox_loss
