#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fast_rcnn.models.vgg_cnn_m_1024 import VGG_CNN_M_1024
import sys
# sys.path.insert(0, 'fast-rcnn/caffe-fast-rcnn/build/install/python')
import caffe
import cPickle as pickle

param_dir = 'data/fast_rcnn_models'
param_fn = '%s/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel' % param_dir
model_dir = 'python/fast_rcnn/models'
model_fn = '%s/test.prototxt' % model_dir

vgg = VGG_CNN_M_1024()
net = caffe.Net(model_fn, param_fn, caffe.TEST)
for name, param in net.params.iteritems():
    layer = getattr(vgg, name)

    print name, param[0].data.shape, param[1].data.shape,
    print layer.W.data.shape, layer.b.data.shape

    layer.W.data = param[0].data
    layer.b.data = param[1].data
    setattr(vgg, name, layer)

with open('data/chainer_models/vgg_cnn_m_1024.chainermodel', 'wb') as f:
    pickle.dump(vgg, f, -1)
