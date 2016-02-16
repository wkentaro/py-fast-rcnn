#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

import caffe
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split

import fast_rcnn
from fast_rcnn.dataset import get_data_dir
from fast_rcnn.dataset import load_APC2015berkeley
from fast_rcnn.dataset import load_batch_APC2015berkeley
from fast_rcnn.models.vgg_cnn_m_1024 import VGG_CNN_M_1024


class FastRCNNTrainer(object):

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        (self.train_fnames, self.test_fnames,
         self.train_target, self.test_target) =\
            train_test_split(dataset.filenames, dataset.target, test_size=.1)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

        self.sum_loss = []

    def batch_loop_train(self, batch_size):
        N = len(self.train_fnames)
        random_index = np.random.randint(0, N, N)
        for i in xrange(0, N, batch_size):
            batch_index = random_index[i:i+batch_size]
            batch_fnames = self.train_fnames[batch_index]
            batch_labels = self.train_target[batch_index]

            blobs, bboxes, t_labels, t_bboxes = load_batch_APC2015berkeley(
                fnames=batch_fnames,
                labels=batch_labels,
                bg_label=self.dataset.background_label,
                n_labels=len(self.dataset.target_names),
            )

            blobs, bboxes, t_labels, t_bboxes =\
                map(cuda.to_gpu, [blobs, bboxes, t_labels, t_bboxes])

            self.optimizer.zero_grads()
            volatile = 'OFF'
            blobs = Variable(blobs, volatile=volatile)
            bboxes = Variable(bboxes, volatile=volatile)
            t_labels = Variable(t_labels, volatile=volatile)
            t_bboxes = Variable(t_bboxes, volatile=volatile)
            loss = self.model(blobs, bboxes, (t_labels, t_bboxes), train=True)
            loss.backward()
            self.optimizer.update()

    def batch_loop_test(self, batch_size):
        sum_loss = 0
        N = len(self.test_fnames)
        random_index = np.random.randint(0, N, N)
        for i in xrange(0, N, batch_size):
            batch_index = random_index[i:i+batch_size]
            batch_fnames = self.test_fnames[batch_index]
            batch_labels = self.test_target[batch_index]

            blobs, bboxes, t_labels, t_bboxes = load_batch_APC2015berkeley(
                fnames=batch_fnames,
                labels=batch_labels,
                bg_label=self.dataset.background_label,
                n_labels=len(self.dataset.target_names),
            )
            loss = self.model(blobs, bboxes, (t_labels, t_bboxes), train=False)
            sum_loss += loss.data
            print(loss.data / batch_size)

        self.sum_loss.append(sum_loss / N)

    def main_loop(self, batch_size, epoch_size):
        for epoch in xrange(1, epoch_size):
            if epoch > 0 and epoch % 10 == 0:
                subprocess.call('python tools/test.py', shell=True)
                self.batch_loop_test(batch_size)
                plt.plot(self.sum_loss)
                plt.xlabel('epoch [time]')
                plt.ylabel('loss')
                plt.savefig('loss.png')

            else:
                self.batch_loop_train(batch_size)


def load_pretrained_model(n_class):
    param_dir = '%s/imagenet_models' % get_data_dir()
    param_fn = '%s/VGG_CNN_M_1024.v2.caffemodel' % param_dir
    model_dir = '%s/python/fast_rcnn/models' % fast_rcnn.get_root_dir()
    model_fn = '%s/test.prototxt' % model_dir

    vgg = VGG_CNN_M_1024(n_class=n_class)
    net = caffe.Net(model_fn, param_fn, caffe.TEST)
    for name, param in net.params.iteritems():
        if 'conv' in name:
            layer = getattr(vgg, name)

            print name, param[0].data.shape, param[1].data.shape,
            print layer.W.data.shape, layer.b.data.shape

            layer.W.data = param[0].data
            layer.b.data = param[1].data
            setattr(vgg, name, layer)

    return vgg


def main():
    dataset = load_APC2015berkeley()

    n_class = len(dataset.target_names)
    model = load_pretrained_model(n_class=n_class)
    model.to_gpu()

    trainer = FastRCNNTrainer(model, dataset=dataset)
    trainer.main_loop(batch_size=10, epoch_size=10000)


if __name__ == '__main__':
    main()
