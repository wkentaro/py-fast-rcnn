#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fast_rcnn
from fast_rcnn.models.vgg_cnn_m_1024 import VGG_CNN_M_1024
from sklearn.cross_validation import train_test_split


class FastRCNNTrainer(object):

    def __init__(self, model, dataset):
        (self.train_blobs, self.test_blobs,
         self.train_labels, self.test_labels,
         self.train_rois, self.test_rois) =\
            train_test_split(dataset.blobs, dataset.labels, dataset.rois,
                             test_size=.1)

    def batch_loop(self, batch_size):
        N = len(self.train_blobs)
        for i in xrange(0, N, batch_size):
            blob_batch = self.blobs[i:i+batch_size]
            # TODO(wkentaro): pass label data
            label_batch = self.labels[i:i+batch_size]
            roi_batch = self.rois[i:i+batch_size]

            inputs = ()  # TODO(wkentaro): pass inputs
            loss = self.model(*inputs)
            loss.backward()

    def main_loop(self, epoch_size):
        for epoch in xrange(1, epoch_size):
            self.batch_loop(batch_size)


def main():
    model = VGG_CNN_M_1024()
    model.to_gpu()

    dataset = fast_rcnn.dataset.load_APC2015berkeley()

    trainer = Trainer(model, dataset=dataset, batch_size=10)
    trainer.main_loop()


if __name__ == '__main__':
    main()
