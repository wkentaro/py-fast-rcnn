#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from fast_rcnn.dataset import load_APC2015berkeley
from fast_rcnn.dataset import load_batch_APC2015berkeley


def main():
    print('Loading APC2015berkeley...')
    dataset = load_APC2015berkeley()
    N = len(dataset.target)
    print('''
N: {0}
dataset: {1}'''.format(N, dataset))
    bg_label = dataset.target_names.index('__background__')
    n_labels = len(dataset.target_names)
    max_batch_size = 10
    for i in xrange(0, N, max_batch_size):
        t_start = time.time()

        fname_batch = dataset.filenames[i:i+max_batch_size]
        label_batch = dataset.target[i:i+max_batch_size]
        blob_batch, bbox_batch, label_batch, roi_delta_batch =\
            load_batch_APC2015berkeley(fname_batch, label_batch,
                                       bg_label, n_labels)

        # show stats
        elapsed_time = time.time() - t_start
        head_fname_batch = ['/'.join(f.split('/')[-2:]) for f in fname_batch]
        print('''
elapsed_time: {0} [s]
fnames: {1}
blob: {2}
bboxes: {3}
label: {4}
roi_delta: {5}'''.format(elapsed_time, head_fname_batch,
                         blob_batch.shape, bbox_batch.shape,
                         label_batch.shape, roi_delta_batch.shape))


if __name__ == '__main__':
    main()
