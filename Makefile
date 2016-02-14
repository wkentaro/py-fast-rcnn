all:
	(cd python && python setup.py develop --user && rm -rf build)
	(cd chainer-fast-rcnn && python setup.py develop --user)

load_net_rcnn:
	(cd data/scripts && bash fetch_fast_rcnn_models.sh)
	(python tools/load_net_rcnn.py)

fetch_chainer_models:
	(cd data/scripts && bash fetch_chainer_models.sh)

load_net_imagenet:
	(cd data/scripts && bash fetch_imagenet_models.sh)
	(python tools/load_net_imagenet.py)

preprocess_APC2015berkeley:
	(cd tools && python preprocess_APC2015berkeley.py)
