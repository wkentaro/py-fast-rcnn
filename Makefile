all:
	(cd python && python setup.py develop --user && rm -rf build)
	(cd chainer-fast-rcnn && python setup.py develop --user)

load_net:
	(cd data/scripts && bash fetch_fast_rcnn_models.sh)
	(python tools/load_net.py)

fetch_chainer_models:
	(cd data/scripts && bash fetch_chainer_models.sh)
