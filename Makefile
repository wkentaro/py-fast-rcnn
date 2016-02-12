all:
	(cd python && python setup.py develop --user)
	(cd chainer-fast-rcnn && python setup.py develop --user)
