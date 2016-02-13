#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext


cmdclass = {}
ext_modules = [
    Extension(
        "fast_rcnn.utils.cython_bbox",
        ["fast_rcnn/utils/bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    ),
    Extension(
        "fast_rcnn.utils.cython_nms",
        ["fast_rcnn/utils/nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
    )
]
cmdclass.update({'build_ext': build_ext})


setup(
    name='fast_rcnn',
    version=__import__('fast_rcnn').__version__,
    packages=find_packages(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/py-fast-rcnn',
    install_requires=open('requirements.txt').readlines(),
    license='MIT',
    keywords='machine-learning',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
)
