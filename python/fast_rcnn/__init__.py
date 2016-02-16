#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp


__version__ = '0.1'


_this_dir = osp.dirname(osp.abspath(__file__))


def get_root_dir():
    return osp.realpath(osp.join(_this_dir, '../..'))
