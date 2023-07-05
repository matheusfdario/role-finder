# -*- coding: utf-8 -*-
"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""

## Add path to library (just for examples; you do not need this)
#import initExample

from pyqtgraph.Qt import QtCore, QtGui
from framework import post_proc
import pyqtgraph.opengl as gl
import numpy as np


def distcenter(i, j, k):
    offset = (i.shape[0]/2, i.shape[1]/2, i.shape[2]/2)
    x = i - offset[0]
    y = j - offset[1]
    z = k - offset[2]
    return np.array(np.sqrt(x**2 + 2*y**2 + z**2))


def bola(i, j, k):
    offset = (i.shape[0]/2, i.shape[1]/2, i.shape[2]/2)
    x = i - offset[0]
    y = j - offset[1]
    z = k - offset[2]
    dist = np.array(np.sqrt(x**2 + 2*y**2 + z**2))
    max_dist = np.max(dist)
    return max_dist - dist

def apply_threshold(data, threshold):
    return np.array(data < threshold, np.float)

def leq_threshold(data, threshold):
    return np.array(data >= threshold, np.float)

def leq_threshold_db(data, threshold):
    threshold = -np.abs(threshold)
    normalized = np.abs(data) / np.max(np.abs(data))
    normalized_log = 10*np.log10(normalized + 1e-20)
    return np.array(normalized_log >= threshold, np.float)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.setBackgroundColor('w')
w.show()
w.setWindowTitle('TFM3D')


#data = np.fromfunction(bola, (100, 100, 100))
data = np.load('result_tfm.npy')
data = data[0]
data_log = 10*np.log10(np.abs(data) + 1e-20)
data_th = leq_threshold_db(data, 9)

d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = 255
d2[..., 1] = 0
d2[..., 2] = 255
d2[..., 3] = data_th*128
#d2[..., 3] = np.abs(data_th)*255




size_box = QtGui.QVector3D(data.shape[0], data.shape[1], data.shape[2])
b = gl.GLBoxItem(size=size_box)
b.translate(-size_box[0]/2, -size_box[1]/2, -size_box[2]/2)
b.setColor('k')
w.addItem(b)

v = gl.GLVolumeItem(d2, smooth=False, sliceDensity=3, glOptions='translucent')
v.translate(-size_box[0]/2, -size_box[1]/2, -size_box[2]/2)
w.addItem(v)


inst = QtGui.QApplication.instance()
inst.exec()