import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from framework.post_proc import envelope, normalize
from imaging import wk_cpwc
import time

# --- Input ---
# File
data = file_civa.read('../data/peca_80_60_25_ensaio_pw_validation.civa')

# --- ROI ---
# Tamanho da ROI
height = 20.0
width = 30.0

# Define a ROI
corner_roi = np.array([[-15.5, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=200, w_len=200)

_ti = time.time()
cpwc_key = wk_cpwc.wk_cpwc_kernel(data, roi)
_tf = time.time()
print(_tf - _ti)
ic = data.imaging_results[cpwc_key].image

# --- Output ---
plt.figure()
plt.imshow(normalize(envelope(ic, -2)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.show()
