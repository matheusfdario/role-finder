import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from framework import data_types, file_civa, file_mat
from framework.post_proc import envelope, normalize, api
from framework.data_types import ImagingROI
from imaging import wk_saft, e_wk_saft
import time

# --- Input ---
# File
data = file_mat.read('../data/DadosEnsaio.mat')

# Parameters
#cl = 5900
cl = 5859.4
data.probe_params.central_freq = 4.6
data.probe_params.bw = 4.4 / 4.6
data.inspection_params.cl = cl

# ROI
height = 20.0
width = 30.0
corner_roi = np.array([[0.0, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width)

print('\nRunning wk-SAFT...')
_ti = time.time()
wk_saft_key = wk_saft.wk_saft_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} ms'.format((_tf - _ti) / 1e-3))

print('\nRunning E-wk-SAFT...')
_ti = time.time()
e_wk_saft_key = e_wk_saft.e_wk_saft_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} ms'.format((_tf - _ti) / 1e-3))

i_wk_saft = data.imaging_results[wk_saft_key].image
i_e_wk_saft = data.imaging_results[e_wk_saft_key].image

# --- Output ---
plt.ion()

xr_i = roi.w_points[0]; xr_f = roi.w_points[-1]
zr_i = roi.h_points[0]; zr_f = roi.h_points[-1]
extent = [xr_i, xr_f, zr_f, zr_i]

plt.figure()
ax = plt.subplot(1, 2, 1)
plt.title('Wavenumber')
plt.imshow(normalize(envelope(i_wk_saft, 0)), aspect='auto', extent=extent)

plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
plt.title('E-wavenumber')
plt.imshow(normalize(envelope(i_e_wk_saft, 0)), aspect='auto', extent=extent)

