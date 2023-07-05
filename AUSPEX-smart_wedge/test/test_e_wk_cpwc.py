import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from framework.post_proc import envelope, normalize
from imaging import e_wk_cpwc, wk_cpwc
import time

# --- Input ---
# File
data = file_civa.read('../data/peca_80_60_25_ensaio_pw_validation.civa')

# --- ROI ---
# Tamanho da ROI
height = 40.0
width = 2*9.45

# Define a ROI, iniciando em (-9.45, 0, 30) e com as dimensões definidas acima.
corner_roi = np.array([[-9.45, 0.0, 10.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=200, w_len=200)

print('\nRunning wk-CPWC...')
_ti = time.time()
cpwc_key = wk_cpwc.wk_cpwc_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} s'.format(_tf - _ti))

print('\nRunning E-wk-CPWC...')
_ti = time.time()
e_cpwc_key = e_wk_cpwc.e_wk_cpwc_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} s'.format(_tf - _ti))

i_wk_cpwc = data.imaging_results[cpwc_key].image
i_e_wk_cpwc = data.imaging_results[e_cpwc_key].image

# --- Output ---
plt.ion()

xr_i = roi.w_points[0]; xr_f = roi.w_points[-1]
zr_i = roi.h_points[0]; zr_f = roi.h_points[-1]
extent = [xr_i, xr_f, zr_f, zr_i]

plt.figure()
ax = plt.subplot(1, 2, 1)
plt.title(r'$\omega k$-CPWC')
plt.imshow(normalize(envelope(i_wk_cpwc, 0)), aspect='auto', extent=extent)

plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
plt.title(r'E-$\omega k$-CPWC')
plt.imshow(normalize(envelope(i_e_wk_cpwc, 0
                              )), aspect='auto', extent=extent)
