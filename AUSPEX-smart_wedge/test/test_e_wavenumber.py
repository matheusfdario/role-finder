import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from framework.post_proc import envelope, normalize
from imaging import e_wavenumber, wavenumber
from guiqt.Utils import Cmaps
import time

# --- Input ---
# File
#data = file_civa.read('../data/peca_80_60_25_ensaio_pw_validation.civa')
data = file_civa.read('/media/marco/hd/Software/VirtualBox/shared/civa/Dados_Sim/peca_80_60_25_ensaio_fmc_validation.civa')

# --- ROI ---
# Tamanho da ROI
height = 20.0
width = 2*9.45

# Define a ROI, iniciando em (-9.45, 0, 30) e com as dimensões definidas acima.
corner_roi = np.array([[-9.45, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=400, w_len=200)

print('\nRunning Wavenumber...')
_ti = time.time()
wavenumber_key = wavenumber.wavenumber_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} s'.format(_tf - _ti))

print('\nRunning E-wavenumber...')
_ti = time.time()
ewavenumber_key = e_wavenumber.e_wavenumber_kernel(data, roi)
_tf = time.time()
print('Run time: {:.2f} s'.format(_tf - _ti))

i_ewavenumber = data.imaging_results[ewavenumber_key].image
i_wavenumber = data.imaging_results[wavenumber_key].image

# --- Output ---
plt.ion()

xr_i = roi.w_points[0]; xr_f = roi.w_points[-1]
zr_i = roi.h_points[0]; zr_f = roi.h_points[-1]
extent = [xr_i, xr_f, zr_f, zr_i]
cmap = matplotlib.colors.ListedColormap(Cmaps.civa/255)

plt.figure()
ax = plt.subplot(1, 2, 1)
plt.title('Wavenumber')
plt.imshow(normalize(envelope(i_wavenumber, 0)), aspect='auto', extent=extent, cmap=cmap, vmin=-1, vmax=1)

plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
plt.title('E-wavenumber')
plt.imshow(normalize(envelope(i_ewavenumber, 0)), aspect='auto', extent=extent, cmap=cmap, vmin=-1, vmax=1)
