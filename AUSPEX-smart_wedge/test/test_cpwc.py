import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from framework.post_proc import envelope, normalize
from imaging import cpwc, tfm
import time
from surface.surface import Surface, SurfaceType, Lineparam

# --- Input ---
# File
data = file_civa.read('D:/2021-03-16 b/peca_80_60_25_ensaio_pw_immersion_5ang_20mm.civa')

# --- ROI ---
# Tamanho da ROI
height = 20.0
width = 35.0

# Define a ROI
corner_roi = np.array([[-13, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
wpath = data.inspection_params.water_path
data.surf = Surface(data, -1, surf_type=SurfaceType.LINE_NEWTON,
               surf_param=Lineparam(a=1e-8, b=wpath, SSE=1e-9))
data.surf.fit(surf_type=SurfaceType.LINE_NEWTON, surf_param=Lineparam(a=1e-8, b=wpath, SSE=1e-9))

_ti = time.time()
cpwc_key = cpwc.cpwc_kernel(data, roi)
_tf = time.time()
print(_tf - _ti)
ic = data.imaging_results[cpwc_key].image

# --- Output ---
plt.ion()
plt.figure()
plt.imshow(normalize(envelope(ic, -2)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])

