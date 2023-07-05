import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from framework.post_proc import envelope
from imaging import cpwc
import time
from surface.surface import Surface, SurfaceType, Lineparam



# --- ROI ---
# Tamanho da ROI
height = 20.0
width = 35.0

plt.ion()
plt.figure()

ic1 = np.load('../contact_TFM.npy')
ic2 = np.load('../10mm_TFM.npy')
ic3 = np.load('../20mm_TFM.npy')
ic4 = np.load('../contact.npy')
ic5 = np.load('../10mm.npy')
ic6 = np.load('../20mm.npy')

allimg = [envelope(ic1, -2), envelope(ic2, -2),
          envelope(ic3, -2), envelope(ic4, -2),
          envelope(ic5, -2), envelope(ic6, -2)]

def normalize(img):
    return img / np.max(allimg)

plt.subplot(231)
corner_roi = np.array([[-13, 0.0, 10.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic1, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('contact' + ' (max='+"{:.4f}".format(np.max(envelope(ic1, -2)))+')')

plt.subplot(232)
corner_roi = np.array([[-13, 0.0, 20.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic2, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('10mm water path' + ' (max='+"{:.4f}".format(np.max(envelope(ic2, -2)))+')')

plt.subplot(233)
corner_roi = np.array([[-13, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic3, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('20mm water path' + ' (max='+"{:.4f}".format(np.max(envelope(ic3, -2)))+')')

plt.subplot(234)
corner_roi = np.array([[-13, 0.0, 10.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic4, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('contact' + ' (max='+"{:.4f}".format(np.max(envelope(ic4, -2)))+')')

plt.subplot(235)
corner_roi = np.array([[-13, 0.0, 20.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic5, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('10mm water path' + ' (max='+"{:.4f}".format(np.max(envelope(ic5, -2)))+')')

plt.subplot(236)
corner_roi = np.array([[-13, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)
plt.imshow(normalize(envelope(ic6, -2)), aspect='auto', vmin=0, vmax=1,
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('20mm water path' + ' (max='+"{:.4f}".format(np.max(envelope(ic6, -2)))+')')

