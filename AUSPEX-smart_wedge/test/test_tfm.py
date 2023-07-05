import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfmcuda as tfmc
from imaging import tfm, cumulative_tfm
from surface.surface import Surface, SurfaceType


data = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/tubo_acrilico_com_dentes_FMC.m2k', freq_transd=5,
                     bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
data = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/ensaios_23_09/peca_dentada.m2k', 5, 0.5, 'gaussian', sel_shots=[0, 1])
L = data.ascan_data.shape[-1]
_ = pre_proc.hilbert_transforms(data, np.arange(L))
corner_roi = np.array([-9.0, 0.0, 0.0])[np.newaxis, :]
size = np.array([40, 20])
res = np.array([0.05, 0.05])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-20, 0, 10]
roi = ImagingROI(corner_roi, height=size[1], width=size[0], h_len=shape[1], w_len=shape[0], depth=1.0, d_len=1)

# data.inspection_params.type_insp = 'contact'
# data.specimen_params.cl = data.inspection_params.coupling_cl
# c = data.inspection_params.coupling_cl

# data.surf = Surface(data)
# data.surf.fit(SurfaceType.CIRCLE_NEWTON)

t0 = time.time()
chave0 = tfmc.tfmcuda_kernel(data, roi=roi)
tg = time.time()-t0
print(tg)

t0 = time.time()
chave2 = cumulative_tfm.cumulative_tfm_kernel(data, roi=roi, sel_shots=np.arange(L))
tg = time.time()-t0
print(tg)
# t0 = time.time()
# chave2 = tfm.tfm_kernel(data, roi=roi, c=c)
# tc = time.time()-t0
# print(tc)
# #
result0 = post_proc.normalize(post_proc.envelope(data.imaging_results[chave0].image, -2))
result2 = post_proc.normalize(post_proc.envelope(data.imaging_results[chave2].image, -2))
# _ = pre_proc.hilbert_transforms(data)

# t0 = time.time()
# chave1 = tfmc.tfmcuda_kernel(data, roi=roi)
# tga = time.time()-t0
# print(tga)

# t0 = time.time()
# chave3 = tfm.tfm_kernel(data, roi=roi, c=c)
# tca = time.time()-t0
# print(tca)
#
# result1 = post_proc.normalize(post_proc.envelope(data.imaging_results[chave1].image, -2))
# result3 = post_proc.normalize(post_proc.envelope(data.imaging_results[chave3].image, -2))
#
# plt.figure()
# plt.imshow(result0, aspect='auto',
#            extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
# plt.title(f'TFM {roi.w_len}x{roi.h_len}pixels em CUDA - {tg:.4f}s')
#
# plt.figure()
plt.imshow(result2, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
# plt.title(f'TFM {roi.w_len}x{roi.h_len}pixels na toolbox - {tc:.4f}s')
# plt.show()
#
# plt.figure()
# plt.imshow(result1, aspect='auto',
#            extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
# plt.title(f'Analytical Signal TFM {roi.w_len}x{roi.h_len}pixels em CUDA - {tga:.4f}s')
#
# plt.figure()
# plt.imshow(result3, aspect='auto',
#            extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
# plt.title(f'Analytical Signal TFM {roi.w_len}x{roi.h_len}pixels na toolbox - {tca:.4f}s')
# plt.show()
