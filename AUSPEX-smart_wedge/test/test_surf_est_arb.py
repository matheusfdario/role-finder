import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc
from framework.data_types import ImagingROI
from imaging import tfm, cumulative_tfm
from framework.pre_proc import hilbert_transforms
from parameter_estimation import intsurf_estimation
from surface.surface import Surface, SurfaceType

def img_line(image):
    aux = np.argmax(image, 0)
    w = np.max(image, 0)
    a = np.asarray([aux, w])
    return a

shots = range(0, 4)
print('Loading File')
data = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/pecas_acrilico_old/peca11.civa', sel_shots=shots)
shots = np.asarray(shots)
print('Applying Hilbert Transform')
out = hilbert_transforms(data, shots, N=2)
data.inspection_params.type_insp = 'contact'
specimen_cl = data.specimen_params.cl
data.specimen_params.cl = data.inspection_params.coupling_cl

size = np.array([60, 10])
res = np.array([0.1, 0.1])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-30, 0, 80]
roi = ImagingROI(corner_roi, size[1], shape[1], size[0], shape[0])
print('Imaging')
t0 = time.time()
cumulative_tfm.cumulative_tfm_kernel(data, roi, output_key=0, shots=shots)
print(time.time()-t0)
eroi = data.imaging_results[0].roi
yn = data.imaging_results[0].image
# yt = yn[:, int(shape[0]/2):int(data.inspection_params.step_points[-1][0]/res[0]+shape[0]/2)+1]
yt = yn[:, 300:3301]
y = yt/yt.max()
# plt.imshow(y)
# Revertendo mudanças data_insp
data.inspection_params.type_insp = 'immersion'
data.specimen_params.cl = specimen_cl
# Surface Estimation
a = img_line(y)
z = eroi.h_points[a[0].astype(int)]
w = np.diag((a[1]))
d = intsurf_estimation.matrix_d2(z, 'mirrored')
lamb = 1/np.mean(np.abs(d@z))
rho = 1
print(f'Estimating Surface')
t0 = time.time()
zf, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb, x0=z, rho=rho, eta=.9, itmax=50, tol=1e-30)
print(time.time()-t0)

data.surf = Surface(data, surf_type=SurfaceType.ARBITRARY)
# w_points = eroi.w_points[int(shape[0]/2):int(data.inspection_params.step_points[-1][0]/res[0]+shape[0]/2)+1]
w_points = eroi.w_points[300:3301]

# Chama TFM pra gerar imagem da superfície + sdh
size2 = np.array([40, 20])
res2 = np.array([0.1, 0.1])
shape2 = (size2/res2).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-20, 0, 80]
roi = ImagingROI(corner_roi, size2[1], shape2[1], size2[0], shape2[0])

# interpola e corrige unidade da superfície estimada
shot = 7
x_interp = np.linspace(roi.w_points[0], roi.w_points[-1], 600)
z_interp = np.interp(x_interp+data.inspection_params.step_points[shot, 0], w_points, zf)
data.surf.z_discr = z_interp
data.surf.x_discr = x_interp


print('Imaging Surface+SDH')
t0 = time.time()
tfm.tfm_kernel(data, roi, output_key=0, sel_shot=shot)
print(time.time()-t0)
result = post_proc.envelope(data.imaging_results[0].image)

# Chama TFM pra gerar imagem do sdh
# size2 = np.array([4, 4])
# res2 = np.array([0.02, 0.02])
# shape2 = (size2/res2).astype(int)
# corner_roi2 = np.zeros((1, 3))
# corner_roi2[0] = [-10.5-2.5, 0, 92]
# roi2 = ImagingROI(corner_roi2, size2[1], shape2[1], size2[0], shape2[0])
# print('Imaging SDH')
# t0 = time.time()
# tfm.tfm_kernel(data, roi2, output_key=0, sel_shot=shot)
# print(time.time()-t0)
# result_sdh = post_proc.envelope(data.imaging_results[0].image)


fig = plt.figure()
# fig.add_subplot(1, 2, 1)
plt.imshow(result, extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]], origin='upper')
# fig.add_subplot(1, 2, 2)
# sdhlog = np.log(result_sdh/result_sdh.max())
# sdhlog[sdhlog < -3] = np.NaN
# plt.imshow(sdhlog, extent=[roi2.w_points[0], roi2.w_points[-1], roi2.h_points[-1], roi2.h_points[0]],
#            origin='upper')
