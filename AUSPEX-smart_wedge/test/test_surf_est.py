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

data = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/tubo_acrilico_com_dentes_FMC.m2k', 5, 0.5, 'gaussian', sel_shots=0)
# data = file_m2k.read('C:/Users/Rossato/Downloads/tubo_acrilico_com_dentes_FMC.m2k', 5, 0.5, 'gaussian', sel_shots=0)
shots = np.asarray(np.arange(0, data.ascan_data.shape[-1]))
out = hilbert_transforms(data, shots)

data.inspection_params.type_insp = 'contact'
specimen_cl = data.specimen_params.cl
data.specimen_params.cl = data.inspection_params.coupling_cl
data.surf = Surface(data)
size = np.array([23, 30])
res = np.array([0.1, 0.1])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-11, 0, 30]
roi = ImagingROI(corner_roi, size[1], shape[1], size[0], shape[0])
print('Imaging')
t0 = time.time()
cumulative_tfm.cumulative_tfm_kernel(data, roi, output_key=0, shots=shots)
print(time.time()-t0)
eroi = data.imaging_results[0].roi
yt = data.imaging_results[0].image
y = yt/yt.max()
# Revertendo mudanças data_insp
data.inspection_params.type_insp = 'immersion'
data.specimen_params.cl = specimen_cl
# Surface Estimation
a = img_line(y)
z = eroi.h_points[a[0].astype(int)]
w = np.diag((a[1]))
d = intsurf_estimation.matrix_d2(z, 'mirrored')
lamb = 1/np.mean(np.abs(d@z))
rho = 20
print(f'Estimating Surface')
t0 = time.time()
zf, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb, x0=z, rho=rho, eta=.9, itmax=50, tol=1e-30)
print(time.time()-t0)

# Rossato completa essa parte
data.surf = Surface(data)
# interpola e corrige unidade da superfície estimada
x_interp = np.linspace(eroi.w_points[0], eroi.w_points[-1], len(eroi.w_points)*1)
z_interp = np.interp(x_interp, eroi.w_points, zf)
data.surf.z_discr = z_interp
data.surf.x_discr = x_interp
data.surf.surfacetype = SurfaceType.ARBITRARY
# Chama TFM pra gerar imagem interna (talvez precise acertar a quantidade de shots e qual shot pega um furo)
size = np.array([23, 30])
res = np.array([0.1, 0.1])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-11, 0, 30]
roi = ImagingROI(corner_roi, size[1], shape[1], size[0], shape[0])
print('Imaging')
t0 = time.time()
tfm.tfm_kernel(data, roi, output_key=0, sel_shot=0)
print(time.time()-t0)
result = post_proc.envelope(data.imaging_results[0].image)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(result)
fig.add_subplot(1, 2, 2)
plt.imshow(result[100:, :])
plt.show()
