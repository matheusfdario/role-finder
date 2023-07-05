import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc
from framework.data_types import ImagingROI
from imaging import tfm, cumulative_tfm
from framework.pre_proc import hilbert_transforms
from parameter_estimation import intsurf_estimation
from surface.surface import Surface, SurfaceType
from framework.post_proc import envelope

amw = np.asarray([2.00, 5.11, 8.22, 11.33, 14.44, 17.56, 20.67, 23.78, 26.89, 30.00])
pos1 = np.array([7.24, 108.12, 67.58, 54.6, 21.62, 10.81, 6.76, 5.41, 4.32, 3.6, 2.7])
pos2 = np.cumsum(pos1)
sigma = np.arctan(2*np.pi*amw[:, np.newaxis]/pos1[np.newaxis, 1:])

def calc_sines(x):
    return -np.sin(np.linspace(0, 2*np.pi, x))

def peca_sines(x):
    ref = np.zeros_like(x)
    for i in range(len(pos2[:-1])):
        ref[(x>=pos2[i]) & (x<=pos2[i+1])] = calc_sines(np.sum((x>=pos2[i]) & (x<pos2[i+1])))
    return ref

def img_line(image):
    aux = np.argmax(image, 0)
    w = np.max(image, 0)
    a = np.asarray([aux, w])
    return a

shots = range(4, 8)
print('Loading File')
data = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/ensaios_23_09/peca_seno.m2k', 5, 0.5, 'gaussian',
                      sel_shots=shots)
aux = np.arange(0, len(shots))*5
data.inspection_params.step_points[:, 0] += aux
shots = np.asarray(shots) - shots[0]
print('Applying Hilbert Transform')
out = hilbert_transforms(data, shots, N=2)
data.inspection_params.type_insp = 'contact'
specimen_cl = data.specimen_params.cl
data.specimen_params.cl = data.inspection_params.coupling_cl

size = np.array([60, 10])
res = np.array([0.1, 0.1])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-30, 0, 30]
roi = ImagingROI(corner_roi, size[1], shape[1], size[0], shape[0])
print('Imaging')
t0 = time.time()
cumulative_tfm.cumulative_tfm_kernel(data, roi, output_key=0, sel_shots=shots)
print(time.time()-t0)
eroi = data.imaging_results[0].roi
yn = envelope(data.imaging_results[0].image)
yt = yn[:, int(shape[0]/2):int(data.inspection_params.step_points[shots[-1]][0]/res[0]+shape[0]/2)+1]
# yt = yn[:, 300:3301]
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
lamb = 150/np.mean(np.abs(d@z))
rho = 1
print(f'Estimating Surface')
t0 = time.time()
zf, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, a[0], lamb, x0=z, rho=rho, eta=.99, itmax=500, tol=1e-30)
print(time.time()-t0)
plt.imshow(y), plt.plot(zf)

data.surf = Surface(data, surf_type=SurfaceType.ARBITRARY)
w_points = eroi.w_points[int(shape[0]/2):int(data.inspection_params.step_points[-1][0]/res[0]+shape[0]/2)+1]
w_points = eroi.w_points[300:3301]

ref = peca_sines(w_points+data.inspection_params.step_points[0][0]) * 0.5 * amw[0] + 85# / eroi.h_step + np.argmax(y[:, 0])

data.surf.x_discr = w_points
data.surf.z_discr = ref
data.surf.fitted = True
# Chama TFM pra gerar imagem da superfície + sdh
# shots = np.arange(22, 3)
size = np.array([60, 10])
res = np.array([0.1, 0.1])
shape = (size/res).astype(int)
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-30, 0, 80]
roi = ImagingROI(corner_roi, size[1], shape[1], size[0], shape[0])
print('Imaging')
t0 = time.time()
cumulative_tfm.cumulative_tfm_kernel(data, roi, output_key=0, sel_shots=shots)
print(time.time()-t0)
eroi = data.imaging_results[0].roi
yn2 = data.imaging_results[0].image
# yt2 = yn2[:, int(shape[0]/2):int(data.inspection_params.step_points[-1][0]/res[0]+shape[0]/2)+1]
# yt2 = yn2[:, 100:1101]
# y2 = yt2/yt2.max()

# plt.figure()
# plt.imshow(y2, extent=[eroi.w_points[100], eroi.w_points[1100], eroi.h_points[-1], eroi.h_points[0]], origin='upper')
# plt.plot(w_points, zf)
