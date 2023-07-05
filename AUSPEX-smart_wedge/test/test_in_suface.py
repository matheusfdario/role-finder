import numpy as np
from matplotlib import pyplot as plt
from framework import file_m2k, file_civa, post_proc, utils
from parameter_estimation import intsurf_estimation
from scipy.signal import hilbert
from framework.data_types import ImagingROI
import time
from imaging import tfm, cumulative_tfm
from surface.surface import SurfaceType
from surface.surface import Surface
plt.rcParams["font.size"] = 14

plt.close('all')
# Parametros da imagem TFM
shots = [0]
width = 32
height = 22
wres = 0.1
hres = 0.10
roi_0 = -16
W = int(width/wres)
H = int(height/hres)

# Profile - Reference
ref, x_inf, x_sup = intsurf_estimation.specimen(roi_0,width,shots)

print('Loading File')
# data = file_m2k.read("/home/hector/PycharmProjects/AUSPEX/data/trecho_reto_dentes_FMC_gate8.m2k",
#                      'contact', 0.0, 4.9, 0.63, 'gaussian')
data = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/kirchoff_25shots.civa', sel_shots=None)
# data = file_civa.read('C:/Users/Tatiana/Desktop/kirchoff_25shots.civa', sel_shots=shots)
# data.ascan_data = data.ascan_data[:, :, :, 0, np.newaxis]
# data.ascan_data = hilbert(np.real(data.ascan_data), axis=0)
# data.ascan_data[-1, :, :, :] = 0


shots = np.asarray(shots) - shots[0]
print('Applying Hilbert Transform')
corner_roi = np.array([roi_0, 0.0, 5.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height, int(height/hres), width=width, w_len=int(width/wres))
# data.surf = Surface(data, surf_type=SurfaceType.ATFM)
# data.surf.fit(roi=roi, shot=shots)

t0 = time.time()
chave = cumulative_tfm.cumulative_tfm_kernel(data, roi=roi, sel_shot=shots, c=data.specimen_params.cl)
print(time.time()-t0)

image = data.imaging_results[chave].image
y = (image)
y /= y.max()
# eroi = data.imaging_results[chave].roi
# plt.figure()
# plt.imshow(im_env, aspect='auto',
#                extent=[eroi.w_points[0], eroi.w_points[-1], eroi.h_points[-1], eroi.h_points[0]])
# plt.show()


# t0 = time.time()
# y = utils.cumulative_tfm(data, corner_roi, height, width, hres, wres, shots=range(0,5), scat_angle=180)
# print(time.time()-t0)
# t0 = time.time()
# y = utils.cmax_tfm(data, corner_roi, height, width, hres, wres, shots=range(0,25), scat_angle=180)
# print(time.time()-t0)
# tfm.tfm_kernel(data, roi, output_key=0)
# y = np.abs(data.imaging_results[0].image)
# y = post_proc.normalize(y[:, :700])
# y = post_proc.normalize(y)
# plt.close('all')



# Limites para os plots
x0_lim = np.int(x_inf*wres)
xf_lim = np.int(x_sup*wres)
y0_lim = height
yf_lim = 0

a = utils.img_line(y)
z = a[0]
w = np.diag((a[1]))
# w = np.ones(600)
# w[0:200] = 0
# w[450:] = 0
# w = np.diag(w)
# w[w > 0.05] = 1
w2 = np.ones_like(w)
w2[w <= 0.05] = w[w <= 0.05]
w3 = w > 0.05
w4 = w@w
w5 = np.ones_like(w)
w5[w4 <= 0.05] = w4[w4 <= 0.05]

# w = w@w
wtw = w.T@w
d = intsurf_estimation.matrix_d2(z, 'mirrored')
dtd = d.T@d
# leg1 = ['Ref', 'Maximums']
leg1 = ['Referência']
leg2 = []

lamb = 0.009
print(f'Estimating Surface - IRLS')
t0 = time.time()
x, res, k = intsurf_estimation.profile_irls(w4, z, lamb, eps=1e-9, itmax=250, tol=1e-15, x0=None)
print(time.time()-t0)
print(f'Residue = {res[-1]:.2f}')
print(f'lambda = {lamb}')
print(f'Iterações = {k}')
erro_irls = (x - ref)/10
print(f'MSE = {((x-ref)**2).mean()}')
leg1.append('IRLS')
leg2.append('IRLS')

lamb = np.abs(d@z).max()
rho = 0.0021
print(f'Estimating Surface - ADMM')
t0 = time.time()
xa, resa, ka = intsurf_estimation.profile_admm(w4, z, lamb, rho=rho, itmax=250, tol=1e-15)
print(time.time()-t0)
print(f'Residue = {resa[-1]:.2f}')
print(f'lambda = {lamb}')
print(f'rho = {rho}')
print(f'Iterações = {ka}')
erro_admm = (xa - ref)/10
print(f'MSE = {((xa-ref)**2).mean()}')
leg1.append('ADMM')
leg2.append('ADMM')

lamb = np.abs(d@z).max()
rho = 0.004
print(f'Estimating Surface - fADMM')
t0 = time.time()
xf, resf, kf, prim, sec = intsurf_estimation.profile_fadmm(w4, z, lamb, rho=rho, eta=.9, itmax=250, tol=1e-15)
print(time.time()-t0)
print(f'Residue = {resf[-1]:.2f}')
print(f'lambda = {lamb}')
print(f'rho = {rho}')
print(f'Iterações = {kf}')
erro_fadmm = (xf - ref)/10
print(f'MSE = {((xf-ref)**2).mean()}')
leg1.append('FADMM')
leg2.append('FADMM')

# print(f'Estimating Surface')
# t0 = time.time()
# xfa, resfa, kfa = intsurf_estimation.profile_fama(w, z, lamb=1e-10, tau=1/8, itmax=250, tol=1e-4)
# print(time.time()-t0)
# print(f'Residue = {resf[-1]:.2f}')
# erro_fama = (xfa - ref)/10
# print(f'MSE = {((xfa-ref)**2).mean()}')
# leg1.append('FAMA')
# leg2.append('FAMA')

#
# print(f'Estimating Surface')
# t0 = time.time()
# xgs, resgs, kgs = intsurf_estimation.gstv_w(z, w4, k=50, d=d, lamb=1e-100, eps=1e-100, itmax=150, tol=1e-400)
# print(time.time()-t0)
# print(f'Residue = {resgs[-1]:.2f}')
# erro_xgs = (xgs - ref)/10
# print(f'MSE = {((xgs-ref)**2).mean()}')
# leg1.append('GSTV')
# leg2.append('GSTV')

plt.figure(figsize=(6,5)) # para 1 Shot
# plt.figure(figsize=(18,5)) # para todos Shots
plt.imshow(y,cmap='binary', extent=[x0_lim, xf_lim, y0_lim, yf_lim])
plt.plot(np.arange(x0_lim, xf_lim, wres), ref*hres, '--g', linewidth=1.5)
plt.yticks(np.arange(y0_lim-2, yf_lim-1, -5))
plt.ylabel('Profundidade [mm]')
plt.xlabel('Posição em x [mm]')
plt.grid(True,which="both",ls="--", color='0.85')
plt.legend(['Referência'])
# plt.savefig('C:/Users/Tatiana/Desktop/Images/TFM_0.pdf')
plt.show()

plt.figure(figsize=(4.8,3.7)) # para 1 Shot
# plt.figure(figsize=(18,2.7)) # para todos Shot
plt.plot(np.arange(x0_lim, xf_lim, wres), ref*hres, '-g', linewidth=2)
plt.gca().invert_yaxis()
plt.plot(np.arange(x0_lim, xf_lim, wres), x*hres, '--r', linewidth=1.5)
plt.plot(np.arange(x0_lim, xf_lim, wres), xa*hres, '-.', color='tab:orange', linewidth=1.5)
plt.plot(np.arange(x0_lim, xf_lim, wres), xf*hres, ':b', linewidth=1.5)
# plt.plot(xfa, 'k')
# plt.plot(xgs, 'g')
plt.legend(leg1, loc='best')
plt.xlim(x0_lim, xf_lim)
plt.ylim(y0_lim, yf_lim)
plt.yticks(np.arange(y0_lim-2, yf_lim-1, -5))
plt.ylabel('Profundidade [mm]')
plt.xlabel('Posição em x [mm]')
plt.grid(True,which="both",ls="--", color='0.85')
plt.subplots_adjust(bottom = 0.2)
# plt.savefig('C:/Users/Tatiana/Desktop/Images/Perfil_0.pdf')
# plt.title('Superfície estimada')
plt.show()

plt.figure(figsize=(5.7,4.2))
plt.plot(np.arange(1,k+1),res, '--r', linewidth=1.5)
plt.plot(np.arange(1,ka+1),resa, '-.', color='tab:orange', linewidth=1.5)
plt.plot(np.arange(1,kf+1),resf[0:kf], ':b', linewidth=1.5)
# plt.plot(resfa)
# plt.plot(resgs)
plt.ylabel('Resíduo')
plt.xlabel('Iterações [un]')
plt.grid(True,which="both",ls="--", color='0.85')
plt.subplots_adjust(left= 0.18, bottom = 0.12)
plt.legend(leg2)
# plt.savefig('C:/Users/Tatiana/Desktop/Images/Res_0.pdf')
plt.show()
#
# plt.figure(figsize=(6,4.5))
# plt.imshow(y,cmap='binary', extent=[x0_lim, xf_lim, y0_lim, yf_lim])
# plt.plot(np.arange(x0_lim, xf_lim, wres), z*hres, '-', color='tab:orange')
# plt.yticks(np.arange(y0_lim-2, yf_lim-1, -5))
# plt.ylabel(r'$z^{*}$')
# plt.xlabel('Posição em x [mm]')
# plt.grid(True,which="both",ls="--", color='0.85')
# plt.subplots_adjust(left= 0.12, bottom = 0.12)
# plt.savefig('C:/Users/Tatiana/Desktop/Images/z_0.pdf')
# plt.show()

# fig, ax1 = plt.subplots(figsize=(6,4.5))
# plt.plot(np.arange(x0_lim, xf_lim, wres), a[1], '-', color='tab:orange', linewidth=0.8)
# plt.ylabel('W')
# plt.xlabel('Posição em x [mm]')
# plt.grid(True,which="both",ls="--", color='0.85')
# plt.subplots_adjust(left= 0.12, bottom = 0.12)
# plt.savefig('C:/Users/Tatiana/Desktop/Images/W_0.pdf')
# plt.show()

# plt.figure()
# plt.plot(erro_irls, 'r')
# plt.plot(erro_admm, '-.g')
# plt.plot(erro_fadmm, '-.m')
# # plt.plot(erro_fama, '-.k')
# # plt.plot(erro_xgs)
# plt.title('Erro absoluto')
# plt.legend(leg2)
# plt.ylabel('mm')
# plt.show()


# plt.figure()
# rx = roi.w_points
# rz = x*roi.h_step
# dx = np.gradient(rx, 1, edge_order=2)
# dz = np.gradient(rz, 1, edge_order=2)
# ang = np.arctan2(dx, dz)
# nor = np.array([np.cos(ang), np.sin(ang)])
#
# ang2 = np.arctan2(rx, -rz)+np.pi/2
# trdir = np.array([np.cos(ang2), np.sin(ang2)])
# plt.plot(rx, rz, '.-')
# plt.plot(0, 0, '.r')
# Sc = 30
# Rp = 1
# arw = 1e-3
# plt.quiver(rx[::Rp], rz[::Rp], nor[0][::Rp], nor[1][::Rp], scale=Sc, width=arw)
# plt.quiver(rx[::Rp], rz[::Rp], trdir[0][::Rp], trdir[1][::Rp], scale=Sc, color='blue', angles='xy', width=arw)
# plt.gca().invert_yaxis()
# plt.legend(['Surface Profile', 'Origin', 'Surface Normal', 'Transducer Center'])
#
#
# alt_ang = np.arctan2(rx, rz)+np.pi/2
# abang = ang2+np.pi/2
# abang[abang>np.pi] -= 2*np.pi
# mi = np.argmax(np.diag(w))
# plt.figure()
# wn = np.diag(w)#/w[:].sum()
# ws = (1/(0.4*np.sqrt(2*np.pi)))*np.exp(-0.5*(((ang-alt_ang))/0.3)**2)#*np.cos(alt_ang-ang)**6
# ws = np.cos(alt_ang-ang)
# # ws /= ws.sum()
# plt.plot(rx, wn)
# plt.plot(rx, ws)
# plt.legend(['Image maximum', 'Gaussian distribution with angles'])
# plt.show()

