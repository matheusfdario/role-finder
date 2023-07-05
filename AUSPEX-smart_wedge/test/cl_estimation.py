import numpy as np
import cv2
import matplotlib.pyplot as plt
from framework import file_civa, file_m2k
from framework import  pre_proc, post_proc
from imaging import saft, wk_saft, tfm, cpwc, wk_cpwc
from parameter_estimation import cl_estimators
from framework.data_types import ImagingROI
import time
from scipy.signal import hilbert
from matplotlib import rc
# rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
rc('text', usetex=True)
# plt.close('all')

# TV tenenbaum
def cl_estimator_ten(image):
    kernel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img1 = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    img2 = cv2.filter2D(src=image, kernel=kernel.T, ddepth=-1)
    out = np.sum(np.square(img1)) + np.sum(np.square(img2))
    return out


# TV Brenner
def cl_estimator_bre(image):
    kernel = np.asarray([-1, 0, 1])
    img1 = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    img2 = cv2.filter2D(src=image, kernel=kernel.T, ddepth=-1)
    out = np.sum(np.square(img1)) + np.sum(np.square(img2))
    return out


# Contrast
def cl_estimator_con(image):
    # image = np.log10(image)
    image = (image - image.min())/abs(image.max()-image.min())
    out = np.mean(image**2)/np.mean(image)**2
    return out


# API
def cl_estimator_api(image, lamb):
    api = post_proc.api(image, roi, lamb)
    return api


# Normalized Variance
def cl_estimator_var(image):
    mean = np.mean(image)
    out = (1/mean) * np.sum(np.square(image-mean))
    return out


def gse_img(imgs, cs, metric_func, arg=None):
    metric = np.zeros_like(cs)
    for i in range(len(cs)):
        if arg is not None:
            metric[i] = metric_func(imgs[i], arg[i])
        else:
            metric[i] = metric_func(imgs[i])
    return metric


def nxtpw2(x):
    return 1 << (x-1).bit_length()


alg_dic = {'tfm': 'TFM', 'cpwc': 'CPWC'}
met_dic = {'ten': 'Tenenbaum', 'var': 'Normalized Variance',
           'con': 'Contrast'}

# data_fmc = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/var_speed_fmc.civa')
data_fmc = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/CP1_Bot_Direct/50.m2k', type_insp='contact',
                     water_path=0, freq_transd=5e6, bw_transd=0.5, tp_transd='gaussian', sel_shots=1)
# data_fmc = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/trecho_reto_dentes_FMC_gate8.m2k',
#                          type_insp='contact', water_path=0, freq_transd=5e6, bw_transd=0.5,
#                          tp_transd='gaussian', sel_shots=0)
# data_pwi = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/var_speed_pwi.civa')
samp = len(data_fmc.ascan_data[:, 0, 0, 0])
data_fmc.ascan_data = hilbert(data_fmc.ascan_data, N=nxtpw2(2*samp), axis=0)[:samp, :, :, :]
data_fmc.ascan_data[-1, :, :, :] = 0
# data_pwi.ascan_data = hilbert(data_pwi.ascan_data, N=nxtpw2(2*samp), axis=0)[:samp, :, :, :]
# data_pwi.ascan_data[-1, :, :, :] = 0

height = 20
h_res = 0.1
h_len = int(height/h_res)
width = 40
w_res = 0.1
w_len = int(width/w_res)
roi_corner = np.array([-20.0, 0.0, 30.0])[np.newaxis]
roi = ImagingROI(roi_corner, height, h_len, width, w_len)
# Number of estimates
N = 101

cl = 6319#data_fmc.specimen_params.cl

cle = np.linspace(cl*0.9, cl*1.1, N)
cle = np.arange(np.rint(0.9*cl), np.rint(1.1*cl), 1)
N = len(cle)
for i in range(alg_dic.__len__()):
    exec(f'img_{list(alg_dic.keys())[i]} = np.zeros((N, h_len, w_len))')
    exec(f'chave_{list(alg_dic.keys())[i]} = np.linspace({i}*N, {i+1}*N-1, N, dtype=np.int)')

print('Imaging')
t = time.time()

trcomb = np.ones((64, 64))
for i in range(32):
    aux = np.zeros(64-i)
    np.fill_diagonal(trcomb[i:], aux)
    np.fill_diagonal(trcomb[:, i:], aux)

for i in range(N):
    print(f'{i}/{N}')
    # saft.saft_kernel(data_fmc, roi, chave_saft[i], c=cle[i])
    # wk_saft.wk_saft_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i])
    tfm.tfm_kernel(data_fmc, roi, chave_tfm[i], c=cle[i])
    # tfm.tfm_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i], trcomb=trcomb)
    cpwc.cpwc_kernel(data_fmc, roi, chave_cpwc[i], c=cle[i], angles=np.linspace(-45, 45, 32))
    # wk_cpwc.wk_cpwc_kernel(data_fmc, roi, chave_wkcpwc[i], c=cle[i], angles=np.linspace(-45, 45, 16))
t = time.time()-t
print(f'Done imaging in {t:.2f}s, extracting images')
t = time.time()
for i in range(N):
    # img_saft[i] = data_fmc.imaging_results[chave_saft[i]].image
    # img_wksaft[i] = data_fmc.imaging_results[chave_wksaft[i]].image
    img_tfm[i] = np.abs(data_fmc.imaging_results[chave_tfm[i]].image)
    img_cpwc[i] = np.abs(data_fmc.imaging_results[chave_cpwc[i]].image)
    # img_wkcpwc[i] = data_fmc.imaging_results[chave_wkcpwc[i]].image
t = time.time() - t
print(f'Done extracting in {t:.2f}s, taking the envelope')
t = time.time()
# img_saft = (post_proc.envelope(img_saft))
# img_wksaft = (post_proc.envelope(img_wksaft))
# img_tfm = (post_proc.envelope(img_tfm))
# img_cpwc = (post_proc.envelope(img_cpwc))
# img_wkcpwc = (post_proc.envelope(img_wkcpwc))
t = time.time() - t
print(f'Done in {t:.2f}s, applying metrics')

t = time.time()
clest = dict()
for i in alg_dic.keys():
    for j in met_dic.keys():
        if j is not 'api':
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j})')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmax()]}})')
        else:
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j}, '
                 f'(cle/data_fmc.probe_params.central_freq*1e6).astype(np.float32))')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmin()]}})')
t = time.time()-t
print(f'Done with metrics in {t:.2f}s')

for i in alg_dic.keys():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(alg_dic[i])
    ax1.grid()
    legend = list()
    for j in met_dic.keys():
        ax1.plot(cle, post_proc.normalize(eval(f'{j}_{i}')))
        aux_alg = eval(f'met_dic.get(\'{j}\')')
        aux_met = eval(f'clest.get(\'{j+i}\')')
        legend.append(f'{aux_alg} = {aux_met:.2f}')
    ax1.axvline(cl, 0, 1, color='black', linestyle='--')
    legend.append(f'True speed = {cl:.2f}')
    ax1.legend(legend)
    ax1.set_xlabel('Assumed Speed (m/s)')
    ax1.set_ylabel('Normalized Focus Function')
    fig.tight_layout()
    plt.savefig(f'/home/hector/Pictures/speed estimation/afcurve_nb_{i}.pdf')

height = 50
h_len = int(height/h_res)
roi_corner = np.array([-20.0, 0.0, 30.0])[np.newaxis]
roi = ImagingROI(roi_corner, height, h_len, width, w_len)

for i in range(alg_dic.__len__()):
    exec(f'img_{list(alg_dic.keys())[i]} = np.zeros((N, h_len, w_len))')
    exec(f'chave_{list(alg_dic.keys())[i]} = np.linspace({i}*N, {i+1}*N-1, N, dtype=np.int)')

print('Imaging')
t = time.time()

trcomb = np.ones((64, 64))
for i in range(32):
    aux = np.zeros(64-i)
    np.fill_diagonal(trcomb[i:], aux)
    np.fill_diagonal(trcomb[:, i:], aux)

for i in range(N):
    print(f'{i}/{N}')
    # saft.saft_kernel(data_fmc, roi, chave_saft[i], c=cle[i])
    # wk_saft.wk_saft_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i])
    tfm.tfm_kernel(data_fmc, roi, chave_tfm[i], c=cle[i])
    # tfm.tfm_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i], trcomb=trcomb)
    cpwc.cpwc_kernel(data_fmc, roi, chave_cpwc[i], c=cle[i], angles=np.linspace(-45, 45, 32))
    # wk_cpwc.wk_cpwc_kernel(data_fmc, roi, chave_wkcpwc[i], c=cle[i], angles=np.linspace(-45, 45, 16))
t = time.time()-t
print(f'Done imaging in {t:.2f}s, extracting images')
t = time.time()
for i in range(N):
    # img_saft[i] = data_fmc.imaging_results[chave_saft[i]].image
    # img_wksaft[i] = data_fmc.imaging_results[chave_wksaft[i]].image
    img_tfm[i] = np.abs(data_fmc.imaging_results[chave_tfm[i]].image)
    img_cpwc[i] = np.abs(data_fmc.imaging_results[chave_cpwc[i]].image)
    # img_wkcpwc[i] = data_fmc.imaging_results[chave_wkcpwc[i]].image
t = time.time() - t
print(f'Done extracting in {t:.2f}s, taking the envelope')
t = time.time()
# img_saft = (post_proc.envelope(img_saft))
# img_wksaft = (post_proc.envelope(img_wksaft))
# img_tfm = (post_proc.envelope(img_tfm))
# img_cpwc = (post_proc.envelope(img_cpwc))
# img_wkcpwc = (post_proc.envelope(img_wkcpwc))
t = time.time() - t
print(f'Done in {t:.2f}s, applying metrics')

t = time.time()
clest = dict()
for i in alg_dic.keys():
    for j in met_dic.keys():
        if j is not 'api':
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j})')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmax()]}})')
        else:
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j}, '
                 f'(cle/data_fmc.probe_params.central_freq*1e6).astype(np.float32))')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmin()]}})')
t = time.time()-t
print(f'Done with metrics in {t:.2f}s')

for i in alg_dic.keys():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(alg_dic[i])
    ax1.grid()
    legend = list()
    for j in met_dic.keys():
        ax1.plot(cle, post_proc.normalize(eval(f'{j}_{i}')))
        aux_alg = eval(f'met_dic.get(\'{j}\')')
        aux_met = eval(f'clest.get(\'{j+i}\')')
        legend.append(f'{aux_alg} = {aux_met:.2f}')
    ax1.axvline(cl, 0, 1, color='black', linestyle='--')
    legend.append(f'True speed = {cl:.2f}')
    ax1.legend(legend)
    ax1.set_xlabel('Assumed Speed (m/s)')
    ax1.set_ylabel('Normalized Focus Function')
    fig.tight_layout()
    plt.savefig(f'/home/hector/Pictures/speed estimation/afcurve_bot_{i}.pdf')

height = 20
h_len = int(height/h_res)
roi_corner = np.array([-20.0, 0.0, 50.0])[np.newaxis]
roi = ImagingROI(roi_corner, height, h_len, width, w_len)

for i in range(alg_dic.__len__()):
    exec(f'img_{list(alg_dic.keys())[i]} = np.zeros((N, h_len, w_len))')
    exec(f'chave_{list(alg_dic.keys())[i]} = np.linspace({i}*N, {i+1}*N-1, N, dtype=np.int)')

print('Imaging')
t = time.time()

trcomb = np.ones((64, 64))
for i in range(32):
    aux = np.zeros(64-i)
    np.fill_diagonal(trcomb[i:], aux)
    np.fill_diagonal(trcomb[:, i:], aux)

for i in range(N):
    print(f'{i}/{N}')
    # saft.saft_kernel(data_fmc, roi, chave_saft[i], c=cle[i])
    # wk_saft.wk_saft_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i])
    tfm.tfm_kernel(data_fmc, roi, chave_tfm[i], c=cle[i])
    # tfm.tfm_kernel(data_fmc, roi, chave_wksaft[i], c=cle[i], trcomb=trcomb)
    cpwc.cpwc_kernel(data_fmc, roi, chave_cpwc[i], c=cle[i], angles=np.linspace(-45, 45, 32))
    # wk_cpwc.wk_cpwc_kernel(data_fmc, roi, chave_wkcpwc[i], c=cle[i], angles=np.linspace(-45, 45, 16))
t = time.time()-t
print(f'Done imaging in {t:.2f}s, extracting images')
t = time.time()
for i in range(N):
    # img_saft[i] = data_fmc.imaging_results[chave_saft[i]].image
    # img_wksaft[i] = data_fmc.imaging_results[chave_wksaft[i]].image
    img_tfm[i] = np.abs(data_fmc.imaging_results[chave_tfm[i]].image)
    img_cpwc[i] = np.abs(data_fmc.imaging_results[chave_cpwc[i]].image)
    # img_wkcpwc[i] = data_fmc.imaging_results[chave_wkcpwc[i]].image
t = time.time() - t
print(f'Done extracting in {t:.2f}s, taking the envelope')
t = time.time()
# img_saft = (post_proc.envelope(img_saft))
# img_wksaft = (post_proc.envelope(img_wksaft))
# img_tfm = (post_proc.envelope(img_tfm))
# img_cpwc = (post_proc.envelope(img_cpwc))
# img_wkcpwc = (post_proc.envelope(img_wkcpwc))
t = time.time() - t
print(f'Done in {t:.2f}s, applying metrics')

t = time.time()
clest = dict()
for i in alg_dic.keys():
    for j in met_dic.keys():
        if j is not 'api':
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j})')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmax()]}})')
        else:
            exec(f'{j}_{i} = gse_img(img_{i}, cle, cl_estimator_{j}, '
                 f'(cle/data_fmc.probe_params.central_freq*1e6).astype(np.float32))')
            exec(f'clest.update({{\'{j+i}\': cle[{j}_{i}.argmin()]}})')
t = time.time()-t
print(f'Done with metrics in {t:.2f}s')

for i in alg_dic.keys():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(alg_dic[i])
    ax1.grid()
    legend = list()
    for j in met_dic.keys():
        ax1.plot(cle, post_proc.normalize(eval(f'{j}_{i}')))
        aux_alg = eval(f'met_dic.get(\'{j}\')')
        aux_met = eval(f'clest.get(\'{j+i}\')')
        legend.append(f'{aux_alg} = {aux_met:.2f}')
    ax1.axvline(cl, 0, 1, color='black', linestyle='--')
    legend.append(f'True speed = {cl:.2f}')
    ax1.legend(legend)
    ax1.set_xlabel('Assumed Speed (m/s)')
    ax1.set_ylabel('Normalized Focus Function')
    fig.tight_layout()
    plt.savefig(f'/home/hector/Pictures/speed estimation/afcurve_onlybot_{i}.pdf')