import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k, file_civa
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (LinearLocator, FormatStrFormatter,
                               AutoMinorLocator)
from framework.post_proc import envelope
from generate_law import *
from parameter_estimation.intsurf_estimation import img_line
from parameter_estimation import intsurf_estimation
from scipy import signal
from scipy import misc

import math
import os.path
import struct
import zlib
import sys
from matplotlib.animation import FuncAnimation

import cv2
import os

def moving_average(a, n=3, padding=0) :
    row = a.shape[0]
    padded_row = 2 * (n//2) + row
    a_out = np.zeros(padded_row)
    a_out[n//2:padded_row - n//2] = a
    ret = np.cumsum(a_out, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def vertical_img_moving_avg(img, n=3, padding=0):
    for j in range(img.shape[1]):
        img[:, j] = moving_average(img[:, j], n=n)
    return img


def find_inner_surface(img, z_seam, x_axis, y_axis, wall_thickness=16, perc_theshold=0.6, n_first=20):
    avg_img = vertical_img_moving_avg(img, n=37)
    ext_surf_idx = np.array([np.argmin(np.power(z - y_axis, 2)) for z in z_seam])
    int_surf_idx = np.array([np.argmin(np.power((z - wall_thickness) - y_axis, 2)) for z in z_seam])
    ext_surf_intensity = np.array([img[ext_surf_idx[i], i] for i in range(img.shape[1])])
    threshold = ext_surf_intensity * perc_theshold

    y_inner = np.zeros_like(z_seam)
    for col in range(img.shape[1]):
        wall_img = np.copy(avg_img[ext_surf_idx[col]:int_surf_idx[col], col])

        pixels_above_threshold = np.copy(wall_img)

        for i in range(pixels_above_threshold.shape[0]):
            if pixels_above_threshold[i] >= threshold[col]:
                pixels_above_threshold[i] = 1
            else:
                pixels_above_threshold[i] = 0


        ext_surf_idx_slack = np.array([np.argmin(np.power(z - 10 - y_axis, 2)) for z in z_seam])
        y_wall = y_axis[ext_surf_idx[col]:int_surf_idx[col]]

        # plt.figure()
        # plt.plot(y_wall, wall_img, color='g')
        # plt.plot(y_wall, threshold[col] * np.ones_like(pixels_above_threshold), color='r')
        # plt.plot(y_wall, pixels_above_threshold, 'o', color='k')
        # plt.show()

        pixels_above_threshold[:ext_surf_idx_slack[col] - ext_surf_idx[col]] = 0

        # plt.plot(pixels_above_threshold)
        # plt.show()
        y_selected = y_wall[np.array(pixels_above_threshold, dtype=bool)]
        if y_selected.shape[0] == 0:
            y_inner[col] = 0
        else:
            y_inner[col] = np.mean(y_selected[:n_first])
    return y_inner

def generate_mean_kernel(n=3):
    return 1/np.power(n,2) * np.ones((n,n))


def generateSobelY(n=3):
    kernel=np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).transpose()
    return kernel
def width2amplitude_filter(img, width=3):
    kernel = generateSobelY(3)
    img_filtered = signal.convolve2d(img, kernel, mode='same', boundary="fill") * img

    return img_filtered

def makevideo(videoname, frames=5):

    image_folder = 'frames'
    video_name = videoname + '.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'3IVD')
    video = cv2.VideoWriter(video_name, fourcc, frames, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()



def civa_pwi(arquivo, n_angles, n_elem, n_passos, _sel_shots):
    """Abre um arquivo binário do formato do civa, que contém uma série de
    A-scans e retorna os dados no formato de um PWI.

    """

    tam_num = np.int32().itemsize

    file = open(arquivo, 'rb')
    data = file.read(4)

    num_inicial = struct.unpack('<i', data)[0]
    posicoes = []
    headers = []

    n_ascan = n_angles * n_elem * n_passos

    # menor numero de amostras antes do gating iniciar
    menor = sys.maxsize
    # maior indice de amostras do A-scan, considerando o gating
    tam_max = 0

    # procura o arquivo para encontrar todos os cabeçalhos e salva as suas posicoes
    # tambem encontra os valores acima
    _i = 0
    j = 0
    while j < n_ascan:
        file.seek(_i)
        data = file.read(5 * tam_num)
        try:
            header = struct.unpack('<5i', data)
        except struct.error:
            # marca essa posicao para ter sempre 0
            header = (-1, -1, -1, -1, -2)
            headers.append(header)

        if header[0] == num_inicial and header[3] != 0:
            # header esperado
            posicoes.append(_i)
            headers.append(header)
            if headers[j][1] < menor:
                menor = headers[j][1]
            if headers[j][2] + headers[j][1] > tam_max:
                tam_max = headers[j][2] + headers[j][1]

            if headers[j][4] == -1:
                _i += headers[j][2] * tam_num + 5 * tam_num
            else:
                _i += headers[j][4] + 5 * tam_num

        elif header[0] == num_inicial:
            # gating manual e elemento fora da peça
            header = (-1, -1, -1, -1, -2)
            headers.append(header)
            posicoes.append(_i)
            _i += 4 * tam_num

        else:
            _i += 1

        j += 1

    # aloca a matriz dos dados
    # n_elem = int(math.sqrt(n_ascan / n_passos))
    out = np.zeros((7999, 37, 64))

    gating = [menor, tam_max]

    n_ascan_salvos = 0

    # preenche a matriz
    for k in _sel_shots:
        for t in range(n_angles):
            for j in range(n_elem):
                index = j + t * n_elem + k * n_elem * n_angles
                if headers[index][4] > 0:
                    # descompacta o A-scan
                    decompress = zlib.decompressobj(-15)
                    n_bytes = headers[index][4]
                    file.seek(posicoes[index] + 5 * tam_num)
                    data = file.read(n_bytes)
                    inflated = decompress.decompress(data)
                    inflated += decompress.flush()
                    ascan = np.frombuffer(inflated, dtype=np.float32)
                    out[:, t, j] = ascan

                elif headers[index][4] == -1:
                    file.seek(posicoes[index] + 5 * tam_num)
                    data = file.read(headers[index][2] * tam_num)
                    ascan = np.frombuffer(data, dtype=np.float32)
                    out[:, t, j] = ascan

                else:
                    # transdutor fora da peça, possui apenas o header
                    ascan = np.zeros(tam_max - menor)
                    headers[index] = (-1, menor, -1, -1, -2)
                    out[:, t, j] = ascan

                # corrige o A-scan de acordo com o gating
                out[:, t, j] = np.pad(out[:, t, j], (headers[index][1] - menor, 0),
                                             'constant')
                # coloca todos com o mesmo tamanho
                out[:, t, j] = np.pad(out[:, t, j], (0, tam_max - menor - len( out[:, t, j])),
                                             'constant')

                n_ascan_salvos += 1
    return out

def GenerateImage(shot):
    # Lê o arquivo a ser reconstruido:

    extension = ".m2k"
    path = data_root + filename + extension
    data = file_m2k.read(path, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian')
    data_4db = data[0]
    data_15db = data[1]



    # Tempo de ensaio em micro segundos:
    t_span = np.linspace(start=data[0].inspection_params.gate_start,
                         stop=data[0].inspection_params.gate_end,
                         num=data[0].inspection_params.gate_samples)

    # Calcula os tempos entre o centro do transdutor e a superfície da tubulação:
    delay_pattern = np.zeros_like(angles, dtype=float)
    time_spent = np.zeros_like(delay_pattern)
    wedge_time = np.zeros_like(delay_pattern)
    for i, bet in enumerate(betas):
        delay_pattern[i] = compensate_time(bet, sm) * 2
        time_spent[i] = delay_pattern[i] / 1e-6
        wedge_time[i] = compensate_time_wedge(bet, sm)/1e-6 * 2
    delay_pattern = delay_pattern - delay_pattern.min()
    delay_pattern = delay_pattern / 1e-6

    nonzero = delay_pattern[0]
    for i in np.arange(0, delay_pattern.shape[0]):
        if delay_pattern[i] < 1e-6:
            delay_pattern[i] = nonzero
        else:
            delay_pattern[i] = 0

    # Transforma os atrasos em indices a deslocar as linhas:
    sample_time = data[0].inspection_params.sample_time
    # Define limites de saturação:
    log_offset = 1e2
    alpha_cte = .6
    # ROI
    beg_time = wedge_time[0] + 1
    beg_idx = np.argmin(np.power(t_span - beg_time, 2))

    v_steel = 5.9
    d = np.zeros_like(t_span)
    for i, t in enumerate(t_span):
        if 40 <= t <= wedge_time[0]:
            d[i] = t * v1 # em mm
            # print(d[i])
        elif wedge_time[0] < t <= time_spent[0]:
            t_prime = t - wedge_time[0]
            d[i] = t_prime * 1.483 + wedge_time[0] * v1 # em mm
        elif time_spent[0] < t <= time_spent[0] + 32/v_steel:
            t_prime = t - time_spent[0]
            d[i] = (t_prime * v_steel + \
                   (wedge_time[0] * v1 +
                    (time_spent[0] - wedge_time[0]) * 1.483
                    ))
        elif time_spent[0] + 32/v_steel < t <= 90:
            t_prime = t - time_spent[0] + 32/v_steel
            d[i] = (t_prime * 1.483 + \
                   (wedge_time[0] * v1 +
                    (time_spent[0] - wedge_time[0]) * 1.483 +
                    16
                    ))
    radii = d[beg_idx:]
    thetas = betas
    total_dist = compute_total_dist(60, sm) * 2e3
    tube_radii = (total_dist - radii)/2

    # Transforma os atrasos em indices a deslocar as linhas:
    sample_time = data[0].inspection_params.sample_time
    # Define limites de saturação:
    log_offset = 1e1
    alpha_cte = .6
    # ROI
    beg_time = wedge_time[0] + 1
    beg_idx = np.argmin(np.power(t_span - beg_time, 2))

    v_steel = 5.9
    d = np.zeros_like(t_span)
    for i, t in enumerate(t_span):
        if 40 <= t <= wedge_time[0]:
            d[i] = t * v1 # em mm
            # print(d[i])
        elif wedge_time[0] < t <= time_spent[0]:
            t_prime = t - wedge_time[0]
            d[i] = t_prime * 1.483 + wedge_time[0] * v1 # em mm
        elif time_spent[0] < t <= time_spent[0] + 32/v_steel:
            t_prime = t - time_spent[0]
            d[i] = (t_prime * v_steel + \
                    (wedge_time[0] * v1 +
                     (time_spent[0] - wedge_time[0]) * 1.483
                     ))
        elif time_spent[0] + 32/v_steel < t <= 90:
            t_prime = t - time_spent[0] + 32/v_steel
            d[i] = (t_prime * 1.483 + \
                    (wedge_time[0] * v1 +
                     (time_spent[0] - wedge_time[0]) * 1.483 +
                     16
                     ))
    radii = d[beg_idx:]
    thetas = betas
    total_dist = compute_total_dist(60, sm) * 2e3
    tube_radii = (total_dist - radii)/2



    img_ref = np.load("media_refencia_sum.npy")
    img_ref_env = envelope(img_ref, axis=0)
    img_ref_log = np.log10(envelope(img_ref, axis=0) + log_offset)


    img_4db = data[0].ascan_data_sum
    img_4db_env = envelope(img_4db, axis=0)
    img_15db = data[1].ascan_data_sum


    i = 18
    idx_beg = 2900
    idx_end = 3500
    gain = np.sum(img_4db_env[idx_beg:idx_end, i])/np.sum(img_ref_env[idx_beg:idx_end, i])


    gain = list()
    for i in range(0, 37):
        if 9 <= i <= 27:
            idx_beg = 300
            idx_end = 650
            gain.append(np.sum(img_4db_env[idx_beg:idx_end, i])/np.sum(img_ref_env[idx_beg:idx_end, i]))
        else:
            idx_beg = 2900
            idx_end = 3500
            gain.append(np.sum(img_4db_env[idx_beg:idx_end, i])/np.sum(img_ref_env[idx_beg:idx_end, i]))
    gain = np.array(gain)

    img_log_4db = np.log10(img_4db_env + 1)

    img_log_ref = np.log10(img_ref_env + 1)


    img_log_ref = np.log10(np.abs(img_4db_env[:, :, 0] - gain * img_ref_env[:, :, 0]) + 1)
    img_shifted_log = shiftImage(img_log_ref[:, :], delay_pattern, sample_time)



    beg_idx = np.argmin(np.power(t_span - beg_time, 2))

    delta = 4
    plt.pcolormesh(thetas, tube_radii, img_shifted_log[beg_idx:, :], cmap="gray")
    img_shifted = img_shifted_log[beg_idx:, :]

    # APlicação do SEAM:
    # SEAM related parameters:
    lambda_param = 1
    rho_param = 100

    # Recorte da superfície externa:
    upper_outer_idx = np.argmin(np.power(tube_radii - (sm.r0 - delta), 2))
    lower_outer_idx = np.argmin(np.power(tube_radii - (sm.r0 + delta), 2))
    outer_surf_img = img_shifted[lower_outer_idx:upper_outer_idx, :]
    # Aplicação do SEAM:
    outer_norm_img = outer_surf_img / outer_surf_img.max()
    y = outer_norm_img
    a = img_line(y)
    zeta = tube_radii[lower_outer_idx:upper_outer_idx]
    z = zeta[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Externa com SEAM")
    outer_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                      eta=.999, itmax=10, tol=1e-3)

    plt.plot(thetas, outer_ext_zk, 'sb', label='SEAM p/ sup. externa')

    # Recorte da superfície interna:
    lambda_param = 10
    rho_param = 100
    upper_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 - 2), 2))
    lower_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 + 10), 2))
    inner_surf_img = img_shifted[lower_inner_idx:upper_inner_idx, :]
    # Aplicação do SEAM:
    inner_norm_img = inner_surf_img / inner_surf_img.max()
    y = inner_norm_img
    a = img_line(y)
    zeta = tube_radii[lower_inner_idx:upper_inner_idx]
    z = zeta[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Interba com SEAM")
    inner_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                      eta=.999, itmax=10, tol=1e-3)

    plt.plot(thetas, inner_ext_zk, 'db', label='SEAM com $\lambda = 10^{1}$')

    # Recorte da superfície interna:
    lambda_param = 1e-10
    rho_param = 100
    upper_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 - 2), 2))
    lower_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 + 10), 2))
    inner_surf_img = img_shifted[lower_inner_idx:upper_inner_idx, :]
    # Aplicação do SEAM:
    inner_norm_img = inner_surf_img / inner_surf_img.max()
    y = inner_norm_img
    a = img_line(y)
    zeta = tube_radii[lower_inner_idx:upper_inner_idx]
    z = zeta[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Interba com SEAM")
    inner_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                      eta=.999, itmax=10, tol=1e-3)

    plt.plot(thetas, inner_ext_zk, 'dg', label='SEAM com $\lambda = 10^{-10}$')

    # Recorte da superfície interna:
    lambda_param = 1e-20
    rho_param = 100
    upper_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 - 2), 2))
    lower_inner_idx = np.argmin(np.power(tube_radii - (sm.r0 - 16 + 10), 2))
    inner_surf_img = img_shifted[lower_inner_idx:upper_inner_idx, :]
    # Aplicação do SEAM:
    inner_norm_img = inner_surf_img / inner_surf_img.max()
    y = inner_norm_img
    a = img_line(y)
    zeta = tube_radii[lower_inner_idx:upper_inner_idx]
    z = zeta[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Interba com SEAM")
    inner_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param,
                                                                      eta=.999, itmax=10, tol=1e-3)

    inner_surf = find_inner_surface(img_shifted_log[beg_idx:, :], outer_ext_zk, angles, tube_radii)

    inner_surf_final = np.zeros_like(inner_surf)
    for i in range(inner_surf.shape[0]):
        if inner_surf[i] == 0:
            inner_surf_final[i] = inner_ext_zk[i]
        else:
            inner_surf_final[i] = inner_surf[i]


    plt.title(f"Ensaio = {filename} \n shot = {shot}")
    plt.plot(thetas, inner_surf_final, 'dr', label='Estimador Novo')
    plt.ylabel("distância do centro da tubulação /[mm]")
    plt.xlabel("ângulo de varredura da tubulação /[°]")
    plt.legend()
    plt.tight_layout()

    int_surf_mat[:, shot] = np.concatenate((inner_surf_final[:-1], inner_surf_final[1:][::-1]))
    ext_surf_mat[:, shot] = np.concatenate((outer_ext_zk[:-1], outer_ext_zk[1:][::-1]))

    return

# Importa módulos associados à smartwedge_old:
from generate_law import *

# GEra um Bscan onde
# Eixo X são os elementos e Eixo Y são os tempos. Varia-se o shot (ângulo fixo).

# Dados da sapata do Guastavo:
pitch = 0.6e-3
coord = np.zeros((64, 2))
coord[:, 0] = np.linspace(-63 * pitch / 2, 63 * pitch / 2, 64)

# Ellipse parameters:
c = 84.28
r0 = 67.15
wc = 6.2
offset = 2
Tprime = 19.5

# Velocidade do som nos materiais em mm/us
v1 = 6.37
v2 = 1.43

# Criação do objeto smartwedge_old:
sm = smartwedge(c, r0, wc, v1, v2, Tprime, offset)

# Define ângulos de varredura no referencial da tubulação:
beg_angle = -90
end_angle = 90
step = 5
betas = np.arange(beg_angle, end_angle + step, step)

# Calcula os ângulos no referncial do transdutor (ângulo de disparo da onda plana):
angles = list()
for beta in betas:
    ang, _ = sm.compute_entrypoints(beta)
    angles.append(ang)
angles = np.rad2deg(np.array(angles))

# Define o caminho dos dados:

data_root = "/media/tekalid/Data/CENPES/junho_2022/"
filename = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1"
path1 = "resultados"
result_foldername = path1
extension = ".m2k"
data_path = data_root + filename + extension

# Parâmetros do Vídeo:
generate_video = True

# O número de shots:
# n_shot = 97
n_shot = 24



video_title = "Bscan-IntSurf_Estimation"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

fig = plt.figure(figsize=(14, 9))

# Cria vetores para superfície interna e externa:
int_surf_mat = np.zeros((2 * angles.shape[0] - 2, n_shot))
ext_surf_mat = np.zeros((2 * angles.shape[0] - 2, n_shot))


shot = 0
if generate_video == True:
    # with writer.saving(fig, result_foldername + "/" + video_title + ".mp4", dpi=300):
    with writer.saving(fig, video_title + ".mp4", dpi=300):
        for shot in range(n_shot):
            GenerateImage(shot)
            print(f"frame = {shot + 1}")
            writer.grab_frame()
            plt.clf()
else:
    GenerateImage(shot)

if generate_video == True:
    np.save('int_surf_mat', int_surf_mat)
    np.save('ext_surf_mat', ext_surf_mat)