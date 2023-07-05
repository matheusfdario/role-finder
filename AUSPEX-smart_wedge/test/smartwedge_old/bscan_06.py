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

import math
import os.path
import struct
import zlib
import sys
from matplotlib.animation import FuncAnimation
from polarImg import *

import cv2
import os




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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def GenerateImage(shot):
    # Lê o arquivo a ser reconstruido:

    extension = ".m2k"
    path = data_root + filename + extension
    data = file_m2k.read(path, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian')
    data_4db = data[0]
    data_15db = data[1]

    # Lê o arquivo de referência das médias:
    smartwedge_mask = np.load("media_referencia.npy")

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



    img_ref = np.sum(smartwedge_mask, 2)
    img_ref = np.log10(np.abs(img_ref) + log_offset)


    # Plotando a figura:
    plt.suptitle(f"Shot : {shot} || Ensaio: {filename}")

    # Imagem do Ensaio a 4 dB
    plt.subplot(2, 2, 1, projection='polar')
    plt.gca().cla()
    img_insp_4db = np.sum(data_4db.ascan_data, 2)
    img_insp_4db = np.log10(np.abs(img_insp_4db) + log_offset)

    img_insp_4db_2 = shiftImage(img_insp_4db[:, :, 0], delay_pattern, sample_time)
    img_insp_4db_2 = img_insp_4db_2[beg_idx:, :]

    vmin_cte = img_insp_4db_2.min()
    vmax_cte = img_insp_4db_2.max()

    plt.pcolormesh(np.deg2rad(thetas), tube_radii, img_insp_4db_2,
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.grid()
    # plt.xlabel("Ângulo de Varredura")
    # plt.ylabel("Distância em mm")
    plt.title(f"Ganho = 4 dB")
    ax = plt.gca()
    ax.set_theta_zero_location("N")

    ax.set_ylim([0, 80])
    ax.set_theta_zero_location("N")
    ax.set_xticks(np.deg2rad(betas))
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.25, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.deg2rad(np.arange(-90, 90 + step * 3, step * 3)), minor=False)
    ax.set_xticks(np.deg2rad(np.arange(-90 - step / 2, 90 + step / 2 + step, step)), minor=True)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )

    plt.plot(np.deg2rad(betas), sm.r0 * np.ones_like(betas), 'x', color="#FF1F5B", alpha=alpha_cte, markersize=2.5,
             label='Sup. Ext.')
    plt.plot(np.deg2rad(betas), (sm.r0 - 16) * np.ones_like(betas), 'x', color="#00CD6C", alpha=alpha_cte,
             markersize=2.5, label='Sup. Int.')
    ax.set_xlim([-np.pi / 2, np.pi / 2])




    # Imagem do Ensaio a 15 dB
    plt.subplot(2, 2, 2, projection='polar')
    plt.gca().cla()
    img_ref_2 = shiftImage(img_ref[:, :, 0], delay_pattern, sample_time)
    img_ref_2 = img_ref_2[beg_idx:, :]

    plt.pcolormesh(np.deg2rad(thetas), tube_radii, img_ref_2,
                   vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.grid()
    plt.title(f"Referência = 7 dB")
    ax = plt.gca()
    ax.set_theta_zero_location("N")

    ax.set_ylim([0, 80])
    ax.set_theta_zero_location("N")
    ax.set_xticks(np.deg2rad(betas))
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.25, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.deg2rad(np.arange(-90, 90 + step * 3, step * 3)), minor=False)
    ax.set_xticks(np.deg2rad(np.arange(-90 - step / 2, 90 + step / 2 + step, step)), minor=True)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )

    plt.plot(np.deg2rad(betas), sm.r0 * np.ones_like(betas), 'x', color="#FF1F5B", alpha=alpha_cte, markersize=2.5,
             label='Sup. Ext.')
    plt.plot(np.deg2rad(betas), (sm.r0 - 16) * np.ones_like(betas), 'x', color="#00CD6C", alpha=alpha_cte,
             markersize=2.5, label='Sup. Int.')
    ax.set_xlim([-np.pi / 2, np.pi / 2])


    # Imagem do Ensaio a 15 dB
    plt.subplot(2, 2, 3, projection='polar')
    plt.gca().cla()
    img_insp_15db = np.sum(data_15db.ascan_data, 2)
    img_insp_15db = np.log10(np.abs(img_insp_15db) + log_offset)

    img_insp_15db_2 = shiftImage(img_insp_15db[:, :, 0], delay_pattern, sample_time)
    img_insp_15db_2 = img_insp_15db_2[beg_idx:, :]

    plt.pcolormesh(np.deg2rad(thetas), tube_radii, img_insp_15db_2,
                   vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.grid()
    plt.title(f"Ganho = 15 dB")
    ax = plt.gca()
    ax.set_theta_zero_location("N")

    ax.set_ylim([0, 80])
    ax.set_theta_zero_location("N")
    ax.set_xticks(np.deg2rad(betas))
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.25, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.deg2rad(np.arange(-90, 90 + step * 3, step * 3)), minor=False)
    ax.set_xticks(np.deg2rad(np.arange(-90 - step / 2, 90 + step / 2 + step, step)), minor=True)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )

    plt.plot(np.deg2rad(betas), sm.r0 * np.ones_like(betas), 'x', color="#FF1F5B", alpha=alpha_cte, markersize=2.5,
             label='Sup. Ext.')
    plt.plot(np.deg2rad(betas), (sm.r0 - 16) * np.ones_like(betas), 'x', color="#00CD6C", alpha=alpha_cte,
             markersize=2.5, label='Sup. Int.')
    ax.set_xlim([-np.pi / 2, np.pi / 2])


    # Imagem do Ensaio a corrigida
    plt.subplot(2, 2, 4, projection='polar')
    plt.gca().cla()
    cte_4db = 10**(4/20)
    cte_7db = 10**(7/20)
    cte_15db = 10**(15/20)
    img_insp_4db_sub = np.sum(data_4db.ascan_data - smartwedge_mask * cte_4db / cte_7db, 2)
    img_insp_4db_sub = np.log10(np.abs(img_insp_4db_sub) + log_offset)


    inspecao =  data[0].ascan_data_sum
    referencia = np.sum(smartwedge_mask, 2)
    plt.imshow(np.log10(np.abs(inspecao - referencia) + 1) , aspect='auto')


    plt.plot(t_span, referencia[:, 0, 0] * cte_4db / cte_7db, label='Wedge com ganho')
    plt.plot(t_span, inspecao[:, 0, 0], label='4dB')
    plt.plot(t_span, data_4db.ascan_data[:, 0, 0, 0] - smartwedge_mask[:, 0, 0, 0] * cte_4db / cte_7db, label='Subtração')

    plt.legend()

    img_insp_15db_sub = np.sum(data_15db.ascan_data - smartwedge_mask * cte_15db / cte_7db, 2)
    img_insp_15db_sub = np.log10(np.abs(img_insp_15db_sub) + log_offset)

    img_combined = np.zeros_like(img_insp_15db_sub)

    for i, delay in enumerate(delay_pattern):
        if delay == 0:
            img_combined[:, i, 0] = img_insp_15db_sub[:, i, 0]
        else:
            img_combined[:, i, 0] = img_insp_4db_sub[:, i, 0]

    img_combined_2 = shiftImage(img_combined[:, :, 0], delay_pattern, sample_time)
    img_combined_2 = img_combined_2[beg_idx:, :]

    vmin_cte = img_combined_2.min()
    vmax_cte = img_combined_2.max()

    t_prime_span = t_span[beg_idx:]

    # SEAM related parameters:
    lambda_param = 1
    rho_param = 100

    # Recorte da superfície externa:
    time_outer = time_spent + delay_pattern
    time_outer = np.mean(time_outer)
    time_idx_cte = 2
    upper_outer_idx = np.argmin(np.power(t_prime_span - ((time_outer-t_span[beg_idx]) - time_idx_cte + t_span[beg_idx]), 2))
    lower_outer_idx = np.argmin(np.power(t_prime_span - ((time_outer-t_span[beg_idx]) + time_idx_cte + t_span[beg_idx]), 2))
    outer_surf_img = img_combined_2[upper_outer_idx:lower_outer_idx, :]
    # Aplicação do SEAM:
    outer_norm_img = outer_surf_img/outer_surf_img.max()
    y = outer_norm_img
    a = img_line(y)
    tube_radii_2 = tube_radii[upper_outer_idx:lower_outer_idx]
    z = tube_radii_2[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Externa com SEAM")
    outer_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param, eta=.999, itmax=10, tol=1e-3)
    # plt.imshow(outer_surf_img, aspect='auto', extent=[betas[0]-step/2, betas[-1]+step/2, t_span_prime[-1], t_span_prime[0]])


    # Recorte da superfície interna:
    time_inner = time_spent + 2 * (16e-3) / (5.9e3) * 1e6 + delay_pattern
    time_inner = np.mean(time_inner)
    upper_inner_idx = np.argmin(np.power(t_prime_span - ((time_inner-t_span[beg_idx]) - time_idx_cte + t_span[beg_idx]), 2))
    lower_inner_idx = np.argmin(np.power(t_prime_span - ((time_inner-t_span[beg_idx]) + time_idx_cte + t_span[beg_idx]), 2))
    inner_surf_img = img_combined_2[upper_outer_idx:lower_outer_idx, :]
    # Aplicação do SEAM:
    inner_norm_img = inner_surf_img/inner_surf_img.max()
    y = inner_norm_img
    a = img_line(y)
    tube_radii_3 = tube_radii[upper_inner_idx:lower_inner_idx]
    z = tube_radii_3[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Externa com SEAM")
    inner_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param, eta=.999, itmax=10, tol=1e-3)
    # plt.imshow(inner_surf_img, aspect='auto')

    vmin_cte = img_combined_2.min()
    vmax_cte = img_combined_2.max()
    plt.pcolormesh(np.deg2rad(thetas), tube_radii, img_combined_2,
                   vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.grid()
    plt.title(f"Imagem Subtraida e Combinada")
    ax = plt.gca()
    ax.set_theta_zero_location("N")

    ax.set_ylim([0, 80])
    ax.set_theta_zero_location("N")
    ax.set_xticks(np.deg2rad(betas))
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.25, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.deg2rad(np.arange(-90, 90 + step * 3, step * 3)), minor=False)
    ax.set_xticks(np.deg2rad(np.arange(-90 - step / 2, 90 + step / 2 + step, step)), minor=True)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )

    plt.plot(np.deg2rad(betas), sm.r0 * np.ones_like(betas), 'x', color="#FF1F5B", alpha=alpha_cte, markersize=2.5,
             label='Sup. Ext.')
    plt.plot(np.deg2rad(betas), (sm.r0 - 16) * np.ones_like(betas), 'x', color="#00CD6C", alpha=alpha_cte,
             markersize=2.5, label='Sup. Int.')
    ax.set_xlim([-np.pi / 2, np.pi / 2])
    plt.plot(np.deg2rad(betas), outer_ext_zk, ':', alpha=0.85, color="#009ADE", label='SEAM Ext.')
    plt.plot(np.deg2rad(betas), inner_ext_zk, ':', alpha=0.85, color="#AF58BA", label='SEAM Int.')
    plt.legend(loc="lower center")

    return None

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
data_root = "C:/Users/Thiago/repos/Dados/AUSPEX/CENPES/jun2022/"
project_root = "C:/Users/Thiago/repos/AUSPEX_new/test/smartwedge_old"
filename = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1"
path1 = "resultados"
result_foldername = path1 + "/" + filename
extension = ".m2k"
data_path = data_root + filename + extension
# #
# data = file_m2k.read(data_path, sel_shots=9, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
#                      tp_transd='gaussian')
# data[0].ascan_data
#
# data_path = "C:/SharedFolder/SmartWedge/SmartWedge_OndaPlana.civa"
# app = "Mephisto"
# ascan_data = civa_pwi(data_path + f'/proc0/channels_signal_{app}_gate_1',
#                                                 len(betas), 64, 1, range(1))
# sectorial_img = np.sum(ascan_data, 2)
# plt.figure()
# plt.imshow(np.log10(envelope(sectorial_img) + 1e-5), aspect='auto', interpolation='none', extent=[-92.5, 92.5, 90, 40],
#            cmap=plt.get_cmap("Greys").reversed())

# Parâmetros do Vídeo:
generate_video = False

# O número de shots:
n_shot = 97


video_title = "VistaSetorial_Combinada_ComReferência_SEAM_Polar"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

fig, ax = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, constrained_layout=True, figsize=(14, 9))


shot = 0
if generate_video == True:
    # with writer.saving(fig, result_foldername + "/" + video_title + ".mp4", dpi=300):
    for shot in range(n_shot):
        GenerateImage(shot)
        print(f"frame = {shot + 1}")
        plt.savefig("frames/" + video_title + "_frame"+ f"{shot:.0f}" + ".jpg")
else:
    GenerateImage(shot)