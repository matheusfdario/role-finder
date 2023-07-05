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

from polarImg import *




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



    img_ref = np.sum(smartwedge_mask, 2)
    img_ref = np.log10(np.abs(img_ref) + log_offset)


    # Plotando a figura:
    plt.suptitle(f"Shot : {shot} || Ensaio: {filename}")

    # Imagem do Ensaio a 4 dB
    plt.subplot(2, 3, 1)
    img_insp_4db = np.sum(data_4db.ascan_data, 2)
    img_insp_4db = np.log10(np.abs(img_insp_4db) + log_offset)

    vmin_cte = img_insp_4db.min()
    vmax_cte = img_insp_4db.max()

    plt.imshow(img_insp_4db, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Ganho = 4 dB")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent+ + 2*(16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5, label='Sup. Int. Teórica')
    vetor_medio = np.array([moving_average(img_insp_4db[:, i, 0], 7) for i in range(img_insp_4db.shape[1])]).transpose()
    max_idx = np.argmax(vetor_medio, 0) + 1
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
    #          'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")


    # Imagem do Ensaio a 15 dB
    plt.subplot(2, 3, 2)
    plt.imshow(img_ref, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Referência (7 dB)")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int. Teórica')


    max_idx = np.argmax(img_ref, 0)
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
    #          'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")


    # Imagem do Ensaio a 15 dB
    plt.subplot(2, 3, 3)
    img_insp_15db = np.sum(data_15db.ascan_data, 2)
    img_insp_15db = np.log10(np.abs(img_insp_15db) + log_offset)

    plt.imshow(img_insp_15db, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Ganho = 15 dB")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int. Teórica')

    max_idx = np.argmax(img_insp_15db, 0)
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
    #          'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")


    # Imagem do Ensaio a corrigida
    plt.subplot(2, 3, 4)
    cte_4db = 10**(4/20)
    cte_7db = 10**(7/20)
    cte_15db = 10**(15/20)
    img_insp_4db_sub = np.sum(data_4db.ascan_data - smartwedge_mask * cte_4db / cte_7db, 2)
    img_insp_4db_sub = np.log10(np.abs(img_insp_4db_sub) + log_offset)
    img_insp_15db_sub = np.sum(data_15db.ascan_data - smartwedge_mask * cte_15db / cte_7db, 2)
    img_insp_15db_sub = np.log10(np.abs(img_insp_15db_sub) + log_offset)

    img_combined = np.zeros_like(img_insp_4db_sub)
    #
    # plt.plot(t_span, np.sum(img_combined[:, 9:27], 1))
    for i, delay in enumerate(delay_pattern):
        if delay == 0:
            img_combined[:, i, 0] = img_insp_15db_sub[:, i, 0]
        else:
            img_combined[:, i, 0] = img_insp_4db_sub[:, i, 0]

    vmin_cte = img_combined.min()
    vmax_cte = img_combined.max()
    plt.imshow(img_combined, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Imagem Combinada e Subtraida")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')

    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int. Teórica')

    max_idx = np.argmax(img_combined, 0)
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
    #          'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")

    # Imagem do Ensaio a corrigida dB
    plt.subplot(2, 3, 5)
    img_shifted = shiftImage(img_combined[:, :, 0], delay_pattern, sample_time)

    # SEAM related parameters:
    lambda_param = 1
    rho_param = 100

    # Recorte da superfície externa:
    time_outer = time_spent + delay_pattern
    time_outer = np.mean(time_outer)
    time_idx_cte = 2
    upper_outer_idx = np.argmin(np.power(t_span - time_outer - time_idx_cte, 2))
    lower_outer_idx = np.argmin(np.power(t_span - time_outer + time_idx_cte, 2))
    outer_surf_img = img_shifted[lower_outer_idx:upper_outer_idx, :]
    # Aplicação do SEAM:
    outer_norm_img = outer_surf_img/outer_surf_img.max()
    y = outer_norm_img
    a = img_line(y)
    t_span_prime = t_span[lower_outer_idx:upper_outer_idx]
    z = t_span_prime[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Externa com SEAM")
    outer_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param, eta=.999, itmax=10, tol=1e-3)
    # plt.imshow(outer_surf_img, aspect='auto', extent=[betas[0]-step/2, betas[-1]+step/2, t_span_prime[-1], t_span_prime[0]])


    # Recorte da superfície interna:
    time_inner = time_spent + 2 * (16e-3) / (6.3e3) * 1e6 + delay_pattern
    time_inner = np.mean(time_inner)
    upper_inner_idx = np.argmin(np.power(t_span - time_inner - time_idx_cte, 2))
    lower_inner_idx = np.argmin(np.power(t_span - time_inner + time_idx_cte, 2))
    inner_surf_img = img_shifted[lower_inner_idx:upper_inner_idx, :]
    # Aplicação do SEAM:
    inner_norm_img = inner_surf_img/inner_surf_img.max()
    y = inner_norm_img
    a = img_line(y)
    t_span_prime = t_span[lower_inner_idx:upper_inner_idx]
    z = t_span_prime[a[0].astype(int)]
    w = np.diag((a[1]))
    print(f"SEAM: Estimando superfíce Externa com SEAM")
    inner_ext_zk, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb=lambda_param, x0=z, rho=rho_param, eta=.999, itmax=10, tol=1e-3)
    # plt.imshow(inner_surf_img, aspect='auto')


    plt.imshow(img_shifted, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Imagem Combinada, Subtraida e Deslocada")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent + delay_pattern, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6 + delay_pattern, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int. Teórica')

    max_idx = np.argmax(img_shifted, 0)
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
    #          'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")
    plt.plot(betas, outer_ext_zk, ':', alpha=0.5, color="#FF1F5B", label='SEAM Ext.')
    plt.plot(betas, inner_ext_zk, ':', alpha=0.5, color="#00CD6C", label='SEAM Int.')


    # Imagem do Ensaio a corrigida dB
    plt.subplot(2, 3, 6)
    img_shifted = shiftImage(img_combined[:, :, 0], delay_pattern, sample_time)
    plt.imshow(img_shifted, aspect='auto', interpolation='bicubic', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Imagem Final Interpolada")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent + delay_pattern, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5,
             label='Sup. Ext. Teórica')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6 + delay_pattern, 'd', color=[.5, 0, .5], alpha=alpha_cte,
             markersize=1.5,
             label='Sup. Int. Teórica')

    max_idx = np.argmax(img_shifted, 0)
    # plt.plot(betas, np.array([t_span[i] for i in max_idx]),
             # 'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")
    plt.plot(betas, outer_ext_zk, ':', alpha=0.5, color="#FF1F5B", label='SEAM Ext.')
    plt.plot(betas, inner_ext_zk, ':', alpha=0.5, color="#00CD6C", label='SEAM Int.')


    plt.legend(loc="upper center")

    plt.tight_layout()


    # # Tempo inicial: 40us tempo final: 90us
    # # t0 ao t1 é alumínio:
    # t0 = 90
    # t1 = wedge_time[0]
    # delta_t_wedge = t1 - 40
    #
    # # t1 ao t2 é água (coluna d'gua):
    # t2 = time_spent[0]
    # delta_t_waterpath = t2 - t1
    #
    # # t2 ao t3 é espessura da tubulação (aço):
    # v_steel = 6300 #em m/s
    # wall_thickness = 16 # em mm
    # delta_t_wall = (2 * wall_thickness * 1e-3)/(v_steel) * 1e6 # em us
    # t3 = t2 + delta_t_wall
    #
    # # t3 ao t4 é o interior da tubulação (água):
    # radius = 2 * (sm.r0 - wall_thickness) # em mm
    # delta_t_center = (radius * 1e-3)/(1483) * 1e6 # em us
    # t4 = t3 + delta_t_center
    #
    # # No sistema de coordenadas centrada na tubulação:
    # # d0 e o valor x de 90 us
    # d0 = (sm.r0 - wall_thickness) * (t4 - 90)/(t4 - t3)
    # d1 = sm.r0 - wall_thickness
    # idx_beg = t_span.shape[0]
    # idx_end = np.argmin(np.power(t_span - t3, 2))
    # x_span_1 = np.linspace(d0, d1, idx_beg - idx_end)
    #
    # # Para a região na parede da tubulação
    # d2 = d1 + 16
    # idx_beg = np.argmin(np.power(t_span - t3, 2)) - 1
    # idx_end = np.argmin(np.power(t_span - t2, 2))
    # x_span_2 = np.linspace(d1, d2, idx_beg - idx_end)
    #
    # # Para a região da coluna d'gua:
    # d3 = d2 + np.linalg.norm(sm.N)-sm.r0
    # idx_beg = np.argmin(np.power(t_span - t2, 2)) - 1
    # idx_end = np.argmin(np.power(t_span - t1, 2))
    # x_span_3 = np.linspace(d2, d3, idx_beg - idx_end)
    #
    # z_coordinates = np.hstack((x_span_1, x_span_2, x_span_3))
    # x_coordinates = betas
    #
    # img2convert = img_shifted[idx_end:, :]
    # cart = CartesianData(
    #     np.linspace(-100, 100, 150),
    #     np.linspace(-10, 80, 150)
    # )
    #
    # thetas = np.linspace(-np.pi/2, np.pi/2, 150)
    # cyl = CilinderData(
    # np.linspace(z_coordinates[0], z_coordinates[-1], img2convert.shape[0]),
    #     thetas
    # )
    #
    # img_converted = impolar(cyl, cart, img_shifted[::-1, :])
    # plt.imshow(np.flip(img_converted.transpose(), axis=0),
    #            aspect='auto', interpolation='none', cmap=plt.get_cmap("Greys").reversed(), vmin=vmin_cte, vmax=vmax_cte,
    #            extent=[cart.x_coords[0], cart.x_coords[-1], cart.z_coords[0], cart.z_coords[-1]]
    #            )
    #
    # r_ext = sm.r0
    # plt.plot(r_ext * np.sin(thetas), r_ext * np.cos(thetas), ':', color='red', alpha=0.5)
    # r_int = sm.r0 - 16
    # plt.plot(r_int * np.sin(thetas), r_int * np.cos(thetas), ':', color=[0.8, 0, 0.4], alpha=0.5)

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
filename = "Smart Wedge 29-06-22 100 V Foco Sup Ext Aquisicao 0h v1"
path1 = "resultados"
result_foldername = path1 + "/" + filename
extension = ".m2k"
data_path = data_root + filename + extension
#
data = file_m2k.read(data_path, sel_shots=9, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                     tp_transd='gaussian')
data[0].ascan_data

data_path = "C:/SharedFolder/SmartWedge/SmartWedge_OndaPlana.civa"
app = "Mephisto"
ascan_data = civa_pwi(data_path + f'/proc0/channels_signal_{app}_gate_1',
                                                len(betas), 64, 1, range(1))
sectorial_img = np.sum(ascan_data, 2)
plt.figure()
plt.imshow(np.log10(envelope(sectorial_img) + 1e-5), aspect='auto', interpolation='none', extent=[-92.5, 92.5, 90, 40],
           cmap=plt.get_cmap("Greys").reversed())

# Parâmetros do Vídeo:
generate_video = False

# O número de shots:
n_shot = 97


video_title = "VistaSetorial_Combinada_ComReferência_SEAM"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
fig = plt.figure(figsize=(14, 9))
# fig = plt.figure(constrained_layout=True, figsize=(14, 9))

shot = 0
if generate_video == True:
    with writer.saving(fig, result_foldername + "/" + video_title + "14dB_" + ".mp4", dpi=300):
        for shot in range(n_shot):
            GenerateImage(shot)
            print(f"frame = {shot + 1}")
            writer.grab_frame()
            plt.clf()
else:
    GenerateImage(shot)