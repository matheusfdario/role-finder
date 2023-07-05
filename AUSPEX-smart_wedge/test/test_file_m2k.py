import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from sys import platform, exit

from framework import file_m2k


# Função para apresentar animação de bscans.
def bscan_animation(d, s, title):
    def bscan_animation_func(i):
        return [plt.imshow(d.ascan_data[:, i, :, s], aspect='auto', animated=True)]

    fig = plt.figure()
    plt.title(title)
    animation.FuncAnimation(fig, bscan_animation_func, frames=d.ascan_data.shape[1], blit=True, repeat=True,
                            interval=100)
    plt.show(block=True)


# Função para apresentar animação de TFMs.
def tfm_animation(d, title):
    def tfm_animation_func(i):
        key = list(d.imaging_results.keys())[i]
        return [plt.imshow(d.imaging_results[key].image, aspect='auto', animated=True,
                           extent=(d.imaging_results[key].roi.w_points[0],
                                   d.imaging_results[key].roi.w_points[-1],
                                   d.imaging_results[key].roi.h_points[-1],
                                   d.imaging_results[key].roi.h_points[0]))]

    fig = plt.figure()
    plt.title(title)
    animation.FuncAnimation(fig, tfm_animation_func, frames=len(d.imaging_results), blit=True, repeat=True,
                            interval=100)
    plt.show(block=True)


# Vários arquivos diferentes para teste da leitura de arquivos MULTISALVO.
if platform == "win32":
    files_panther_path = "C:/Users/GiovanniAlfredo/Google Drive/AUSPEX/Dados Panther/"
elif platform == "linux":
    files_panther_path = "/home/giovanni/Documents/Panther/"
elif platform == "darwin":
    files_panther_path = "AUSPEX/Dados Panther/"
else:
    files_panther_path = ""
    print("Plataforma não suportada.")
    exit(1)

# Nomes do arquivo para testes.
filenames = [#"Encoder/Ensaio2_z_var.m2k",
             #"Encoder/Encoder_sem_movimento.m2k",
             "Encoder/Encoder_endvale.m2k",
             #"Encoder/Encoder_time_scan.m2k",
             #"Encoder/Ensaio1_cobrinha.m2k",
             #"WCNDT/CERNN/TESTE COM 2 TRANSDUTORES/FLUIDO_27_TESTE_MEIO.m2k",  # Transdutores divididos (não resolvido)
             #"Testes_giovanni/fmc.m2k",  # 8
             #"WCNDT/Ensaios_WCNDT_Passarin_Kalid/ensaio_4_antigo.m2k",
             #"ensaio_rugosidade_imersao_rugosidade_entrada.m2k",
             "Encoder/Motor_quad_s1_lento.m2k",
             "CP1_Bot_50_Direct.m2k",  # 0
             "fmc_gate6.m2k",  # 1
             "pwi(-10_10_20p).m2k",  # 2
             "tfm-pwi.m2k",  # 3
             "unisequencial/Aluminio_0db.m2k",  # 4
             "multisalvo/multisalvo_tfm_pwi.m2k",  # 5
             "multisalvo/multisalvo.m2k",  # 6
             "multisalvo/multisalvo_unisequencial_fmc.m2k",  # 7
             "Testes_giovanni/fmc.m2k",  # 8
             "Testes_giovanni/fmc_tfm.m2k",  # 9
             "Testes_giovanni/pwi.m2k",  # 10
             "Testes_giovanni/pwi_tfm.m2k",  # 11
             "Testes_giovanni/fmc_4_shots_1_perdido.m2k"]  # 12

# Carrega arquivo de testes
idx_teste = 0
# shots = None
shots = range(0, 10)
# shots = np.random.permutation(32)[0:4].tolist()
# shots = np.random.permutation(67)[0:4].tolist()

data_list = file_m2k.read(files_panther_path + filenames[idx_teste],
                          sel_shots=shots,
                          type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

if type(data_list) is list:
    print("Número de ``DataInsp`` = ", len(data_list))
    for data in data_list:
        print("Número de shots p/ '" + data.dataset_name + "' = ", data.ascan_data.shape[-1])
        print("Tipo de captura: " + data.inspection_params.type_capt)

        # Apresenta os B-scans importados.
        shots_in_file = range(data.ascan_data.shape[-1])
        if shots is None:
            shots = shots_in_file
        for shot in shots:
            shot_idx = shots.index(shot)
            bscan_animation(data, shot_idx, "'" + data.dataset_name + "', Shot[" + str(shot)
                            + "] - (" + str(shot_idx + 1) + "/" + str(data.ascan_data.shape[-1]) + ")")

        # Apresenta as imagens TFM, se existirem.
        if len(data.imaging_results) > 0:
            tfm_animation(data, "'" + data.dataset_name + "' - Imagens TFM")

else:
    print("Número de ``DataInsp`` = 1")
    data = data_list
    print("Número de shots no ``DataInsp`` = ", data.ascan_data.shape[-1])
    print("Tipo de captura: " + data.inspection_params.type_capt)

    # Apresenta os B-scans importados.
    shots_in_file = range(data.ascan_data.shape[-1])
    if shots is None:
        shots = shots_in_file
    for shot in shots:
        shot_idx = shots.index(shot)
        bscan_animation(data, shot_idx, "'" + data.dataset_name + "' - Shot[" + str(shot) + "] - ("
                        + str(shot_idx + 1) + "/" + str(data.ascan_data.shape[-1]) + ")")

    # Apresenta as imagens TFM, se existirem.
    if len(data.imaging_results) > 0:
        tfm_animation(data, "'" + data.dataset_name + "' - Imagens TFM")
