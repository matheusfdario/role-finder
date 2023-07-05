import zipfile
import io
import os
import numpy as np
from framework import file_m2k


def append_npz(filename, data_name, data):
    data_npz = np.load(filename + ".npz", allow_pickle=True)
    data_npz2 = dict(data_npz)
    data_npz2[data_name] = data
    np.savez(filename, **data_npz2)
    return None


def save_data(filename, raw_data, idx, save_ascan_data=True, save_ascan_data_sum=True):
    if type(raw_data) is not list:
        data = [raw_data]
    else:
        data = raw_data

    for multisalvo_idx, data in enumerate(raw_data):
        if save_ascan_data:
            data_name = "ascan_" + f'{idx}' + "_multisalvo_" + f'{multisalvo_idx}'
            append_npz(filename, data_name, data.ascan_data)
        if save_ascan_data_sum:
            data_name = "ascan_sum_" + f'{idx}' + "_multisalvo_" + f'{multisalvo_idx}'
            append_npz(filename, data_name, data.ascan_data_sum)
    return None


def convert_m2k_to_npz(data_filename, npz_filename, n_shots, n_shots_ram, save_ascan_data=True,
                       save_ascan_data_sum=True):
    if n_shots_ram > n_shots:
        raise ValueError('O número de shots não pode ser menor do que o número de shots da partição.')
    if n_shots < 0 or n_shots < 0:
        raise ValueError('O número de shots tem que ser maior ou igual a zero.')
    if (type(n_shots) is not int) or (type(n_shots_ram) is not int):
        raise


    np.savez(npz_filename, array_none=None, allow_pickle=True)  # cria um npz vazio

    for i in range(0, n_shots // n_shots_ram):
        try:
            print(f"Salvando shot {i}")
            beg_idx = i * n_shots_ram
            end_idx = (i + 1) * n_shots_ram
            raw_data = file_m2k.read(data_filename, sel_shots=range(beg_idx, end_idx),
                                     type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                                     tp_transd='gaussian')
            save_data(npz_filename, raw_data, i, save_ascan_data, save_ascan_data_sum)
        except:
            print(f"Falha na leitura do shot {i}.")

    if n_shots % n_shots_ram:
        try:
            j = (n_shots // n_shots_ram)
            print(f"Salvando shot {j}")
            beg_idx = j * n_shots_ram
            end_idx = beg_idx + n_shots % n_shots_ram
            raw_data = file_m2k.read(data_filename, sel_shots=range(beg_idx, end_idx),
                                     type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                                     tp_transd='gaussian')
            save_data(npz_filename, raw_data, j, save_ascan_data, save_ascan_data_sum)
        except:
            print(f"Falha na leitura do shot {j}.")
    print("Finalizando o salvamento do .npz")
    return None


def create_memmap_from_obj_datainsp(memmap_filename, data_insp, n_shots):
    if type(data_insp) is list:
        ascan_shape_list = [None] * len(data_insp)
        ascan_sum_shape_list = [None] * len(data_insp)
        for i, _ in enumerate(ascan_shape_list):
            ascan_shape = data_insp[0].ascan_data.shape
            ascan_shape = (*ascan_shape[:-1], n_shots)
            ascan_shape_list[i] = ascan_shape
            ascan_shape_sum = data_insp[0].ascan_data.shape
            ascan_shape_sum = (*ascan_shape_sum[:-1], n_shots)
            ascan_sum_shape_list[i] = data_insp[0].ascan_data_sum.shape
    else:
        ascan_shape = data_insp.ascan_data.shape
        ascan_sum_shape = data_insp.ascan_data_sum.shape

    ascan_mmap = np.mmap(memmap_filename, dtype='float32', mode='w+', shape=ascan_shape)
    ascan_sum_mmap = np.mmap(memmap_filename + '_sum', dtype='float32', mode='w+', shape=ascan_sum_shape)
    return ascan_mmap, ascan_sum_mmap

def convert_m2k_to_npz(data_filename, n_shots, n_shots_ram, save_ascan_data=True,  memmap_filename=None,
                       save_ascan_data_sum=True):
    if n_shots_ram > n_shots:
        raise ValueError('O número de shots não pode ser menor do que o número de shots da partição.')
    if n_shots < 0 or n_shots < 0:
        raise ValueError('O número de shots tem que ser maior ou igual a zero.')
    if (type(n_shots) is not int) or (type(n_shots_ram) is not int):
        raise
    if memmap_filename is None:
        memmap_filename = data_filename


    raw_data = file_m2k.read(data_filename, sel_shots=1,
                                     type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                                     tp_transd='gaussian')

    ascan_mmap, ascan_sum_mmap = create_memmap_from_obj_datainsp(memmap_filename, raw_data, n_shots)

    for i in range(0, n_shots // n_shots_ram):
        try:
            print(f"Salvando shot {i}")
            beg_idx = i * n_shots_ram
            end_idx = (i + 1) * n_shots_ram
            raw_data = file_m2k.read(data_filename, sel_shots=range(beg_idx, end_idx),
                                     type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                                     tp_transd='gaussian')
            ascan_mmap[:, :, :, beg_idx:end_idx, :] = raw_data[:, :, :, beg_idx:end_idx, :
                                                      ]
            ascan_sum_mmap[:, :, beg_idx:end_idx, :] = raw_data[:, :, beg_idx:end_idx, :]
        except:
            print(f"Falha na leitura do shot {i}.")

    if n_shots % n_shots_ram:
        try:
            j = (n_shots // n_shots_ram)
            print(f"Salvando shot {j}")
            beg_idx = j * n_shots_ram
            end_idx = beg_idx + n_shots % n_shots_ram
            raw_data = file_m2k.read(data_filename, sel_shots=range(beg_idx, end_idx),
                                     type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                                     tp_transd='gaussian')
            ascan_mmap[:, :, :, beg_idx:end_idx, :] = raw_data[:, :, :, beg_idx:end_idx, :
                                                      ]
            ascan_sum_mmap[:, :, beg_idx:end_idx, :] = raw_data[:, :, beg_idx:end_idx, :]
        except:
            print(f"Falha na leitura do shot {j}.")
    print("Finalizando o salvamento do .npz")
    return None



if __name__ == '__main__':
    # File_root:
    data_root = "G:/.shortcut-targets-by-id/1T3yHfBX35wLz0UjXeyUCLflWgR_wB0yv/AUSPEX_OFICIAL/Dados/CENPES/2022 Junho/CENPES 2022-06-29/"
    # file_name = 'MultiSalvo Smart Wedege 12-08-22 with TCG 31_25 Pos 0 v1.mit2k'
    file_name = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1.m2k"

    # file_name = "ensaio_frio_falha1.m2k"
    path = data_root + file_name
    #
    # n_shots = 650  # Número total de shots no ensaio
    # n_shots_ram = 100  # Número máximo de shots suportados pela ram
    out_n_shots = 60   # Número total de shots no ensaio
    out_n_shots_ram = 5  # Número máximo de shots suportados pela ram

    convert_m2k_to_npz(path, out_n_shots_ram, out_n_shots_ram, save_ascan_data_sum=True, memmap_filename='teste')
