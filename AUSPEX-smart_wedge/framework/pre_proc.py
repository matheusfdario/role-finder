# -*- coding: utf-8 -*-
"""
Módulo ``pre_proc``
===================

Neste módulo estão implementados os algoritmos para realizar o processamento nos dados carregados pelo *framework*. 
Todas as funções recebem um objeto do tipo :class:`.data_types.DataInsp` e um array do tipo :class:`numpy.ndarray` com o nome *shots*.

"""

import numpy as np
import scipy.signal as sig
from framework import file_civa


def remove_media(data_insp, shots=np.asarray([0])):
    """Remove a media de um sinal.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Objeto com os dados carregados pelo *framework*.

    shots : :class:`np.ndarray`
        Vetor com os disparos a serem processados.

    Returns
    -------
    :class:`numpy.ndarray`
        Dados processados.

    """
    shots = shots.astype(int)
    data_insp.ascan_data[:, :, :, shots] -= data_insp.ascan_data[:, :, :, shots].mean()
    return data_insp.ascan_data[:, :, :, shots]


def add_noise(data_insp, snr=50, shots=np.asarray([0])):
    """Adiciona ruído com SNR desejada.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Objeto com os dados carregados pelo *framework*.

    snr : :class:`float`
        SNR desejada.

    shots : :class:`np.ndarray`
        Vetor com os disparos a serem processados.

    Returns
    -------
    :class:`numpy.ndarray`
        Dados com ruido.

    """
    shots = shots.astype(int)
    n = data_insp.ascan_data.shape[1] * data_insp.ascan_data.shape[2]
    pw_shot = (np.abs(data_insp.ascan_data[:, :, :, shots])**2).sum()/n
    pw_noise = pw_shot/(10**(snr/10))
    # noise = np.random.normal(0, np.sqrt(pw_noise)/2, [2, *data_insp.ascan_data[:, :, :, shots].shape])
    for shot in shots:
        if np.iscomplexobj(data_insp.ascan_data):
            data_insp.ascan_data[:, :, :, shot] += np.random.normal(0, np.sqrt(pw_noise)/2,
                                                                     data_insp.ascan_data[:, :, :, shot].shape)
            data_insp.ascan_data[:, :, :, shot] += 1j*np.random.normal(0, np.sqrt(pw_noise)/2,
                                                                        data_insp.ascan_data[:, :, :, shot].shape)
        else:
            data_insp.ascan_data[:, :, :, shot] += np.random.normal(0, np.sqrt(pw_noise),
                                                                     data_insp.ascan_data[:, :, :, shot].shape)
    # noise = (noise[0]+1j*noise[1]).astype(data_insp.ascan_data.dtype)
    # data_insp.ascan_data[:, :, :, shots] += noise
    # del noise
    return data_insp.ascan_data[:, :, :, shots]


def sum_shots(data_insp, shots=np.asarray([0]),):
    """Soma vários disparos diferentes. Salva o resultado no primeiro disparo da lista.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Objeto com os dados carregados pelo *framework*.

    shots : :class:`np.ndarray`
        Vetor com os disparos a serem somados.

    Returns
    -------
    :class:`numpy.ndarray`
        Disparo com os dados somados.

    """
    shots = shots.astype(int)
    data_insp.ascan_data[:, :, :, shots[0]] = data_insp.ascan_data[:, :, :, shots].sum(-1)
    return data_insp.ascan_data[:, :, :, [shots[0]]]


def matched_filter(data_insp, shots=np.asarray([0])):
    """Filtra o sinal com um filtro casado para melhorar a SNR :cite:`Turin60`
    (`referência <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1057571>`_)

    Assume-se que os ecos tenham o formato de um pulso gaussiano com as características do transdutor, frequência
    central e largura de banda. As características do transdutor são as mesmas presentes na estrutura ``DataInsp``.
    Esse pulso gaussiano é a resposta em frequência do filtro aplicado a cada A-scan.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Objeto com os dados carregados pelo *framework*.

    shots : :class:`np.ndarray`
        Vetor com os disparos a serem somados.

    Returns
    -------
    :class:`numpy.ndarray`
        Dados processados.

    """
    shots = shots.astype(int)
    t0 = 5e-6
    t = np.linspace(-t0/2, t0/2, t0*data_insp.inspection_params.sample_freq*1e6)
    gp = sig.gausspulse(t, data_insp.probe_params.central_freq*1e6, data_insp.probe_params.bw)
    data_insp.ascan_data[:, :, :, shots] = np.apply_along_axis(sig.correlate, 0,
                                                               data_insp.ascan_data[:, :, :, shots], gp, 'same')
    return data_insp.ascan_data[:, :, :, shots]



def hilbert_transforms(data, shots=np.array([0]), N=2):
    """
    Transformada de Hilbert para arquivos muito pesados, visando reduzir o custo. A forma mais rápida encontrada foi
    fazendo a transformada 2 shots por vez em um FMC.

    Parameters
    ----------
    data: o arquivo a ser aplicada a transformada, deve ser tipo data_insp ou FMC;

    shots: os shots nos quais será realizada a transformada

    Returns
    -------

    data: retorna um ponteiro do data.ascan_data
    """
    shots = np.asarray(shots).astype(np.int)
    n_shots = len(shots)
    sr = np.arange(0, data.ascan_data.shape[-1])
    if not np.all(np.isin(sr, shots)):
        data.ascan_data = np.delete(data.ascan_data, np.where(~np.isin(sr, shots)), axis=-1)
    if not np.iscomplexobj(data.ascan_data):
        data.ascan_data = data.ascan_data.astype(np.complex64)
    data.inspection_params.step_points = data.inspection_params.step_points[shots] #Revisar
    for i in range(int(data.ascan_data.shape[-1] / N + 0.5)):
        if N * i == min(N * i + N, n_shots):
            pass
        else:
            data.ascan_data[:-1, :, :, shots[N * i:min(N * i + N, n_shots)]] = \
                sig.hilbert(np.real(data.ascan_data[:-1, :, :, shots[N * i:min(N * i + N, n_shots)]]), axis=0)\
                .astype(np.complex64)

    return data.ascan_data
