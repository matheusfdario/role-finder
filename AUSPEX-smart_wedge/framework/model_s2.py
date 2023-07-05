# -*- coding: utf-8 -*-
"""
Módulo ``model_s2``
===============

O módulo :mod:`.model_s2` contém as funções de suporte para a criação de algoritmos
de reconstrução de imagens baseados em resolução de problemas inversos.

"""

import numpy as np
from framework.fk_mig import f_k_sweep, nextpow2
from framework.linterp_oper import linterp_numba


def model_s2_direct(image, nt0, nu0, roi=None, dt=1e-8, du=1e-3, c=5900.0, tau0=0.0,
                    model_transd=None, coord_orig=np.zeros(3)):
    """
    Essa função é responsável por calcular o modelo do operador direto da matriz H.

    Parameters
    ----------
    image : :class:`.np.ndarray`
        imagem resultante da reconstrução na iteração anterior.

    nt0 : :class:`int` ou :class:`float`
        Número de amostras de A-scan a ser usado.

    nu0 : :class:`int` ou :class:`float`
        Número de posições do transdutor usada no ensaio.

    roi : :class:`.data_types.ImagingROI`
        Região de interesse na qual o algoritmo será executado. As dimensões
        da ROI devem estar em mm.

    dt : :class:`int` ou :class:`float`
        Tamanho da amostra de tempo.

    du : :class:`int` ou :class `float`
        Amostra de tempo.

    c : :class:`int` ou :class:`float`
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    tau0 : :class:`int` ou :class:`float`
        Grade de tempo do algoritmo.

    model_transd : :class:`model_point"
        Modelo do transutor.

    coord_orig : :class:`.np.array`
        Coordenada de origem do sistema.

    """

    # Zero-padding para interpolação no domínio do tempo.
    if model_transd is not None:
        nt, nu = model_transd.shape
    else:
        nt = 0
        nu = 0

    if nt < nt0:
        nt = int(2.0 ** (nextpow2(nt0) + 1))

    if nu < nu0:
        nu = int(2.0 ** (nextpow2(nu0) + 1))

    # Converte a imagem para o domínio da frequência.
    if roi is not None:
        nx = int(2.0 ** (nextpow2(roi.w_len) + 1))
        nz = int(2.0 ** (nextpow2(roi.h_len) + 1))
        ftimage = np.fft.fftshift(np.fft.fft2(image, s=(nz, nx)))
    else:
        nx = nu
        nz = nt
        ftimage = np.fft.fftshift(np.fft.fft2(image, s=(nt, nu)))

    # Calcula os grids necessários para a execução do algoritmo.
    # Grids dos dados.
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    ku = np.fft.fftshift(np.fft.fftfreq(nu, d=du))

    # Grids da imagem.
    if roi is not None:
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=(roi.w_step * 1e-3)))
        kz = np.fft.fftshift(np.fft.fftfreq(nz, d=(roi.h_step * 1e-3)))
    else:
        kx = ku
        kz = f / (c / 2.0)

    # Ajusta o espectro da imagem em relação ao deslocamento da ROI.
    # Aqui é necessária a coordenada de origem primeiro elemento/posição do transdutor.
    if roi is not None:
        x0 = (roi.w_points[0] - coord_orig[0]) * 1.e-3
        if x0 != 0.0:
            ftimage = ftimage * np.exp(-2j * np.pi * kx * x0)[np.newaxis, :]

        z0 = (roi.h_points[0] - coord_orig[2]) * 1.e-3
        if z0 != 0.0:
            ftimage = ftimage * np.exp(-2j * np.pi * kz * z0)[:, np.newaxis]

    else:
        z0 = 0.0

    # Grids para a Transformada de Stolt.
    ku_kz, kz_ku = np.meshgrid(ku, kz)
    f_kz = f_k_sweep(c, kz_ku, ku_kz)

    # Gera espectro dos dados pelo operador adjunto da interpolação linear 2D.
    if roi is not None:
        kx_kz, _ = np.meshgrid(kx, kz)
        ftdata = linterp_numba(f, f_kz, linterp_numba(ku, kx_kz.T, ftimage.T, op='a').T, op='a')
    else:
        ftdata = linterp_numba(f, f_kz, ftimage, op='a')

    # Insere o modelo do transdutor (se existir).
    if model_transd is not None:
        if model_transd.shape == ftdata.shape:
            # Modelo no formato dos dados.
            ftdata = ftdata * model_transd
        else:
            # Monta o modelo no formato dos dados.
            row_shift = np.floor((nt - model_transd.shape[0]) // 2)
            col_shift = np.floor((nu - model_transd.shape[1]) // 2)
            model_transd = np.lib.pad(model_transd,
                                      ((0, nt - model_transd.shape[0]), (0, nu - model_transd.shape[1])),
                                      "constant",
                                      constant_values=(0,))
            model_transd = np.roll(model_transd, (row_shift, col_shift))
            ftdata = ftdata * model_transd

    # Ajusta o espectro em relação ao tempo ``tau0``.
    if tau0 != 0.0:
        ftdata = ftdata * np.exp(2j * np.pi * f * tau0 * 1e-6)[:, np.newaxis]

    # Retorna para o domínio do tempo.
    data = np.real(np.fft.ifft2(np.fft.ifftshift(ftdata)))
    if roi is not None:
        ermv = c / 2.0
        ze = (roi.h_points[-1] - coord_orig[2]) * 1e-3
        # É necessário calcular os dois índices para evitar erros de arredondamento.
        # Esses índices são em relação ao grid de tempo, não ao grid da matriz ``data``.
        # É importante lembrar que a primeira linha da matriz ``data`` contém os dados da posição ``z0``.
        idx_t0 = np.floor((z0 / ermv) / dt + 0.5).astype(int)
        idx_te = np.floor((ze / ermv) / dt + 0.5).astype(int)
        data = data[0: idx_te - idx_t0, 0: nu0]
    else:
        data = data[0: nt0, 0: nu0]

    # Calcula o ganho.
    np.seterr(divide='ignore', invalid='ignore')
    en_image = np.linalg.norm(image.flatten()) ** 2
    en_data = np.linalg.norm(data.flatten()) ** 2
    ganho = np.sqrt(en_image / en_data)

    return data, ganho


def model_s2_adjoint(data, nt0, nu0, roi=None, dt=1e-8, du=1e-3, c=5900.0, tau0=0.0,
                     filter_transd=None, coord_orig=np.zeros(3)):
    """
    Essa função é responsável por calcular o modelo do operador direto da matriz H.

    Parameters
    ----------
    data : :class:`.np.ndarray`
        matriz da imagem resultante.

    nt0 : :class:`int` ou :class:`float`
        Número de amostras de A-scan a ser usado.

    nu0 : :class:`int` ou :class:`float`
        Número de posições do transdutor usada no ensaio.

    roi : :class:`.data_types.ImagingROI`
        Região de interesse na qual o algoritmo será executado. As dimensões
        da ROI devem estar em mm.

    dt : :class:`int` ou :class:`float`
        Tamanho da amostra de tempo.

    du : :class:`int` ou :class `float`
        Amostra de tempo.

    c : :class:`int` ou :class:`float`
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    tau0 : :class:`int` ou :class:`float`
        Grade de tempo do algoritmo.

    filter_transd : :class:`float`
        Modelo do filtro casado.

    coord_orig : :class:`.np.array`
        Coordenada de origem do sistema.

    """

    # Zero-padding para interpolação no domínio do tempo.
    if filter_transd is not None:
        nt, nu = filter_transd.shape
    else:
        nt = 0
        nu = 0

    if nt < nt0:
        nt = int(2.0 ** (nextpow2(nt0) + 1))

    if nu < nu0:
        nu = int(2.0 ** (nextpow2(nu0) + 1))

    # Calcula a FFT 2D dos sinais A-scan.
    ftdata = np.fft.fftshift(np.fft.fft2(data, s=(nt, nu)))

    # Calcula os grids necessários para a execução do algoritmo.
    # Grids dos dados.
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    ku = np.fft.fftshift(np.fft.fftfreq(nu, d=du))

    # Grids da imagem.
    if roi is not None:
        nx = int(2.0 ** (nextpow2(roi.w_len) + 1))
        nz = int(2.0 ** (nextpow2(roi.h_len) + 1))
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=(roi.w_step * 1e-3)))
        kz = np.fft.fftshift(np.fft.fftfreq(nz, d=(roi.h_step * 1e-3)))
    else:
        kx = ku
        kz = f / (c / 2.0)

    # Ajusta o espectro em relação ao tempo ``tau0``.
    if tau0 != 0.0:
        ftdata = ftdata * np.exp(-2j * np.pi * f * tau0 * 1e-6)[:, np.newaxis]

    # Aplica o filtro (se existir).
    if filter_transd is not None:
        if filter_transd.shape == ftdata.shape:
            # Filtro no formato dos dados de entrada.
            ftdata = ftdata * filter_transd
        else:
            # Monta o filtro no formato dos dados de entrada.
            row_shift = np.floor((nt - filter_transd.shape[0]) // 2)
            col_shift = np.floor((nu - filter_transd.shape[1]) // 2)
            filter_transd = np.lib.pad(filter_transd,
                                       ((0, nt - filter_transd.shape[0]), (0, nu - filter_transd.shape[1])),
                                       "constant",
                                       constant_values=(0,))
            filter_transd = np.roll(filter_transd, (row_shift, col_shift))
            ftdata = ftdata * filter_transd

    # Grids para a Transformada de Stolt.
    ku_kz, kz_ku = np.meshgrid(ku, kz)
    f_kz = f_k_sweep(c, kz_ku, ku_kz)

    # Aplica a Transformada de Stolt (migração f-k), via interpolação linear 2D.
    if roi is not None:
        kx_kz, _ = np.meshgrid(kx, kz)
        ftimage = linterp_numba(ku, kx_kz.T, linterp_numba(f, f_kz, ftdata).T).T
    else:
        ftimage = linterp_numba(f, f_kz, ftdata)

    # Ajusta o espectro da imagem em relação ao deslocamento da ROI.
    # Aqui é necessária a coordenada do transdutor.
    if roi is not None:
        x0 = (roi.w_points[0] - coord_orig[0]) * 1e-3
        if x0 != 0.0:
            ftimage = ftimage * np.exp(2j * np.pi * kx * x0)[np.newaxis, :]

        z0 = (roi.h_points[0] - coord_orig[2]) * 1e-3
        if z0 != 0.0:
            ftimage = ftimage * np.exp(2j * np.pi * kz * z0)[:, np.newaxis]

    # Retorna para o domínio da imagem.
    image = np.real(np.fft.ifft2(np.fft.ifftshift(ftimage)))
    if roi is not None:
        image = image[0: roi.h_len, 0: roi.w_len]
    else:
        image = image[0: nt0, 0: nu0]

    # Calcula o ganho.
    np.seterr(divide='ignore', invalid='ignore')
    en_image = np.linalg.norm(image.flatten()) ** 2
    en_data = np.linalg.norm(data.flatten()) ** 2
    ganho = np.sqrt(en_image / en_data)

    return image, ganho


def shrinkage(data, value):

    """
    Essa função é responsável por calcular o algoritmo de shrinkage treshold.

    Parameters
    ----------
    data : :class:`.np.ndarray`
        Matriz a ser aplicado o algoritmo.

    value : :class:`int` ou :class:`float`
        Valor do treshold.

    """

    value = np.maximum(value, 0.0)
    return np.sign(data) * np.maximum(np.abs(data) - value, 0.0)
