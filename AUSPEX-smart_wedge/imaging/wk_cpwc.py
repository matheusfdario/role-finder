# -*- coding: utf-8 -*-
r"""
Módulo ``wk_cpwc``
==================

O :math:`\omega k`-CPWC é um algoritmo de reconstrução de imagens para
inspeções que utilizam ondas planas (*plane waves*) na aquisição dos dados. O
algoritmo é baseado na migração de Stolt, sendo análogo aos algoritmos
:math:`\omega k`-SAFT e *Wavenumber*.

O algoritmo consiste em obter imagens para cada ângulo de disparo, formando uma
imagem final a partir da soma das imagens parciais. Como o algoritmo é
desenvolvido no domínio da frequência, seu desempenho pode ser melhor do que
algoritmos no domínio do tempo, quando o número de pontos da região de interesse
é elevado.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo :math:`\omega k`-CPWC para a
reconstrução de uma imagem a partir de dados sintéticos, oriundos do simulador
CIVA. (Assume-se que os dados estão na mesma pasta em que o *script* é
executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.wk_cpwc`;
e o pós-processamento de dados, utilizando o módulo
:mod:`framework.post_proc`. 

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo :math:`\omega k`-CPWC. Além disso, a imagem
mostra o resultado do :math:`\omega k`-CPWC com pós-processamento.

.. plot:: plots/imaging/wk_cpwc_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>
    
"""

import numpy as np
import time

from framework.data_types import DataInsp, ImagingROI, ImagingResult
from framework.linterp_oper import linterp
from framework.fk_mig import nextpow2
from framework.utils import pwd_from_fmc


def wk_cpwc_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None,
                   angles=np.arange(-10, 10 + 1, 1), pad_t='auto', pad_x=True,
                   print_time=False, print_fft_size=False):
    r"""Processa dados de A-scan utilizando o algoritmo :math:`\omega k`-CPWC.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Dados de inspeção, contendo parâmetros de inspeção, da peça e do
        transdutor, além da estrutura para salvar os resultados obtidos.

    roi : :class:`.data_types.ImagingROI`
        Região de interesse na qual o algoritmo será executado. As dimensões
        da ROI devem estar em mm.

    output_key : None ou int
        Chave identificadora do resultado de processamento. O atributo
        :attr:`.data_types.DataInsp.imaging_results` é um dicionário, capaz
        de armazenar diversos resultados de processamento. A chave (*key*) é
        um valor numérico que representa o ID do resultado, enquanto que o
        valor (*value*) é o resultado do processamento. Se ``output_key`` for
        ``None``, uma nova chave aleatória é gerada e o resultado é armazenado
        no dicionário. Se ``int``, o resultado é armazenado sob a chave
        especificada, criando uma nova entrada caso a chave não exista no
        dicionário ou sobrescrevendo os resultados anteriores caso a chave já
        exista. Por padrão, é ``None``.

    description : str
        Texto descritivo para o resultado. Por padrão, é uma *string* vazia.

    sel_shot : int
        Parâmetro que refere-se ao disparo caso o transdutor tenha sido
        deslocado. Por padrão, é ``0``.

    c : int ou float
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é ``None`` e nesse caso é obtido o valor do data_insp.

    angles : np.ndarray
        Vetor com ângulos para executar o algoritmo de CPWC a partir de dados
        de FMC. Por padrão, é um vetor [-10, -9,..., 10].
    
    pad_t : str ou int
        Define se o método para aplicar zero-padding no eixo `t` dos dados de
        aquisição. Se 'auto', a quantidade de zero-padding a ser aplicada é
        determinada a partir da dimensão dos dados de aquisição. Se 'none',
        zero-padding não é aplicado. Se um valor numérico, o valor define a
        quantidade de zero-padding a ser aplicada. Por padrão, é 'auto'.

    pad_x : bool
        Define se deve ser aplicado zero-padding no eixo `x` dos dados de
        aquisição. Por padrão, é ``True``.

    print_time : bool
        Exibe os tempos de execução da FFT, do cálculo da migração e do passo
        de  interpolação. Por padrão, é ``False``.

    print_fft_size : bool
        Exibe o tamanho da FFT. Por padrão, é ``False``.
    
    Returns
    -------
    int
        Chave de identificação do resultado (``output_key``).

    Raises
    ------
    TypeError
        Se ``data_insp`` não for do tipo :class:`.data_types.DataInsp`.

    TypeError
        Se ``roi`` não for do tipo :class:`.data_types.ImagingROI`.

    TypeError
        Se ``output_key`` não for do tipo :class:`NoneType` ou se não for
        possível realizar sua conversão para :class:`np.int32`.

    TypeError
        Se ``description`` não for do tipo :class:`str` ou se não for possível
        realizar sua conversão para :class:`str`.

    TypeError
        Se ``sel_shot`` não for do tipo :class:`int` ou se não for possível
        realizar sua conversão para :class:`int`.

    TypeError
        Se ``c`` não for do tipo :class:`float` ou se não for possível
        realizar sua conversão para :class:`float`.

    TypeError
        Se ``angles`` não for do tipo :class:`np.ndarray`.
    
    NotImplementedError
        Se o tipo de captura (:attr:`.data_types.InspectionParams.type_capt`)
        não for ``PWI`` ou ``FMC``.

    """

    # Teste dos tipos dos parâmetros.
    if type(data_insp) is not DataInsp:
        raise TypeError("O argumento ``data_insp`` não é um objeto do tipo ``DataInsp``.")

    if type(roi) is not ImagingROI:
        raise TypeError("O argumento ``roi`` não é um objeto do tipo ``ImagingROI``.")

    if output_key is not None:
        try:
            output_key = np.int32(output_key)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``output_key`` para ``numpy.int32``.")

    try:
        description = str(description)
    except Exception:
        raise TypeError("Não foi possível converter o argumento ``description`` para o tipo ``str``.")

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    if type(angles) is not np.ndarray:
        raise TypeError("O argumento ``angles`` não é do tipo ``np.ndarray``")
    
    # --- Extração dos dados necessários para a execução do algoritmo ---
    # Posições transdutores
    xt = 1e-3 * data_insp.probe_params.elem_center[:, 0]
    
    # Amostragem e gate
    ts = 1e-6 * data_insp.inspection_params.sample_time
    tgs = 1e-6 * data_insp.inspection_params.gate_start
    
    # Extração dos dados de A-scan. Se o ensaio for do tipo FMC, os dados de PWI
    # são gerados a partir do conjunto de ângulos informado.
    if data_insp.inspection_params.type_capt == "PWI":
        # Inverte os dados para as dimensões da matriz de dados ficar [emissão, a-scan, transdutor]
        theta = data_insp.inspection_params.angles / 180 * np.pi
        data = np.swapaxes(data_insp.ascan_data[:, :, :, sel_shot], 0, 1)

    elif data_insp.inspection_params.type_capt == "FMC":
        theta = angles / 180 * np.pi
        fmcdata = data_insp.ascan_data[:, :, :, sel_shot]
        data = pwd_from_fmc(fmcdata, angles, xt, c, ts)
        
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``PWI`` e ``FMC``.")

    # Dados da ROI
    xr = 1e-3 * roi.w_points
    zr = 1e-3 * roi.h_points

    # Índices para selecionar dados referentes à ROI
    idx_i = np.round((zr[0] / (c / 2)) / ts - tgs / ts).astype(int)
    if idx_i < 0:
        idx_i = 0
    ti = idx_i * ts + tgs
    idx_f = np.round((zr[-1] / (c / 2)) / ts - tgs / ts).astype(int)

    # Ajuste dos dados para região da ROI
    n_f_data = data.shape[1]
    data = data[:, idx_i:idx_f, :]
    
    # --- Início do algoritmo wk-CPWC, desenvolvido por Marco ---
    # Imagem
    img = np.zeros((zr.shape[0], xr.shape[0]))

    # --- Grids estáticas ---
    # Grid de dados. O número de pontos é aumentado para fazer zero-padding.
    if type(pad_t) is int or type(pad_t) is float:
        n_f = int(pad_t)
    else:
        if pad_t == 'none':
            n_f = data.shape[1]
        else:
            n_f = 2 ** (nextpow2(n_f_data) + 1)
    if pad_x is True:
        n_k = 2 ** (nextpow2(data.shape[2]) + 2)
    else:
        n_k = data.shape[2]
    
    dk = xt[1] - xt[0]
    k = (1 / dk / n_k) * np.arange(-int(n_k / 2), n_k / 2)
    f = (1 / ts / n_f) * np.arange(-int(n_f / 2), n_f / 2)

    # Grid da ROI
    n_kx = xr.shape[0]
    n_kz = zr.shape[0]
    dx = xr[1] - xr[0]
    kx = (1 / dx / n_kx) * np.arange(-int(n_kx / 2), n_kx / 2)

    # k_mig does not change as a function of theta
    k_mig = np.tile(kx, (n_kz, 1))
        
    t_fft = 0
    t_mig = 0
    t_interp = 0
    
    # Forma uma imagem para cada ângulo e soma o resultado em I
    for kt, thetak in enumerate(theta):
        # ------ Algoritmo do Garcia -------
        # Parâmetros da migração
        alpha = 1 / np.sqrt(1 + np.cos(thetak) + np.sin(thetak) ** 2)
        beta = np.sqrt(1 + np.cos(thetak)) ** 3 / (1 + np.cos(thetak) + np.sin(thetak) ** 2)
        gamma = np.sin(thetak) / (2 - np.cos(thetak))

        # Nova velocidade
        ch = alpha * c

        # --- Grids dinâmicas ---
        # Grid da ROI
        dz = beta * (zr[1] - zr[0])
        kz = (1 / dz / n_kz) * np.arange(-int(n_kz / 2), n_kz / 2).reshape(-1, 1)

        # Grid de interpolação
        _ti = time.time()
        f_mig = ch * np.sign(kz) * np.sqrt(k ** 2 + kz ** 2)
        t_mig += (time.time() - _ti)

        # --- FFT ---
        # x_i é a transformada de Fourier do B-scan
        # FFT axial (no tempo)
        _ti = time.time()
        x_i = np.fft.fftshift(np.fft.fft(data[kt], n_f, axis=0), axes=0)
        t_fft += (time.time() - _ti)

        # Ajuste do deslocamento no tempo
        x_i = x_i * np.exp(-1j * 2 * np.pi * f.reshape(-1, 1) * ti)

        # Ajuste devido ao ângulo de disparo
        x_i = x_i * np.exp(1j * 2 * np.pi * f.reshape(-1, 1) * xt * np.sin(thetak) / c)

        # FFT lateral (no espaço)
        _ti = time.time()
        x_i = np.fft.fftshift(np.fft.fft(x_i, n_k, axis=1), axes=1)
        t_fft += (time.time() - _ti)

        # --- Interpolação ---
        # Imagem reconstruída no domínio de Fourier
        _ti = time.time()
        x_ir = linterp(f, f_mig, x_i)
        if ((k.shape == kx.shape) and (np.allclose(k, kx) is True)) is False:
            x_ir = linterp(k, k_mig.T, x_ir.T).T
        t_interp += (time.time() - _ti)

        # Ajuste do deslocamento da ROI
        z0 = beta * zr[0]
        x0 = xr[0] - xt[0]
        x_ir = x_ir * np.exp(1j * 2 * np.pi * kz * z0)
        x_ir = x_ir * np.exp(1j * 2 * np.pi * kx * x0)

        # --- IFFT ---
        # IFFT axial (na frequência temporal)
        xir = np.fft.ifft(np.fft.ifftshift(x_ir, axes=0), axis=0)
        xir = xir * np.exp(1j * 2 * np.pi * kx * gamma * zr.reshape(-1, 1))

        # IFFT lateral (na frequência espacial) 
        xir = np.fft.ifft(np.fft.ifftshift(xir, axes=1), axis=1)

        img += np.real(xir)
        
    f = img

    if print_time is True or print_fft_size is True:
        print('\nAlgorithm wk-CPWC')
        print('-------------------------------')
    if print_fft_size is True:
        print('FFT size: ', (n_f, n_k))
        print('-------------------------------')
    if print_time is True:
        print('|{:^13}|{:<15}|'.format('Step', 'Exec. time (s)'))
        print('|{:<13}|{:^15.4f}|'.format('FFT', t_fft))
        print('|{:<13}|{:^15.4f}|'.format('Migration', t_mig))
        print('|{:<13}|{:^15.4f}|'.format('Interpolation', t_interp))
        print('-------------------------------')
    
    # --- Fim do algoritmo wk-CPWC ---

    # Salva o resultado.
    if output_key is None:
        # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
        result = ImagingResult(roi=roi, description=description)
        result.image = f

        # Gera uma chave aleatória para inserção no dicionário de resultados.
        ii32 = np.iinfo(np.int32)
        while True:
            output_key = np.random.randint(low=ii32.min, high=ii32.max, dtype=np.int32)

            # Insere o resultado na lista apropriada do objeto DataInsp
            if output_key in data_insp.imaging_results:
                # Chave já existe. Como deve ser uma chave nova, repete.
                continue
            else:
                # Chave inexistente. Insere o resultado no dicionário e sai do laço.
                data_insp.imaging_results[output_key] = result
                break
    else:
        # Salva o resultado em um objeto ImagingResult já existente em DataInsp.
        # Busca o resultado no dicionário baseado na chave.
        try:
            result = data_insp.imaging_results[output_key]
            result.roi = roi
            result.description = description
        except KeyError:
            # Objeto não encontrado no dicionário. Cria um novo.
            # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
            result = ImagingResult(roi=roi, description=description)

        # Salva o novo resultado.
        result.image = f

        # Guarda o resultado no dicionário.
        data_insp.imaging_results[output_key] = result

    # Retorna o valor da chave
    return output_key


def wk_cpwc_params():
    r"""Retorna os parâmetros do algoritmo :math:`\omega k`-CPWC.

    Returns
    -------
    :class:`dict`
        Dicionário, em que a chave ``roi`` representa a região de interesse
        utilizada pelo algoritmo, a chave ``output_key`` representa a chave
        de identificação do resultado, a chave ``description`` representa a
        descrição do resultado, a chave ``sel_shot`` representa o disparo
        do transdutor e a chave ``c`` representa a velocidade de propagação
        da onda na peça.
    
    """

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0,
            "angles": np.arange(-10, 10 + 1, 1)}
