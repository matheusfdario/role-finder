# -*- coding: utf-8 -*-
r"""
Módulo ``cpwc``
===============

O CPWC (*Coherent Plane Wave Compounding*) é um algoritmo utilizado para
reconstruir imagens quando o tipo de inspeção é por ondas planas (*plane
waves*). Nesse método, todos os elementos de um transdutor do tipo *array*
linear são disparados simultaneamente, criando uma única frente de onda que
ilumina o objeto sob inspeção, conforme ilustrado na
:numref:`fig_imaging_nde_pwi`.

.. figure:: figures/imaging/nde-pwi.png
    :name: fig_imaging_nde_pwi
    :width: 35 %
    :align: center

    Inspeção com ondas planas.

É possível transmitir ondas planas com inclinações, aplicando *delays* no
disparo dos elementos do transdutor, conforme indicado na 
:numref:`fig_imaging_nde_pwi_inc`. 

.. figure:: figures/imaging/nde-pwi-inc.png
    :name: fig_imaging_nde_pwi_inc
    :width: 35 %
    :align: center

    Inspeção com ondas planas.

O algoritmo CPWC produz uma imagem final somando as diversas imagens obtidas
a partir de ondas com ângulos diferentes. Cada imagem individual é formada
aplicando *delay*-e-soma nos sinais de A-scan, sendo que os *delays* aplicados
dependem da posição do ponto da imagem e do ângulo da onda plana. A
:numref:`fig_imaging_pwi` ilustra as distâncias percorridas por uma onda
emitida com uma inclinação :math:`\theta` em relação à superfície da peça.

.. figure:: figures/imaging/pwi.png
    :name: fig_imaging_pwi
    :width: 35 %
    :align: center

    Inspeção com ondas planas.

A distância :math:`d_\text{i}` que a onda percorre até atingir um ponto
:math:`(x, z)` é função da posição do ponto e da inclinação da onda:

.. math::
    
    d_\text{i} = z\cos{\theta} + x\sin{\theta}.

Após atingir o ponto :math:`(x, z)`, a onda pode ser refletida. A distância
:math:`d_\text{v}` percorrida pela onda, do ponto :math:`(x, z)` até um
transdutor posicionado em :math:`(x_t, 0)` é: 

.. math::
    
    d_\text{v} = \sqrt{(x - x_t)^2 + z^2}.
    
O *delay* :math:`\tau` aplicado ao sinal do transdutor em :math:`x_t` é obtido
a partir da distância total percorrida pela onda e a sua velocidade :math:`c`
no meio:

.. math::
    
    \tau_{x_t} = \frac{d_i + d_v}{c}.

Exemplo
-------

O *script* abaixo mostra o uso do algoritmo CPWC para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo de
simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento de
dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.cpwc`; e o
pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`. 

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo CPWC. Além disso, a imagem mostra o
resultado do CPWC com pós-processamento.

.. plot:: plots/imaging/cpwc_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>
    
"""

import numpy as np
import numba

from framework.data_types import DataInsp, ImagingROI, ImagingResult
from framework.utils import pwd_from_fmc
from surface.surface import Surface, SurfaceType, Lineparam


def cpwc_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None,
                cmed=None, angles=np.arange(-10, 10 + 1, 1)):
    """Processa dados de A-scan utilizando o algoritmo CPWC.

    Parameters
    ----------
    data_insp : :class:`.data_types.DataInsp`
        Dados de inspeção, contendo parâmetros de inspeção, da peça e do
        transdutor, além da estrutura para salvar os resultados obtidos.

    roi : :class:`.data_types.ImagingROI`
        Região de interesse na qual o algoritmo será executado. As dimensões
        da ROI devem estar em mm.

    output_key : :class:`None` ou :class:`int`
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
        deslocado.

    c : int ou float
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    cmed : int ou float
        Velocidade de propagação da onda no meio acoplante. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    angles : :class:`np.ndarray`
        Vetor com ângulos para executar o algoritmo de CPWC a partir de dados
        de FMC. Por padrão, é um vetor [-10, -9,..., 10].

    Returns
    -------
    :class:`int`
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

    if cmed is None:
        cmed = data_insp.inspection_params.coupling_cl
    else:
        try:
            cmed = float(cmed)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``cmed`` para o tipo ``float``.")

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    if type(angles) is not np.ndarray:
        raise TypeError("O argumento ``angles`` não é do tipo ``np.ndarray``")

    # --- Extração dos dados necessários para a execução do algoritmo. ---
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
        pwdata = np.swapaxes(data_insp.ascan_data[:, :, :, sel_shot], 0, 1)

    elif data_insp.inspection_params.type_capt == "FMC":
        theta = angles / 180 * np.pi
        fmcdata = data_insp.ascan_data[:, :, :, sel_shot]
        pwdata = pwd_from_fmc(fmcdata, angles, xt, c, ts)

    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``PWI`` e ``FMC``.")

    # Dados da ROI
    xr = 1e-3 * roi.w_points
    zr = 1e-3 * roi.h_points

    # --- INÍCIO DO ALGORITMO CPWC, desenvolvido por Marco. ---
    # Dimensões dados
    m = pwdata.shape[1]
    n = pwdata.shape[2]

    # Dimensões ROI
    m_r = zr.shape[0]
    n_r = xr.shape[0]

    # Imagem
    img = np.zeros((m_r * n_r, 1), dtype=pwdata.dtype)

    for k, thetak in enumerate(theta):
        data = np.vstack((pwdata[k], np.zeros((1, n)))).astype(pwdata.dtype)

        # Calcula a distância percorrida pela onda até cada ponto da ROI e de
        # volta para cada transdutor. As distâncias são convertidas em delays
        # e então em índices. Cada linha da variável j representa um pixel na
        # imagem final, contendo o índice das amostras do sinal de Ascan para
        # todos os transdutores que contribuem para esse pixel.
        if data_insp.inspection_params.type_insp == 'immersion':
            j = cpwc_roi_dist_immersion(xr, zr, xt, thetak, c, cmed, ts, tgs, data_insp.surf)
        else:
            j = cpwc_roi_dist(xr, zr, xt, thetak, c, ts, tgs)
        j = j.reshape(m_r * n_r, n)
        j[j >= m] = -1
        j[j < 0] = -1

        # Soma as amostras de Ascan coerentemente
        aux = np.zeros(j.shape[0], dtype=pwdata.dtype)
        img[:, 0] += cpwc_sum(data, aux, j)

    f = img.reshape((m_r, n_r), order='F')

    # --- FIM DO ALGORITMO CPWC. ---

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


@numba.njit(parallel=True)
def cpwc_roi_dist(xr, zr, xt, theta, c, ts, tgs):
    r"""Calcula os *delays* para o DAS do algoritmo CPWC.

    Os *delays* são convertidos para índices, a partir do período de
    amostragem. Os *delays* são calculados conforme a trajetória da onda
    plana, desde o transdutor até o ponto da ROI e de volta para o transdutor.

    Parameters
    ----------
    xr : :class:`np.ndarray`
        Vetor com os valores de :math:`x` da ROI, em m.

    zr : :class:`np.ndarray`
        Vetor com os valores de :math:`z` da ROI, em m.

    xt : :class:`np.ndarray`
        Vetor com os valores de :math:`x` dos elementos do transdutor, em m.

    theta : :class:`int`, :class:`float`
        Ângulo de inclinação da onda plana, em radianos.

    c : :class:`int`, :class:`float`
        Velocidade de propagação da onda no meio.

    ts : :class:`int`, :class:`float`
        Período de amostragem do transdutor.

    tgs : :class:`int`, :class:`float`
        Tempo do gate inicial.

    Returns
    -------
    :class:`np.ndarray`
        Uma matriz de números inteiros :math:`M_r \cdot N_r` por :math:`N`, em
        que :math:`M_r` é a quantidade de elementos do vetor :math:`x`,
        :math:`N_r` é a quantidade de elementos do vetor :math:`z` e :math:`N`
        é a quantidade de elementos do transdutor.

    """

    m_r = zr.shape[0]
    n_r = xr.shape[0]
    n = xt.shape[0]

    ti_i = np.int64(tgs / ts)

    j = np.zeros((n_r * m_r, n), dtype=np.int64)
    for i in numba.prange(n_r):
        for jj in range(m_r):
            i_i = i * m_r + jj
            di = (zr[jj] * np.cos(theta) + xr[i] * np.sin(theta))
            dv = np.sqrt(zr[jj] ** 2 + (xr[i] - xt) ** 2)
            d = np.rint((di + dv) / (c * ts))
            j[i_i, :] = d - ti_i

    return j


def cpwc_roi_dist_immersion(xr, zr, xt, theta, c, cmed, ts, tgs, surf):
    r"""Calcula os *delays* para o DAS do algoritmo CPWC.

    Os *delays* são convertidos para índices, a partir do período de
    amostragem. Os *delays* são calculados conforme a trajetória da onda
    plana, desde o transdutor até o ponto da ROI e de volta para o transdutor.

    Parameters
    ----------
    xr : :class:`np.ndarray`
        Vetor com os valores de :math:`x` da ROI, em m.

    zr : :class:`np.ndarray`
        Vetor com os valores de :math:`z` da ROI, em m.

    xt : :class:`np.ndarray`
        Vetor com os valores de :math:`x` dos elementos do transdutor, em m.

    theta : :class:`int`, :class:`float`
        Ângulo de inclinação da onda plana, em radianos.

    c : :class:`int`, :class:`float`
        Velocidade de propagação da onda no meio.

    ts : :class:`int`, :class:`float`
        Período de amostragem do transdutor.

    tgs : :class:`int`, :class:`float`
        Tempo do gate inicial.

    surf : :class:`Surface`
        Objeto com informações sobre a superfície externa.

    Returns
    -------
    :class:`np.ndarray`
        Uma matriz de números inteiros :math:`M_r \cdot N_r` por :math:`N`, em
        que :math:`M_r` é a quantidade de elementos do vetor :math:`x`,
        :math:`N_r` é a quantidade de elementos do vetor :math:`z` e :math:`N`
        é a quantidade de elementos do transdutor.

    """

    m_r = zr.shape[0]
    n_r = xr.shape[0]
    n = xt.shape[0]

    ti_i = np.int64(tgs / ts)

    j = np.zeros((n_r * m_r, n), dtype=np.int64)
    yr = np.zeros_like(xr)
    roi_coord = np.zeros((m_r * n_r, 3))
    for i in range(n_r):
        roi_coord[i*m_r:(i+1)*m_r, 0] = xr[i]
        for jj in range(m_r):
            roi_coord[i*m_r+jj, 2] = zr[jj]
    roi_coord *= 1e3
    # Distância dos pixels aos respectivos pontos de entrada
    di_mat = (roi_coord[:, 2] - surf.surfaceparam.b) / np.cos(theta)
    #di_mat = roi_coord[:, 2] * np.cos(theta) + roi_coord[:, 0] * np.sin(theta)
    # Localização dos pontos de entrada
    entrypoints = np.zeros_like(roi_coord)
    entrypoints[:, 2] = surf.surfaceparam.b
    entrypoints[:, 0] = roi_coord[:, 0] - di_mat * np.sin(theta)
    # Transformação do ângulo theta pela Lei de Snell
    theta_med = np.arcsin(np.sin(theta) * cmed / c)
    print(np.array([theta, theta_med])*180/np.pi)
    # Distância dos pontos de entrada à origem da frente de onda plana
    di_med = entrypoints[:, 2] * np.cos(theta_med) + entrypoints[:, 0] * np.sin(theta_med)
    # Distância total de ida
    #di = di_med + di_mat
    # A Distância de volta é caulculada diretamente pela classe Seurface
    elem_center = np.zeros((len(xt), 3))
    elem_center[:, 0] = xt * 1e3
    [dv_med, dv_mat] = surf.cdist_medium(elem_center, roi_coord)
    dv_med = dv_med.transpose()
    dv_mat = dv_mat.transpose()
    #dv = dv_med + dv_mat
    # Dividir distâncias por 1e3
    di_med *= 1e-3
    di_mat *= 1e-3
    dv_med *= 1e-3
    dv_mat *= 1e-3
    #d_med = np.copy()
    #d = np.rint((di_mat + dv_mat) / (c * ts)) + \
    #    np.rint((di_med + dv_med) / (cmed * ts))
    # Faz a operação de soma
    d = np.zeros_like(dv_med)
    d += dv_med / (cmed * ts)
    d += dv_mat / (c * ts)
    for i in range(len(xt)):
        d[:, i] += di_med / (cmed * ts)
        d[:, i] += di_mat / (c * ts)
    j = np.array(np.rint(d) - ti_i, dtype=np.int64)

    # for i in range(n_r):
    #     for jj in range(m_r):
    #         i_i = i * m_r + jj
    #         # Tem que fazer isso aqui (linha abaixo) para o ponto de entrada
    #         # e somar do ponto de entrada ao foco
    #         di = (zr[jj] * np.cos(theta) + xr[i] * np.sin(theta))
    #         dv = np.sqrt(zr[jj] ** 2 + (xr[i] - xt) ** 2)
    #         d = np.rint((di + dv) / (c * ts))
    #         j[i_i, :] = d - ti_i

    return j


@numba.njit(parallel=True)
def cpwc_sum(data, img, j):
    r"""Realiza a soma para o DAS do algoritmo CPWC.

    Parameters
    ----------
    data : :class:`np.ndarray`
        Matriz :math:`M` por :math:`N` contendo os dados de aquisição.

    img : :class:`np.ndarray`
        Vetor :math:`N_r` para acumular os dados.

    j : :class:`np.ndarray`
        Matriz com os *delays* para cada ponto da ROI. Deve ser uma matriz
        :math:`M_r \cdot N_r` por :math:`N`, em que :math:`M_r` é a quantidade
        de elementos do vetor :math:`x`, :math:`N_r` é a quantidade de
        elementos do vetor :math:`z` e :math:`N` é a quantidade de elementos
        do transdutor.

    Returns
    -------
    :class:`np.ndarray`
        Vetor 1 por :math:`M_r \cdot N_r` contendo a soma no eixo 1 da matriz.

    """

    i = np.arange(j.shape[1])
    # img = np.zeros(j.shape[0])
    for jj in numba.prange(j.shape[0]):
        idx = j[jj, :]
        for ii in range(i.shape[0]):
            img[jj] += data[idx[ii], ii]

    return img


def cpwc_params():
    """Retorna os parâmetros do algoritmo CPWC.

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
