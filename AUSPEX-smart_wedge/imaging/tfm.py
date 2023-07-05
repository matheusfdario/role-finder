# -*- coding: utf-8 -*-
r"""
Módulo ``tfm``
==============

O TFM (*Total Focusing Method* - Método de Focalização Total) é um algoritmo
de reconstrução de imagens para ensaios não destrutivos, quando o
sistema de inspeção utiliza transdutores ultrassônicos *phased array* e o
sistema de captura é FMC (*Full Matrix Capture* - Matriz Completa de Captura).
No TFM, o feixe é focalizado em todos os pontos da ``roi`` (*Region of
Interest* - Região de Interesse).
A primeira etapa do algoritmo consiste em discretizar a roi no plano
:math:`(x, z)` em uma grade definida. Então, os sinais de todos os
elementos da matriz são somados para sintetizar um foco em todos os pontos
da grade. Calcula-se a intensidade da imagem, :math:`I(x, z)` em qualquer
ponto da varredura através da Equação :eq:`eq-i-fxz`:


.. math:: I(x,z) = \left|\sum h_{tx,rx}\left(\frac{\sqrt{(x_{tx}-x)^2+z^2} + \sqrt{(x_{rx}-x)^2+z^2}}{c}\right)\right|,
    :label: eq-i-fxz


sendo :math:`c` a velocidade do som no meio, :math:`x_{tx}` e :math:`x_{rx}` as posições laterais dos elementos
transmissores e receptores, respectivamente :cite:`Holmes2005`.

Devido a necessidade de realizar a interpolação linear dos sinais do domínio
do tempo, anteriormente amostrados discretamente, a soma é realizada para
cada par transmissor-receptor possível e, portanto, usa a quantidade máxima
de informações disponíveis para cada ponto.

Essa técnica tem como principal limitante o tempo de computação.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo TFM para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.tfm`; e o
pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`.

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo TFM. Além disso, a imagem mostra o
resultado do TFM com pós-processamento.

.. plot:: plots/imaging/tfm_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>

"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import hilbert
from framework.data_types import DataInsp, ImagingROI, ImagingResult, ElementGeometry
from numba import njit, prange

initialize = True


def tfm_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None, trcomb=None,
               scattering_angle=None, elem_geometry=ElementGeometry.RECTANGULAR, analytic=False):
    centros = data_insp.probe_params.elem_center[:, 1]
    # se existe algum transdutor ou ponto da ROI desalinhado no eixo Y, usa o TFM 3D
    if any(centros != centros[0]) or any(roi.d_points != centros[0]):
        return tfm3d_kern(data_insp, roi, output_key, description, sel_shot,
                          c, trcomb, scattering_angle, elem_geometry, analytic)
    else:
        return tfm2D_kern(data_insp, roi, output_key, description, sel_shot,
                          c, trcomb, scattering_angle, elem_geometry, analytic)

def tfm_params():
    """Retorna os parâmetros do algoritmo TFM.

    Returns
    -------
    dict
        Dicionário, em que a chave ``roi`` representa a região de interesse
        utilizada pelo algoritmo, a chave ``output_key`` representa a chave
        de identificação do resultado, a chave ``description`` representa a
        descrição do resultado, a chave ``sel_shot`` representa o disparo
        do transdutor, a chave ``c`` representa a velocidade de propagação
        da onda na peça e ``trcomb`` representa as combinações de transmis
        sores e receptores usados.

    """

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0,
            "scattering_angle": 180}


def tfm2D_kern(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None, trcomb=None,
                 scattering_angle=None, elem_geometry=ElementGeometry.RECTANGULAR, analytic=False):
    """Processa dados de A-scan utilizando o algoritmo TFM.

    Parameters
    ----------
        data_insp : :class:`.data_types.DataInsp`
            Dados de inspeção, contendo parâmetros de inspeção, da peça e do
            transdutor, além da estrutura para salvar os resultados obtidos.

        roi : :class:`.data_types.ImagingROI`
            Região de interesse na qual o algoritmo será executado. As
            dimensões da ROI devem estar em mm.

        output_key : None ou int
            Chave identificadora do resultado de processamento.
            O atributo :attr:`.data_types.DataInsp.imaging_results` é um
            dicionário, capaz de armazenar diversos resultados de
            processamento. A chave (*key*) é um valor numérico que representa
            o ID do resultado, enquanto que o valor (*value*) é o resultado
            do processamento. Se ``output_key`` for ``None``, uma nova chave
            aleatória é gerada e o resultado é armazenado no dicionário. Se
            ``int``, o resultado é armazenado sob a chave especificada, criando
            uma nova entrada caso a chave não exista no dicionário ou
            sobrescrevendo os resultados anteriores caso a chave já exista.
            Por padrão, é ``None``.

        description : str
            Texto descritivo para o resultado. Por padrão, é uma *string*
            vazia.

        sel_shot : int
            Parâmetro que refere-se ao disparo caso o transdutor tenha sido
            deslocado.

        c : int ou float
            Velocidade de propagação da onda no objeto sob inspeção. Por
            padrão, é None e nesse caso é obtido o valor do data_insp.

        trcomb : None ou 2d-array int
            Especifica quais as combinações de elementos Transmissores e Receptores usar.

        scattering_angle : None, float, ou 2d-array bool
            Fornece um ângulo a partir do qual é gerado um mapa de pontos que influenciam o A-scan. Opcionalmente pode
            fornecido o mapa diretamente.

        elem_geometry : framework.data_types.ElementGeometry
            Geometria dos elementos do transdutor, que é utilizada no cálculo do
            conjunto de elementos enquadrados no ``scattering_angle``.
            Por padrão, é ``RECTANGULAR``.

        analytic : bool
            Se ``True``, o cálculo do TFM é feito sobre o sinal analítico,
            gerando uma imagem com valores complexos. Se ``false``, o
            cálculo de TFM é feito sobre os dados brutos, gerando uma
            imagem com valores reais.

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

    NotImplementedError
        Se o tipo de captura (:attr:`.data_types.InspectionParams.type_capt`)
        não for ``sweep`` ou ``FMC``.

    """

    # Teste dos tipos dos parâmetros.
    if type(data_insp) is not DataInsp:
        raise TypeError("O argumento ``data_insp`` não é um objeto do tipo ``DataInsp``.")

    if data_insp.surf is None and data_insp.inspection_params.type_insp == 'immersion':
        raise ValueError("Surface não inicializado")

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

    try:
        sel_shot = int(sel_shot)
    except Exception:
        raise TypeError("Não foi possível converter o argumento ``sel_shot`` para o tipo ``int``.")

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "FMC":
        if analytic is True:
            g = hilbert(data_insp.ascan_data[:, :, :, sel_shot], axis=0)
        else:
            g = data_insp.ascan_data[:, :, :, sel_shot]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo TFM.")

    nb_elements = data_insp.probe_params.num_elem
    if trcomb is None:
        trcomb = np.ones((nb_elements, nb_elements), dtype=bool)
    else:
        try:
            trcomb = np.asarray(trcomb, bool)
        except ValueError:
            raise TypeError("O argumento trcomb não é compatível com formato int 2D-array")
        if not(trcomb.shape.__len__() == 2 and trcomb.shape[0] == trcomb.shape[1] and trcomb.shape[0] == nb_elements):
            strerr = "O argumento trcomb não tem o formato exigido ("+str(nb_elements)+"x"+str(nb_elements)+")"
            raise TypeError(strerr)

    if scattering_angle is None:
        scatfilt = np.zeros((nb_elements, roi.h_len*roi.w_len), dtype=bool)
    else:
        try:
            scattering_angle = float(scattering_angle)
            scatfilt = roi.get_coord() * 1e-3
            scatfilt = scatfilt[:, np.newaxis] - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
            scatfilt = np.angle(scatfilt[:, :, 0] + scatfilt[:, :, 2] * 1j, True) - 90
            scatfilt = abs(scatfilt.T) > (scattering_angle / 2)
        except TypeError:
            scatfilt = np.asarray(scattering_angle, bool)
            if not (scatfilt.shape.__len__() == 2 and scatfilt.shape[0] == data_insp.probe_params.num_elem and
                    scatfilt.shape[1] == roi.h_len*roi.w_len):
                strerr = "O argumento scatfilt não tem o formato exigido (" + str(nb_elements) + "x" + str(
                    roi.h_len*roi.w_len) + ")"
                raise TypeError(strerr)

    # --- INÍCIO DO ALGORITMO TFM, desenvolvido por Hector. ---
    f = np.zeros((1, roi.h_len * roi.w_len), dtype=g.dtype)
    combs = np.argwhere(trcomb.T)
    tx_elements = combs[:, 0]
    rx_elements = combs[:, 1]
    nb_combs = combs.shape[0]
    # Calcula a distância entre os pontos da ROI e os centros dos transdutores.
    if data_insp.inspection_params.type_insp == 'immersion':
        # from surface.surface import Surface
        # surf = Surface(data_insp, xdczerototal=0)
        dist = data_insp.surf.cdist_medium(data_insp.probe_params.elem_center, roi.get_coord(), roi=roi, sel_shot=sel_shot) * 1e-3
        dist_correction = 1.0 / (np.asarray([data_insp.inspection_params.coupling_cl, c]) *
                                 data_insp.inspection_params.sample_time * 1e-6)
        samp_dist = dist[0] * dist_correction[0] + dist[1] * dist_correction[1]
    else:
        dist = cdist(data_insp.probe_params.elem_center * 1e-3, roi.get_coord() * 1e-3)
        dist_correction = 1.0 / (c * data_insp.inspection_params.sample_time * 1e-6)
        samp_dist = dist * dist_correction
    dr = (roi.get_coord()[:, :, np.newaxis]-data_insp.probe_params.elem_center.T[np.newaxis])
    dg = np.arctan2(dr[:, 0, :], dr[:, 2, :])#/(np.pi/2)*90
    if data_insp.inspection_params.type_insp == 'contact':
        k = data_insp.probe_params.central_freq*1e6/data_insp.specimen_params.cl
    else:
        k = data_insp.probe_params.central_freq*1e6/data_insp.inspection_params.coupling_cl
    a = data_insp.probe_params.elem_dim*1e-3
    wt = np.sinc(k*a/2*np.sin(dg))*np.cos(dg)
    wt = wt[np.newaxis]
    samp_dist[scatfilt] = g.shape[0]
    samp_dist -= int(data_insp.inspection_params.gate_start*data_insp.inspection_params.sample_freq)//2
    # f = kernel(f, g, rx_elements, tx_elements, nb_combs, samp_dist.astype(np.int32))
    f = kernel(f, g, rx_elements, tx_elements, wt, nb_combs, samp_dist.astype(np.int32))
    # Ajusta o tamanho da imagem ``f``.
    f = (f.reshape((roi.w_len, roi.h_len))).T
    # --- FIM DO ALGORITMO TFM.
    # Salva o resultado.
    if output_key is None:
        # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
        result = ImagingResult(roi=roi, description=description)

        # Gera uma chave aleatória para inserção no dicionário de resultados.
        ii32 = np.iinfo(np.int32)
        while True:
            output_key = np.random.randint(low=ii32.min, high=ii32.max, dtype=np.int32)

            # Insere o resultado na lista apropriada do objeto DataInsp
            if output_key in data_insp.imaging_results:
                # Chave já existe. Como deve ser uma chave nova, repete.
                continue
            else:
                # Chave inexistente. Sai do laço.
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
    if data_insp.inspection_params.type_insp == 'immersion':
        result.surface = data_insp.surf.get_points_in_roi(roi, sel_shot)
    # Guarda o resultado no dicionário.
    data_insp.imaging_results[output_key] = result

    # Retorna o valor da chave
    return output_key


@njit(parallel=True, cache=True)
def kernel(f, g, rx_elements, tx_elements, wt, nb_comb, samp_dist):
    for comb in prange(nb_comb):
        j = samp_dist[rx_elements[comb], :] + samp_dist[tx_elements[comb], :]
        j[j < 0] = -1
        j[j >= g.shape[0]] = -1
        f += g[j, tx_elements[comb], rx_elements[comb]]*wt[:, :, rx_elements[comb]]*wt[:, :, tx_elements[comb]]
    return f


########################## 3D #####################################

def tfm3d_kern(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None, trcomb=None,
                 scattering_angle=None, elem_geometry=ElementGeometry.RECTANGULAR, analytic=False):
    """Processa dados de A-scan utilizando o algoritmo TFM.

    Parameters
    ----------
        data_insp : :class:`.data_types.DataInsp`
            Dados de inspeção, contendo parâmetros de inspeção, da peça e do
            transdutor, além da estrutura para salvar os resultados obtidos.

        roi : :class:`.data_types.ImagingROI`
            Região de interesse na qual o algoritmo será executado. As
            dimensões da ROI devem estar em mm.

        output_key : None ou int
            Chave identificadora do resultado de processamento.
            O atributo :attr:`.data_types.DataInsp.imaging_results` é um
            dicionário, capaz de armazenar diversos resultados de
            processamento. A chave (*key*) é um valor numérico que representa
            o ID do resultado, enquanto que o valor (*value*) é o resultado
            do processamento. Se ``output_key`` for ``None``, uma nova chave
            aleatória é gerada e o resultado é armazenado no dicionário. Se
            ``int``, o resultado é armazenado sob a chave especificada, criando
            uma nova entrada caso a chave não exista no dicionário ou
            sobrescrevendo os resultados anteriores caso a chave já exista.
            Por padrão, é ``None``.

        description : str
            Texto descritivo para o resultado. Por padrão, é uma *string*
            vazia.

        sel_shot : int
            Parâmetro que refere-se ao disparo caso o transdutor tenha sido
            deslocado.

        c : int ou float
            Velocidade de propagação da onda no objeto sob inspeção. Por
            padrão, é None e nesse caso é obtido o valor do data_insp.

        trcomb : None ou 2d-array int
            Especifica quais as combinações de elementos Transmissores e Receptores usar.

        scattering_angle : None, float, ou 2d-array bool
            Fornece um ângulo a partir do qual é gerado um mapa de pontos que influenciam o A-scan. Opcionalmente pode
            fornecido o mapa diretamente.

        elem_geometry : framework.data_types.ElementGeometry
            Geometria dos elementos do transdutor, que é utilizada no cálculo do
            conjunto de elementos enquadrados no ``scattering_angle``.
            Por padrão, é ``RECTANGULAR``.

        analytic : bool
            Se ``True``, o cálculo do TFM é feito sobre o sinal analítico,
            gerando uma imagem com valores complexos. Se ``false``, o
            cálculo de TFM é feito sobre os dados brutos, gerando uma
            imagem com valores reais.

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

    NotImplementedError
        Se o tipo de captura (:attr:`.data_types.InspectionParams.type_capt`)
        não for ``sweep`` ou ``FMC``.

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

    try:
        sel_shot = int(sel_shot)
    except Exception:
        raise TypeError("Não foi possível converter o argumento ``sel_shot`` para o tipo ``int``.")

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "FMC":
        if analytic is True:
            g = hilbert(data_insp.ascan_data[:, :, :, sel_shot], axis=0)
        else:
            g = data_insp.ascan_data[:, :, :, sel_shot]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo TFM.")

    nb_elements = data_insp.probe_params.num_elem
    if trcomb is None:
        trcomb = np.ones((nb_elements, nb_elements), dtype=bool)
    else:
        try:
            trcomb = np.asarray(trcomb, np.int32)
        except ValueError:
            raise TypeError("O argumento trcomb não é compatível com formato int 2D-array")
        if not(trcomb.shape.__len__() == 2 and trcomb.shape[0] == trcomb.shape[1] and trcomb.shape[0] == nb_elements):
            strerr = "O argumento trcomb não tem o formato exigido ("+str(nb_elements)+"x"+str(nb_elements)+")"
            raise TypeError(strerr)

    # Calcula a distância entre os pontos da ROI e os centros dos transdutores.
    if data_insp.inspection_params.type_insp == 'immersion':
        from surface.surface import Surface
        if data_insp.surf == None:
            data_insp.surf = Surface(data_insp, xdczerototal=0)
        dist = data_insp.surf.cdist_medium(data_insp.probe_params.elem_center, roi.get_coord()) * 1e-3
        dist_correction = 1.0 / (np.asarray([data_insp.inspection_params.coupling_cl, c]) *
                                 data_insp.inspection_params.sample_time * 1e-6)
        samp_dist = dist[0] * dist_correction[0] + dist[1] * dist_correction[1]
    else:
        dist = cdist(data_insp.probe_params.elem_center * 1e-3, roi.get_coord() * 1e-3)
        dist_correction = 1.0 / (c * data_insp.inspection_params.sample_time * 1e-6)
        samp_dist = dist * dist_correction

    if scattering_angle is None:
        scatfilt = np.zeros((nb_elements, roi.h_len*roi.w_len*roi.d_len), dtype=bool)
    else:
        try:
            scattering_angle = float(scattering_angle)
            if data_insp.inspection_params.type_insp == 'contact':
                scatfilt = roi.get_coord() * 1e-3
                scatfilt = scatfilt[:, np.newaxis] - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
                scatfilt = np.angle(scatfilt[:, :, 0] + scatfilt[:, :, 2] * 1j, True) - 90
                scatfilt = abs(scatfilt.T) > (scattering_angle / 2)
            else:
                if elem_geometry == ElementGeometry.RECTANGULAR:
                    scatfilt_1 = surf.entrypoints.transpose(1, 0, -1) * 1e-3
                    scatfilt_1 = scatfilt_1 - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
                    scatfilt_1 = np.angle(scatfilt_1[:, :, 0] + scatfilt_1[:, :, 2] * 1j, True) - 90
                    scatfilt_1 = abs(scatfilt_1.T) > (scattering_angle / 2)
                    scatfilt_2 = surf.entrypoints.transpose(1, 0, -1) * 1e-3
                    scatfilt_2 = scatfilt_2 - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
                    scatfilt_2 = np.angle(scatfilt_2[:, :, 1] + scatfilt_2[:, :, 2] * 1j, True) - 90
                    scatfilt_2 = abs(scatfilt_2.T) > (scattering_angle / 2)
                    scatfilt = np.any([scatfilt_1, scatfilt_2], 0)
                else:
                    scatfilt = surf.entrypoints.transpose(1, 0, -1) * 1e-3
                    scatfilt = scatfilt - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
                    scatfilt = scatfilt/np.linalg.norm(scatfilt, 2, -1)[:, :, np.newaxis]
                    scatfilt = np.arccos(scatfilt[:, :, 2])*180/np.pi
                    scatfilt = abs(scatfilt.T) > (scattering_angle / 2)
        except TypeError:
            scatfilt = np.asarray(scattering_angle, bool)
            if not (scatfilt.shape.__len__() == 2 and scatfilt.shape[0] == data_insp.probe_params.num_elem and
                    scatfilt.shape[1] == roi.h_len*roi.w_len):
                strerr = "O argumento scatfilt não tem o formato exigido (" + str(nb_elements) + "x" + str(
                    roi.h_len*roi.w_len) + ")"
                raise TypeError(strerr)

    # --- INÍCIO DO ALGORITMO TFM, desenvolvido por Hector. ---
    f = np.zeros((1, roi.h_len * roi.w_len * roi.d_len), dtype=g.dtype)
    combs = np.argwhere(trcomb.T)
    tx_elements = combs[:, 0]
    rx_elements = combs[:, 1]
    nb_combs = combs.shape[0]
    wt = np.ones((*f.shape, data_insp.probe_params.num_elem))
    samp_dist[scatfilt] = g.shape[0]
    samp_dist -= int(data_insp.inspection_params.gate_start * data_insp.inspection_params.sample_freq) // 2
    f = kernel(f, g, rx_elements, tx_elements, wt, nb_combs, samp_dist.astype(np.int32))

    # Ajusta o tamanho da imagem ``f``.
    f = (f.reshape((roi.w_len, roi.d_len, roi.h_len,))).T
    # --- FIM DO ALGORITMO TFM.
    # Salva o resultado.
    if output_key is None:
        # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
        result = ImagingResult(roi=roi, description=description)

        # Gera uma chave aleatória para inserção no dicionário de resultados.
        ii32 = np.iinfo(np.int32)
        while True:
            output_key = np.random.randint(low=ii32.min, high=ii32.max, dtype=np.int32)

            # Insere o resultado na lista apropriada do objeto DataInsp
            if output_key in data_insp.imaging_results:
                # Chave já existe. Como deve ser uma chave nova, repete.
                continue
            else:
                # Chave inexistente. Sai do laço.
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
    result.surface = data_insp.surf
    # Guarda o resultado no dicionário.
    data_insp.imaging_results[output_key] = result

    # Retorna o valor da chave
    return output_key

