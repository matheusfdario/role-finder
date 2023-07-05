# -*- coding: utf-8 -*-
r"""
Módulo ``vtfm``
===============

O VTFM (*vector-TFM*) é um algoritmo adaptado do ``TFM`` (*Total Focusing Method* - Método de Focalização Total).
São definidas sub-aberturas do transdutor, e a imagem TFM para cada uma é gerada, todas utilizando a mesma ``ROI``
(*Region of Interest* - Região de Interesse). Em seguida, é calulada a especularidade de acordo com a variação da
intensidade em função das sub-aberturas. Finalmente, a imagem final (sem sub-abertura) é gerada, e a partir do mapa de
especularidade é possível realçar os refletores especulares ou puntuais.

Esse método é capaz de estimar a direção normal da superfície dos refletores, mas como algoritmo de imageamento esta
funcionalidade não está disponível.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo VTFM para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.vtfm`; e o
pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`.

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo VTFM. Além disso, a imagem mostra o
resultado do VTFM com pós-processamento.

.. plot:: plots/imaging/vtfm_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>

"""

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import cdist

from framework import post_proc
from framework.data_types import ImagingROI, ImagingResult, DataInsp


def vtfm_kernel(data_insp, aperture_size, aperture_step, roi=ImagingROI(), output_key=None, description="", sel_shot=0,
                c=None, beta=1):
    """Processa dados de A-scan utilizando o algoritmo VTFM.

    Parameters
    ----------
        data_insp : :class:`.data_types.DataInsp`
            Dados de inspeção, contendo parâmetros de inspeção, da peça e do
            transdutor, além da estrutura para salvar os resultados obtidos.

        aperture_size : int
            Tamanho da sub-abertura a ser utilizada em número de elementos.

        aperture_step : int
            Passo entre duas sub-aberturas em número de elementos.

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

        beta : float
            Parâmetro para focar em refletores especulares ou não-especulares.
            Valores positivos dão ênfase em refletores especulares, enquanto valores negativos dão enfase em refletores
            puntuais. Por padrão, beta=1. Os valores recomendados estão na faixa de -5 a 5.

    Returns
    -------
    int
        Chave de identificação do resultado (``output_key``).

    Raises
    ------
    TypeError
        Se ``data_insp`` não for do tipo :class:`.data_types.DataInsp`.

    TypeError
        Se ``aperture_size`` não for do tipo :class:`int` ou se não for possível
        realizar sua conversão para :class:`int`.

    TypeError
        Se ``aperture_step`` não for do tipo :class:`int` ou se não for possível
        realizar sua conversão para :class:`int`.

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
        Se ``beta`` não for do tipo :class:`float` ou se não for possível
        realizar sua conversão para :class:`float`.

    NotImplementedError
        Se o tipo de captura (:attr:`.data_types.InspectionParams.type_capt`)
        não for ``FMC``.

    """

    # Teste dos tipos dos parâmetros.
    if type(data_insp) is not DataInsp:
        raise TypeError("O argumento ``data_insp`` não é um objeto do tipo ``DataInsp``.")

    try:
        aperture_size = int(aperture_size)
    except ValueError:
        raise TypeError("Não foi possível converter o argumento ``aperture_size`` para o tipo ``int``.")

    try:
        aperture_step = int(aperture_step)
    except ValueError:
        raise TypeError("Não foi possível converter o argumento ``aperture_step`` para o tipo ``int``.")

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

    try:
        beta = float(beta)
    except ValueError:
        raise TypeError("Não foi possível converter o argumento ``beta`` para o tipo ``float``.")

    if data_insp.inspection_params.type_capt != "FMC":
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo VTFM.")

    apertures = []
    n = (data_insp.probe_params.num_elem - aperture_size) // aperture_step
    for i in range(n + 1):
        apertures.append(range(i * aperture_step, i * aperture_step + aperture_size))

    g = data_insp.ascan_data[:, :, :, sel_shot]
    ts = data_insp.inspection_params.sample_time
    pos_trans = data_insp.probe_params.elem_center
    elem_width = data_insp.probe_params.elem_dim
    gate_start = data_insp.inspection_params.gate_start
    center_freq = data_insp.probe_params.central_freq
    roi_coord = roi.get_coord()
    dist = cdist(pos_trans, roi.get_coord()) * 1e-3
    dist = dist / (c * ts * 1e-6)
    directivity = np.zeros((len(pos_trans), roi.h_len * roi.w_len))
    beam_spread = np.zeros((len(pos_trans), roi.h_len * roi.w_len))
    for i in range(len(pos_trans)):
        directivity[i, :] = np.arctan2(roi_coord[:, 0] - pos_trans[i, 0], roi_coord[:, 2] - pos_trans[i, 2])
    directivity = np.sinc(elem_width * 1e-3 * np.sin(directivity) * center_freq * 1e6 / c)
    directivity_correction = np.expand_dims(directivity, axis=1) * np.expand_dims(directivity, axis=0)
    directivity_correction = np.sum(directivity_correction, axis=(0, 1)).reshape((roi.h_len, roi.w_len)).T
    for i in range(len(pos_trans)):
        beam_spread[i, :] = 1 / np.sqrt(np.sqrt(((roi_coord[:, 0] - pos_trans[i, 0]) * 1e-3) ** 2 +
                                                ((roi_coord[:, 2] - pos_trans[i, 2]) * 1e-3) ** 2))
    beam_spread_correction = np.expand_dims(beam_spread, axis=1) * np.expand_dims(beam_spread, axis=0)
    beam_spread_correction = np.sum(beam_spread_correction, axis=(0, 1)).reshape((roi.h_len, roi.w_len)).T
    beam_spread_correction[0, :] = beam_spread_correction[1, :]
    images = []
    angles = []
    apertures.append(np.asarray(range(len(pos_trans))))
    for aperture in apertures:
        aperture = np.asarray(aperture)
        central_pos = np.average(pos_trans[aperture], axis=0)
        img = np.zeros((1, roi.h_len * roi.w_len))
        angle = np.zeros((1, roi.h_len * roi.w_len))
        d = dist[aperture]
        combs = []
        for i in range(len(aperture)):
            for j in range(len(aperture)):
                combs.append(np.asarray([j, i]))

        combs = np.asarray(combs)
        index = (d[combs[:, 0]] + d[combs[:, 1]]).astype(int) - int(gate_start / ts)
        index[index >= g.shape[0]] = -1

        _vtfm_kernel(g, combs, index, aperture, img, angle, roi_coord, central_pos)

        img = img.reshape((roi.w_len, roi.h_len)).T
        angle = angle.reshape((roi.w_len, roi.h_len)).T

        img = post_proc.envelope(img)
        img /= img.max()

        images.append(img)
        angles.append(angle)

    tfm_image = directivity_correction * beam_spread_correction * images.pop(-1)
    angles.pop(-1)

    images = np.asarray(images)
    # angles = np.asarray(angles)
    # vector = np.sum((images ** alfa) * np.exp(1j * angles), axis=0)
    specularity = np.std(np.abs(images), axis=0) / (
            np.mean(np.abs(images), axis=0) + 1e-30)  # soma 1e-30 para evitar nan
    specularity /= np.sqrt(len(apertures) - 1)

    # vector = vector ** (1 / alfa)

    # final_vector = vector / (np.abs(vector) + 1e-30) * tfm_image

    out_img = tune_image(tfm_image, specularity, beta)

    # --- FIM DO ALGORITMO VTFM.
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
    result.image = out_img
    result.surface = None

    # Guarda o resultado no dicionário.
    data_insp.imaging_results[output_key] = result

    return output_key  # , final_vector, specularity, images, angles, tfm_image, directivity, directivity_correction, beam_spread_correction


@njit(parallel=True)
def _vtfm_kernel(g, combs, index, aperture, img, angle, roi_coord, central_pos):
    for pos in prange(img.shape[1]):
        for i in range(combs.shape[0]):
            img[0, pos] += g[index[i, pos], aperture[combs[i, 0]], aperture[combs[i, 1]]]
        angle[0, pos] = np.arctan2((roi_coord[pos, 0] - central_pos[0]) * 1e-3,
                                   (roi_coord[pos, 2] - central_pos[2]) * 1e-3)


def tune_image(image, specularity, beta):
    """Realça uma imagem de tfm de acordo com o mapa de especularidade.

    Parameters
    ----------
        image : 2d-array
            Imagem do tfm.
        specularity : 2d-array
            Mapa de especularidade.
        beta : int ou float
            Expoente aplicado no mapa de especularidade. Valores positivos dão ênfase em refletores especulares,
            enquanto valores negativos dão enfase em refletores puntuais.

    Returns
    -------
    2d-array
        Imagem realçada.
    """
    out = (specularity ** beta) * image
    return out


def vtfm_params():
    """Retorna os parâmetros do algoritmo TFM.

        Returns
        -------
        dict
            Dicionário com os parâmetros utilizados pelo algoritmo VTFM e valores padrões para sua utilização.
    """
    return {"aperture_size": 32, "aperture_step": 4, "roi": ImagingROI(), "output_key": None,
            "description": "", "c": 5900.0, "beta": 1.0}
