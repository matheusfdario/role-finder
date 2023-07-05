# -*- coding: utf-8 -*-
r"""
Módulo ``cumulative_tfm``
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
from framework import post_proc
from framework.data_types import DataInsp, ImagingROI, ImagingResult
from imaging import tfm


initialize = False


def cumulative_tfm_kernel(data_insp, roi, output_key=None, description="", sel_shots=np.array([0]),
                          c=None, trcomb=None, scattering_angle=None, combine_mean=False, use_cuda=False):
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

        sel_shots : int
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

        combine_mean : bool
            If True, each resulting pixel will be the mean of the corresponding overlapped pixels.
            If False, the resulting pixel will be the maximum value between the overlapped pixels.

        use_cuda : bool
            Habilita o uso de GPU CUDA-compatible.


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
        Se ``sel_shots`` não for do tipo :class:`int` ou se não for possível
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
        sel_shots[0]
    except TypeError:
        sel_shots = [sel_shots]
    sel_shots = np.array(sel_shots, dtype=int)

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt != "FMC":
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo TFM.")

    surf = data_insp.surf
    steps = data_insp.inspection_params.step_points
    corner_roi = roi.coord_ref
    try:
        step_dx = int((steps[1] - steps[0])[0] / roi.w_step)
    except IndexError:
        step_dx = 0
    w = roi.w_len
    h = roi.h_len
    if combine_mean:
        alpha = np.zeros((h, int((steps[sel_shots.max()][0] - steps[sel_shots.min()][0]) / roi.w_step + w)),
                         dtype=np.float32)
    tfm_img = np.zeros((h, int((steps[sel_shots.max()][0] - steps[sel_shots.min()][0]) / roi.w_step + w)),
                       dtype=np.complex64)
    for i, shot in enumerate(sel_shots):
        print(f'Cumulative TFM shot:{shot}')
        # roi = ImagingROI(corner_roi, height=roi, width=roi_size[0], h_len=h, w_len=w)
        if use_cuda:
            from imaging import tfmcuda
            chave = tfmcuda.tfmcuda_kernel(data_insp, roi=roi, sel_shot=shot, scattering_angle=scattering_angle,
                                           output_key=i, c=c, trcomb=trcomb)
        else:
            chave = tfm.tfm_kernel(data_insp, roi=roi, sel_shot=shot, scattering_angle=scattering_angle,
                                   output_key=i, c=c, trcomb=trcomb)
        rel_shot = shot - sel_shots[0]
        if combine_mean:
            image = data_insp.imaging_results.pop(chave).image
            alpha[:, rel_shot * step_dx:w + rel_shot * step_dx] += 1
            tfm_img[:, rel_shot * step_dx:w + rel_shot * step_dx] += image
        else:
            image = post_proc.envelope(data_insp.imaging_results.pop(chave).image)
            tfm_img[:, rel_shot * step_dx:w + rel_shot * step_dx] = \
                np.maximum(image, tfm_img[:, rel_shot * step_dx:w + rel_shot * step_dx])
    if combine_mean:
        alpha[alpha == 0] = np.Inf
        tfm_img *= 1 / alpha
    roi = ImagingROI(corner_roi, roi.height, roi.h_len, tfm_img.shape[1] * roi.w_step, tfm_img.shape[1],
                     roi.depth, roi.d_len)
    # tfm_img = post_proc.envelope(tfm_img)
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
    # result.median = median
    result.image = tfm_img
    if data_insp.inspection_params.type_insp == 'immersion':
        result.surface = data_insp.surf.get_points_in_roi(roi, sel_shots[0])
    # Guarda o resultado no dicionário.
    data_insp.imaging_results[output_key] = result

    # Retorna o valor da chave
    return output_key


def cumulative_tfm_params():
    """Retorna os parâmetros do algoritmo TFM.

    Returns
    -------
    dict
        Dicionário, em que a chave ``roi`` representa a região de interesse
        utilizada pelo algoritmo, a chave ``output_key`` representa a chave
        de identificação do resultado, a chave ``description`` representa a
        descrição do resultado, a chave ``sel_shots`` representa o disparo
        do transdutor, a chave ``c`` representa a velocidade de propagação
        da onda na peça e ``trcomb`` representa as combinações de transmis
        sores e receptores usados.

    """

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shots": np.array([0]), "c": 5900.0,
            "scattering_angle": 180}
