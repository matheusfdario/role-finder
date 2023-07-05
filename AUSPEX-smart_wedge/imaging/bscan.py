# -*- coding: utf-8 -*-
"""
Módulo ``bscan``
================
O B-scan é um algoritmo de reconstrução de imagens para ensaios não
destrutivos que recebe todos os elementos dos sinais ultrassônicos
da abertura. Esses sinais, no domínio do tempo, são então combinados para
produzir a imagem final do B-Scan.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo B-scan para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA (Assume-se
que os dados estão na mesma pasta em que o *script* é executado).

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando o módulo :mod:`imaging.bscan`; e o pós-processamento de
dados, utilizando o módulo :mod:`framework.post_proc`. 

O resultado do *script* é uma imagem, exibindo o resultado do algoritmo e o
resultado com pós-processamento.

.. plot:: plots/imaging/bscan_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>
    
"""

import numpy as np

from framework.data_types import DataInsp, ImagingROI, ImagingResult


def bscan_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None):
    """Processa dados de A-scan utilizando o algoritmo B-scan.

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
            deslocado. Por padrão, é 0.

        c : int ou float
            Velocidade de propagação da onda no objeto sob inspeção. Por
            padrão, é None e nesse caso é obtido o valor do data_insp.

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

    if c is None:
        c = data_insp.specimen_params.cl
    else:
        try:
            c = float(c)
        except ValueError:
            raise TypeError("Não foi possível converter o argumento ``c`` para o tipo ``float``.")

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "sweep":
        g = data_insp.ascan_data[:, 0, 0, :]
    elif data_insp.inspection_params.type_capt == "FMC":
        g = np.zeros((data_insp.inspection_params.gate_samples, data_insp.probe_params.num_elem))
        for i in range(data_insp.probe_params.num_elem):
            g[:, i] = data_insp.ascan_data[:, i, i, sel_shot]
    elif data_insp.inspection_params.type_capt == "PWI":
        # Pega apenas o Bscan da emissão com menor ângulo
        i = np.argmin(abs(data_insp.inspection_params.angles))
        g = data_insp.ascan_data[:, i, :, sel_shot]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``sweep`` e ``FMC``.")

    # --- INÍCIO DO ALGORITMO BSCAN, desenvolvido por Giovanni. ---
    # Calcula os índices da matriz de dados ``g`` que correspondem com a ROI.
    if data_insp.inspection_params.type_capt == "sweep":
        # Há movimentação de transdutor.
        pos_x_inic_transd = data_insp.inspection_params.step_points[0, 0]
        pos_x_fin_transd = data_insp.inspection_params.step_points[-1, 0]
        num_steps = len(data_insp.inspection_params.step_points)
        pitch_steps = data_insp.inspection_params.step_points[1, 0] - data_insp.inspection_params.step_points[0, 0]
    else:
        # É um *array* estático.
        pos_x_inic_transd = data_insp.probe_params.elem_center[0, 0]
        pos_x_fin_transd = data_insp.probe_params.elem_center[-1, 0]
        num_steps = data_insp.probe_params.num_elem
        pitch_steps = data_insp.probe_params.pitch

    min_x = roi.w_points[0] - pos_x_inic_transd
    min_x = int(min_x / pitch_steps) if roi.w_points[0] >= pos_x_inic_transd else 0
    max_x = roi.w_points[-1] - pos_x_inic_transd
    max_x = int(max_x / pitch_steps) if roi.w_points[-1] <= pos_x_fin_transd else num_steps

    min_z = int((roi.h_points[0]*1e-3/(c/2) - data_insp.time_grid[0, 0]*1e-6) /
                (data_insp.inspection_params.sample_time*1e-6))
    max_z = int((roi.h_points[-1]*1e-3/(c/2) - data_insp.time_grid[0, 0]*1e-6) /
                (data_insp.inspection_params.sample_time*1e-6))

    # Pega a imagem Bscan.
    f = g[min_z:max_z, min_x:max_x]

    # Ajusta a ROI para ficar com o tamanho da imagem.
    roi = ImagingROI(roi.coord_ref, width=roi.width, height=roi.height, h_len=f.shape[0], w_len=f.shape[1])

    # --- FIM DO ALGORITMO Bscan.
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


def bscan_params():
    """Retorna os parâmetros do algoritmo B-scan.

    Returns
    -------
    dict
        Dicionário, em que a chave ``roi`` representa a região de interesse
        utilizada pelo algoritmo, a chave ``output_key`` representa a chave
        de identificação do resultado, a chave ``description`` representa a
        descrição do resultado, a chave ``sel_shot`` representa o disparo
        do transdutor e a chave ``c`` representa a velocidade de propagação
        da onda na peça.
    
    """
    
    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0}
