# -*- coding: utf-8 -*-
"""
Módulo ``saft``
===============

O SAFT (*Synthetic Aperture Focusing Technique* - Técnica de Focalização de
Abertura Sintética) é uma ferramenta que tem sido usada para restaurar
imagens ultrassônicas obtidas de *B-scans* com distorção de foco. Com o uso
desta técnica, há uma melhoria da resolução da imagem pode obtida, sem o uso
das lentes ultrassônicas tradicionais.

O foco sintético é baseado na reflexão geométrica ou no modelo acústico de
raios.

O modelo do algoritmo considera que o foco do transdutor ultrassônico é
assumido como sendo um ponto de fase constante pelo qual todos os raios
sonoros passam antes de divergir em um cone cujo ângulo é determinado pelo
diâmetro do transdutor e pela distância focal.

Se um alvo refletivo estiver localizado abaixo do ponto focal e dentro do
cone, serão calculados o comprimento do caminho e o tempo de trânsito para um
sinal viajando ao longo do raio. A largura do cone em um determinado
intervalo corresponde à largura de abertura que pode ser sintetizada, e o
comprimento do caminho que o sinal deve percorrer corresponde ao deslocamento
de fase visto no sinal para essa posição do transdutor.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo SAFT para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.saft`; e
o pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`. 

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo SAFT. Além disso, a imagem mostra o
resultado do SAFT com pós-processamento.

.. plot:: plots/imaging/saft_example.py
    :include-source:
    :width: 100 %
    :align: center

.. raw:: html

    <hr>

"""
import numpy as np
from scipy.spatial.distance import cdist
from surface.surface import Surface

from framework.data_types import DataInsp, ImagingROI, ImagingResult


def saft_oper_direct(image, roi, coord_transd, nt, nu, c=5900.0, dt=1e-8, tau0=0.0):
    """Calcula a modelagem de Kirchhoff (Claerbout, 2004, p. 108).

    Calcula como cada ponto da imagem influencia nos sinais A-scan. Faz isso realizando a operação de *scattering* de
    cada ponto da ROI nos sinais A-scan.
    """
    # Calcula a distância entre os pontos da ROI e os centros dos transdutores.
    dist = cdist(coord_transd * 1e-3, roi.get_coord() * 1e-3)

    # Calcula os índices dos sinais ``A-scan`` relativos a cada ponto da ROI.
    dist_correction = 2.0 / (c * dt)
    j = np.rint(dist_correction * dist - tau0 / dt).astype(int).T
    j[j >= nt] = (nt*nu)

    # Realiza o "espalhamento" de cada ponto da imagem para a amplitudes dos sinais ``A-scan``.
    data = np.bincount((j + np.arange(0, nt * nu, nt, dtype=int)).flatten('F'),
                       weights=np.tile(image.flatten('F'), nu),
                       minlength=(nt * nu + 1))
    data = np.reshape(data[0: nt * nu], (nt, nu), order='F')

    return data


def saft_oper_adjoint(data, roi, coord_transd, c=5900.0, dt=1e-8, tau0=0.0):
    """Calcula a migração de Kirchhoff (Claerbout, 2004, p.108).

    Equivale ao algoritmo do SAFT. Toma um conjunto de sinais A-scan e faz a operação *Delay-and-Sum* para obter uma
    imagem.
    """
    # Calcula a distância entre os pontos da ROI e os centros dos transdutores.
    dist = cdist(coord_transd * 1e-3, roi.get_coord() * 1e-3)
    dist = dist.reshape(dist.shape[0], roi.w_len, roi.h_len)
    dist = np.transpose(dist, (-1, 1, 0))

    # Calcula os índices dos sinais ``A-scan`` relativos a cada ponto da ROI.
    data = np.append(data, np.zeros([1, data.shape[-1]]), axis=0)
    dist_correction = 2.0 / (c * dt)
    j = np.rint(dist_correction * dist - tau0 / dt).astype(int)
    j[j >= data.shape[0]] = -1

    # Realiza a soma das amplitudes dos sinais ``A-scan`` para cada ponto da ROI.
    nx = np.arange(data.shape[-1], dtype=int)
    nx = np.tile(nx.transpose(), [j.shape[0], roi.w_len, 1])
    o = data[j, nx]
    image = o.sum(2)

    return image


def saft_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None,
                scattering_angle=None):
    """Processa dados de A-scan utilizando o algoritmo SAFT.

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

        scattering_angle : float ou nd-array
            Filtro para considerar a abertura do feixe emitido por cada elemetno transdutor. Por padrão é None para ser
            definido posteriormente.

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

    if data_insp.surf is None and data_insp.inspection_params.type_insp == 'immersion':
        raise ValueError("Surface não inicializado")

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "sweep":
        g = data_insp.ascan_data[:, 0, 0, :]
        nb_elements = data_insp.inspection_params.step_points.shape[0]
    elif data_insp.inspection_params.type_capt == "FMC":
        g = np.zeros((data_insp.inspection_params.gate_samples, data_insp.probe_params.num_elem),
                     dtype=data_insp.ascan_data.dtype)
        nb_elements = data_insp.probe_params.num_elem
        for i in range(data_insp.probe_params.num_elem):
            g[:, i] = data_insp.ascan_data[:, i, i, sel_shot]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``sweep`` e ``FMC``.")

    if scattering_angle is None:
        scatfilt = np.zeros((nb_elements, roi.h_len * roi.w_len), dtype=bool)
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
                    scatfilt.shape[1] == roi.h_len * roi.w_len):
                strerr = "O argumento scatfilt não tem o formato exigido (" + str(nb_elements) + "x" + str(
                    roi.h_len * roi.w_len) + ")"
                raise TypeError(strerr)

    # --- INÍCIO DO ALGORITMO SAFT, desenvolvido por Hector. ---
    # Calcula a distância entre os pontos da ROI e os centros dos transdutores.
    if data_insp.inspection_params.type_capt == "sweep":
        # Há movimentação de transdutor.
        if data_insp.inspection_params.type_insp == 'immersion':
            # surf = Surface(data_insp, -1)
            # dist = surf.cdist(data_insp.probe_params.elem_center + data_insp.inspection_params.step_points[sel_shot] -
            #                  data_insp.inspection_params.step_points[0], roi.get_coord()) * 1e-3
            dist = data_insp.surf.cdist_medium(
                data_insp.probe_params.elem_center, roi.get_coord()) * 1e-3
            dist_correction = 2.0 / (np.asarray([data_insp.inspection_params.coupling_cl, c]) *
                                     data_insp.inspection_params.sample_time * 1e-6)
            samp_dist = dist[0] * dist_correction[0] + dist[1] * dist_correction[1]
        else:
            dist = cdist(data_insp.inspection_params.step_points * 1e-3, roi.get_coord() * 1e-3)
            dist_correction = 2.0 / (c * data_insp.inspection_params.sample_time * 1e-6)
            samp_dist = dist * dist_correction
    else:
        # É um *array* estático.
        if data_insp.inspection_params.type_insp == 'immersion':
            # surf = Surface(data_insp, -1)
            # dist = surf.cdist(data_insp.probe_params.elem_center + data_insp.inspection_params.step_points[sel_shot] -
            #                  data_insp.inspection_params.step_points[0], roi.get_coord()) * 1e-3
            dist = data_insp.surf.cdist_medium(
                data_insp.probe_params.elem_center, roi.get_coord()) * 1e-3
            dist_correction = 2.0 / (np.asarray([data_insp.inspection_params.coupling_cl, c]) *
                                     data_insp.inspection_params.sample_time * 1e-6)
            samp_dist = dist[0] * dist_correction[0] + dist[1] * dist_correction[1]
        else:
            dist = cdist(data_insp.probe_params.elem_center * 1e-3, roi.get_coord() * 1e-3)

            dist_correction = 2.0 / (c * data_insp.inspection_params.sample_time * 1e-6)
            samp_dist = dist * dist_correction
    samp_dist[scatfilt] = g.shape[0]
    # Calcula os índices dos sinais ``A-scan`` relativos a cada ponto da ROI.

    j = np.rint(samp_dist - data_insp.inspection_params.gate_start*data_insp.inspection_params.sample_freq).astype(int)
    j[j < 0] = -1
    j[j >= g.shape[0]] = -1
    nt = j.shape
    j = j.reshape(nt[0], roi.w_len, roi.h_len)
    j = np.transpose(j, (-1, 1, 0))
    # Realiza a soma das amplitudes dos sinais ``A-scan`` para cada ponto da ROI.
    nx = np.arange(g.shape[-1], dtype=int)
    # nx = np.tile(nx.transpose(), [j.shape[0], roi.w_len, 1])
    o = g[j, nx]
    f = o.sum(2)

    # --- FIM DO ALGORITMO SAFT.
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


def saft_params():
    """Retorna os parâmetros do algoritmo SAFT.

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

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0, "scattering_angle":
                                                                                                    180.0}
