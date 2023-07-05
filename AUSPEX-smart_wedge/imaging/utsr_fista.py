# -*- coding: utf-8 -*-
r"""
Módulo ``utsr_fista``
=====================

O FISTA (*Fast Iteractive Shrinkage-Thresholding Algorithm*) constitui um método de otimização que
utiliza o operador Shrinkage-Thresholding, definido conforme as Equações :eq:`eq-1-fista`: e :eq:`eq-2-fista`:
para um valor de x escalar.

.. math:: Sa(x) = 0, a >= |x|
    :label: eq-1-fista

.. math:: Sa(x) = (x-a)\text{sign}(x), a < |x|
    :label: eq-2-fista


No algoritmo, :math:`x` é um vetor multidimensional. Entretanto, o operador deve ser aplicado
individualmente a cada um de seus elementos.

O método é mais simples e têm se mostrado eficiente na minimização de normas
:math:`\ell_1-\ell_2`, especialmente em problemas de grandes dimensões. Ele está baseado em
um gradiente na ordem de :math:`(1/k^2)`.

No FISTA, calcula-se o vetor de resíduos :math:`g - Hf` no espaço dos dados e sua projeção de
volta ao espaço da imagem pela multiplicação por :math:`H` adjunta. Em seguida, aplica-se o operador
de *shrinkage-thresholding*, estimando um novo para a figura. O cálculo é iterativo e é executado
até que a diferença entre imagens consecutivas seja menor que o critério de parada.

O vetor passado como argumento ao operador é obtido pela soma de um ponto :math:`y^k` ao negativo do
gradiente no mesmo ponto, porém escalonado pelo parâmetro :math:`c`.
O ponto :math:`y^k` é resultado da combinação de duas soluções anteriores.
Isso faz com que, enquanto as iterações evoluem, o vetor :math:`y^k` modifica o ponto da
solução sobre o qual o operador é aplicado. Isso faz com que o algoritmo seja mais rápido.

A convergência do FISTA é garantida quando :math:`c` é maior que a constante de Lipschitz do
gradiente do termo diferenciável da função custo, ou seja, o termo quadrático.
A constante de Lipschitz está relacionada ao maior valor singular da multiplicação da matrizes
:math:`H` adjunta e :math:`H`.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo UTSR FISTA para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.utsr_fista`; e
o pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`.

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo UTSR FISTA. Além disso, a imagem mostra o
resultado do UTSR FISTA com pós-processamento.

.. plot:: plots/imaging/utsr_fista_example.py
    :include-source:
    :width: 100 %
    :align: center

.. raw:: html

    <hr>

"""
import numpy as np

from framework.data_types import DataInsp, ImagingROI, ImagingResult
import framework.schmerr_model as sm
from framework.model_s2 import model_s2_direct, model_s2_adjoint, shrinkage


def model_s2(x, nt, nu, roi, dt, du, c, tau0, _model, _filt, coord_orig):
    # Aplica o operador direto do modelo de aquisição.
    g, _ = model_s2_direct(np.reshape(x, newshape=(roi.h_len, roi.w_len), order="F"),
                           nt0=nt, nu0=nu,
                           dt=dt, du=du,
                           roi=roi,
                           c=c, tau0=tau0, model_transd=_model, coord_orig=coord_orig)

    # Aplica o operador adjunto do  modelo de aquisição.
    y, _ = model_s2_adjoint(g, nt0=nt, nu0=nu,
                            dt=dt, du=du,
                            roi=roi,
                            c=c, tau0=tau0, filter_transd=_filt, coord_orig=coord_orig)

    # Sem regularização, apenas solução por mínimos quadrados

    y = y.flatten("F")

    return y


def utsr_fista_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0,
                      c=None, _model=-1, alpha=1.0, tol=2e-4, max_stag_count=-1, debug=False):
    """Processa dados de A-scan utilizando o algoritmo UTSR FISTA.

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

    description : :class:`str`
        Texto descritivo para o resultado. Por padrão, é uma *string* vazia.

    sel_shot : :class:`int`
        Parâmetro que refere-se ao disparo caso o transdutor tenha sido
        deslocado.

    c : :class:`int` ou :class:`float`
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    _model : :class:`int` ou :class:`np.ndarray`
        Modelo de transdutor que será usado. Por padrão recebe -1.

    alpha : :class:`int` ou :class:`float`
        Parâmetro usado para cálculo do treshold. Por padrão recebe 1.

    tol : :class:`int` ou :class:`float`
        Tolerância de erro no resultado do algoritmo. Por padrão recebe 2e-4.

    max_stag_count : :class:`int`
        Número máximo de iterações consecutivas em que o resultado do
        algoritmo não reduz em 10% do valor da tolerância. Por padrão, é -1.

    debug : :class:`boolean`
        Depurador do algoritmo. Por padrão é `False`.

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
        
    # TODO: implementar os testes de verificação dos outros parâmetros da função.

    # --- INÍCIO DO ALGORITMO FISTA, desenvolvido por Andréia. ---
    # Calcula os instantes de tempo que definem a região da ROI nos sinais A-scan.
    dt = (data_insp.time_grid[1, 0] - data_insp.time_grid[0, 0]) * 1e-6
    ermv = c / 2.0
    z0 = (roi.h_points[0] - data_insp.probe_params.elem_center[0, 2]) * 1e-3
    ze = (roi.h_points[-1] - data_insp.probe_params.elem_center[0, 2]) * 1e-3
    idx_t0 = np.floor((z0 / ermv) / dt + 0.5).astype(int)
    idx_te = np.floor((ze / ermv) / dt + 0.5).astype(int)

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "sweep":
        s_t_u = data_insp.ascan_data[idx_t0: idx_te, 0, 0, :]
        nu0 = data_insp.inspection_params.step_points.shape[0]
        du = (data_insp.inspection_params.step_points[1, 0] - data_insp.inspection_params.step_points[0, 0]) * 1e-3
        coord_orig = data_insp.inspection_params.step_points[0]

    elif data_insp.inspection_params.type_capt == "FMC":
        s_t_u = np.zeros((idx_te - idx_t0, data_insp.probe_params.num_elem))
        for i in range(data_insp.probe_params.num_elem):
            s_t_u[:, i] = data_insp.ascan_data[idx_t0: idx_te, i, i, sel_shot]

        nu0 = data_insp.probe_params.num_elem
        du = data_insp.probe_params.pitch * 1e-3
        # TODO: verificar se esse cálculo de coordenada está adequado com o padrão do ``framework``.
        coord_orig = data_insp.probe_params.elem_center[0]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``sweep`` e ``FMC``.")

    # Gera o modelo e o filtro casado.
    if (_model is None) or ((type(_model) is not np.ndarray) and (_model == -1)):
        _model, _filt, _, _, _ = sm.generate_model_filter(data_insp, c=c)
    else:
        _filt = np.conjugate(_model)

    # Faz a reconstrução utilizando o método FISTA para resolução do problema inverso.
    htg, _ = model_s2_adjoint(s_t_u,
                              nt0=data_insp.inspection_params.gate_samples,
                              nu0=nu0,
                              dt=dt,
                              du=du,
                              roi=roi,
                              tau0=data_insp.time_grid[idx_t0, 0],
                              c=c,
                              filter_transd=_filt,
                              coord_orig=coord_orig)
    image_shape = htg.shape
    htg = htg.flatten("F")
    if (alpha is None) or (alpha == -1):
        alpha = np.sqrt(0.2 * np.linalg.norm(htg, ord=np.inf))

    x_k_1 = htg
    num_iter = 0
    stag_count = 0
    y_k = x_k_1
    c_fista = 2e0
    restart = 0
    t = [1.0]
    t1 = [1.0]
    dif_fista = [0.0]

    while True:
        # Calcula o gradiente de y_k.
        hth_y_k = model_s2(y_k, nt=data_insp.inspection_params.gate_samples,
                           nu=nu0,
                           dt=dt,
                           du=du,
                           roi=roi,
                           c=c,
                           tau0=data_insp.time_grid[idx_t0, 0],
                           _model=_model,
                           _filt=_filt,
                           coord_orig=coord_orig)

        grad = htg - hth_y_k
        x_k = shrinkage(data=((1 / c_fista) * grad + y_k), value=(alpha ** 2.0) / c_fista)
        res = x_k - x_k_1
        num_iter = num_iter + 1

        # Combination Parameter
        t.append((1.0 + np.sqrt(1.0 + 4.0 * t[num_iter - 1] ** 2.0)) / 2.0)
        t1.append((t[num_iter - 1] - 1) / t[num_iter])

        if (restart == 1) and (np.dot((y_k - x_k).T, res) > 0):
            y_k = x_k
            t[num_iter] = 1.0
        else:
            y_k = x_k + t1[num_iter] * res

        # Calcula a 'diferença' entre imagens de iterações consecutivas.
        dif_fista.append(np.linalg.norm(res) / np.linalg.norm(x_k))
        if debug:
            print(' iter = %4d | alpha = %f | difFISTA = %f |' % (num_iter, alpha, dif_fista[num_iter]))

        # Tolerância do algoritmo FISTA encontrada.
        if dif_fista[num_iter] < tol:  # or (err_mse < tol)
            if debug:
                print('Tolerância atingida')
            break

        # Iterações estagnadas.
        if (max_stag_count is not None) and (max_stag_count > 0):
            if num_iter > 1 and (np.abs(dif_fista[num_iter] - dif_fista[num_iter - 1]) < (tol / 10.0)):
                stag_count = stag_count + 1
                if stag_count >= max_stag_count:
                    if debug:
                        print('Iterações estagnadas')
                    break
            else:
                stag_count = 0

        # Ajusta variáveis para a próxima iteração.
        x_k_1 = x_k

    # Retorna para o domínio da imagem.
    image = np.reshape(x_k, image_shape, order="F")

    # --- FIM DO ALGORITMO FISTA.
    # Salva o resultado.
    if output_key is None:
        # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
        result = ImagingResult(roi=roi, description=description)
        result.image = image

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
        result.image = image

        # Guarda o resultado no dicionário.
        data_insp.imaging_results[output_key] = result

    # Retorna o valor da chave
    return output_key


def utsr_fista_params():
    """ Retorna os parâmetros do algoritmo UTSR FISTA.

    Returns
    -------
    dict
        Dicionário, em que a chave ``roi`` representa a região de interesse
        utilizada pelo algoritmo, a chave ``output_key`` representa a chave
        de identificação do resultado, a chave ``description`` representa a
        descrição do resultado, a chave ``sel_shot`` representa o disparo
        do transdutor, a chave ``c`` representa a velocidade de propagação
        da onda na peça, a chave ``_model`` representa o modelo de transdutor
        que será usado, a chave ``alpha`` representa o parâmetro usado para cálculo
        do treshold, a chave ``tol`` representa a tolerância de erro no resultado do
        algoritmo, a chave ``max_stag_count`` representa o número máximo de iterações
        consecutivas em que o resultado do algoritmo não reduz em 10% do valor da
        tolerância e a chave ``debug`` representa o depurador do algoritmo.

    """

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0, "_model": -1,
            "alpha": 1.0, "tol": 2e-4, "max_stag_count": -1, "debug": False}
