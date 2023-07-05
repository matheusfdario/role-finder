# -*- coding: utf-8 -*-
r"""
Módulo ``sparsa``
=================

O SpaRSA (*SPArse Reconstruction by a Separable Approximation* -
Reconstrução Esparsa por Algoritmos de Aproximação Separável) é um método
de reconstrução de imagens em aplicações ENDs que se baseia em uma
reconstrução inversa regularizada.

Essa técnica busca melhorar a eficiência computacional de algoritmos
baseados em problemas inversos, além de solucionar o problema de
resolução lateral e temporal de imagens em algoritmos B-Scan que possuem
limitações devido ao tamanho do transdutor finito e da ressonância do
transdutor, auxiliando na melhor caracterização dos defeitos.

O algoritmo utiliza um modelo linear de imagens que considera a difração
acústica e os efeitos elétricos baseados na resposta espacial ao impulso
(SIR) e na resposta elétrica do impulso. Em sua formulação, há um problema
inverso regularizado, estabelecido com a distribuição esparsa de defeitos
do modelo, e uma função objetivo inversa composta por normas :math:`\ell_1` e
:math:`\ell_2`, que são resolvidas pela aproximação separável.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo SpaRSA para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.sparsa`; e
o pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`.

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo SpaRSA. Além disso, a imagem mostra o
resultado do SpaRSA com pós-processamento.

.. plot:: plots/imaging/sparsa_example.py
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


def model_s_t_s(x, nt, nu, roi, dt, du, c, tau0, _model, _filt, coord_orig):
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


def sparsa_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0,
                  c=None, _model=-1, alpha=1.0, tol=1e-5, debug=False):
    """Processa dados de A-scan utilizando o algoritmo UTSR.

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
        deslocado. Por padrão, é 0.

    c : :class:`int`, :class:`float`, :class:`NoneType`
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    _model : :class:`int` ou :class:`np.ndarray`
        Modelo de transdutor que será usado. Por padrão recebe -1.

    alpha : :class:`int`, :class:`float`
        Parâmetro usado para cálculo do threshold. Por padrão, é 1.0.

    tol : :class:`int`, :class:`float`
        Tolerância de erro no resultado do algoritmo. Por padrão é 1e-5.

    debug : :class:`bool`
        Depurador do algoritmo. Por padrão, é `False`.

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

    # --- INÍCIO DO ALGORITMO SpaRSA, desenvolvido por Andréia. ---
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

    # Faz a reconstrução SpaRSA.
    st_y, _ = model_s2_adjoint(s_t_u,
                               nt0=data_insp.inspection_params.gate_samples,
                               nu0=nu0,
                               dt=dt,
                               du=du,
                               roi=roi,
                               tau0=data_insp.time_grid[idx_t0, 0],
                               c=c,
                               filter_transd=_filt,
                               coord_orig=coord_orig)
    image_shape = st_y.shape
    st_y = st_y.flatten("F")
    s_t_u = s_t_u.flatten("F")
    if (alpha is None) or (alpha == -1):
        alpha = np.sqrt(0.2 * np.linalg.norm(st_y, ord=np.inf))

    f_temp = st_y
    f_temp_1 = f_temp
    h_f, _ = model_s2_direct(np.reshape(st_y, newshape=image_shape, order="F"),
                             nt0=data_insp.inspection_params.gate_samples,
                             nu0=nu0,
                             dt=dt,
                             du=du,
                             roi=roi,
                             c=c,
                             tau0=data_insp.time_grid[idx_t0, 0],
                             model_transd=_model,
                             coord_orig=coord_orig)
    h_f = h_f.flatten("F")
    h_f_1 = h_f

    num_iter = 0
    sigma = 2.0
    eta = 2.0

    t_l2 = [0.0]
    t_l1 = [0.0]
    j_f = [0.0]
    dif_sparsa = [0.0]

    # Laço externo
    while True:
        # Cálculo do passo(eta)
        if num_iter > 0:
            s = (f_temp - f_temp_1)
            eta = (np.linalg.norm(h_f_1 - h_f) ** 2) / (np.linalg.norm(s) ** 2)

        # Calcula o gradiente
        sts_f = model_s_t_s(f_temp, nt=data_insp.inspection_params.gate_samples,
                            nu=nu0,
                            dt=dt,
                            du=du,
                            roi=roi,
                            c=c,
                            tau0=data_insp.time_grid[idx_t0, 0],
                            _model=_model,
                            _filt=_filt,
                            coord_orig=coord_orig)
        grad = sts_f - st_y
        h_f_1 = h_f

        # Laço interno
        while True:
            # Faz o soft-thresold, calculando f_i+1
            w = f_temp - (1 / eta) * grad
            f = shrinkage(data=w, value=(alpha ** 2.0) / eta)

            # Calcula os termos da função custo
            h_f, _ = model_s2_direct(np.reshape(f, newshape=image_shape, order="F"),
                                     nt0=data_insp.inspection_params.gate_samples,
                                     nu0=nu0,
                                     dt=dt,
                                     du=du,
                                     roi=roi,
                                     c=c,
                                     tau0=data_insp.time_grid[idx_t0, 0],
                                     model_transd=_model,
                                     coord_orig=coord_orig)
            h_f = h_f.flatten("F")
            _t_l2 = (1.0 / 2.0) * np.linalg.norm(s_t_u - h_f) ** 2.0
            _t_l1 = (alpha ** 2) * np.linalg.norm(f, ord=1)
            _j_f = _t_l2 + _t_l1

            # Verifica se J(f_i + 1) < J(f_i)
            if debug:
                print('num_iter = %d | alpha = %f | eta = %f | J(f_i+1) = %f | J(f_i) = %f' % (num_iter,
                                                                                               alpha,
                                                                                               eta,
                                                                                               _j_f,
                                                                                               j_f[num_iter]))

            if (num_iter == 0) or (_j_f < j_f[num_iter]):
                t_l2.append(_t_l2)
                t_l1.append(_t_l1)
                j_f.append(_j_f)
                break
            else:
                eta = sigma * eta

        # Calcula a 'diferença' entre imagens de iterações consecutivas.
        if num_iter > 0:
            dif_sparsa.append(np.abs(j_f[num_iter + 1] - j_f[num_iter]) / j_f[num_iter])
        else:
            dif_sparsa.append(1.0)

        if debug:
            print('difSpaRSA = %f' % (dif_sparsa[num_iter + 1]))

        # Tolerância do algoritmo SpaRSA encontrada.
        if dif_sparsa[num_iter + 1] < tol:  # or num_iter > 200:  # or (err_mse < tol)
            if debug:
                print('Tolerância atingida')
            break

        # Ajusta as variáveis para a próxima iteração.
        f_temp_1 = f_temp
        f_temp = f
        num_iter = num_iter + 1

    # Retorna para o domínio da imagem.
    image = np.reshape(f, image_shape, order="F")

    # --- FIM DO ALGORITMO SpaRSA.
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


def sparsa_params():
    """ Retorna os parâmetros do algoritmo SpaRSA.

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
            algoritmo, e a chave ``debug`` representa o depurador do algoritmo.

        """

    return {"roi": ImagingROI(), "output_key": None, "description": "", "sel_shot": 0, "c": 5900.0, "_model": -1,
            "alpha": 1.0, "tol": 1e-5, "debug": False}
