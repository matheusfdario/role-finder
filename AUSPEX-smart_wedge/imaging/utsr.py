# -*- coding: utf-8 -*-
"""
Módulo ``utsr``
===============

O UTSR (*Ultrasonic Sparse Reconstruction*) é um algoritmo
de reconstrução de imagens em aplicações ENDs que se baseia na
resolução de um problema de mínimos quadrados regularizados,
com norma :math:`\ell_1` na regularização termo, criando uma reconstrução
esparsa da imagem.

Para a reconstrução da imagem, resolve-se um problema de otimização de
normas mistas, usando um algoritmos de mínimos quadrados iterativamente
reescritos (IRLS) e gradiente conjugado (CG).

A imagem recontruída pelo UTSR apresenta uma melhor resolução quando
comparada às respostas de algoritmos tradicionais.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo UTSR para a reconstrução de uma
imagem a partir de dados sintéticos, oriundos do simulador CIVA. (Assume-se
que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e :mod:`imaging.utsr`; e
o pós-processamento de dados, utilizando o módulo :mod:`framework.post_proc`.

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo UTSR. Além disso, a imagem mostra o
resultado do UTSR com pós-processamento.

.. plot:: plots/imaging/utsr_example.py
    :include-source:
    :width: 100 %
    :align: center

.. raw:: html

    <hr>

"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from framework.data_types import DataInsp, ImagingROI, ImagingResult
from framework.model_s2 import model_s2_direct, model_s2_adjoint
import framework.schmerr_model as sm


def model_s2(x, w, alpha, nt, nu, roi, dt, du, c, tau0, _model, _filt, coord_orig):
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

    # Aplica a regularização.
    if alpha != 0.0:
        # Norma L1 se W != I e L = I, ou Tikhonov se W = I e L = I.
        y = y.flatten("F") + w * x * alpha ** 2

    else:
        # Sem regularização, só solução por Mínimos Quadrados
        y = y.flatten("F")

    return y


""" Variável global criada para contagem das iterações do algoritmo de gradiente conjugado."""
num_cg_iter = 0


def utsr_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0,
                c=5900.0, _model=-1, cg_tol=1e-2, cg_max_iter=1000,
                alpha=1e-3, beta=2.23e-16, tol=1e-2,
                max_stag_count=6, neg_cut_out=True, result_norm_l2=True, debug=False):
    r"""Processa dados de A-scan utilizando o algoritmo UTSR .

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

    c : :class:`int`, :class:`float`
        Velocidade de propagação da onda no objeto sob inspeção. Por
        padrão, é None e nesse caso é obtido o valor do data_insp.

    _model : :class:`int` ou :class:`np.ndarray`
        Modelo de transdutor que será usado. Por padrão recebe -1.

    cg_tol: :class:`int` ou :class:`float`
        Tolerância de erro no resultado do gradiente conjugado. Por padrão, é
        1e-2.

    cg_max_iter: :class:`int`
        Número máximo de iterações consecutivas para o algoritmo gradiente
        conjugado. Por padrão, é 1000.

    alpha : :class:`int` ou :class:`float`
        Parâmetro usado para cálculo do threshold. Por padrão, é 1e-3.

    beta : :class:`int` ou :class:`float`
        Parâmetro usado para cálculo do threshold. Por padrão, é 2.23e-16.

    tol : :class:`int` ou :class:`float`
        Tolerância de erro no resultado do algoritmo. Por padrão, é 1e-2.

    max_stag_count : :class:`int`
        Número máximo de iterações consecutivas em que o resultado do
        algoritmo não reduz em 10% do valor da tolerância. Por padrão, é 6.

    neg_cut_out : :class:`bool`
        Elimina os valores negativos da imagem. Por padrão é `True`

    result_norm_l2 : :class:`bool`
        Normaliza a resposta pela norma :math:`\ell_2`. Por padrão, é `True`.
    
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

    # --- INÍCIO DO ALGORITMO UTSR, desenvolvido por Giovanni. ---
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

    # Faz a reconstrução do UTSR utilizando o método IRLS para resolução do problema inverso.
    # Calcula a solução *backprojection*. É utilizada como resposta inicial e como vetor *b* para calcular o CG.
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

    if (beta is None) or (beta == -1):
        beta = np.finfo(np.float64).eps

    f_temp = htg
    num_iter = 1
    stag_count = 0
    err_irls = []
    beta_iter = []
    cg_iters = []
    global num_cg_iter

    while True:
        # Calcula a matriz de ponderação (norma L1).
        if num_iter == 1:
            w = np.ones(f_temp.shape)

        else:
            if np.isfinite(beta):
                # Procura o beta adequado.
                while True:
                    w = 1. / (np.abs(f_temp) + beta)
                    l1_f_temp = np.linalg.norm(f_temp, 1)
                    l1_f_temp_appr = np.linalg.norm(np.sqrt(w) * f_temp) ** 2.0
                    l1_err = (l1_f_temp - l1_f_temp_appr) / l1_f_temp
                    if np.abs(l1_err - cg_tol) > (cg_tol / 10.0):
                        beta = beta / (l1_err / cg_tol)
                    else:
                        break

        beta_iter.append(beta)

        # Aplica o algoritmo do gradiente conjugado (CG).
        # TODO: verificar como otimizar o algoritmo aqui.
        def hh(x):
            return model_s2(x, w=w, alpha=alpha,
                            nt=data_insp.inspection_params.gate_samples,
                            nu=nu0,
                            dt=dt,
                            du=du,
                            roi=roi,
                            c=c,
                            tau0=data_insp.time_grid[idx_t0, 0],
                            _model=_model,
                            _filt=_filt,
                            coord_orig=coord_orig)

        # noinspection PyUnusedLocal
        def iter_count(x):
            global num_cg_iter
            num_cg_iter += 1

        f, flag = cg(LinearOperator((f_temp.shape[0], f_temp.shape[0]), hh),
                     htg,
                     x0=f_temp,
                     tol=cg_tol,
                     callback=iter_count,
                     maxiter=cg_max_iter)

        # Normaliza o vetor-resposta pela norma l2.
        if result_norm_l2:
            f /= np.linalg.norm(f)
            coef_ampl = 1.0
        else:
            coef_ampl = np.linalg.norm(f)

        # Aplica projeções das informações a priori da imagem.
        # Elimina os valores negativos da imagem.
        if neg_cut_out:
            f[f < 0.0] = 0.0

        # Calcula a 'diferença' entre imagens de iterações consecutivas.
        err_irls.append(np.linalg.norm(f - f_temp)/coef_ampl)
        if debug:
            print('errIRLS = %f | alpha = %f | iter = %4d | flag = %d' % (err_irls[-1], alpha, num_cg_iter, flag))

        cg_iters.append(num_cg_iter)
        num_cg_iter = 0

        # Critérios de parada do algoritmo.
        # Busca somente solução por mínimos quadrados (LS).
        if alpha == 0:
            break

        # Busca solução regularizada por norma L2 na regularização.
        if np.isinf(beta):
            break

        # Tolerância do algoritmo IRLS encontrada.
        if err_irls[-1] < tol:  # or (err_mse < tol)
            break

        # Tolerância IRLS foi maior que na iteração anterior.
        if num_iter > 1 and err_irls[-1] >= err_irls[-2]:
            stag_count = stag_count + 1
            if stag_count >= max_stag_count:
                break

        # Ajusta variáveis para a próxima iteração.
        f_temp = f
        num_iter = num_iter + 1

    # Retorna para o domínio da imagem.
    image = np.reshape(f, image_shape, order="F")

    # --- FIM DO ALGORITMO UTSR.
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


def utsr_params():
    """ Retorna os parâmetros do algoritmo UTSR IRLS.

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
            "cg_tol": 1e-2, "cg_max_iter": 1000, "alpha": 1e-3, "beta": 2.23e-16,
            "tol": 1e-2, "max_stag_count": 6, "neg_cut_out": True, "result_norm_l2": True, "debug": False}
