# -*- coding: utf-8 -*-
"""
Módulo ``wavenumber``
=====================

O *Wavenumber* é um algoritmo de reconstrução de imagens para Ensaios Não
Destrutivos (ENDs), implementado no domínio da frequência, quando o sistema
de inspeção utiliza transdutores ultrassônicos *phased array* e o sistema de
captura é FMC (*Full Matrix Capture* - Matriz Completa de Captura). Surgiu a
partir de técnicas aplicadas a radares e sonares. É comparado ao algoritmo TFM,
no domínio do tempo.

O *Wavenumber* fornece uma solução matematicamente rigorosa para o problema
inverso para o modelo de propagação de onda, enquanto o TFM emprega a forma
heurística de forma de feixe de soma e atraso. Consequentemente, o algoritmo
*Wavenumber* tem uma função aprimorada de difusão de ponto e fornece uma
melhor imagem. Outra vantagem é o desempenho computacional superior.
Entretanto, o TFM é mais flexivel, pois enquanto o *Wavenumber* requer uma
matriz linear regularmente amostrada, o TFM pode manipular geometrias de
imagens arbitrárias.

Exemplo
-------
O *script* abaixo mostra o uso do algoritmo *Wavenumber* para a reconstrução
de uma imagem a partir de dados sintéticos, oriundos do simulador CIVA.
(Assume-se que os dados estão na mesma pasta em que o *script* é executado)

O *script* mostra o procedimento para realizar a leitura de um arquivo
de simulação, utilizando o módulo :mod:`framework.file_civa`; o processamento
de dados, utilizando os módulos :mod:`imaging.bscan` e
:mod:`imaging.wavenumber`; e o pós-processamento de dados, utilizando o módulo
:mod:`framework.post_proc`. 

O resultado do *script* é uma imagem, comparando a imagem reconstruída com o
algoritmo B-scan e com o algoritmo *Wavenumber*. Além disso, a imagem mostra o
resultado do *Wavenumber* com pós-processamento.

.. plot:: plots/imaging/wavenumber_example.py
    :include-source:
    :scale: 100

.. raw:: html

    <hr>

"""

import numpy as np
import time

from framework.data_types import DataInsp, ImagingROI, ImagingResult
from framework.linterp_oper import linterp
from framework.fk_mig import nextpow2, f_k_fmc


def wavenumber_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None,
                      pad_t='auto', pad_x=True, print_time=False, print_fft_size=False):
    """Processa dados de A-scan utilizando o algoritmo wavenumber.

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
        Velocidade de propagação da onda no objeto sob inspeção. Por padrão,
        é ``None`` e nesse caso é obtido o valor do data_insp.

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
    
    NotImplementedError
        Se o tipo de captura (:attr:`.data_types.InspectionParams.type_capt`)
        não for ``FMC``.
        
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
    
    # --- Início do algoritmo Wavenumber, desenvolvido por Giovanni ---
    # Calcula os instantes de tempo que definem a região da ROI nos sinais A-scan.
    ermv = c / 2.0
    x0 = (roi.w_points[0] - data_insp.probe_params.elem_center[0, 0]) * 1e-3
    z0 = (roi.h_points[0] - data_insp.probe_params.elem_center[0, 2]) * 1e-3
    ze = (roi.h_points[-1] - data_insp.probe_params.elem_center[0, 2]) * 1e-3
    t0 = z0 / ermv
    te = ze / ermv
    idx_t0 = np.searchsorted(data_insp.time_grid[:, 0], t0 / 1e-6)
    idx_te = np.searchsorted(data_insp.time_grid[:, 0], te / 1e-6)

    # Extração dos sinais ``A-scan`` necessários para a execução do algoritmo.
    if data_insp.inspection_params.type_capt == "FMC":
        e_t_u_v = data_insp.ascan_data[idx_t0: idx_te, :, :, sel_shot]
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo WAVENUMBER.")
        
    # Zero-padding para interpolação no domínio do tempo.
    if type(pad_t) is int or type(pad_t) is float:
        nt = int(pad_t)
    else:
        if pad_t == 'none':
            nt = e_t_u_v.shape[0]
        else:
            gs = data_insp.inspection_params.gate_samples
            nt = int(2.0 ** (nextpow2(gs) + 1))
    if pad_x is True:
        nu = 2 * e_t_u_v.shape[1]
        nv = 2 * e_t_u_v.shape[2]
    else:
        nu = e_t_u_v.shape[1]
        nv = e_t_u_v.shape[2]

    # Calcula a FFT 3D dos sinais A-scan.
    _ti = time.time()
    e_w_ku_kv = np.fft.fftshift(np.fft.fftn(e_t_u_v, s=(nt, nu, nv)))
    t_fft = time.time() - _ti
    
    # Calcula os grids necessários para a execução do algoritmo.
    # Grids dos dados medidos.
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=(data_insp.inspection_params.sample_time * 1e-6)))
    ku = np.fft.fftshift(np.fft.fftfreq(nu, d=(data_insp.probe_params.pitch * 1e-3)))
    kv = np.fft.fftshift(np.fft.fftfreq(nv, d=(data_insp.probe_params.pitch * 1e-3)))

    # Ajusta o espectro em relação ao tempo referente a posição ``z`` inicial da ROI.
    if t0 != 0.0:
        for idx_kv in range(nv):
            e_w_ku_kv[:, :, idx_kv] = e_w_ku_kv[:, :, idx_kv] *\
                                      np.exp(-2j * np.pi * f * data_insp.time_grid[idx_t0] * 1e-6)[:, np.newaxis]

    # Grids da imagem reconstruída.
    kx = np.fft.fftshift(np.fft.fftfreq(roi.w_len, d=(roi.w_step * 1e-3)))
    kz = np.fft.fftshift(np.fft.fftfreq(roi.h_len, d=(roi.h_step * 1e-3)))
    #_, kz_kv = np.meshgrid(kv, kz)
    kx_kz, _ = np.meshgrid(kx, kz)

    t_mig = 0
    t_interp = 0    
    f_kx_kz = np.zeros((roi.h_len, roi.w_len), dtype=e_w_ku_kv.dtype)
    # Aplica a Transformada de Stolt em cada *slice*.
    for idx_ku in range(nu):
        kui = kv[idx_ku]
        # Pega o *slice* referente ao ``ku``.
        e_w_kv = e_w_ku_kv[:, idx_ku, :]

        # Grids para a Transformada de Stolt.
        _ti = time.time()
        f_kz = f_k_fmc(c, kz.reshape(-1, 1), kv + kui, kui)
        t_mig += (time.time() - _ti)

        # Aplica a Transformada de Stolt (migração f-k), via interpolação linear 2D.
        _ti = time.time()
        f_kx_kz = f_kx_kz - linterp(kv + ku[idx_ku], kx_kz.T, linterp(f, f_kz, e_w_kv).T).T * (4 * np.pi) ** 2
        t_interp += (time.time() - _ti)
        
    # Ajusta o espectro da imagem em relação ao deslocamento da ROI.
    if z0 != 0.0:
        f_kx_kz = f_kx_kz * np.exp(2j * np.pi * kz * z0)[:, np.newaxis]

    if x0 != 0.0:
        f_kx_kz = f_kx_kz * np.exp(2j * np.pi * kx * x0)[np.newaxis, :]

    # Retorna para o domínio da imagem.
    image = np.real(np.fft.ifft2(np.fft.ifftshift(f_kx_kz)))

    if print_time is True or print_fft_size is True:
        print('\nAlgorithm Wavenumber')
        print('-------------------------------')
    if print_fft_size is True:
        print('FFT size: ', (nt, nu, nv))
        print('-------------------------------')
    if print_time is True:
        print('|{:^13}|{:<15}|'.format('Step', 'Exec. time (s)'))
        print('|{:<13}|{:^15.4f}|'.format('FFT', t_fft))
        print('|{:<13}|{:^15.4f}|'.format('Migration', t_mig))
        print('|{:<13}|{:^15.4f}|'.format('Interpolation', t_interp))
        print('-------------------------------')
    
    # --- Fim do algoritmo wavenumber ---
    
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


def wavenumber_params():
    """Retorna os parâmetros do algoritmo wavenumber.

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
