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
from framework.data_types import DataInsp, ImagingROI, ImagingResult, ElementGeometry
from framework.utils_gpu import cdist_gpu, directivity_weights
try:
    import pycuda.autoinit
    import pycuda.cumath as cm
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as culinalg
    import skcuda.misc as misc
    from pycuda.compiler import SourceModule as SM
    import time

    culinalg.init()
    misc.init()


    initialize = False

    def tfmcuda_kernel(data_insp, roi=ImagingROI(), output_key=None, description="", sel_shot=0, c=None,
                   scattering_angle=None, trcomb=None):
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
            g = data_insp.ascan_data[:, :, :, sel_shot]
        else:
            raise NotImplementedError("Tipo de captura inválido. Só é permitido ``FMC`` para o algoritmo TFM.")

        nb_elements = data_insp.probe_params.num_elem
        # if trcomb is None:
        #     trcomb = np.ones((nb_elements, nb_elements), dtype=bool)
        # else:
        #     try:
        #         trcomb = np.asarray(trcomb, bool)
        #     except ValueError:
        #         raise TypeError("O argumento trcomb não é compatível com formato int 2D-array")
        #     if not (trcomb.shape.__len__() == 2 and trcomb.shape[0] == trcomb.shape[1] and trcomb.shape[0] == nb_elements):
        #         strerr = "O argumento trcomb não tem o formato exigido (" + str(nb_elements) + "x" + str(nb_elements) + ")"
        #         raise TypeError(strerr)
        #
        # if scattering_angle is None:
        #     scatfilt = np.zeros((nb_elements, roi.h_len * roi.w_len), dtype=bool)
        # else:
        #     try:
        #         scattering_angle = float(scattering_angle)
        #         scatfilt = roi.get_coord() * 1e-3
        #         scatfilt = scatfilt[:, np.newaxis] - data_insp.probe_params.elem_center[np.newaxis] * 1e-3
        #         scatfilt = np.angle(scatfilt[:, :, 0] + scatfilt[:, :, 2] * 1j, True) - 90
        #         scatfilt = abs(scatfilt.T) > (scattering_angle / 2)
        #     except TypeError:
        #         scatfilt = np.asarray(scattering_angle, bool)
        #         if not (scatfilt.shape.__len__() == 2 and scatfilt.shape[0] == data_insp.probe_params.num_elem and
        #                 scatfilt.shape[1] == roi.h_len * roi.w_len):
        #             strerr = "O argumento scatfilt não tem o formato exigido (" + str(nb_elements) + "x" + str(
        #                 roi.h_len * roi.w_len) + ")"
        #             raise TypeError(strerr)
            # Prepara os valores para a chamada
        a = gpuarray.to_gpu(np.ascontiguousarray(data_insp.probe_params.elem_center*1e-3).astype(np.float32))
        b = gpuarray.to_gpu(np.ascontiguousarray(roi.get_coord()*1e-3).astype(np.float32))
        if data_insp.inspection_params.type_insp == 'immersion':
            samp_dist = data_insp.surf.cdist_medium(data_insp.probe_params.elem_center, roi.get_coord(), roi=roi,
                                                    sel_shot=sel_shot) * 1e-3
            dist_correction = 1.0 / (np.asarray([data_insp.inspection_params.coupling_cl, c]) *
                                     data_insp.inspection_params.sample_time * 1e-6)
            samp_dist = samp_dist[0] * dist_correction[0] + samp_dist[1] * dist_correction[1]
            samp_dist = gpuarray.to_gpu(np.ascontiguousarray(samp_dist.astype(np.float32)))
            cf_coef = np.float32(1.0)
        else:
            samp_dist = cdist_gpu(a, b)
            cf_coef = np.float32(data_insp.inspection_params.sample_freq*1e6 / c)
        ## Implementar o uso de filtro de combinacao para reduzir o tempo de processamento
        ## Implementar o uso de filtro de angulo (wt) como no TFM padrão

        if data_insp.inspection_params.type_insp == 'contact':
            ka = data_insp.probe_params.central_freq * 1e6 / data_insp.specimen_params.cl * \
                data_insp.probe_params.elem_dim * 1e-3
        else:
            ka = data_insp.probe_params.central_freq * 1e6 / data_insp.inspection_params.coupling_cl * \
                data_insp.probe_params.elem_dim * 1e-3

        # t = time.time()
        wt = directivity_weights(a, b, np.float32(ka))
        # print(time.time()-t)


        gate_start = np.float32(data_insp.inspection_params.gate_start*data_insp.inspection_params.sample_freq)
        fmc = data_insp.ascan_data[:, :, :, sel_shot]

        fmc_gpu = gpuarray.to_gpu(np.ascontiguousarray(fmc))
        # t = time.time()
        f = tfmcuda_kern(fmc_gpu, samp_dist, cf_coef, gate_start, wt, np.iscomplexobj(fmc)).get()
        # print(time.time()-t)
        f = f.reshape(roi.h_len, roi.w_len)

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
                result = data_insp.imaging_results.pop(output_key)
                result.roi = roi
                result.description = description
            except KeyError:
                # Objeto não encontrado no dicionário. Cria um novo.
                # Cria um objeto ImagingResult com o resultado do algoritmo e salva a imagem reconstruída.
                result = ImagingResult(roi=roi, description=description)

        # Salva o novo resultado.
        result.image = (f.reshape((roi.w_len, roi.h_len))).T
        if data_insp.inspection_params.type_insp == 'immersion':
            result.surface = data_insp.surf.get_points_in_roi(roi, sel_shot)
        # Guarda o resultado no dicionário.
        data_insp.imaging_results[output_key] = result

        # Retorna o valor da chave
        return output_key

    tfm_kernel_complex = """
    #include <pycuda-complex.hpp>
    typedef pycuda::complex<float> cmplx;
    __global__ void tfm_kern(cmplx *image, cmplx *fmc, float *samp_dist, float *wt, float *coefs, int *S)
    {
        const int tx = threadIdx.x;
        const int bx = blockIdx.x;
        const int BSZ = blockDim.x;
        const int ind_px = bx*BSZ+tx;
        const float cf_coef = coefs[0];
        const float gate_start = coefs[1];
        const int nb_elem = S[0];
        const int num_px = S[1];
        const int max_sample = S[2];
        int sample;
        cmplx acc = 0;
        if (ind_px<num_px){
        #pragma unroll
            for (int t=0;t<nb_elem;t++){
            #pragma unroll
                for (int r=t;r<nb_elem;r++){
                    sample = (int)((samp_dist[t*num_px+ind_px] + samp_dist[r*num_px+ind_px])*cf_coef - gate_start);
                    if (sample<max_sample && sample>=0){
                        acc += float((t!=r)+1)*fmc[sample*nb_elem*nb_elem + t*nb_elem + r]*wt[ind_px*nb_elem+t]*wt[ind_px*nb_elem+r];
                    }
                }
            }
        }
        image[ind_px] = acc;
        __syncthreads();
    }
    """

    tfm_kernel_float = """
    __global__ void tfm_kern(float *image, float *fmc, float *samp_dist, float *wt, float *coefs, int *S)
    {
        const int tx = threadIdx.x;
        const int bx = blockIdx.x;
        const int BSZ = blockDim.x;
        const int ind_px = bx*BSZ+tx;
        const float cf_coef = coefs[0];
        const float gate_start = coefs[1];
        const int nb_elem = S[0];
        const int num_px = S[1];
        const int max_sample = S[2];
        int sample;
        float acc = 0;
        if (ind_px<num_px){
        #pragma unroll
            for (int t=0;t<nb_elem;t++){
            #pragma unroll
                for (int r=t;r<nb_elem;r++){
                    sample = (int)((samp_dist[t*num_px+ind_px] + samp_dist[r*num_px+ind_px])*cf_coef - gate_start);
                    if (sample<max_sample && sample>=0){
                        acc += float((t!=r)+1)*fmc[sample*nb_elem*nb_elem + t*nb_elem + r]*wt[ind_px*nb_elem+t]*wt[ind_px*nb_elem+r];
                    }
                }
            }
        }
        image[ind_px] = acc;
        __syncthreads();
    }
    """

    def tfmcuda_kern(fmc_gpu, samp_dist, cf_coef, gate_start, wt, iscomplex):
        f = gpuarray.empty((samp_dist.shape[1]), dtype=fmc_gpu.dtype)
        M = 512
        N = samp_dist.shape[1]//M + 1
        if iscomplex:
            kernel = SM(tfm_kernel_complex).get_function("tfm_kern")
        else:
            kernel = SM(tfm_kernel_float).get_function("tfm_kern")
        floats = np.asarray([cf_coef, gate_start])
        S = np.int32([fmc_gpu.shape[1], samp_dist.shape[1], fmc_gpu.shape[0]])
        kernel(f, fmc_gpu.ravel(), samp_dist.ravel(), wt.ravel(), drv.In(floats), drv.In(S),
               block=(M, 1, 1), grid=(N, 1, 1))
        return f

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
except ImportError:
    print("TFMCUDA Import Error. Missing PyCUDA/SKCUDA packages.")