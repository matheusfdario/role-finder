# -*- coding: utf-8 -*-
"""
Módulo ``file_omniscan``
========================

Neste módulo é feita a leitura arquivos de inspeção
do equipamento comercial ``OmniScan MX2`` e disponibiliza-los no formato
do ``framework`` do projeto AUSPEX. Para isso, utiliza-se a biblioteca
**NDT Data Access Library**, disponibilizada pela ``Olympus``, que
é um ``kit`` de desenvolvimento de ``software`` que permite ler dados
adquiridos usando instrumentos da ``Olympus`` e um aplicativo
personalizado.

A biblioteca **NDT Data Access Library**, disponibilizada pela ``Olympus``,
é um ``kit`` de desenvolvimento de ``software``. Neste caso, foi usada para ler os arquivos de inspeção
do equipamento comercial ``OmniScan MX2`` e disponibiliza-los no formato
do ``framework`` do projeto AUSPEX.

Tipos de Inspeção
-----------------

O processamento dos dados é dependente do tipo de inspeção escolhido.
O OmniScan pode realizar inspeções usando transdutores do tipo ``mono``
ou ``linear``. Para transdutores ``mono``, o tipo de inspeção é ``sweep``.
Já para transdutores do tipo ``linear``, os tipos de inspeção podem ser
``PhasedArrayMerged`` ou ``PhasedArray``, cuja diferença está na possibilidade
de usar uma fusão linear entre os elementos.

PhasedArray
-----------

Neste tipo de inspeção, o número de ``beams`` é o mesmo que o número de elementos
do transdutor linear, ou seja, os A-Scan dependem do feixe selecionado, e possuí apenas
um ``Index``, que controla o deslocamento do elemento ativo do transdutor.
Para a obtenção das coordenadas de cada elemento no transdutor linear, em inspeções por
varredura, foi usada a leitura ``ReferenceIndexOffset`` que fornece a coordenada *x* a
esquerda do elemento, conforme ilustrado na figura :numref:`fig_phased_array`.

.. figure:: figures/phasedarray.png
    :name: fig_phased_array
    :width: 90 %
    :align: center

    Arranjo da organização dos dados para inspeções do tipo ``PhasedArray``.

PhasedArrayMerged
-----------------

Neste tipo de inspeção, é possível realizar a fusão linear dos elementos. O arquivo possui
apenas um ``beam`` e os A-Scan são selecionados a partir do ``Index``, que têm o mesmo número
do de elementos ativos considerados.
As coordenadas de cada elemento no transdutor linear, em inspeções por varredura, foram calculadas
considerando o número de elementos ativos  e o valor do ``pitch``.
O parâmeto ``.DataInsp.ProbeParams.pitch`` é obtido pela leitura do ``IndexResolution``.
Na figura :numref:`fig_phased_array_merged`, está exemplificado o sistema de coordenadas de um transdutor com
elementos fundidos 2 a 2.

.. figure:: figures/phasedarraymerged.png
    :name: fig_phased_array_merged
    :width: 90 %
    :align: center

    Arranjo da organização dos dados para inspeções do tipo ``PhasedArrayMerged``.

Observação
----------

O parâmetro ``DataInsp.SpecimenParams.step_points`` foi corrigido considerando que as
coordenadas dos elemento ativos estão localizadas no centro e que a origem é relativa
ao elemento central do transdutor.
O parâmetro ``DataInsp.ProbeParams.dim`` é considerado igual ao ``pitch``, já que o parâmetro
``DataInsp.ProbeParams.inter_elem`` é nulo, devido a característica desse tipo de transdutor.

.. raw:: html

    <hr>

"""
import numpy as np
from framework.data_types import DataInsp, InspectionParams, SpecimenParams, ProbeParams
from sys import platform
if platform == "win32":
    import win32com.client


def read(filename, sel_shots, freq, bw, pulse_type):
    # As informações referentes a frequência do
    # trandutor, banda, pitch, espaçamento entre
    # os trandutores, dimensão e tipo de pulso não
    # estão disponíveis, portanto devem ser informadas

    """
    Abre e analisa um arquivo .opd, retornando os dados da inspeção.

    Os dados são retornados como um objeto da classe :class:`.DataInsp`,
    contendo os parâmetros de inspeção, do transdutor, da peça e os
    resultados do ensaio utilizando o equipamento OminiScan.

    Parameters
    ==========
        filename : str
            Informa o caminho do arquivo .opd a ser lido.

        sel_shots : NoneType, int, list ou range
            *Shots* para a leitura. Se for ``None``, lê todos os *shots*
            disponíveis. Se ``int``, lê o índice especificado. Se ``list``
            ou ``range``, lê os índices especificados. Por padrão, é
            ``None``.

        freq: float
            Frequência nominal do transdutor, em MHz. Por padrão, é
            5.0 MHz.

        bw: float
            Largura de banda do transdutor, expressa em percentual
            da frequência central. Por padrão é 0,5, que corresponde a uma banda de 50%.

        pulse_type: str
            Tipo do pulso de excitação do transdutor. Por padrão, é
            ``gaussian``.

    Returns
    ========
    :class:`.DataInsp`
        Dados do arquivo lido, contendo parâmetros de inspeção, do transdutor,
        da peça e os dados dos ensaios.

    Raises
    ======

    FileNotFoundError
        Gera exceção de ``FileNotFoundError`` se o arquivo não existir.

    SelShotError
        Gera exceção de ``TypeError`` se o parâmetro ``sel_shots`` não é do
        tipo ``NoneType``, ``int``, ``list`` ou ``range``.
        Se ``sel_shots`` for do tipo ``list`` ou ``range``, gera exceção
        ``NotImplemented``.

    TypeInspectionError
        Gera exceção de ``TypeError`` se o *tipo de inspeção* for diferente
        de ``PhasedArrayMerged`` ou ``PhasedArray``.



      """
    if platform != "win32":
        raise OSError('Only implemented for Windows')
    # Cria um objeto COM para leitura da DLL
    rdtiff_access = win32com.client.Dispatch('RDTiffDataAccess.RDTiffData')

    # Usa a interface IRDTiff para obter informações dos arquivos de dados
    # usando os métodos do DataFile

    rdtiff1 = rdtiff_access.RDTiffDataFile

    # Abre um arquivo de dados do OminiScan no formato .opd
    rdtiff1.OpenFile(filename)

    # Inicialização das variáveis usadas para acessar os parâmetros e
    # resultados dos ensaios

    # Leitura dos parâmetros de inspeção e os parâmetros da peça inspecionada
    # Só trabalhamos com 1 canal. Quando existem mais canais ????
    assert rdtiff1.channels.Count == 1

    # sample_freq depende dos canais usados na inspeção, recebe o valor da
    # frequência do digitalizador em MHz dividida pela taxa de compressão
    sample_freq = float(
        rdtiff1.channels.Item(1).DigitizingFrequency / (rdtiff1.channels.Item(1).Compression * 1e6))

    # cl depende dos canais usados na inspeção, recebe o valor da
    # velocidade de propagação das ondas longitudinais no material
    cl = rdtiff1.channels.Item(1).PartParameters.MaterialSoundVelocity

    # water_cl depende dos canais usados na inspeção, recebe o valor da
    # velocidade de propagação das ondas longitudinais na agua
    water_cl = rdtiff1.channels.Item(1).PartParameters.InterfaceSoundVelocity

    # Processamento dependente do tipo de inspeção no Omniscan.
    if rdtiff1.channels.Item(1).Type == 1:
        # Inspeções do tipo PhasedArrayMerged salvam os A-Scan em um beam e n index
        # É para só ter um beam
        assert rdtiff1.channels.Item(1).Beams.Count == 1
        beam_object = rdtiff1.Channels.Item(1).Beams.Item(1)

        # Não é necessário usar um Loop para os gates, pois apenas o Gate 1 (Gate Main (A-Scan) acessa o grupo de
        # dados dos A-Scan
        gate_object = beam_object.Gates.Item(1)
        assert gate_object.Type == 3  # Verifica se é o 'Gate Main (A-Scan)'

        # gate_start é a posição inicial da porta selecionada (us)
        gate_start = gate_object.Start * 1e06

        # gate_end é a posição final da porta selecionada (us)
        # é obtido pela soma da posição inicial com o tamanho
        # da porta
        gate_end = (gate_object.Width + gate_object.Start) * 1e06

        # impact_angle é o ângulo de incidência ou de
        # inclinação do feixe de ultrassom.
        # Valor complementar ao ângulo ``skew``
        impact_angle = (90. - beam_object.Skew)

        #  Acessando o grupo de dados dos A-Scan - é sempre o primeiro Data Group.
        datagroup_object = gate_object.DataGroups.Item(1)
        assert datagroup_object.Type == 0

        # gate_samples recebe a quantidade de
        # amostras que compõe a aquisição dos A-Scan
        gate_samples = datagroup_object.SampleQuantity

        # num_elem é o número de elementos do transdutor linear, mas caso o tipo de
        # inspeção seja ``PhasedArrayMerged`` é igual ao número de indices da aquisição
        num_elem = datagroup_object.IndexQuantity

        # Calcula a coordenada de cada elemento do array em relação a seu centro geométrico.
        # step_points é a matriz com as coordenadas do
        # transdutor durante a inspeção. Cada linha
        # dessa matriz corresponde a posição do
        # transdutor e equivale a um elemento na
        # dimensão ``passo`` do DataInsp.ascan_data.
        # Devido a retirada das amostras correspondentes
        # a sapata, a origem z deve ser deslocada em
        # proporção ao tamanho da sapata, mas em
        # referência a velocidade na peça
        step_points = np.zeros((num_elem, 3))
        pitch = round(datagroup_object.IndexResolution * 1e3, 2)
        dim = pitch  # Isso é característica desse transdutor.
        inter_elem = 0  # Isso é uma característica desse transdutor.

        for index in range(num_elem):
            step_points[index, :] = [((index - (num_elem / 2 - 1)) * pitch) - pitch / 2.0, 0.0, 0.0]

        # Ajusta os tamanhos do array de dados
        dim1 = datagroup_object.SampleQuantity
        dim2 = 1
        dim3 = 1

        # Leitura de todos os *shots*
        if sel_shots is None:
            dim4 = datagroup_object.IndexQuantity * datagroup_object.ScanQuantity
            ascan_data = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)

            # Percorre todos os shots na leitura de dados dos A-Scan
            for scan in range(datagroup_object.ScanQuantity):
                for index in range(datagroup_object.IndexQuantity):
                    ascan_data[:, 0, 0, num_elem * scan + index] = datagroup_object.DataAccess.ReadAscan(scan, index)

        elif type(sel_shots) is int:
            dim4 = datagroup_object.IndexQuantity
            ascan_data = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)

            for index in range(datagroup_object.IndexQuantity):
                ascan_data[:, 0, 0, index] = datagroup_object.DataAccess.ReadAscan(sel_shots, index)

        elif type(sel_shots) is list:
            # É uma lista, não faz nada por enquanto.
            # TODO: Implementar leitura para uma lista de shots.
            raise NotImplemented("Implementar leitura para uma lista de shots.")

        elif type(sel_shots) is range:
            # É um ``range``, não faz nada por enquanto.
            # TODO: Implementar leitura para um range().
            raise NotImplemented("'Implementar leitura para um range().")

        else:
            # Tipo inválido.
            raise TypeError("``sel_shots`` deve ser um inteiro, uma lista ou um range().")

    elif rdtiff1.channels.Item(1).Type == 2:
        # Inspeções do tipo PhasedArray salvam os A-Scan em um n beams e 1 index
        # Mas existem algumas informações que são iguais para todos os beams.
        # Pegar elas do primeiro beam
        beam_object = rdtiff1.Channels.Item(1).Beams.Item(1)

        # Não é necessário usar um Loop para os gates, pois apenas o Gate 1 (Gate Main (A-Scan) acessa o grupo de
        # dados dos A-Scan
        gate_object = beam_object.Gates.Item(1)
        assert gate_object.Type == 3  # Verifica se é o 'Gate Main (A-Scan)'

        # gate_start é a posição inicial da porta selecionada (us)
        gate_start = gate_object.Start * 1e06

        # gate_end é a posição final da porta selecionada (us)
        # é obtido pela soma da posição inicial com o tamanho
        # da porta
        gate_end = (gate_object.Width + gate_object.Start) * 1e06

        # impact_angle é o ângulo de incidência ou de
        # inclinação do feixe de ultrassom.
        # Valor complementar ao ângulo ``skew``
        impact_angle = (90. - beam_object.Skew)

        #  Acessando o grupo de dados dos A-Scan - é sempre o primeiro Data Group.
        datagroup_object = gate_object.DataGroups.Item(1)
        assert datagroup_object.Type == 0

        # gate_samples recebe a quantidade de
        # amostras que compõe a aquisição dos A-Scan
        gate_samples = datagroup_object.SampleQuantity

        # num_elem é o número de elementos do transdutor linear, mas caso o tipo de
        # inspeção seja ``PhasedArray`` é igual ao número de beams
        num_elem = rdtiff1.channels.Item(1).Beams.Count

        dim1 = datagroup_object.SampleQuantity
        dim2 = 1
        dim3 = 1

        # Leitura de todos os *shots*
        if sel_shots is None:
            dim4 = num_elem * datagroup_object.ScanQuantity
            ascan_data = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)

        elif type(sel_shots) is int:
            dim4 = num_elem
            ascan_data = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)

        elif type(sel_shots) is list:
            # É uma lista, não faz nada por enquanto.
            # TODO: Implementar leitura para uma lista de shots.
            raise NotImplemented("Implementar leitura para uma lista de shots.")

        elif type(sel_shots) is range:
            # É um ``range``, não faz nada por enquanto.
            # TODO: Implementar leitura para um range().
            raise NotImplemented("'Implementar leitura para um range().")

        else:
            # Tipo inválido.
            raise TypeError("``sel_shots`` deve ser um inteiro, uma lista ou um range().")

        # step_points é a matriz com as coordenadas do
        # transdutor durante a inspeção. Cada linha
        # dessa matriz corresponde a posição do
        # transdutor e equivale a um elemento na
        # dimensão ``passo`` do DataInsp.ascan_data.
        # Devido a retirada das amostras correspondentes
        # a sapata, a origem z deve ser deslocada em
        # proporção ao tamanho da sapata, mas em
        # referência a velocidade na peça
        step_points = np.zeros((num_elem, 3))

        # Loop para buscar as informações dos beams
        for beam in range(num_elem):
            beam_object = rdtiff1.Channels.Item(1).Beams.Item(beam + 1)

            # Obtém a coordenada central do elemento transdutor
            step_points[beam, :] = [beam_object.ReferenceIndexOffset * 1e3, 0.0, 0.0]

            # Não é necessário usar um Loop para os gates, pois apenas o Gate 1 (Gate Main (A-Scan) acessa o grupo
            # de dados dos A-Scan
            gate_object = beam_object.Gates.Item(1)

            #  Acessando o grupo de dados dos A-Scan - é sempre o primeiro Data Group.
            datagroup_object = gate_object.DataGroups.Item(1)
            assert datagroup_object.Type == 0

            # Leitura de todos os *shots*
            if sel_shots is None:
                # scan é usado para percorrer os shots na leitura de dados dos A-Scan
                for scan in range(datagroup_object.ScanQuantity):
                    ascan_data[:, 0, 0, num_elem * scan + beam] = datagroup_object.DataAccess.ReadAscan(scan, 0)

            elif type(sel_shots) is int:
                ascan_data[:, 0, 0, beam] = datagroup_object.DataAccess.ReadAscan(sel_shots, 0)

            elif type(sel_shots) is list:
                # É uma lista, não faz nada por enquanto.
                # TODO: Implementar leitura para uma lista de shots.
                raise NotImplemented("Implementar leitura para uma lista de shots.")

            elif type(sel_shots) is range:
                # É um ``range``, não faz nada por enquanto.
                # TODO: Implementar leitura para um range().
                raise NotImplemented("'Implementar leitura para um range().")

            else:
                # Tipo inválido.
                raise TypeError("``sel_shots`` deve ser um inteiro, uma lista ou um range().")

        # Ajusta as coordenadas do array step_points.
        step_points = step_points.round(2)
        pitch = np.max(np.diff(step_points, axis=0)[:, 0]).round(2)
        dim = pitch  # Isso é característica desse transdutor.
        inter_elem = 0  # Isso é uma característica desse transdutor.
        transd_center = np.array([step_points[num_elem // 2 - 1, 0] - pitch / 2.0, 0.0, 0.0])[np.newaxis, :].round(2)
        step_points = step_points - transd_center

    else:
        # Tipo inválido.
        raise TypeError("O tipo de inspeção deve ser ``PhasedArrayMerged`` ou ``PhasedArray``.")

    # ========== Cria uma instância do objeto ``InspectionParams`` ==========

    # type_insp recebe o valor ``contact``

    # type_capt tem como valores possíveis ``sweep`` ou ``FMC`` (CIVA),
    # mas os tipos de inspeção possíveis no omniscan são ``PulseEcho``,
    # ``Tofd``, ``PitchCatch`` ou ``ThroughTransmission``, por isso será
    # definido como inspeção por contato

    # point_origin é a posição no espaço indicando
    # a origem do sistemas de coordenadas para a
    # inspeção.Todas as outras posições de pontos
    # são relativas a este ponto no espaço.
    # Vetor linha (x, y, z) recebe o valor padrão
    # (0, 0, 0)
    inspection_params = InspectionParams(type_insp="contact",
                                         type_capt="sweep",
                                         sample_freq=sample_freq,
                                         gate_end=gate_end,
                                         gate_start=gate_start,
                                         gate_samples=gate_samples)
    inspection_params.coupling_cl = water_cl
    inspection_params.water_path = 0
    inspection_params.impact_angle = impact_angle
    inspection_params.step_points = step_points
    inspection_params.coupling_cl = water_cl

    # ========== Cria uma instância do objeto ``SpecimenParams`` ==========

    # a velocidade das ondas transversais ``cs``
    # e a rugosidade ``roughness`` não estão entre
    # os parâmetros que podem ser acessados pela
    # DataAccess
    specimen_params = SpecimenParams(cl=cl)

    # ========== Cria uma instância do objeto ``ProbeParams`` ==========

    # Tipo de canal --> ajustar --> valor padrão linear

    probe_params = ProbeParams(tp="linear",
                               num_elem=num_elem,
                               pitch=pitch,
                               dim=dim,
                               inter_elem=inter_elem,
                               freq=freq,
                               bw=bw,
                               pulse_type=pulse_type)

    # Ajusta os centros dos elementos aos step_points.
    probe_params.elem_center = step_points

    # ========== Cria uma instância do objeto ``DataInsp`` ==========
    dados = DataInsp(inspection_params, specimen_params, probe_params)
    dados.ascan_data = ascan_data

    # Fecha um arquivo de dados do OminiScan no formato .opd
    rdtiff1.CloseFile()

    return dados
