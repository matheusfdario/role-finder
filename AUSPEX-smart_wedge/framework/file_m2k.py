# -*- coding: utf-8 -*-
"""
Módulo ``file_m2k``
===================

O módulo :mod:`file_m2k` é responsável pela leitura de arquivos que possuem a
extensão ``.m2k``. Atualmente, o módulo é capaz de realizar a leitura dos
arquivos ``.m2k`` gerados por inspeções com os equipamentos Multix++ e Panther.

Os dados de A-scan são obtidos a partir da decodificação dos arquivos binários
que são gerados pela inspeção com os equipamentos. Os parâmetros de inspeção
são obtidos a partir do processamento dos arquivos ``.xml`` que acompanham os
arquivos binários, possuindo informações como frequência de amostragem,
informações de *gate*, tipo de captura, entre outros. 

O módulo também suporta a leitura de arquivos produzidos pelo equipamento
Panther que possuem múltiplas aquisições e tipos de captura. Se salvas, as
imagens produzidas pelo equipamento também se encontram nos arquivos binários
e são automaticamente detectadas e disponibilizadas no atributo
:attr:`.data_types.DataInsp.imaging_results`.

.. raw:: html

    <hr>

"""

import numpy as np
import struct
from xml.etree.ElementTree import parse, ParseError
from framework.data_types import DataInsp, InspectionParams, SpecimenParams, ProbeParams, ImagingROI, ImagingResult


class DataDescSave:
    """Classe que contém as informações referentes a posição dos dados armazenados no arquivo binário ``acq_data.bin``.

    Attributes
    ----------
    id : :class:`numpy.uint32`
        Identificador.

    file_pointer : :class:`list`
        Lista de ponteiros no arquivo binário de dados.

    index : :class:`numpy.uint64`
        Indexador (não sei ainda a função desse campo).

    carto : :class:`numpy.array`
        Informações da cartografia do movimento capturado pelos encoders (mecânicos ou tempo).

    pad : :class:`numpy.uint32`
        Pad (esses bytes sempre guardam o número 4).

    bytes_per_channel : :class:`list`
        Quantidade de bytes em cada canal armazanado.

    type50 : :class:`numpy.uint32`
        Informação "tipo 50". Quantidade de bytes em uma imagem TFM.

    type51 : :class:`numpy.uint32`
        Informação "tipo 51". Metade da quantidade do *offset* em uma imagem TFM.

    type52 : :class:`numpy.uint32`
        Informação "tipo 52". Metade da quantidade do *offset* em uma imagem TFM.

    """

    def __init__(self):
        # Atribuição dos atributos da instância.
        # Identificador.
        self.id = np.uint32(0)

        # Lista de ponteiros no arquivo binário de dados.
        self.file_pointer = list()

        # Indexador (não sei ainda a função desse campo).
        self.index = np.uint64(0)

        # Informação da cartografia.
        self.carto = np.float32(0.0)

        # Pad (esses bytes sempre guardam o número 4).
        self.pad = np.uint32(0)

        # Quantidade de bytes em cada canal armazenado.
        self.bytes_per_channel = list()

        # Informação "tipo 50". Quantidade de bytes em uma imagem TFM.
        self.type50 = np.uint32(0)

        # Informação "tipo 51". Metade da quantidade do *offset* em uma imagem TFM.
        self.type51 = np.uint32(0)

        # Informação "tipo 52". Metade da quantidade do *offset* em uma imagem TFM.
        self.type52 = np.uint32(0)

        # Informação "tipo 5". Quantidade de bytes em um canal com a soma dos Ascans recebidos.
        self.type05 = np.uint32(0)

        # Informação "tipo 21". Quantidade de bytes adicionados ao final do shot com alguma informação relacionada
        # com o canal da soma dos Ascans. Não sei ainda o que significa, mas pode ser alguma amplitude.
        self.type21 = np.uint32(0)

        # Informação dos encoders. Quantidade de bytes com as informações registradas dos encoders.
        # É um array de float32.
        self.type03 = np.uint32(0)

    def total_bytes(self):
        """Método que retorna o total de bytes armazenados por todos os canais.

        Returns
        -------
        :class:`int`
            Soma de todos os bytes contidos na lista ``bytes_per_channel``.

        """
        total_channels = np.sum(np.asarray(self.bytes_per_channel)) if len(self.bytes_per_channel) > 0 else 0
        return total_channels + self.type05 + self.type21 + self.type03

    def has_recep_ascan(self):
        """Método que retorna se foram armazenados dados o total de bytes armazenados sinais *A-scan* recebidos.

        Returns
        -------
        :class:`bool`
            Verdadeiro se houve armazenamento dos sinais *A-scan* de recepção.

        """
        return len(self.bytes_per_channel) > 0

    def has_sum_ascan(self):
        """Método que retorna se foi armazenado o sinal *A-scan* da soma de todos os *A-scans* recebidos.

        Returns
        -------
        :class:`bool`
            Verdadeiro se houve armazenamento do sinal *A-scan* de soma.

        """
        return (self.type21 + self.type05) > 0

    def has_tfm(self):
        """Método que retorna se foi armazenada uma imagem TFM pelo Acquire.

        Returns
        -------
        :class:`bool`
            Verdadeiro se houve armazenamento de image TFM pelo Acquire.

        """
        return (self.type50 + self.type51 + self.type52) > 0

    def has_encoders_info(self):
        """Método que retorna se foram armazenadas informações dos encoders.

        Returns
        -------
        :class:`bool`
            Verdadeiro se houve armazenamento de informações dos encoder.

        """
        return self.type03 > 0


def read(filename, freq_transd, bw_transd, tp_transd, sel_shots=None, read_ascan=True, type_insp="contact",
         water_path=0.0):
    """Abre um arquivo .m2k e preenche um objeto :class:`.DataInsp`.

    Considera-se que as amplitudes dos dados de A-scan são de 2 bytes.

    Parameters
    ----------
    filename : :class:`str`
        Caminho do arquivo .m2k.

    sel_shots : :class:`NoneType`, :class:`int`, :class:`list` ou :class:`range`
        *Shots* para a leitura. Se for ``None``, lê todos os *shots*
        disponíveis. Se ``int``, lê o índice especificado. Se ``list``
        ou ``range``, lê os índices especificados. Por padrão, é
        ``None``.

    read_ascan : :class:`bool`
        *Flag* que indica a leitura dos sinais *A-scan*. É ``True`` por padrão.

    type_insp : :class:`str`
        Tipo de inspeção. Pode ser ``immersion`` ou ``contact``. É
        ``contact`` por padrão.

    water_path : :class:`float`
        Se a inspeção é do tipo ``immersion``, ``water_path`` define o
        tamanho da coluna d'água que separa o transdutor da peça, em mm.
        Por padrão é 0 mm.

    freq_transd : :class:`float`
        Frequência nominal do transdutor, em MHz. Por padrão, é
        5.0 MHz

    bw_transd : :class:`float`
        Largura de banda do transdutor, em porcentagem da frequência
        central. Por padrão, é 0.5%.

    tp_transd : :class:`str`
        Tipo do pulso de excitação do transdutor. Por padrão, é
        ``gaussian``.

    Returns
    -------
    :class:`.DataInsp`
        Dados do ensaio realizando, podendo conter parâmetros de inspeção, do
        transdutor e da peça, além dos dados de *A-scan*.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``sel_shots`` não é do
        tipo ``NoneType``, ``int``, ``list`` ou ``range``.

    IndexError
        Gera exceção de ``IndexError`` se o *shot* especificado não existe.

    """

    # Abre o arquivo XML de configuração e pega as entradas que contém informações relevantes do processo de inspeção.
    tree = parse(filename + "/M2kConfig.xml")
    control_configuration = tree.getroot().find("ConfigurationDeControle")

    # Verifica se foi passada uma lista de *shots* para leitura.
    if sel_shots is None:
        # É para ler todos os *shots*.
        # None indica que é para ler todos os shots.
        pass
    else:
        if type(sel_shots) is int:
            # É somente um shot, cria uma lista com esse índice.
            sel_shots = [sel_shots]
        elif type(sel_shots) is list:
            # É uma lista, não faz nada.
            pass
        elif type(sel_shots) is range:
            # É um ``range``, não faz nada.
            pass
        else:
            # Tipo inválido.
            raise TypeError("``sel_shots`` deve ser um inteiro, uma lista ou um range().")

    # Elemento contendo informações sobre o transdutor.
    probe = control_configuration.find("Probe")
    salvo_name_list = probe.get("salvoName").split(';')[:-1]

    # Abre o arquivo XML de configuração auxiliar "M2kConfig.xml", na subpasta M2kConfigFiles.
    # Isso é para uso com o Panther.
    try:
        tree_sub = parse(filename + "/M2kConfig_files/M2kConfig.xml")
        root_sub = tree_sub.getroot()
    except ParseError:
        # Erro no parsing, não utilizar as informações desse arquivo.
        root_sub = None

    # Abre o arquivo XML com a descrição dos dados (A-scan e imagens TFM) gravados no arquivo binário.
    tree_acq_desc = parse(filename + "/acq_files/acq_desc.xml")
    shot_desc_list = tree_acq_desc.getroot().findall("ShotDescSave")
    carto_by_shot = int(tree_acq_desc.getroot().get("nbCartoByShot"))

    # Abre o arquivo binário com as informações sobre os dados armazenados no arquivo binário com dados de inspeção.
    data_desc_save_list = list()
    with open(filename + "/acq_files/acq_desc.bin", "rb") as f_desc:
        f_desc.seek(0, 2)
        filesize_desc = f_desc.tell()
        f_desc.seek(0, 0)

        # Processa o arquivo binário até o fim. Esse procedimento detecta a quantidade de shots no arquivo.
        num_shots = 0
        idx_shot = 0
        while f_desc.tell() < filesize_desc:
            for shot_desc in shot_desc_list:
                shot_desc_index = shot_desc_list.index(shot_desc)
                # Verifica se esse ``shot_desc`` já foi processado em um ``shot`` anterior.
                try:
                    data_desc_save = data_desc_save_list[shot_desc_index]
                    new_data_desc_save = False

                except IndexError:
                    # Cria um novo item "DataDescSave".
                    data_desc_save = DataDescSave()
                    new_data_desc_save = True

                # Obtém as informações do novo ``DataDescSave``
                ddsave_list = shot_desc.findall("DataSave/DataDescSave")
                for ddsave in ddsave_list:
                    ddsave_class_type = np.int(ddsave.get("classType"))
                    ddsave_type = np.int(ddsave.get("type"))
                    if ddsave_class_type == 2 and ddsave_type == 16:
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(24 + 4*carto_by_shot)
                        sads = "=IQQ" + "f"*carto_by_shot + "I"
                        unpacked_data = struct.unpack(sads, buffer)

                        if new_data_desc_save:
                            data_desc_save.id = unpacked_data[0]
                            data_desc_save.file_pointer.append(unpacked_data[1])
                            data_desc_save.index = unpacked_data[2]
                            data_desc_save.carto = np.array(unpacked_data[3:-1])
                            data_desc_save.pad = unpacked_data[-1]
                        else:
                            data_desc_save.file_pointer.append(unpacked_data[1])

                    elif ddsave_class_type == 1 and ddsave_type == 6:
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            index = np.int(ddsave.get("indice")) - 1
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.bytes_per_channel.insert(index, unpacked_data[0])

                    elif ddsave_class_type == 2 and ddsave_type == 50:
                        # Não sei ainda o que é.
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type50 = unpacked_data[0]

                    elif ddsave_class_type == 2 and ddsave_type == 51:
                        # Não sei ainda o que é.
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type51 = unpacked_data[0]

                    elif ddsave_class_type == 2 and ddsave_type == 52:
                        # Não sei ainda o que é.
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type52 = unpacked_data[0]

                    elif ddsave_class_type == 1 and ddsave_type == 48:
                        # Não sei ainda o que é, mas está presente nos arquivos do Multix2000.
                        # Lê mas não faz nada, só para ajustar o ponteiro do arquivo.
                        f_desc.seek(4, 1)

                    elif ddsave_class_type == 0 and ddsave_type == 10:
                        # Não sei ainda o que é, mas está presente nos arquivos do Multix2000.
                        # Lê mas não faz nada, só para ajustar o ponteiro do arquivo.
                        f_desc.seek(4, 1)

                    elif ddsave_class_type == 4 and ddsave_type == 3:
                        # Quantidade de bytes do array de float32 que guarda as informações dos encoders.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type03 = unpacked_data[0]

                    elif ddsave_class_type == 1 and ddsave_type == 5:
                        # Ascan com a soma de todos os canais.
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            # Aparentemente a informação de ``index`` não é importante.
                            # index = np.int(ddsave.get("indice")) - 1
                            # unpacked_data = struct.unpack("=I", buffer)
                            # data_desc_save.bytes_per_channel.insert(index, unpacked_data[0])
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type05 = unpacked_data[0]

                    elif ddsave_class_type == 1 and ddsave_type == 21:
                        # Não sei ainda o que é. Mas aparece sempre junto com o "tipo 5".
                        # Lê os dados do cabeçalho no arquivo binário.
                        buffer = f_desc.read(4)
                        if new_data_desc_save:
                            unpacked_data = struct.unpack("=I", buffer)
                            data_desc_save.type21 = unpacked_data[0]

                    else:
                        # Tipo desconhecido ainda.
                        raise TypeError("DataDescSave desconhecido - class[%d], type[%d]" % (ddsave_class_type,
                                                                                             ddsave_type))

                # Guarda a descrição na lista
                if new_data_desc_save:
                    data_desc_save_list.append(data_desc_save)
                else:
                    data_desc_save_list[shot_desc_index] = data_desc_save

            num_shots = num_shots + 1
            idx_shot = idx_shot + 1

    # Fecha o arquivo binário.
    f_desc.close()

    # Busca, na configuração do arquivo m2k, os parâmetros relativos ao transdutor e cria uma instância do objeto
    # ``ProbeParams``. Primeiro, tentamos buscar as informações no arquivo "M2kConfig.xml",
    # na subpasta M2kConfigFiles. Isso é para uso com o Panther. Se falhar, segue com a versão original para
    # os arquivos m2k.
    if root_sub is not None:
        # Tipo de inspeção
        position = root_sub.find("ModeleControle/Positionnement")
        if position is not None:
            type_insp = position.get("typeCnt")

        # Coluna de água
        pos_immersion = position.find("PosImmersion")
        if pos_immersion is not None:
            water_path = float(pos_immersion.get("Heau"))

        # Se o valor da coluna de água for 0, o ensaio é por contato.
        if water_path == 0.0:
            type_insp = "contact"

        # Parâmetros do transdutor
        # Descobre antes o tipo do transdutor
        transd_past = root_sub.find("ReseauTraducteur/Traducteur/DecoupagePastille")
        type_transd = transd_past.get("decoupage")
        if type_transd == "lineaire":
            # Array linear
            decoup_lin = root_sub.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageLineaire")
            num_elem = int(decoup_lin.get("nbElements"))
            tp = "linear" if num_elem > 1 else "mono"
            inter_elem = float(decoup_lin.get("spaceInterIncid"))
            dim = float(decoup_lin.get("widthIncid"))
            pitch = dim + inter_elem
        elif type_transd == "monoelement":
            # Monoelemento
            decoup_mono = root_sub.find(
                "ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMono/PastilleCirculaire")
            num_elem = 1
            tp = "mono"
            inter_elem = 0.0
            dim = float(decoup_mono.get("rayon"))
            pitch = 0.0
        elif type_transd == 'circulaire':
            # Circular matricial (rho-theta)
            decoup_ellip = root_sub.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti"
                                         "/DecoupageElliptique")
            rings = decoup_ellip.findall("Piste")
            shape = 0
            elem_list = list()
            for ring in rings:
                r_max = float(ring.get("r3max"))
                r_min = float(ring.get("r2min"))
                sectors = ring.findall("Secteur")
                for sector in sectors:
                    shape += 1
                    start_angle = float(sector.get("startAngle"))
                    arc = 2.0 * np.pi / float(len(sectors))
                    center_angle = start_angle + arc / 2
                    if r_min < 1e-6:
                        center_radius = 0.0
                    else:
                        center_radius = (r_max + r_min) / 2
                    elem_dict = {"center_angle": center_angle, "center_radius": center_radius}
                    elem_list.append(elem_dict)
            shape = (shape,)
            num_elem = shape[0]
            tp = "circular" if num_elem > 1 else "mono"
            inter_elem = 0.0
            dim = 0.0
            pitch = 0.0
            # probe_args["num_elem"] = shape[0]
            # probe_args["elem_list"] = elem_list
        else:
            # Tipo não suportado.
            raise TypeError("Tipo de transdutor não suportado (atualmente só 'lineaire', 'monoelement' e'"
                            "'circulaire').")

    else:
        num_elem = int(probe.get("nbElement"))
        tp = "linear" if num_elem > 1 else "mono"
        dim = float(probe.get("largeurElement"))
        inter_elem = float(probe.get("interElement"))
        pitch = float(probe.get("interElement")) + dim

    # Verificação sobre os eixos de movimentação do transdutor.
    step_x = 0.0
    step_y = 0.0
    step_z = 0.0
    unit_scales = {'UNIT_MILLIMETER': 1.0, 'UNIT_CENTIMETER': 10.0, 'UNIT_METER': 1000.0}
    mechanic_params = control_configuration.find("ParametresMecanique")
    if int(mechanic_params.get("modeConfigurationCoder")) == 1:
        # Pega a lista de "coders".
        coder_list = mechanic_params.findall("Coder")

        # Tem eixos de movimentação.
        for traj in mechanic_params.findall("Trajectoire"):
            # Processa somente as trajetórias associadas a movimentação (despreza o "eixo do tempo").
            axe = traj.get("axe")
            if axe[0] == "C":
                for coder in coder_list:
                    if coder.get("name") == axe and coder.get("used") == 'true':
                        try:
                            scale = unit_scales[traj.get("unit")]
                        except IndexError:
                            scale = 1.0

                        # Nós convencionamos que "C1" é o eixo x, "C2" é o eixo y e "C3" é o eixo z.
                        if axe == "C1":
                            step_x = float(traj.get("pas")) * scale
                        elif axe == "C2":
                            step_y = float(traj.get("pas")) * scale
                        elif axe == "C3":
                            step_z = float(traj.get("pas")) * scale

    # Elementos contendo informações sobre o processo eletrônico de disparo do transdutor.
    rafale_list = control_configuration.findall("ReglagesUs/Controle/ControleZone/Rafale")

    # Cria uma lista de objetos DataInsp
    lista_dados = list()

    # Abre o arquivo com os dados de inspeção.
    with open(filename + "/acq_files/acq_data.bin", "rb") as f:
        # Processa todos os *shots* pedidos. A quantidade de *shots* foi encontrada anteriormente.
        shots_in_file = range(num_shots)
        idx_shot = 0
        if sel_shots is None:
            sel_shots = shots_in_file
        for shot in sel_shots:
            if shot not in shots_in_file:
                # Pula o *shot*.
                continue

            # Calcula o deslocamento do shot.
            step_gap = np.zeros((1, 3))
            step_gap[0, 0] = step_x * shot  # Precisa usar o 'índice absoluto' do shot.
            # TODO: Não foi feito o tratamento para movimentação nos eixos y e z. Futuramente devemos avaliar.

            # Laço para processar cada parte de um arquivo "MULTISALVO" como um data_insp diferente
            for rafale in rafale_list:
                general_parameters = rafale.find("ParametresGeneraux")
                dac = general_parameters.find("Dac")
                us_speed = general_parameters.find("UsSpeed")
                gate = general_parameters.find("DefPortes/Porte")
                sequence_list = rafale.findall("Sequence")
                rafale_idx = rafale_list.index(rafale)

                # Verifica se o ensaio está configurado para ter o canal de soma dos *A-scans*.
                sum_ascan = gate.get("saveAscan") == 'true'

                # Verifica se esse ``Rafale`` já foi processado em um ``shot`` anterior.
                try:
                    dados = lista_dados[rafale_idx]
                    new_rafale = False
                except IndexError:
                    # Processa a primeira vez o ``Rafale``.
                    new_rafale = True

                if not new_rafale:
                    # Busca a coordenada do centro do transdutor.
                    point_center_trans = np.zeros((1, 3))
                    point_center_trans[0, 0] = float(rafale.get("ancrageProbeY"))
                    point_center_trans[0, 1] = float(rafale.get("ancrageProbeX"))
                    point_center_trans[0, 2] = 0.0

                    # Ajusta todos os pontos dos passos para a mesma coordenada de centro do transdutor.
                    try:
                        dados.inspection_params.step_points[idx_shot, :] = point_center_trans + step_gap
                    except IndexError:
                        dados.inspection_params.step_points = np.concatenate((dados.inspection_params.step_points,
                                                                              point_center_trans + step_gap))

                    # Concatena mais um ``shot`` no array ascan_data.
                    if read_ascan:
                        shape_fmc = (dados.ascan_data.shape[0], dados.ascan_data.shape[1], dados.ascan_data.shape[2], 1)
                        dados.ascan_data = np.concatenate((dados.ascan_data,
                                                           np.zeros(shape_fmc, dtype=np.float32)),
                                                          axis=-1)

                        # Cria ou concatena ``shot`` ao array ascan_data_sum.
                        if sum_ascan:
                            shape_fmc_sum = (dados.ascan_data.shape[0], dados.ascan_data.shape[1], 1)
                            if dados.ascan_data_sum is None:
                                # *Array* ``ascan_data_sum`` não existe ainda, é preciso criá-lo.
                                dados.ascan_data_sum = np.zeros(shape_fmc_sum, dtype=np.float32)
                            else:
                                # Concatena mais um ``shot`` no array ascan_data_sum.
                                dados.ascan_data_sum = np.concatenate((dados.ascan_data_sum,
                                                                       np.zeros(shape_fmc_sum, dtype=np.float32)),
                                                                      axis=-1)

                    # Inicializa as variáveis locais com informações do DataInsp.
                    gate_samples = dados.inspection_params.gate_samples

                else:
                    # Processa a primeira vez o ``Rafale``.
                    # Cria uma instância do objeto ``ProbeParams``.
                    probe_params = ProbeParams(tp=tp,
                                               num_elem=num_elem,
                                               pitch=pitch,
                                               dim=dim,
                                               inter_elem=inter_elem,
                                               freq=freq_transd,
                                               bw=bw_transd,
                                               pulse_type=tp_transd)
                    if probe_params.type_probe == 'circular':
                        probe_params.elem_list = elem_list

                    # Busca as coordenadas cartesianas dos centros geométricos de cada elemento do *array*.
                    # O equipamento M2K considera o eixo X como Y e vice versa.
                    # pos_trans_x_rafale = np.asarray(rafale.get("posSequencesY").split(';')[:-1], dtype=np.float32)
                    # pos_trans_y_rafale = np.asarray(rafale.get("posSequencesX").split(';')[:-1], dtype=np.float32)

                    # Busca, na configuração do arquivo mk2, os parâmetros relativos ao especimen (peça)
                    # inspecionado e cria uma instância do objeto ``SpecimenParams``.
                    speed_cl = float(us_speed.get("speedL"))
                    speed_cs = float(us_speed.get("speedT"))
                    specimen_params = SpecimenParams(cl=speed_cl, cs=speed_cs)

                    # Busca, na configuração do arquivo m2k, os parâmetros relativos ao processo de inspeção e
                    # cria uma instância do objeto ``InspectionParams``.
                    # Busca frequência de amostragem.
                    sample_freq = float(dac.get("freq")) / 1E6

                    # Busca os ganhos. Lê todos mas salva somente o Balayage e o Numerique (data_types.py)
                    gains = {"Apodisation": float(general_parameters.get("gainApodisation")),
                             "Balayage": float(general_parameters.get("gainBalayage")),
                             "Complementaire": float(general_parameters.get("gainComplementaire")),
                             "Numerique": float(general_parameters.get("gainNumerique"))}

                    # Busca as informações referentes ao *gate*.
                    gate_samples = int(float(gate.get("largeurCour").split(';')[0]) + 0.5)
                    gate_start = float(gate.get("debut").split(';')[0]) + float(
                        general_parameters.get("retardNumerisation"))
                    gate_end = float(gate.get("largeur").split(';')[0]) + gate_start

                    # Cria uma instância do objeto ``InspectionParams``.
                    inspection_params = InspectionParams(type_insp=type_insp,
                                                         sample_freq=sample_freq,
                                                         gate_start=gate_start,
                                                         gate_end=gate_end,
                                                         gate_samples=gate_samples,
                                                         gains=gains, )
                    inspection_params.coupling_cl = 1483.0
                    # Ajusta o valor da coluna d'água se for inspeção por imersão.
                    if type_insp == "immersion":
                        inspection_params.water_path = water_path

                    # Busca informações sobre o tipo de captura para verificar se é PWI ou unissequencial. Se
                    # for PWI, busca informações sobre os ângulos de disparo e altera a
                    # estrutura inspection_params.
                    theta_i = float(general_parameters.get("angleStartDefinitionMS"))
                    theta_f = float(general_parameters.get("angleEndDefinitionMS"))
                    if len(sequence_list) == 1:
                        if (int(probe.get("nbElement")) > 1) and (abs(theta_i) < 1000.0) and (abs(theta_f) < 1000):
                            # É uma varredura por PWI.
                            inspection_params.type_capt = 'PWI'
                            tir_list = sequence_list[0].findall("Tir")
                            theta_n = len(tir_list)
                            inspection_params.angles = np.zeros(theta_n)
                        else:
                            # É uma captura 'Unisequential'.
                            inspection_params.type_capt = 'Unisequential'
                    else:
                        # É uma varredura FMC.
                        inspection_params.type_capt = 'FMC'

                    # Busca a coordenada do centro do transdutor.
                    if probe_params.type_probe == "mono":
                        point_center_trans = np.zeros((1, 3))
                        inspection_params.step_points[0, :] = point_center_trans + step_gap

                    elif probe_params.type_probe == 'circular':
                        probe_params.elem_center = np.zeros((shape[0], 3))
                        for i in range(len(probe_params.elem_list)):
                            elem_dict = probe_params.elem_list[i]
                            x_center = np.cos(elem_dict["center_angle"]) * elem_dict["center_radius"]
                            y_center = np.sin(elem_dict["center_angle"]) * elem_dict["center_radius"]
                            probe_params.elem_center[i, :] = np.array([x_center, y_center, 0.0])

                        elem_center_mean = probe_params.elem_center.mean((0, 1))
                        probe_params.elem_center = probe_params.elem_center - elem_center_mean

                    else:
                        point_center_trans = np.zeros((1, 3))
                        point_center_trans[0, 0] = float(rafale.get("ancrageProbeY"))
                        point_center_trans[0, 1] = float(rafale.get("ancrageProbeX"))
                        point_center_trans[0, 2] = 0.0
                        inspection_params.step_points[0, :] = point_center_trans + step_gap

                        # Ajusta as coordenadas dos centros dos elementos emissores em relação a coordenada central
                        # do transdutor.
                        # Isso é sempre relativo ao primeiro *shot*. Anteriormente, o centro geométrico do transdutor
                        # era obtido a partir dos dados disponíveis no próprio arquivo de inspeção. Agora, o centro dos
                        # elementos é calculado a partir dos parâmetros de pitch (p), gap (g) e tamanho do elemento (a).
                        pb_a = probe_params.elem_dim
                        pb_g = probe_params.inter_elem
                        pb_p = probe_params.pitch
                        pb_w = num_elem * (pb_a + pb_g) - pb_g
                        probe_params.elem_center[:, 0] = pb_p * np.arange(num_elem) + pb_a / 2 - pb_w / 2
                        probe_params.elem_center[:, 1] = inspection_params.step_points[0, 1]
                        probe_params.elem_center[:, 2] = inspection_params.step_points[0, 2]

                    # Ajusta a coordenada de origem como a coordenada do centro do transdutor no primeiro *step*.
                    inspection_params.point_origin = inspection_params.step_points[0, :]

                    # Cria uma instância do objeto ``DataInsp``.
                    dados = DataInsp(inspection_params, specimen_params, probe_params)

                    # Atribui o nome do conjunto de dados.
                    salvo_name_idx = int(rafale.get("index"))
                    dados.dataset_name = salvo_name_list[salvo_name_idx]

                if read_ascan:
                    # Pega os parâmetros referentes às imagens TFM, se existirem.
                    rafale_tfm_params = rafale.find("TFMParam")
                    if rafale_tfm_params is not None:
                        tfm_min_z = float(rafale_tfm_params.get("zoneMinY"))
                        tfm_max_z = float(rafale_tfm_params.get("zoneMaxY"))
                        tfm_min_x = float(rafale_tfm_params.get("zoneMinX"))
                        tfm_max_x = float(rafale_tfm_params.get("zoneMaxX"))
                        tfm_corner = np.asarray([tfm_min_x, 0, tfm_min_z])[np.newaxis, :]
                        tfm_corner = tfm_corner - dados.inspection_params.step_points[idx_shot, :]
                        tfm_len_h = tfm_max_z - tfm_min_z
                        tfm_len_w = tfm_max_x - tfm_min_x
                        tfm_w = int(rafale_tfm_params.get("zoneWidthPt"))
                        tfm_h = int(rafale_tfm_params.get("zoneHeightPt"))
                    else:
                        tfm_corner = np.zeros((1, 3))
                        tfm_len_h = 0
                        tfm_len_w = 0
                        tfm_w = 0
                        tfm_h = 0

                    # Faz a leitura dos sinais ``A-scan`` diretamente do arquivo binário.
                    versionstr = tree.getroot().get('versionStr')
                    if versionstr == '8.0.2':
                        header_inicial = 2 * np.int16().itemsize  # 2 words
                        header_final = 8 * np.int16().itemsize  # 8 words
                    elif versionstr == '8.2.0':
                        header_inicial = 3 * np.int16().itemsize  # 3 words
                        header_final = 8 * np.int16().itemsize  # 8 words
                    else:
                        header_inicial = 2 * np.int16().itemsize  # 2 words
                        header_final = 0 * np.int16().itemsize  # 0 words

                    # Processa cada sequência.
                    # ascan_seq_list = list()
                    #
                    # if sum_ascan:
                    #     ascan_sum_seq_list = list()

                    for sequence in sequence_list:
                        # Pega a lista com todos os disparos (``Tir``) da sequência.
                        tir_list = sequence.findall("Tir")

                        # Calcula quantos bytes tem cada disparo.
                        for tir in tir_list:
                            # Pega a informação do ângulo se a captura for ``PWI``.
                            if dados.inspection_params.type_capt == 'PWI':
                                index_angle = tir_list.index(tir)
                                dados.inspection_params.angles[index_angle] = \
                                    float(tir.find("Reception").get("angleDeg"))
                                emmit_index = index_angle

                            elif dados.inspection_params.type_capt == 'Unisequential':
                                # Captura por emissor único, desconsidera dados dos emissores.
                                emmit_index = 0

                            else:
                                # Outros tipos de captura.
                                emmit_index = list(map(int, tir.find("Emission").get("voieEmission").split(';')[:-1]))
                                if len(emmit_index) == 1:
                                    emmit_index = emmit_index[0]

                            # Pega as informações referentes aos receptores do disparo.
                            recep_index = list(map(int, tir.find("Reception").get("voieReception").split(';')[:-1]))
                            num_recep = len(recep_index)

                            # Pega o índice do disparo e obtém as informações do armazenamento
                            # no arquivo ``acq_data.bin``.
                            tir_index = int(tir.get("index"))
                            data_desc_save_tir = data_desc_save_list[tir_index]

                            # Calcula o número de bytes do disparo no arquivo binário.
                            num_bytes_tir = data_desc_save_tir.total_bytes()
                            if num_bytes_tir > 0:
                                num_bytes_tir = header_inicial + num_bytes_tir + header_final

                                # Cria um ndarray para acomodar todos os sinais **A-scan** recebidos no disparo.
                                num_tot_recep = num_recep if data_desc_save_tir.has_recep_ascan() else 0
                                num_tot_recep += 1 if data_desc_save_tir.has_sum_ascan() else 0
                                ascan_tir_data = np.zeros((gate_samples, num_tot_recep), dtype=np.float32)

                                # Lê os bytes referentes a sequência do arquivo binário.
                                f.seek(data_desc_save_tir.file_pointer[shot], 0)
                                buffer = f.read(num_bytes_tir)
                                tam_tir_data = gate_samples * num_tot_recep
                                ascan_tir_data[:, :] = np.reshape(np.frombuffer(buffer,
                                                                                dtype=np.int16,
                                                                                count=tam_tir_data,
                                                                                offset=header_inicial),
                                                                  (gate_samples, num_tot_recep), order='F')

                                # Inclui os A-scans na estrutura ``DataInsp``.
                                if data_desc_save_tir.has_sum_ascan():
                                    if dados.ascan_data_sum is None:
                                        # *Array* ``ascan_data_sum`` não existe ainda, é preciso criá-lo.
                                        shape_fmc_sum = (dados.ascan_data.shape[0], dados.ascan_data.shape[1], 1)
                                        dados.ascan_data_sum = np.zeros(shape_fmc_sum, dtype=np.float32)

                                    # Armazena o sinal de soma no array específico.
                                    dados.ascan_data_sum[:, emmit_index, idx_shot] = ascan_tir_data[:, 0]
                                    if data_desc_save_tir.has_recep_ascan():
                                        dados.ascan_data[:, emmit_index, recep_index, idx_shot] = ascan_tir_data[:, 1:]

                                elif data_desc_save_tir.has_recep_ascan():
                                    dados.ascan_data[:, emmit_index, recep_index, idx_shot] = ascan_tir_data

                                # Acrescenta o disparo na lista
                                # if data_desc_save_tir.has_sum_ascan():
                                #     ascan_sum_seq_list.append(ascan_tir_data[:, 0])
                                #     if data_desc_save_tir.has_recep_ascan():
                                #         ascan_seq_list.append(ascan_tir_data[:, 1:])
                                # elif data_desc_save_tir.has_recep_ascan():
                                #     ascan_seq_list.append(ascan_tir_data)

                                # Lê informação dos encoders se existir.
                                if data_desc_save_tir.has_encoders_info() > 0:
                                    offset_buffer = header_inicial + tam_tir_data * np.int16().itemsize \
                                                    + data_desc_save_tir.type21
                                    encoders_info = np.frombuffer(buffer,
                                                                  dtype=np.float32,
                                                                  offset=offset_buffer)
                                    dados.encoders_info.append(encoders_info)

                            # Lê uma imagem TFM se existir
                            if data_desc_save_tir.has_tfm() > 0:
                                f.seek(data_desc_save_tir.file_pointer[shot] + int(num_bytes_tir), 0)
                                buffer = f.read(data_desc_save_tir.type50 +
                                                data_desc_save_tir.type51 +
                                                data_desc_save_tir.type52)

                                # Cria a ROI da imagem
                                tfm_roi = ImagingROI(tfm_corner, tfm_len_h, tfm_h, tfm_len_w, tfm_w)

                                # Cria uma "chave" para a imagem
                                ii32 = np.iinfo(np.uint32)
                                while True:
                                    k = np.random.randint(low=ii32.min, high=ii32.max, dtype=np.uint32)
                                    if k not in dados.imaging_results:
                                        break

                                # Cria o objeto da imagem
                                tfm_image = ImagingResult(roi=tfm_roi, description=str(k))
                                tfm_image.image = np.reshape(np.frombuffer(buffer, dtype=np.int32, count=tfm_h * tfm_w),
                                                             (tfm_w, tfm_h), order='F').T

                                # Coloca a imagem no dicionário imaging_result
                                dados.imaging_results[k] = tfm_image

                    # Coloca os sinais ``A-scan`` do ``shot`` processado na estrutura ``DataInsp``.
                    # if len(ascan_seq_list) > 0:
                    #     dados.ascan_data[:, :, :, idx_shot] = np.asarray(ascan_seq_list).transpose((1, 0, 2))
                    #
                    #     # Limpa a lista de sequências de sinais ``A-scan`` antes de processar o próximo ``shot``.
                    #     ascan_seq_list.clear()

                    # if sum_ascan and len(ascan_sum_seq_list) > 0:
                    #     if dados.ascan_data_sum is None:
                    #         # *Array* ``ascan_data_sum`` não existe ainda, é preciso criá-lo.
                    #         shape_fmc_sum = (dados.ascan_data.shape[0], dados.ascan_data.shape[1], 1)
                    #         dados.ascan_data_sum = np.zeros(shape_fmc_sum, dtype=np.float32)
                    #
                    #     dados.ascan_data_sum[:, :, idx_shot] = np.asarray(ascan_sum_seq_list).transpose((1, 0))
                    #
                    #     # Limpa a lista antes de processar o próximo ``shot``.
                    #     ascan_sum_seq_list.clear()

                # Atualiza a lista de DataInsp.
                if new_rafale:
                    lista_dados.append(dados)
                else:
                    lista_dados[rafale_idx] = dados

            # Processou todos os ``Rafales``, os dados restantes do arquivo são novos ``shots``.
            idx_shot = idx_shot + 1

    # Fecha o arquivo binário
    f.close()

    # Retorna a lista
    if len(lista_dados) == 1:
        return lista_dados[0]
    else:
        return lista_dados
