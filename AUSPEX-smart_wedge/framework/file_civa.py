# -*- coding: utf-8 -*-
"""
Módulo ``file_civa``
=====================

O módulo :mod:`.file_civa` é dedicado a realizar a leitura de arquivos do
simulador CIVA.

O CIVA é desenvolvido pelo CEA e parceiros da área de simulação de Ensaios
Não Destrutivos (ENDs).

É uma plataforma de *software* composta por seis módulos com múltiplos
conhecimentos destinados ao desenvolvimento e otimização de métodos de
ENDs e concepção de sondas. Tem por objetivos melhorar a qualidade
das técnicas de ENDs e auxiliar na interpretação dos complexos resultados
das inspeções.

As simulações desempenham um papel importante desde a identificação de
potenciais defeitos a partir do desenho da peça, durante o ensaio,
qualificando os métodos, otimizando parâmetros e analisando fatores de
perturbação, até o desenvolvimento de amostras para ENDs de geometrias
de amostras ou de estruturas semelhantes aos *designs* iniciais.

Como o formato de saída dos dados de simulação são em arquivos do tipo
texto, o que torna a leitura de dados excessivamente lenta, optou-se por
ler os arquivos no formato *.CIVA*.

Esses arquivos *.CIVA* possuem uma série de diretórios denominados *procN*,
em que *N* corresponde ao número do *gating* da simulação. Neste módulo do
*framework*, considera-se apenas o *gating* 0, e os demais são
desconsiderados.

No diretório *proc0*, as configurações da simulação (parâmetros da peça,
transdutor e da inspeção) estão em um arquivo *model.xml*. Como o *XML* é um
formato de dados inerentemente hierárquico, a maneira mais natural de
representá-lo é com uma árvore. Para efetuar a leitura desses arquivos,
utiliza-se a biblioteca *etree*. A *etree* tem duas classes: a
primeira é a *ElementTree* que representa o documento *XML* inteiro como
uma árvore; a segunda é a *Element* que representa um único nó nessa árvore.

Já os *A-scan* são salvos no arquivo *channels_signal_Mephisto_gate_1* quando
a inspeção é feita com transdutor linear, e *sum_signal_Mephisto_gate_1*,
para inspeções com transdutor mono.

Os valores de amplitude dos *A-scan* são do formato *float32*. Ainda, cada
*A-scan* é precedido por um cabeçalho com 5 valores no formato *Int32*.
Destes valores, o segundo representa o número da amostra inicial do *A-scan*,
considerando que a amostragem ocorre desde a emissão do pulso de ultrassom.
O terceiro representa o número da amostra final e o último valor representa
o número de bytes do *A-scan* compresso, ou -1 caso não exista compressão.
Os *A-scans* são compressos utilizando o padrão *gzip*. Os outros dois
valores no cabeçalho são referentes ao CIVA, e não possuem informação sobre
a simulação.

.. raw:: html

    <hr>

"""

import math
import os.path
import struct
import zlib
import sys
import numpy as np
import xml.etree.ElementTree

from framework.data_types import DataInsp, InspectionParams, SpecimenParams, ProbeParams


def read(filename, sel_shots=None, read_ascan=True):
    """Abre e analisa um arquivo .civa, retornando os dados da simulação.

    Os dados são retornados como um objeto da classe :class:`.DataInsp`,
    contendo os parâmetros de inspeção, do transdutor, da peça e os
    dados da simulação.

    Parameters
    ----------
        filename : str
            Caminho do arquivo .civa.

        sel_shots : NoneType, int, list ou range
            *Shots* para a leitura. Se for ``None``, lê todos os *shots*
            disponíveis. Se ``int``, lê o índice especificado. Se ``list``
            ou ``range``, lê os índices especificados.

    Returns
    -------
    :class:`.DataInsp`
        Dados do arquivo lido, contendo parâmetros de inspeção, do transdutor,
        da peça e os dados de simulação.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``sel_shots`` não é do
        tipo ``NoneType``, ``int``, ``list`` ou ``range``.

    IndexError
        Gera exceção de ``IndexError`` se o *shot* especificado não existe.

    FileNotFoundError
        Gera exceção de ``FileNotFoundError`` se o arquivo não existir.

    """

    def civa_matricial(arquivo, n_ascan, n_passos, _sel_shots):
        """Docstring da função ``civa_fmc``

        Abre um arquivo binário do formato do civa, que contém uma série de
        A-scans e retorna os dados no formato de um fmc.

        """

        tam_num = np.int32().itemsize

        file = open(arquivo, 'rb')
        data = file.read(4)

        num_inicial = struct.unpack('<i', data)[0]
        posicoes = []
        headers = []

        # menor numero de amostras antes do gating iniciar
        menor = sys.maxsize
        # maior indice de amostras do A-scan, considerando o gating
        tam_max = 0

        # procura o arquivo para encontrar todos os cabeçalhos e salva as suas posicoes
        # tambem encontra os valores acima
        _i = 0
        j = 0
        while j < n_ascan:
            file.seek(_i)
            data = file.read(5 * tam_num)
            try:
                header = struct.unpack('<5i', data)
            except struct.error:
                # marca essa posicao para ter sempre 0
                header = (-1, -1, -1, -1, -2)
                headers.append(header)

            if header[0] == num_inicial and header[3] != 0:
                # header esperado
                posicoes.append(_i)
                headers.append(header)
                if headers[j][1] < menor:
                    menor = headers[j][1]
                if headers[j][2] + headers[j][1] > tam_max:
                    tam_max = headers[j][2] + headers[j][1]

                if headers[j][4] == -1:
                    _i += headers[j][2] * tam_num + 5 * tam_num
                else:
                    _i += headers[j][4] + 5 * tam_num

            elif header[0] == num_inicial:
                # gating manual e elemento fora da peça
                header = (-1, -1, -1, -1, -2)
                headers.append(header)
                posicoes.append(_i)
                _i += 4 * tam_num

            else:
                _i += 1

            j += 1

        # aloca a matriz dos dados
        n_elem = int(math.sqrt(n_ascan / n_passos))
        out = []

        _gating = [menor, tam_max]

        n_ascan_salvos = 0

        # preenche a matriz
        for k in _sel_shots:
            for _i in range(n_elem):
                for j in range(n_elem):
                    index = j + _i * n_elem + k * n_elem ** 2
                    if headers[index][4] > 0:
                        # descompacta o A-scan
                        decompress = zlib.decompressobj(-15)
                        n_bytes = headers[index][4]
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(n_bytes)
                        inflated = decompress.decompress(data)
                        inflated += decompress.flush()
                        ascan = np.frombuffer(inflated, dtype=np.float32)
                        out.append(ascan)

                    elif headers[index][4] == -1:
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(headers[index][2] * tam_num)
                        ascan = np.frombuffer(data, dtype=np.float32)
                        out.append(ascan)

                    else:
                        # transdutor fora da peça, possui apenas o header
                        ascan = np.zeros(tam_max - menor)
                        headers[index] = (-1, menor, -1, -1, -2)
                        out.append(ascan)

                    # corrige o A-scan de acordo com o gating
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (headers[n_ascan_salvos][1] - menor, 0),
                                                 'constant')
                    # coloca todos com o mesmo tamanho
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (0, tam_max - menor - len(out[n_ascan_salvos])),
                                                 'constant')

                    n_ascan_salvos += 1

        file.close()

        out = np.array(out)
        out = np.reshape(out, (len(out) * len(out[0])))
        out = np.reshape(out, (tam_max - menor, n_elem, n_elem, len(_sel_shots)), order='F')
        out = np.swapaxes(out, 1, 2)

        return [out, _gating]

    def civa_fmc(arquivo, n_ascan, n_passos, _sel_shots):
        """Abre um arquivo binário do formato do civa, que contém uma série de
        A-scans e retorna os dados no formato de um FMC.

        """

        tam_num = np.int32().itemsize

        file = open(arquivo, 'rb')
        data = file.read(4)

        num_inicial = struct.unpack('<i', data)[0]
        posicoes = []
        headers = []

        # menor numero de amostras antes do gating iniciar
        menor = sys.maxsize
        # maior indice de amostras do A-scan, considerando o gating
        tam_max = 0

        # procura o arquivo para encontrar todos os cabeçalhos e salva as suas posicoes
        # tambem encontra os valores acima
        _i = 0
        j = 0
        while j < n_ascan:
            file.seek(_i)
            data = file.read(5 * tam_num)
            try:
                header = struct.unpack('<5i', data)
            except struct.error:
                # marca essa posicao para ter sempre 0
                header = (-1, -1, -1, -1, -2)
                headers.append(header)

            if header[0] == num_inicial and header[3] != 0:
                # header esperado
                posicoes.append(_i)
                headers.append(header)
                if headers[j][1] < menor:
                    menor = headers[j][1]
                if headers[j][2] + headers[j][1] > tam_max:
                    tam_max = headers[j][2] + headers[j][1]

                if headers[j][4] == -1:
                    _i += headers[j][2] * tam_num + 5 * tam_num
                else:
                    _i += headers[j][4] + 5 * tam_num

            elif header[0] == num_inicial:
                # gating manual e elemento fora da peça
                header = (-1, -1, -1, -1, -2)
                headers.append(header)
                posicoes.append(_i)
                _i += 4 * tam_num

            else:
                _i += 1

            j += 1

        # aloca a matriz dos dados
        n_elem = int(math.sqrt(n_ascan / n_passos))
        out = []

        _gating = [menor, tam_max]

        n_ascan_salvos = 0

        # preenche a matriz
        for k in _sel_shots:
            for _i in range(n_elem):
                for j in range(n_elem):
                    index = j + _i * n_elem + k * n_elem ** 2
                    if headers[index][4] > 0:
                        # descompacta o A-scan
                        decompress = zlib.decompressobj(-15)
                        n_bytes = headers[index][4]
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(n_bytes)
                        inflated = decompress.decompress(data)
                        inflated += decompress.flush()
                        ascan = np.frombuffer(inflated, dtype=np.float32)
                        out.append(ascan)

                    elif headers[index][4] == -1:
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(headers[index][2] * tam_num)
                        ascan = np.frombuffer(data, dtype=np.float32)
                        out.append(ascan)

                    else:
                        # transdutor fora da peça, possui apenas o header
                        ascan = np.zeros(tam_max - menor)
                        headers[index] = (-1, menor, -1, -1, -2)
                        out.append(ascan)

                    # corrige o A-scan de acordo com o gating
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (headers[index][1] - menor, 0),
                                                 'constant')
                    # coloca todos com o mesmo tamanho
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (0, tam_max - menor - len(out[n_ascan_salvos])),
                                                 'constant')

                    n_ascan_salvos += 1

        file.close()

        out = np.array(out)
        out = np.reshape(out, (len(out) * len(out[0])))
        out = np.reshape(out, (tam_max - menor, n_elem, n_elem, len(_sel_shots)), order='F')
        out = np.swapaxes(out, 1, 2)

        return [out, _gating]

    def civa_bscan(arquivo, n_passos, _sel_shots):
        """Abre um arquivo binário do formato do civa, que contém uma série de A-scans
        e retorna os dados no formato de um B-scan.

        """

        tam_num = 4

        file = open(arquivo, 'rb')
        data = file.read(4)

        num_inicial = struct.unpack('<i', data)[0]
        posicoes = []
        headers = []

        # menor numero de amostras antes do gating iniciar
        menor = sys.maxsize
        # maior indice de amostras do A-scan, considerando o gating
        tam_max = 0

        # procura o arquivo para encontrar todos os cabeçalhos e salva as suas posicoes
        # tambem encontra os valores acima
        _i = 0
        j = 0
        while j < n_passos:
            file.seek(_i)
            data = file.read(5 * tam_num)
            try:
                header = struct.unpack('<5i', data)
            except struct.error:
                # Acontece quando o arquivo acaba, mas o numero esperado de headers nao foi encontrado
                # marca essa posicao para ter sempre 0
                header = (-1, -1, -1, -1, -2)
                headers.append(header)

            if header[0] == num_inicial:
                posicoes.append(_i)
                headers.append(header)
                if headers[j][1] < menor:
                    menor = headers[j][1]
                if headers[j][2] + headers[j][1] > tam_max:
                    tam_max = headers[j][2] + headers[j][1]

                if headers[j][4] == -1:
                    _i += headers[j][2] * tam_num + 5 * tam_num
                else:
                    _i += headers[j][4] + 5 * tam_num
            j += 1

        # aloca a matriz dos dados
        out = np.zeros((tam_max - menor, 1, 1, len(_sel_shots)))

        _gating = [menor, tam_max]

        n_ascan_salvos = 0

        # preenche a matriz
        for _i in _sel_shots:
            if headers[_i][4] > 0:
                # descompacta o A-scan
                decompress = zlib.decompressobj(-15)
                n_bytes = headers[_i][4]
                file.seek(posicoes[_i] + 5 * tam_num)
                data = file.read(n_bytes)
                inflated = decompress.decompress(data)
                inflated += decompress.flush()
                ascan = np.frombuffer(inflated, dtype=np.float32)
                out[0:len(ascan), 0, 0, n_ascan_salvos] = ascan

            elif headers[_i][4] == -1:
                file.seek(posicoes[_i] + 5 * tam_num)
                data = file.read(headers[_i][2] * tam_num)
                ascan = np.frombuffer(data, dtype=np.float32)
                out[0:len(ascan), 0, 0, n_ascan_salvos] = ascan

            else:
                # transdutor está fora da peça e todas as amostras são zero
                ascan = np.zeros(tam_max - menor)
                out[0:len(ascan), 0, 0, n_ascan_salvos] = ascan

            # corrige o A-scan de acordo com o gating
            out[:, 0, 0, n_ascan_salvos] = np.roll(out[:, 0, 0, n_ascan_salvos], headers[_i][1] - menor, axis=0)

            n_ascan_salvos += 1

        file.close()

        return [out, _gating]

    def civa_pwi(arquivo, n_angles, n_elem, n_passos, _sel_shots):
        """Abre um arquivo binário do formato do civa, que contém uma série de
        A-scans e retorna os dados no formato de um PWI.

        """

        tam_num = np.int32().itemsize

        file = open(arquivo, 'rb')
        data = file.read(4)

        num_inicial = struct.unpack('<i', data)[0]
        posicoes = []
        headers = []

        n_ascan = n_angles * n_elem * n_passos

        # menor numero de amostras antes do gating iniciar
        menor = sys.maxsize
        # maior indice de amostras do A-scan, considerando o gating
        tam_max = 0

        # procura o arquivo para encontrar todos os cabeçalhos e salva as suas posicoes
        # tambem encontra os valores acima
        _i = 0
        j = 0
        while j < n_ascan:
            file.seek(_i)
            data = file.read(5 * tam_num)
            try:
                header = struct.unpack('<5i', data)
            except struct.error:
                # marca essa posicao para ter sempre 0
                header = (-1, -1, -1, -1, -2)
                headers.append(header)

            if header[0] == num_inicial and header[3] != 0:
                # header esperado
                posicoes.append(_i)
                headers.append(header)
                if headers[j][1] < menor:
                    menor = headers[j][1]
                if headers[j][2] + headers[j][1] > tam_max:
                    tam_max = headers[j][2] + headers[j][1]

                if headers[j][4] == -1:
                    _i += headers[j][2] * tam_num + 5 * tam_num
                else:
                    _i += headers[j][4] + 5 * tam_num

            elif header[0] == num_inicial:
                # gating manual e elemento fora da peça
                header = (-1, -1, -1, -1, -2)
                headers.append(header)
                posicoes.append(_i)
                _i += 4 * tam_num

            else:
                _i += 1

            j += 1

        # aloca a matriz dos dados
        # n_elem = int(math.sqrt(n_ascan / n_passos))
        out = []

        gating = [menor, tam_max]

        n_ascan_salvos = 0

        # preenche a matriz
        for k in _sel_shots:
            for _i in range(n_angles):
                for j in range(n_elem):
                    index = j + _i * n_elem + k * n_elem * n_angles
                    if headers[index][4] > 0:
                        # descompacta o A-scan
                        decompress = zlib.decompressobj(-15)
                        n_bytes = headers[index][4]
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(n_bytes)
                        inflated = decompress.decompress(data)
                        inflated += decompress.flush()
                        ascan = np.frombuffer(inflated, dtype=np.float32)
                        out.append(ascan)

                    elif headers[index][4] == -1:
                        file.seek(posicoes[index] + 5 * tam_num)
                        data = file.read(headers[index][2] * tam_num)
                        ascan = np.frombuffer(data, dtype=np.float32)
                        out.append(ascan)

                    else:
                        # transdutor fora da peça, possui apenas o header
                        ascan = np.zeros(tam_max - menor)
                        headers[index] = (-1, menor, -1, -1, -2)
                        out.append(ascan)

                    # corrige o A-scan de acordo com o gating
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (headers[index][1] - menor, 0),
                                                 'constant')
                    # coloca todos com o mesmo tamanho
                    out[n_ascan_salvos] = np.pad(out[n_ascan_salvos], (0, tam_max - menor - len(out[n_ascan_salvos])),
                                                 'constant')

                    n_ascan_salvos += 1

        file.close()

        out = np.array(out)
        out = np.reshape(out, (len(out) * len(out[0])))
        out = np.reshape(out, (tam_max - menor, n_elem, n_angles, len(_sel_shots)), order='F')
        out = np.swapaxes(out, 1, 2)

        return [out, gating]

    def check_pwi_capt(insp_args):
        """Verifica se o tipo de captura pode ser PWI.

        Essa função não retorna nenhum resultado. O tipo de captura é escrito
        diretamente no dicionário ``insp_args``.

        Parameters
        ----------
        insp_args : dict
            Dicionário com o tipo de captura.

        """
        if insp_args["type_capt"] == "FTP":
            _node = root.find("ModeleReglage/Initialisation/OptionsFTP/Ftp")
            ftp_options = {"FMC": "FMC", "PWI": "PWI", "Manuel": "Any"}
            if _node.get("typeExcitation") in ftp_options:
                inspection_args["type_capt"] = ftp_options[_node.get("typeExcitation")]
        elif inspection_args["type_capt"] == "Manual" or inspection_args["type_capt"] == "Eletronic sweep":
            _node = root.find("ModeleReglage/Comportement")
            if _node.get("typeFocalisation") == "focalisationBalayageAng":
                inspection_args["type_capt"] = "PWI"
        elif inspection_args["type_capt"] == "ImportedLaws":
            # Para "ImportedLaws", apesar do tipo de captura ser similar ao PWI, não existe a informação dos ângulos
            # de disparo. Assim, esta informação será substituída pelo número de disparos obtidos pela leitura do
            # arquivo ``law''
            _node = root.find("ModeleReglage/Initialisation/ImportedLaws/LocalFile")
            inspection_args["laws_file"] = _node.get("fileName")
            inspection_args["type_capt"] = "PWI"

    # Busca, na configuração do arquivo civa, os parâmetros relativos ao transdutor e cria uma instância do objeto
    # ``ProbeParams``
    tree = xml.etree.ElementTree.parse(filename + "/proc0/model.xml")
    root = tree.getroot()
    if root.get('choixApplication') == 'App_CSAthena':
        app = 'CS_COUPLAGE'
    else:
        app = 'Mephisto'
    # Calcula as coordenadas do centro do transdutor para os vários *steps*, se existirem
    node = root.find("ModeleControle/Deplacement/DeplacementSuivConfiguration")
    dep_conf_type = node.get('typeDeplaceSuivConfig').split('piece')
    node = node.find("DepConf" + dep_conf_type[1] + "/DeplacementConfigurationBase/DepBalayageIncrement/DepSynchrone")

    # Busca a quantidade de passos dados pelo transdutor
    n_pas_x = 1 if node is None else int(node.get('nbPasBalayage')) + 1
    n_pas_y = 1 if node is None else int(node.get('nbPasIncremental')) + 1
    n_pas = n_pas_y * n_pas_x
    if app == 'CS_COUPLAGE':
        node = root.find("ModeleReglage/Initialisation/BalayageElectroniqueLineaire/BalayageSimple")
        n_pas = int(node.get('nbPasBE'))

    # Busca o passo em cada direção
    # node = node.find("DepConf" + dep_conf_type[1] + "/DeplacementConfigurationBase/DepBalayageIncrement/DepSynchrone")
    pas_x = 1 if node is None else float(node.get('pasBalayage'))
    pas_y = 1 if node is None else float(node.get('pasIncremental'))

    # =========== Busca os parâmetros para instanciar ``SpecimenParams`` ===========
    specimen_args = dict()

    # Velocidade das ondas longitudinais
    node = root.find("Piece/Volume/Material/SimpleMaterial/SimpleParamED/SimpleIsotropicED")
    specimen_args["cl"] = None if node is None else float(node.get("LWaveVelocity_ms"))

    # Velocidade das ondas transversais
    node = root.find("Piece/Volume/Material/SimpleMaterial/SimpleParamED/SimpleIsotropicED")
    specimen_args["cs"] = None if node is None else float(node.get("TWaveVelocity_ms"))

    # Rugosidade
    node = root.find("Piece/Rugosite/RugositeGlobale")
    specimen_args["roughness"] = None if node is None else float(node.get("ra_mm"))

    # Retira todos os elementos da lista ``specimen_args`` cujo valor é ``None``
    specimen_args = {k: v for k, v in specimen_args.items() if v is not None}

    # Cria objeto com os parâmetros da peça. Os argumentos que não foram encontrados são retirados da lista são adotados
    # os valores padrão do construtor
    specimen_params = SpecimenParams(**specimen_args)

    # =========== Busca os parâmetros para instanciar ``ProbeParams`` ===========
    probe_args = dict()

    # Tipo do transdutor
    types_trad = {"monoelement": "mono", "lineaire": "linear", "matriciel": "matricial", "exotique": "generic",
                  "circulaire": "circular"}
    node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille")
    probe_args["tp"] = None if node is None else types_trad.get(node.get("decoupage"))

    # Verifica o tipo do transdutor
    if probe_args["tp"] == "mono":
        # Transdutor monoelemento
        # Formato do transdutor
        shapes_trad = {"circulaire": "circle"}
        node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMono")
        probe_args["shape"] = None if node is None else shapes_trad.get(node.get("formePastille"))
        probe_args["num_elem"] = n_pas

        # Verifica o formato da pastilha
        if probe_args["shape"] == "circle":
            # Pastilha circular
            # Dimensão do transdutor
            node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMono/PastilleCirculaire")
            probe_args["dim"] = None if node is None else (2 * float(node.get("rayon")))

        else:
            # Tratar outros casos
            pass

    elif probe_args["tp"] == "linear":
        # Transdutor *array* linear
        # Número de elementos, dimensão dos elementos e *pitch*
        node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageLineaire")
        probe_args["num_elem"] = None if node is None else int(node.get("nbElements"))
        probe_args["dim"] = None if node is None else float(node.get("widthIncid"))
        probe_args["inter_elem"] = None if node is None else float(node.get("spaceInterIncid"))
        probe_args["pitch"] = None if node is None else probe_args["dim"] + float(node.get("spaceInterIncid"))

    elif probe_args["tp"] == "matricial":
        # Transdutor matricial
        node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageMatriciel")
        lines = node.find("Lines")
        shape = []
        for line in lines.text.split('\n'):
            shape.append(np.array(line.split(';')[:-1], dtype=np.int16))
        shape = np.array(shape).shape
        probe_args["num_elem"] = shape[0] * shape[1]
        ortho_dim = float(node.get("widthOrtho"))
        incid_dim = float(node.get("widthIncid"))
        probe_args["dim"] = (ortho_dim, incid_dim)
        ortho_inter = float(node.get("spaceInterOrtho"))
        incid_inter = float(node.get("spaceInterIncid"))
        probe_args["inter_elem"] = (ortho_inter, incid_inter)

    elif probe_args["tp"] == "circular":
        # Transdutor circular
        node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageElliptique")
        rings = node.findall("Piste")
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
        probe_args["num_elem"] = shape[0]
        probe_args["elem_list"] = elem_list

    elif probe_args["tp"] == "generic":
        node = root.find("ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageExotique")
        nos = node.findall("ListeElements/Element")
        n_elem = nos.__len__()
        probe_args["dim"] = []
        for i_elem in range(n_elem):
            probe_args["dim"].append((float(nos[i_elem].get('dimension1')), float(nos[i_elem].get('dimension2')), 0))
        probe_args["dim"] = np.array(probe_args["dim"])
    else:
        # Tratar outros casos
        pass

    # Tipo, frequência e banda-passante do sinal de excitação do transdutor
    signal_type = {"0": "hanning", "1": "gaussian"}
    node = root.find("ReseauTraducteur/Traducteur/Signal/SignalQuelconque")
    probe_args["pulse_type"] = None if node is None else signal_type.get(node.get("type"))
    probe_args["freq"] = None if node is None else float(node.get("fCentrale"))
    probe_args["bw"] = None if node is None else float(node.get("largeurBande")) / 100.0

    # Guarda o elemento ``shape`` e elimina ele do dicionário
    shape = None
    if probe_args["tp"] != "matricial" and probe_args["tp"] != "circular":
        shape = probe_args.pop("shape") if "shape" in probe_args else None

    # Retira todos os elementos da lista ``probe_args`` cujo valor é ``None``
    probe_args = {k: v for k, v in probe_args.items() if v is not None}

    # Cria objeto com os parâmetros do transdutor
    probe_params = ProbeParams(**probe_args)

    # Ajusta ``shape`` se houver
    if shape is not None:
        probe_params.shape = shape

    # =========== Busca os parâmetros para instanciar ``InspectionParams`` ===========
    inspection_args = dict()

    # Tipo de inspeção
    node = root.find("ModeleControle/Positionnement")
    inspection_args["type_insp"] = None if node is None else node.get("typeCnt")

    # Tipo de captura
    capt_type = {"FMA": "FMC", "Unisequentiel": "sweep", "FTP": "FTP", "newExpert": "Manual",
                 "BalayageSimple": "Eletronic sweep", "ImportedLaws": "ImportedLaws"}
    node = root.find("ModeleReglage/Initialisation")

    inspection_args["type_capt"] = None if node is None else capt_type.get(node.get("choixFonction"))
    if inspection_args["type_capt"] == "FTP" or inspection_args["type_capt"] == "Manual" or inspection_args[
            "type_capt"] == "Eletronic sweep" or inspection_args["type_capt"] == "ImportedLaws":
        check_pwi_capt(inspection_args)

    # Frequência de amostragem
    node = root.find("ReseauTraducteur/Traducteur/Signal/SignalQuelconque")
    inspection_args["sample_freq"] = None if node is None else float(node.get("freqEchant"))

    # Início e fim do *gate*
    # É obtido do binário
    # node = root.find("OptionsCalculMephisto/OptionsMephisto/ParamGates/Gate")
    # inspection_args["gate_start"] = None if node is None else float(node.get("start_us"))
    # inspection_args["gate_end"] = None if node is None else float(node.get("end_us"))

    # Retira todos os elementos da lista ``inspection_args`` cujo valor é ``None``
    inspection_args = {k: v for k, v in inspection_args.items() if v is not None}

    # Cria objeto com os parâmetros da inspeção
    inspection_params = InspectionParams(**inspection_args)

    # Busca o ponto de origem do sistema de coordenadas
    node = root.find("ModeleControle/Positionnement/PosControle/PointCoord/CoordPlanes")
    if node is not None:
        inspection_params.point_origin[0, 0] = float(node.get("x"))
        inspection_params.point_origin[0, 1] = float(node.get("y"))
        inspection_params.point_origin[0, 2] = float(node.get("z"))

    # Ajusta parâmetros específicos para inspeções por contato e por imersão
    if inspection_params.type_insp == "contact":
        # Inspeção por contato
        # Busca o ângulo de incidência
        node = root.find("ModeleControle/Positionnement/PosContact")
        inspection_params.impact_angle = 0 if node is None else float(node.get("orientation"))

        # Define como velocidade da agua a velocidade coupling_cl
        inspection_params.coupling_cl = 1483.0

        # Ponto central do transdutor para o primeiro *step*
        node = root.find("ModeleControle/Positionnement/PosContact/PointCoord/CoordPlanes")
        inspection_params.step_points[0, 0] = 0 if node is None else float(node.get("x"))
        inspection_params.step_points[0, 1] = 0  # if node is None else float(node.get("y"))
        inspection_params.step_points[0, 2] = 0  # CONSIDERAR SEMPRE O PLANO DE CONTATO COMO z=0 ????

    elif inspection_params.type_insp == "immersion":
        # Inspeção por imersão
        # Busca o comprimento da coluna d'água (ou ar, ou sapata, ou ...)
        node = root.find("ModeleControle/Positionnement/PosImmersion")
        inspection_params.water_path = 0 if node is None else float(node.get("Heau"))

        # Busca a velocidade de propagação no meio (água, ar, material da sapata, ...)
        # node = root.find("ModeleControle/MilieuEnvironnant/Material/SimpleMaterial/SimpleParamED/FluidED")
        node = root.find("ModeleControle/MilieuCouplant/Material/SimpleMaterial/SimpleParamED/FluidED")
        inspection_params.coupling_cl = 1483.0 if node is None else float(node.get("PWaveVelocity_ms"))

        # Ponto central do transdutor para o primeiro *step*
        node = root.find("ModeleControle/Positionnement/PosImmersion/PointCoord/CoordPlanes")
        inspection_params.step_points[0, 0] = 0 if node is None else float(node.get("x"))
        inspection_params.step_points[0, 1] = 0 if node is None else float(node.get("y"))
        inspection_params.step_points[0, 2] = 0 if node is None else float(node.get("z"))

    # Busca os ângulos de disparo, caso o ensaio seja por ondas planas
    if inspection_params.type_capt == "PWI":
        # Verifica se existe um arquivo ``law'', indicando que é uma captura ``ImportedLaws''
        if hasattr(inspection_params, "laws_file"):
            # Abre o arquivo ``law'' para obter a quantidade de disparos
            with open(filename + f'/proc0/{inspection_params.laws_file}', "rb") as f_law:
                # Go to the end of the file before the last break-line
                f_law.seek(-2, os.SEEK_END)

                # Keep reading backward until you find the next break-line
                while f_law.read(1) != b'\n':
                    f_law.seek(-2, os.SEEK_CUR)

                # Pega a quantidade de disparos no terceiro elemento da última linha
                n_trigger = int(f_law.readline().decode().split()[2]) + 1
                inspection_params.angles = np.arange(n_trigger)

        # Agora, verifica se existe a informação dos ângulos de disparo
        node = root.find("ModeleReglage/Comportement/FocalisationBalayageAng")
        if node:
            theta_i = float(node.get("angleDepart"))
            theta_f = float(node.get("angleArrivee"))
            theta_n = int(node.get("nbPas"))
            inspection_params.angles = np.linspace(theta_i, theta_f, theta_n + 1)

    # Ajusta a coordenada de origem como a coordenada do centro do transdutor no primeiro *step*.
    inspection_params.point_origin = inspection_params.step_points[0, :]

    # Calcula o numero de A-scans presentes na inspeção
    try:
        if inspection_params.type_capt == "FMC":
            n_ascans = n_pas * probe_params.num_elem ** 2
        elif inspection_params.type_capt == "PWI":
            # Busca o tipo de armazenamento (Somatório, Somatório+Canais ou Canais)
            node = root.find("OptionsCalculMephisto/OptionsMephisto/ParamGates/Gate")
            inspection_args["type_stockage"] = None if node is None else node.get("typeStockage")

            # Número de ascans dos canais
            n_ascans = n_pas * probe_params.num_elem * len(inspection_params.angles)

            # Número de ascans da soma dos canais
            n_ascans_sum = n_pas * len(inspection_params.angles)

    except AttributeError:
        n_ascans = n_pas

    # Verifica se foi passada uma lista de *shots* para leitura.
    if sel_shots is None:
        # É para ler todos os *shots*.
        sel_shots = range(n_pas)
    else:
        if type(sel_shots) is int:
            # É somente um shot, cria uma lista com esse índice.
            sel_shots = [sel_shots]
        elif type(sel_shots) is list:
            # É uma lista, não faz nada.
            # sel_shots.sort()
            pass
        elif type(sel_shots) is range:
            # É um ``range``, não faz nada.
            pass
        else:
            # Tipo inválido.
            raise TypeError("``sel_shots`` deve ser um inteiro, uma lista ou um range().")

    # Verifica se os *shots* selecionados estão no arquivo de configuração
    for i in sel_shots:
        if i in range(n_pas):
            new_point = np.zeros((1, 3))
            new_point[0, 0] = float((n_pas_x - i % n_pas_x - 1) if (i // n_pas_x) % 2 else (i % n_pas_x)) * pas_x
            new_point[0, 1] = (i // n_pas_x) * pas_y
            new_point = new_point + inspection_params.step_points[0]
            try:
                inspection_params.step_points[i, :] = new_point
            except IndexError:
                inspection_params.step_points = np.concatenate((inspection_params.step_points, new_point))
        else:
            raise IndexError(str(i) + " não é um índice possível para ``sel_shots``.")

    # Ajusta as coordenadas dos centros dos elementos emissores em relação a coordenada central do transdutor.
    # Isso é sempre relativo ao primeiro *shot*.

    if inspection_params.type_capt == "Eletronic sweep":
        # Corrige para que o centro do transdutor esteja em [0,0,0]
        starting_pos = [-probe_params.num_elem * probe_params.pitch / 2 + probe_params.elem_dim / 2, 0, 0]
        diff = inspection_params.step_points[0, :] - starting_pos
        for i in range(n_pas):
            inspection_params.step_points[i, :] = inspection_params.step_points[i, :] - diff
            probe_params.elem_center[i, :] = inspection_params.step_points[i, :]
            probe_params.elem_center[i, 1] = 0

        # Salva o sweep eletronico como sweep normal, com trandutor mono
        inspection_params.type_capt = "sweep"
        probe_params.type_probe = "mono"

    elif probe_params.type_probe == "mono":
        # Corrige para que o shot central esteja em [0,0,0]
        starting_pos = [-(n_pas - 1) * pas_x / 2, 0, 0]
        diff = inspection_params.step_points[0, :] - starting_pos
        for i in range(n_pas):
            inspection_params.step_points[i, :] = inspection_params.step_points[i, :] - diff
            probe_params.elem_center[i, :] = inspection_params.step_points[i, :]
            probe_params.elem_center[i, 1] = 0

    elif probe_params.type_probe == "generic":
        nos = root.findall(
            "ReseauTraducteur/Traducteur/DecoupagePastille/DecoupageMulti/DecoupageExotique/ListeElements/Element")
        if nos.__len__() > 0:
            probe_params.num_elem = nos.__len__()
            probe_params.elem_center = np.zeros((probe_params.num_elem, 3))
            for i_elem in range(probe_params.num_elem):
                probe_params.elem_center[i_elem, 0] = float(nos[i_elem].get('positionX'))
                probe_params.elem_center[i_elem, 1] = float(nos[i_elem].get('positionY'))
        else:
            # Compensa elem_dim / 2 para as coordenadas ficarem simétricas em relação ao centro do transdutor
            width_probe = probe_params.num_elem * probe_params.pitch
            elem_dim = probe_params.elem_dim
            probe_params.elem_center[:, 0] = probe_params.elem_center[:, 0] - width_probe / 2 + elem_dim / 2

    # elif probe_params.type_probe == 'matricial':
    #    probe_params.elem_center = np.zeros((*shape, 3))
    #    for i in range(len(probe_params.elem_center[:, 0, 0])):
    #        probe_params.elem_center[i, :, 0] = (ortho_inter + ortho_dim) * i
    #    for i in range(len(probe_params.elem_center[:, 0, 0])):
    #        probe_params.elem_center[:, i, 1] = (incid_inter + incid_dim) * i
    #
    #    central_idx = (np.array(probe_params.elem_center[:, :, 0].shape) - 1) // 2
    #    probe_params.elem_center -= probe_params.elem_center[central_idx[0], central_idx[1], :]
    #    probe_params.elem_center = probe_params.elem_center.reshape(shape[0]*shape[1], 3)

    elif probe_params.type_probe == 'matricial':
        probe_params.elem_center = np.zeros((*shape, 3))
        for i in range(len(probe_params.elem_center[0, :, 0])):
            probe_params.elem_center[:, i, 1] = (ortho_inter + ortho_dim) * i
        for i in range(len(probe_params.elem_center[:, 0, 0])):
            probe_params.elem_center[i, :, 0] = (incid_inter + incid_dim) * i

        elem_center_mean = probe_params.elem_center.mean((0, 1))
        probe_params.elem_center = probe_params.elem_center - elem_center_mean

        probe_params.elem_center = probe_params.elem_center.reshape(shape[0] * shape[1], 3)

    elif probe_params.type_probe == 'circular':
        probe_params.elem_center = np.zeros((shape[0], 3))
        for i in range(len(elem_list)):
            elem_dict = elem_list[i]
            x_center = np.cos(elem_dict["center_angle"]) * elem_dict["center_radius"]
            y_center = np.sin(elem_dict["center_angle"]) * elem_dict["center_radius"]
            probe_params.elem_center[i, :] = np.array([x_center, y_center, 0.0])

        elem_center_mean = probe_params.elem_center.mean((0, 1))
        probe_params.elem_center = probe_params.elem_center - elem_center_mean

    elif probe_params.type_probe == "linear":
        width_probe = probe_params.num_elem * probe_params.pitch
        inter_elem = probe_params.inter_elem

        # Compensa elem_dim / 2 para as coordenadas ficarem simétricas em relação ao centro do transdutor
        probe_params.elem_center[:, 0] = probe_params.elem_center[:, 0] - width_probe / 2 + inter_elem / 2

    if read_ascan:
        # =========== Faz a leitura dos arquivos binários do CIVA ===========
        # Abre o binário do primeiro *gating*
        if os.path.exists(filename + f'/proc0/channels_signal_{app}_gate_1'):
            if inspection_params.type_capt == 'FMC' and probe_params.type_probe != 'matricial' \
                    and probe_params.type_probe != 'generic':
                [ascan_data, gating] = civa_fmc(filename + f'/proc0/channels_signal_{app}_gate_1',
                                                n_ascans, n_pas, sel_shots)

            elif inspection_params.type_capt == 'PWI':
                [ascan_data, gating] = civa_pwi(filename + f'/proc0/channels_signal_{app}_gate_1',
                                                len(inspection_params.angles), probe_params.num_elem, n_pas, sel_shots)

            elif inspection_params.type_capt == 'sweep':
                [ascan_data, gating] = civa_bscan(filename + f'/proc0/channels_signal_{app}_gate_1', n_pas, sel_shots)

            elif probe_params.type_probe == 'matricial':
                n_ascans = probe_args["num_elem"] ** 2
                [ascan_data, gating] = civa_matricial(filename + f'/proc0/channels_signal_{app}_gate_1',
                                                      n_ascans, n_pas, sel_shots)

            elif probe_params.type_probe == 'generic':
                n_ascans = probe_params.num_elem ** 2
                [ascan_data, gating] = civa_matricial(filename + f'/proc0/channels_signal_{app}_gate_1',
                                                      n_ascans, n_pas, sel_shots)

        if os.path.exists(filename + f'/proc0/sum_signal_{app}_gate_1'):
            if inspection_params.type_capt == 'PWI':
                [ascan_data_sum, gating_sum] = civa_pwi(filename + f'/proc0/sum_signal_{app}_gate_1',
                                                        len(inspection_params.angles), 1, n_pas, sel_shots)

            elif inspection_params.type_capt == 'sweep':
                [ascan_data_sum, gating_sum] = civa_bscan(filename + f'/proc0/sum_signal_{app}_gate_1', n_pas,
                                                          sel_shots)

        # Início e fim do *gate*
        if 'gating' in locals():
            inspection_params.gate_start = gating[0] / inspection_args["sample_freq"]
            inspection_params.gate_end = gating[1] / inspection_args["sample_freq"]
            inspection_params.gate_samples = gating[1] - gating[0]
        elif 'gating_sum' in locals():
            inspection_params.gate_start = gating_sum[0] / inspection_args["sample_freq"]
            inspection_params.gate_end = gating_sum[1] / inspection_args["sample_freq"]
            inspection_params.gate_samples = gating_sum[1] - gating_sum[0]

    # =========== Cria uma instância do objeto ``DataInsp`` ===========
    dados = DataInsp(inspection_params, specimen_params, probe_params)
    if read_ascan:
        if 'ascan_data' in locals():
            dados.ascan_data = ascan_data
        else:
            dados.ascan_data = None

        if 'ascan_data_sum' in locals():
            dados.ascan_data_sum = np.reshape(ascan_data_sum, (ascan_data_sum.shape[0],
                                                               ascan_data_sum.shape[1],
                                                               ascan_data_sum.shape[3]))

    return dados
