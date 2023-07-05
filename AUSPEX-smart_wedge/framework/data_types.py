# -*- coding: utf-8 -*-
"""
Módulo ``data_types``
=====================

O módulo :mod:`.data_types` contém classes que definem estruturas de dados
que são utilizadas pelo :mod:`framework`. As estruturas de
dados são utilizadas para armazenar dados de inspeção (como parâmetros 
de inspeção, da peça e do transdutor), gerar regiões de interesse (ROI -
*Region of Interest*) e armazenar resultados de algoritmos de imageamento.

Parâmetros de inspeção
----------------------

A classe :class:`InspectionParams` define uma estrutura de dados para
armazenar os diversos parâmetros relacionados a um procedimento de
inspeção. 

O processo de inspeção pode ser categorizado de duas formas: ensaios
por contato e ensaios por imersão. A classe :class:`InspectionParams`
permite definir o tipo de ensaio por meio do atributo
:attr:`InspectionParams.type_insp`, que pode ser ``contact`` ou
``immersion``. Além disso, o ensaio pode ser realizado com tipos de captura
diferentes, definidos pelo atributo :attr:`InspectionParams.type_capt`. Os
tipos de captura podem ser ``sweep``, para transdutores mono; ``FMC``, para
aquisição com uma *array* linear; e ``PWI``, para ensaios com ondas planas
(*plane waves*).

No ensaio por contato, o transdutor pode ser posicionado junto a peça ou
acoplado por meio de uma sapata. A :numref:`fig_framework_nde_contact` mostra
o primeiro caso. Conforme indicado na figura, o transdutor é posicionado
paralelamente à superfície da peça sob inspeção, sendo que as ondas emitidas
pelo transdutor incidem na peça com um ângulo normal a sua superfície.

.. figure:: figures/framework/nde-contact.*
    :name: fig_framework_nde_contact
    :width: 35 %
    :align: center

    Ensaio por contato.

A :numref:`fig_framework_nde_contact_wedge` mostra o caso em que o transdutor
é acoplado ao objeto sob inspeção por meio de uma sapata. A sapata tem como
objetivo conduzir as ondas sonoras emitidas pelo transdutor, transmitindo-as
para a peça. A partir da :numref:`fig_framework_nde_contact_wedge`, é possível
notar que, no caso do contato pela sapata, as ondas incidentes na peça possuem
um ângulo em relação à normal da superfície. Além disso, a onda conduzida
pela sapata sofre uma refração ao passar para a peça, devido a diferença de
material entre a peça e a sapata. A classe :class:`InspectionParams` permite
definir o ângulo de incidência quando a inspeção é realizada por meio de uma
sapata, com o atributo :attr:`InspectionParams.impact_angle`.

.. figure:: figures/framework/nde-contact-wedge.*
    :name: fig_framework_nde_contact_wedge
    :width: 35 %
    :align: center

    Ensaio por contato com sapata.
    
Por outro lado, na inspeção por imersão, o transdutor é acoplado à peça
por um meio acoplante, comumente água, conforme indicado na
:numref:`fig_framework_nde_immersion`. A figura mostra um objeto e o
transdutor submersos em água, sendo que o transdutor está posicionado
a uma altura :math:`h` da superfície da peça.

.. figure:: figures/framework/nde-immersion_v2.*
    :name: fig_framework_nde_immersion
    :width: 35 %
    :align: center

    Ensaio por imersão.

No caso em que o ensaio é do tipo imersão em água, é possível definir dois
parâmetros adicionais, que são a velocidade do som na água e o tamanho da
coluna d'água que separa o transdutor da superfície da peça. Esses parâmetros
são definidos nos atributos :attr:`InspectionParams.coupling_cl` e
:attr:`InspectionParams.water_path`.

A classe :class:`InspectionParams` permite armazenar as coordenadas
da posição inicial do transdutor no processo de inspeção, por meio
do atributo :attr:`InspectionParams.point_origin`, conforme indicado
na :numref:`fig_framework_coord_ref`. A figura mostra a posição inicial de um
transdutor sob uma peça, que possui sua origem no canto superior esquerdo.
Nesse caso, o transdutor está posicionado em uma posição
(:math:`x`, :math:`y`, :math:`z`) da origem da peça, que é utilizado como
ponto de referência. 

.. figure:: figures/framework/coord_ref.*
    :name: fig_framework_coord_ref
    :width: 30 %
    :align: center

    Coordenadas de referência para a inspeção.

As posições de inspeção são definidas pelo atributo
:class:`InspectionParams.step_points`, que é uma matriz com os
deslocamentos percorridos pelo transdutor durante o processo de inspeção.
Cada linha da matriz representa um deslocamento, em relação a posição
inicial do transdutor.

A classe também permite armazenar aspectos eletrônicos do processo de
inspeção. É possível definir a frequência de amostragem, com a qual a
aquisição de dados do transdutor é realizada. Além disso, a classe define
o período de amostragem, sendo o inverso da frequência de amostragem. Esses
parâmetros podem ser acessados a partir dos atributos
:attr:`InspectionParams.sample_freq` e :attr:`InspectionParams.sample_time`.

As informações de *gate* podem ser acessadas a partir dos atributos
:attr:`InspectionParams.gate_start`, :attr:`InspectionParams.gate_end`
e :attr:`InspectionParams.gate_samples`. Os parâmetros de *gate* definem o
início e fim da aquisição dos dados do transdutor, bem como a quantidade
de amostras tomadas nesse intervalo de tempo. Dessa forma, é possível
definir uma janela para aquisição dos dados provenientes do transdutor,
o que permite rejeitar dados considerados irrelevantes para o
processamento.

A :numref:`fig_framework_gate_example` mostra um exemplo de aquisição de
dados de um transdutor. Quando o transdutor emite um sinal ultrassônico, 
parte da onda é refletida pela superfície da peça sob inspeção. Essa
reflexão pode ser vista nos dados de A-scan, conforme ilustrado na
:numref:`fig_framework_gate_example`, em que os dados nos primeiros
microssegundos da aquisição são provenientes da reflexão da superfície
da peça. Esses dados podem não carregar informações relevantes para o
processamento de dados e posterior reconstrução de uma imagem. Dessa
forma, é possível definir um tempo inicial e um tempo final no qual o
sinal do transdutor será amostrado. No caso da
:numref:`fig_framework_gate_example`, esse intervalo de dados pode estar
compreendido entre 10 us e 25 us. Com isso, o valor inicial e final do
*gate* poderia ser 10 us e 25 us, respectivamente.

.. figure:: figures/framework/gate_example.*
    :name: fig_framework_gate_example
    :width: 50 %
    :align: center

    Coordenadas de referência para a inspeção.

Embora o *gate* possui três parâmetros, a partir de dois parâmetros
é possível determinar o terceiro. Por exemplo, determinado o valor inicial e
final do *gate*, é possível obter o número de amostras, uma vez que a
frequência de amostragem é fixa. Da mesma forma, com o valor inicial
do *gate* e a quantidade de amostras, é possível determinar o valor
final. Na implementação atual do :mod:`framework`, o tratamento automático
dos parâmetros de *gate* está sendo definido.

Para ensaios utilizando ondas planas, o atributo
:attr:`InspectionParams.angles` é utilizado para armazenar os ângulos de
disparo utilizados no ensaio. Esse atributo é apenas utilizado quando o tipo
de captura é ``PWI``, sendo atribuído o valor de zero para outros tipos de
captura.

Parâmetros da peça
------------------
A classe :class:`SpecimenParams` permite definir parâmetros da peça sob
inspeção.

Atualmente, é possível armazenar a velocidade das ondas longitudinais
e transversais e a rugosidade da peça, com os atributos
:attr:`SpecimenParams.cl`, :attr:`SpecimenParams.cs` e
:attr:`SpecimenParams.roughness`, respectivamente. 

Parâmetros do transdutor
------------------------
O transdutor utilizado em um ensaio possui diversos parâmetros, que definem
aspectos geométricos e elétricos. Esses parâmetros podem ser definidos e
armazenados com a classe :class:`ProbeParams`.

Entre os diversos tipos de transdutores disponíveis, a implementação atual
permite definir um transdutor como sendo do tipo *mono* ou do tipo *array*.
O tipo do transdutor pode ser armazenado e acessado a partir do atributo
:attr:`ProbeParams.type_probe`.

O transdutor mono consiste em apenas um elemento, de material piezoelétrico,
capaz de emitir ondas e receber os sinais de eco. Entre as diversas formas
possíveis, a implementação atual considera que o transdutor pode ter um
formato retangular ou circular, definido a partir do atributo
:attr:`ProbeParams.shape`. Dependendo do formato do transdutor, as dimensões
para caracterizá-lo são diferentes. Se o transdutor for do tipo retangular,
sua caracterização é feita a partir de seu comprimento e da sua largura. Se
o elemento for circular, o raio é suficiente para caracterizá-lo. O atributo
:attr:`ProbeParams.elem_dim` define as dimensões do transdutor, podendo ser
uma tupla ou um número, dependendo da geometria do transdutor.

Um transdutor do tipo *array* linear é composto por diversos elementos,
dispostos lado a lado. Embora a disposição dos elementos pode assumir
diversas formas, a implementação atual considera apenas transdutores do tipo
*array* com elementos retangulares. A :numref:`fig_framework_linear_array`
ilustra um transdutor do tipo *array* linear, composto por elementos
retangulares, indicando seus principais parâmetros. Considera-se que todos
os elementos possuem o mesmo comprimento :math:`L` e a mesma largura
:math:`d`, enquanto a espessura é desconsiderada. Os elementos são separados
por uma distância :math:`g`, enquanto a distância entre o centro de um
elemento para o centro do próximo elemento é :math:`p`. A distância centro a
centro também é conhecida como *pitch*.

.. figure:: figures/framework/linear_array.*
    :name: fig_framework_linear_array
    :width: 50 %
    :align: center

    Transdutor do tipo *array* linear.

A quantidade de elementos do transdutor do tipo *array* é definido pelo
atributo :attr:`ProbeParams.num_elem`. O espaçamento entre os elementos,
a distância centro a centro e a largura de cada elemento podem
ser definidos com os atributos :attr:`ProbeParams.inter_elem`,
:attr:`ProbeParams.pitch` e :attr:`ProbeParams.elem_dim`, respectivamente.

Para os algoritmos de imageamento, pode ser mais conveniente trabalhar com
coordenadas relativas à posição do transdutor. Dessa forma, a classe
:class:`ProbeParams` permite definir as coordenadas do centro geométrico
do transdutor, em relação à peça, a partir do atributo
:attr:`ProbeParams.elem_center`. No caso do transdutor mono, as
coordenadas referem-se ao seu centro geométrico, conforme indicado na
:numref:`fig_framework_elem_center`, que mostra a vista superior de um
transdutor mono posicionado sob uma peça. O centro geométrico do transdutor
está localizado em uma posição (:math:`x`, :math:`y`, :math:`z`), que é usada
como ponto de referência nos algoritmos de imageamento.

.. figure:: figures/framework/elem_center.*
    :name: fig_framework_elem_center
    :width: 30 %
    :align: center

    Atributo :attr:`ProbeParams.elem_center` para transdutor mono.

No caso de um transdutor do tipo *array*, :attr:`ProbeParams.elem_center` é um
*array* contendo as coordenadas dos centros geométricos de todos os elementos,
sendo que essas coordenadas são relativas ao centro geométrico do transdutor. A
:numref:`fig_framework_elem_center_array` ilustra um transdutor do tipo
*array* linear sob uma peça, em que o centro geométrico do primeiro elemento
está em uma posição (:math:`x`, :math:`y`, :math:`z`).

.. figure:: figures/framework/elem_center_array.*
    :name: fig_framework_elem_center_array
    :width: 30 %
    :align: center

    Atributo :attr:`ProbeParams.elem_center` para transdutor *array* linear.

A classe :class:`ProbeParams` permite definir aspectos elétricos do transdutor,
como sua frequência central, banda passante e tipo de pulso de excitação.

A frequência do transdutor refere-se a frequência de ressonância do
cristal piezoelétrico utilizado. Ao ser excitado com um impulso, o elemento
piezoelétrico irá vibrar, emitindo ondas sonoras na sua frequência de
ressonância. A frequência do transdutor pode ser definida a partir do
parâmetro :attr:`ProbeParams.central_freq`.

A largura de banda do transdutor está relacionada com sua sensibilidade.
Transdutores com uma largura de banda maior apresentam maior sensibilidade,
uma vez que conseguem detectar uma maior gama de frequências. A largura
de banda de um transdutor é geralmente definida como sendo um percentual
da frequência central. O atributo :attr:`ProbeParams.bw` define a banda
passante do transdutor.

O sinal de excitação do transdutor define como será a onda sonora emitida,
fazendo parte de seu modelo. Na implementação atual, é possível definir
o pulso de excitação como sendo ``gaussian``, ``cossquare``, ``hanning``
e ``hamming``, a partir do atributo :attr:`ProbeParams.pulse_type`.

.. **Preciso estudar mais sobre esses aspectos elétricos para documentá-los.**

Dados de inspeção
-----------------
A classe :class:`DataInsp` apresenta todos os dados relacionados a um
processo completo de ensaio não-destrutivo. A classe contém os parâmetros
do procedimento, armazenando os parâmetros da peça (:class:`SpecimenParams`),
do transdutor (:class:`ProbeParams`) e da inspeção
(:class:`InspectionParams`).

Além dos parâmetros relacionados ao procedimento de inspeção, a classe ainda
possui os dados de A-scan, *grid* de tempo dos dados de A-scan e resultados
dos algoritmos de imageamento.

Os dados de A-scan são genericamente representados por uma matriz de 4
dimensões, considerando os dados de amplitude, sequência de disparo e de
recepção dos elementos do transdutor, bem como as posições do transdutor
na peça. Os dados podem ser acessados a partir do atributo
:attr:`DataInsp.ascan_data`.

Considere a aquisição de dados do caso em que o sistema é o pulso-eco,
conforme indicado na :numref:`fig_framework_pulse_echo_ascan`. Nesse caso, é
possível emitir uma onda sonora e monitorar as ondas refletidas, obtendo
um vetor de dados da amplitude do sinal do transdutor no tempo, o A-scan.
Deslocando o transdutor e realizando novamente o processo de aquisição,
obtém-se um novo vetor de dados de A-scan, sendo possível combinar os dois
vetores obtidos em uma matriz de duas dimensões. Nesse caso, a matriz
possui tantas linhas quanto amostras do sinal de amplitude do transdutor
e tantas colunas quanto a quantidade de posições em que a aquisição do
sinal foi realizada. A :numref:`fig_framework_pulse_echo_ascan` indica esse
processo, em que o transdutor realiza a aquisição do sinal em *n* posições e,
como resultado, uma matriz de duas dimensões é obtida, sendo que cada
coluna da matriz representa uma posição do transdutor.

.. figure:: figures/framework/single_element_v3.*
    :name: fig_framework_pulse_echo_ascan
    :width: 60 %
    :align: center

    Aquisição com o sistema pulso-eco.

É possível estender essa matriz de aquisição para o caso em que o transdutor
não é composto por um único elemento, mas por uma matriz de elementos. A
:numref:`fig_framework_fmc_ascan` indica o processo de aquisição para um
transdutor que possui diversos elementos e realiza a aquisição de dados
em uma única posição. No caso em que há um disparo do transdutor, acionando
um ou mais de seus elementos, é possível obter uma matriz de dados similar
a matriz indicada na :numref:`fig_framework_pulse_echo_ascan`. Porém, enquanto
que na :numref:`fig_framework_pulse_echo_ascan` cada coluna da matriz representa
uma posição diferente do transdutor, nesse caso cada coluna da matriz
representa um elemento diferente, dispostos em posições diferentes.
Dessa forma, a matriz indicada na :numref:`fig_framework_pulse_echo_ascan`
é obtida a partir do movimento do transdutor, enquanto que uma matriz
similar pode ser obtida com apenas um disparo do transdutor no caso
em que o transdutor é composto por diversos elementos. Se um novo
disparo é realizado, uma nova matriz de dados é obtida, sendo possível formar
um cubo com os dados de cada disparo, conforme indicado na
:numref:`fig_framework_fmc_ascan`.

.. figure:: figures/framework/phase_array_v3.*
    :name: fig_framework_fmc_ascan
    :width: 60 %
    :align: center

    Aquisição com um transdutor composto por diversos elementos.


O transdutor contendo múltiplos elementos pode ainda deslocar-se e realizar
um novo procedimento de aquisição de dados. Nesse caso, é possível condensar
os dados de aquisição em uma matriz de 4 dimensões, em que a última dimensão
representa a posição do transdutor. 


A classe :class:`DataInsp` possui também um vetor contendo os instantes de
tempo em que os sinais de A-scan foram obtidos. Como a frequência de
amostragem é fixa, os transdutores são amostrados no mesmo instante e,
por isso, o vetor de tempo é comum para todas as aquisições de A-scan. O
vetor de tempo pode ser acessado a partir do atributo
:attr:`DataInsp.time_grid`.

Os resultados de imageamento podem ser salvos juntamente aos dados de
inspeção e A-scan, sendo acessados pelo atributo
:attr:`DataInsp.imaging_results`.

Região de Interesse
-------------------
A classe :class:`ImagingROI` permite criar e armazenar parâmetros referentes
à região de interesse (ROI) para a reconstrução de imagens. Na implementação
atual, a classe permite a criação de ROIs de duas dimensões. 

Considerando um processo de inspeção, a ROI é uma região na qual se deseja
realizar a aquisição e/ou processamento dos dados, sendo que a ROI do
processo de inspeção pode ser diferente da ROI do processamento de dados.

A :numref:`fig_framework_nde_contact_roi` mostra um processo de inspeção, com um
transdutor do tipo mono e com uma ROI definida. O transdutor realiza a
aquisição dos dados de forma a cobrir toda a ROI. No caso da
:numref:`fig_framework_nde_contact_roi`, a ROI compreende apenas uma parte da
peça, implicando no movimento do transdutor, que será restrito, e
na janela de aquisição de dados, que deve ignorar os sinais de eco
recebidos antes e após a região de interesse. 

.. figure:: figures/framework/nde-contact-roi.*
    :name: fig_framework_nde_contact_roi
    :width: 30 %
    :align: center

    Região de interesse para um ensaio.

A ROI pode ser definida para a reconstrução da imagem, podendo ser diferente
da ROI definida para a aquisição de dados. No exemplo da
:numref:`fig_framework_nde_contact_roi`, a falha a ser detectada está apenas
em uma parte da ROI de aquisição. Uma vez identificado o ponto de interesse
para o processamento e reconstrução da imagem, é possível definir uma nova
ROI, exclusiva para os algoritmos de imageamento. A
:numref:`fig_framework_nde_contact_roi_proc` ilustra uma nova ROI, utilizada
para a reconstrução da imagem naquela região da peça. Como o processamento
dos dados é realizado de forma a gerar uma imagem, a ROI para o processamento
de dados consiste em uma grade, em que cada ponto da grade representa a
posição de um pixel.

.. figure:: figures/framework/nde-contact-roi-proc.*
    :name: fig_framework_nde_contact_roi_proc
    :width: 30 %
    :align: center

    Região de interesse para processamento.

Para definir uma região de interesse, é necessário informar a altura
:math:`h` e a largura :math:`w`, bem como a quantidade de pontos
em cada dimensão. A :numref:`fig_framework_roi_def` ilustra como a ROI é
formada, indicando a grade e a disposição dos pixels na grade. No eixo
vertical, cada ponto (ou pixel) da ROI está separado por uma distância
:math:`h/m`, assim como cada ponto no eixo horizontal está separado por
uma distância :math:`w/n`.

.. figure:: figures/framework/roi-def.*
    :name: fig_framework_roi_def
    :width: 50 %
    :align: center

    Região de interesse para processamento.
    
A classe :class:`ImagingROI` permite criar uma ROI, informando os parâmetros
``height``, ``h_len``, ``width`` e ``w_len``. Os parâmetros ``height`` e
``h_len`` definem a altura  :math:`h` e a quantidade de pixels nessa dimensão.
De maneira análoga, os parâmetros ``width`` e ``w_len`` definem a largura
:math:`w` e a quantidade de pixels, na direção da dimensão da largura.
Esses parâmetros ficam armazenados no objeto da ROI e podem ser acessados a
partir dos atributos :attr:`ImagingROI.height`, :attr:`ImagingROI.h_len`,
:attr:`ImagingROI.width` e :attr:`ImagingROI.w_len`.

Outro parâmetro necessário para a definição de uma ROI é a sua posição na
peça, representada por coordenadas. Esse parâmetro é informado na criação de
um objeto da ROI e pode ser acessado posteriormente pelo atributo
:attr:`ImagingROI.coord_ref`.

Após definido a ROI, com suas dimensões e posição na peça, as coordenadas
absolutas de cada ponto da malha  podem ser acessados por meio do método
:meth:`ImagingROI.get_coord`, que retorna uma matriz de 3 dimensões com
as coordenadas de cada ponto da ROI. 

Resultados de reconstrução
--------------------------
A classe :class:`ImagingResult` é utilizada para armazenar as imagens
reconstruídas a partir dos algoritmos de imageamento. Os resultados
da reconstrução podem ser resumidos à imagem gerada e a ROI.

O atributo :attr:`ImagingResult.image` fornece uma matriz do tipo
:class:`np.ndarray` para armazenar a imagem reconstruída. O
tamanho da imagem depende do tamanho da ROI.

A ROI na qual a imagem foi reconstruída também é armazenada no objeto
da classe, no atributo :attr:`ImagingResult.roi`.

Além da imagem e da ROI, é possível definir uma descrição do resultado,
em forma de texto, com o atributo :attr:`ImagingResult.description`.

.. raw:: html

    <hr>

"""
import numpy as np
from enum import Enum


class InspectionParams:
    """Classe contendo os parâmetros referentes ao processo de inspeção.

    Parameters
    ----------
        type_insp : str
            Tipo de inspeção. Apresenta dois valores possíveis: ``immersion``
            ou ``contact``. O valor padrão é ``immersion``.

        type_capt : str
            Tipo de captura. Indica o tipo de captura dos sinais ``A-scan``.
            Aqui é necessário verificar os tipos configurados no CIVA.
            Atualmente, os valores possíveis são: ``sweep``, ``FMC`` e
            ``PWI``. O valor padrão é ``FMC``.

        sample_freq : int, float
            Frequência de amostragem dos sinais ``A-scan``, em MHz. Por
            padrão, é 100 MHz.

        gate_start : int, float
            Valor inicial do *gate*, em us. Por padrão, é 0.

        gate_end : int, float
            Valor final do *gate*, em us. Por padrão, é 30.

        gate_samples : int
            Número de amostras por canal da aquisição. Por padrão, é 3000.

    Attributes
    ----------
        type_insp : str
            Tipo de inspeção. Apresenta dois valores possíveis: ``immersion``
            ou ``contact``. O valor padrão é ``immersion``.

        type_capt : str
            Tipo de captura. Indica o tipo de captura dos sinais ``A-scan``.
            Aqui é necessário verificar os tipos configurados no CIVA.
            Atualmente, os valores possíveis são: ``sweep``, ``FMC`` e ``PWI``.
            O valor padrão é ``FMC``.

        point_origin : :class:`np.ndarray`
            Posição no espaço indicando a origem do sistemas de coordenadas
            para a inspeção. Todas as outras posições de pontos são
            relativas a este ponto no espaço. Os pontos cartesianos são
            vetores-linhas, em que a primeira coluna é a coordenada
            :math:`x`, a segunda coluna é a coordenada :math:`y` e a terceira
            coluna é a coordenada :math:`z`.

        step_points : :class:`np.ndarray`
            Matriz com as coordenadas do transdutor durante a inspeção.
            Cada linha dessa matriz corresponde a posição do transdutor
            e equivale a um elemento na dimensão ``passo`` do *array*
            :attr:`DataInsp.ascan_data` em :class:`DataInsp`.

        water_path : int, float
            Comprimento da coluna d'água. Exclusivo para inspeções do tipo
            ``immersion``.

        coupling_cl : int, float
            Velocidade de propagação do som no acoplante, em m/s. Exclusivo
            para inspeções do tipo ``immersion``.

        impact_angle : int, float
            Ângulo de incidência. Exclusivo para inspeções do tipo ``contact``.

        sample_freq : int, float
            Frequência de amostragem dos sinais ``A-scan``, em MHz.

        sample_time : int, float
            Período de amostragem dos sinais ``A-scan``, em us.

        gate_start : int, float
            Valor inicial do *gate*, em us.

        gate_end : int,float
            Valor final do *gate*, em us.

        gate_samples : int, float
            Número de amostras por canal da aquisição.

        angles : :class:`np.ndarray`
            Ângulos de disparo para ensaios com ondas planas, em graus.

    """

    def __init__(self, type_insp="immersion", type_capt="FMC", sample_freq=100.,
                 gate_start=0.0, gate_end=30.0, gate_samples=3000, **kwargs):

        # Atribuição dos atributos da instância.
        # Tipo de inspeção. Apresenta dois valores possíveis: ``immersion``
        # ou ``contact``. O valor padrão é ``immersion``.
        self.type_insp = type_insp

        # Tipo de captura. Indica o tipo de captura dos sinais ``A-scan``.
        # Aqui é necessário verificar os tipos configurados no CIVA. Os
        #  valores possíveis nessa primeira versão são: ``sweep`` e ``FMC``.
        # O valor padrão é ``FMC``.
        self.type_capt = type_capt

        # Posição no espaço indicando a origem do sistemas de coordenadas
        # para a inspeção. Todas as outras posições de pontos são
        # relativas a este ponto no espaço. Os pontos cartesianos são
        # vetores-linhas, em que a primeira coluna é a coordenada **x**,
        # a segunda coluna é a coordenada **y** e a terceira coluna é a coordenada **z**.
        self.point_origin = np.zeros((1, 3))

        # Matriz com as coordenadas do transdutor durante a inspeção.
        # Cada linha dessa matriz corresponde a posição do transdutor
        # e equivale a um elemento na dimensão ``passo`` do *array*
        # :attr:`DataInsp.ascan_data` em :class:`DataInsp`.
        self.step_points = np.zeros((1, 3))

        # Parâmetros exclusivos para inspeções do tipo ``immersion``
        if self.type_insp == "immersion":
            # Comprimento da coluna d'água.
            self.water_path = 0
            # Velocidade de propagação do som na água.
            self.coupling_cl = 1483.0
            self.impact_angle = None
        else:
            # Parâmetros exclusivos para inspeções do tipo ``contact``.
            # Ângulo de incidência.
            self.impact_angle = 0
            self.water_path = None
            self.coupling_cl = None

        # Frequência de amostragem dos sinais ``A-scan``, em MHz.
        self.sample_freq = sample_freq

        # Período de amostragem dos sinais ``A-scan``, em us.
        self.sample_time = 1 / sample_freq

        # FALTA A DEFINIÇÃO DO GATE (temporária) .....
        self.gate_start = gate_start
        self.gate_end = gate_end
        self.gate_samples = gate_samples

        # Ângulos de disparo caso o ensaio seja por ondas planas
        if self.type_capt == "PWI":
            self.angles = np.zeros(1)
            if "laws_file" in kwargs.keys():
                self.laws_file = kwargs["laws_file"]

        if "gains" in kwargs.keys():
            self.gain_hw = kwargs["gains"]["Balayage"]  # Salva o ganho do amplificador (hardware)
            self.gain_sw = kwargs["gains"]["Numerique"]  # Salva o ganho digital (soft)
        else:
            self.gain_hw = 0.0
            self.gain_sw = 0.0


class ElementGeometry(Enum):
    CIRCULAR = 1
    RECTANGULAR = 2


class SpecimenParams:
    """Classe contendo os parâmetros da peça inspecionada.

    Parameters
    ----------
        cl : int, float
            Velocidade de propagação das ondas longitudinais na peça, em m/s.
            Por padrão, é 5900 m/s.

        cs : int, float
            Velocidade de propagação das ondas transversais na peça, em m/s.
            Por padrão, é 3230 m/s.

        roughness : int, float
            Rugosidade. Por padrão, é 0.0.

    Attributes
    ----------
        cl : int, float
            Velocidade de propagação das ondas longitudinais na peça, em m/s.

        cs : int, float
            Velocidade de propagação das ondas transversais na peça, em m/s.

        roughness : int, float
            Rugosidade.

    """

    def __init__(self, cl=5900, cs=3230, roughness=0.0):
        # Atribuição dos atributos da instância.
        # Velocidade de propagação das ondas longitudinais na peça.
        self.cl = cl

        # Velocidade de propagação das ondas transversais na peça.
        self.cs = cs

        # Rugosidade.
        self.roughness = roughness


class ProbeParams:
    """Classe contendo os parâmetros do transdutor.

    Na implementação atual, os tipos suportados são mono e linear. 

    Parameters
    ----------
        tp : str
            Tipo de transdutor. O tipos possíveis são: ``mono`` (único
            elemento) e ``linear`` (*array* linear). Por padrão, é do tipo
            ``linear``.

        num_elem : int
            Número de elementos. Exclusivo para transdutores do tipo
            ``linear``. Por padrão, é 32.

        pitch : int, float
            Espaçamento entre os centros dos elementos, em mm. Exclusivo para
            transdutores do tipo ``linear``. Exclusivo para transdutores do
            tipo ``linear``. Por padrão, é 0.6 mm.

        dim : int, float
            Dimensão dos elementos do transdutor, em mm. Se o elemento ativo
            for circular, o valor representa o diâmetro. Se o elemento ativo
            for retangular, o valor é uma tupla no formato ``(dim_x, dim_y)``.
            Se o elemento ativo for retangular para um *array* linear, o valor
            é a menor dimensão do elemento ativo. Por padrão, é 0.5 mm.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Exclusivo para transdutores
            do tipo ``linear``. Por padrão é 0.1 mm.

        freq : int, float
            Frequência central, em MHz. Por padrão, é 5 MHz.

        bw : int, float
            Banda passante, em percentual da frequência central. Por padrão,
            é 0.5 (50%).

        pulse_type : str
            Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
            ``cossquare``, ``hanning`` e ``hamming``. Por padrão, é
            ``gaussian``.

    Attributes
    ----------
        type_probe : str
            Tipo de transdutor. O tipos possíveis são: ``mono`` (único
            elemento) e ``linear`` (*array* linear).

        num_elem : int
            Número de elementos. Exclusivo para transdutores do tipo ``linear``.

        inter_elem : int, float
            Espaçamento entre elementos, em mm. Exclusivo para transdutores do
            tipo ``linear``.
        
        pitch: int, float
            Espaçamento entre os centros dos elementos, em mm. Exclusivo para
            transdutores do tipo ``linear``. Exclusivo para transdutores do
            tipo ``linear``.

        elem_center : :class:`np.ndarray`
            Se o transdutor é do tipo ``linear``, é uma matriz com as
            coordenadas cartesianas do centro geométrico de cada elemento, em
            mm. Essas coordenadas são relativas ao centro geométrico do
            transdutor. Se o transdutor é do tipo ``mono``, é a posição
            central do elemento ativo do transdutor, em mm.
            
        shape : str
            Formato do transdutor. Os valores possíveis são ``circle`` e
            ``rectangle``. O valor padrão é ``circle``. Exclusivo para
            transdutores do tipo ``mono``.

        elem_dim : int, float
            Dimensão dos elementos do transdutor, em mm. Se o elemento ativo
            for circular, o valor representa o diâmetro. Se o elemento ativo
            for retangular, o valor é uma tupla no formato ``(dim_x, dim_y)``.
            Se o elemento ativo for retangular para um *array* linear, o valor
            é a menor dimensão do elemento ativo.

        central_freq : int, float
            Frequência central, em MHz.

        bw : int, float
            Banda passante, em percentual da frequência central.

        pulse_type : str
            Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
            ``cossquare``, ``hanning`` e ``hamming``.

    """

    def __init__(self, tp="linear", num_elem=32, pitch=0.6, dim=0.5, inter_elem=0.1,
                 freq=5., bw=0.5, pulse_type="gaussian", elem_list=None):

        # Atribuição dos atributos da instância.
        # Tipo de transdutor. O tipos possíveis são: 'mono' (único elemento)
        # e 'linear' (*array* linear). O padrão é do tipo 'mono'.
        self.type_probe = tp

        # Parâmetros exclusivos para transdutores do tipo ``linear``
        if self.type_probe == "linear":
            # Número de elementos.
            self.num_elem = num_elem

            # Espaçamento entre elementos.
            self.inter_elem = inter_elem

            # Espaçamento entre os centros dos elementos.
            self.pitch = pitch

            # Matriz com as coordenadas cartesianas do centro geométrico
            # de cada elemento. Essas coordenadas são
            # relativas ao centro geométrico do transdutor.
            self.elem_center = np.zeros((num_elem, 3))
            for i in range(num_elem):
                self.elem_center[i, 0] = dim / 2 + i * pitch

        elif self.type_probe == 'matricial':
            # Número de elementos.
            self.num_elem = num_elem

            # Posição central do elemento ativo do transdutor.
            self.elem_center = np.zeros((num_elem, 3))

            # Espaçamento entre elementos.
            self.inter_elem = inter_elem

        elif self.type_probe == 'circular':
            # Número de elementos.
            self.num_elem = num_elem

            # Posição central do elemento ativo do transdutor.
            self.elem_center = np.zeros((num_elem, 3))

            # Espaçamento entre elementos.
            self.inter_elem = inter_elem

            self.elem_list = elem_list

        else:
            # Parâmetros exclusivos para transdutores do tipo ``mono``.
            # Formato do transdutor. Os valores possíveis são ``circle``
            # e ``rectangle``. O valor padrão é ``circle``.
            self.shape = "circle"

            # Posição central do elemento ativo do transdutor.
            self.elem_center = np.zeros((num_elem, 3))

            # Espaçamento entre elementos.
            self.inter_elem = inter_elem

            # Espaçamento entre os centros dos elementos.
            self.pitch = pitch

        # Dimensão dos elementos do transdutor.
        # Se o elemento ativo for circular, o valor representa o diâmetro.
        # Se o elemento ativo for retangular, o valor é uma tupla no
        # formato ``(dim_x, dim_y)``. Se o elemento ativo for retangular
        # para um *array* linear, o valor é a menor dimensão do elemento
        # ativo.
        self.elem_dim = dim

        # Parâmetros referentes ao sinal de excitação do transdutor.
        # Frequência central, em MHz.
        self.central_freq = freq

        # Banda passante, em percentual da frequência central.
        self.bw = bw

        # Tipo do pulso de excitação. Os tipos possíveis são: ``gaussian``,
        # ``cossquare``, ``hanning`` e ``hamming``.
        self.pulse_type = pulse_type


class ImagingROI:
    """Classe que armazena os parâmetros da *Region of Interest* (ROI) para
    a reconstrução de uma imagem.

    Objetos desse tipo são utilizados como parâmetros para os algoritmos
    de reconstrução de imagem e devem ser armazenados juntos com os resultados
    desses algoritmos.


    Parameters
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em
            mm. Por padrão, é (0.0, 0.0, 0.0) mm.

        height : int, float
            Altura da ROI, em mm. Por padrão, é 20.0 mm.

        h_len : int
            Quantidade de pontos na dimensão de altura ROI. Por padrão, é 200.

        width : int, float
            Largura da ROI, em mm. Por padrão, é 20.0 mm.

        w_len : int
            Quantidade de pontos na dimensão de largura ROI. Por padrão, é 200.

        depth : int, float
            Profundidade da ROI (em um transdutor linear, tipicamente corresponde à
            direção passiva). Por padrão, é 0.0 mm (ROI de duas dimensões).

        d_len : int
            Quantidade de pontos na dimensão de profundidade ROI. Por padrão, é 1
            (ROI de duas dimensões).

    Attributes
    ----------
        coord_ref : :class:`np.ndarray`
            Ponto cartesiano indicando a coordenada de referência da ROI, em mm.

        h_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da altura (dimensão 1)
            da imagem, em mm.

        h_len : int
            Quantidade de pontos da ROI no sentido da altura.

        h_step : float
            Tamanho do passo dos pontos da ROI no sentido da altura, em mm.

        height : float
            Altura da ROI, em mm.

        w_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da largura (dimensão 2)
            da imagem.

        w_len : int
            Quantidade de pontos da ROI no sentido da largura.

        w_step : float
            Tamanho do passo dos pontos da ROI no sentido da largura, em mm.

        width : float
            Largura da ROI, em mm.

        d_points : :class:`np.ndarray`
            Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 1)
            da imagem.

        d_len : int
            Quantidade de pontos da ROI no sentido da profundidade.

        d_step : float
            Tamanho do passo dos pontos da ROI no sentido da profundidade, em mm.

        depth : float
            Profundidade da ROI, em mm.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se ``coord_ref`` não for do tipo
        :class:`np.ndarray` e/ou não possuir 1 linha e três colunas.

    Notes
    -----
    Esta classe aplica-se a ROIs em duas e três dimensões.

    """

    def __init__(self, coord_ref=np.zeros((1, 3)), height=20.0, h_len=200, width=20.0, w_len=200, depth=0.0, d_len=1):
        if (type(coord_ref) is not np.ndarray) and (coord_ref.shape != (1, 3)):
            raise TypeError("``coord_ref`` deve ser um vetor-linha de 3 elementos [shape = (1,3)]")

        # Atribuição dos atributos da instância.
        # Ponto cartesiano indicando a coordenada de referência da ROI.
        self.coord_ref = coord_ref

        # Vetor com as coordenadas da ROI no sentido da altura (dimensão 1) da imagem.
        self.h_points = np.linspace(coord_ref[0, 2], coord_ref[0, 2] + height, num=int(h_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da altura.
        self.h_len = self.h_points.size

        # Passo dos pontos da ROI no sentido da altura.
        self.h_step = height / h_len  # self.h_points[1] - self.h_points[0]

        # Altura da ROI.
        self.height = self.h_points[-1] + self.h_points[1] - 2 * self.h_points[0]

        # Vetor com as coordenadas da ROI no sentido da largura (dimensão 2) da imagem.
        self.w_points = np.linspace(coord_ref[0, 0], coord_ref[0, 0] + width, num=int(w_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da largura.
        self.w_len = self.w_points.size

        # Passo dos pontos da ROI no sentido da largura.
        self.w_step = width / w_len  # self.w_points[1] - self.w_points[0]

        # Largura da ROI.
        self.width = width  # self.w_points[-1] + self.w_points[1] - 2 * self.w_points[0]

        # Vetor com as coordenadas da ROI no sentido da profundidade (dimensão 1) da imagem.
        self.d_points = np.linspace(coord_ref[0, 1], coord_ref[0, 1] + depth, num=int(d_len), endpoint=False)

        # Quantidade de pontos da ROI no sentido da profundidade.
        self.d_len = self.d_points.size
        self.depth = depth

        # Passo dos pontos da ROI no sentido da largura.
        if d_len > 1:
            self.d_step = self.d_points[1] - self.d_points[0]

        # Profundidade da ROI.
        if d_len > 1:
            self.depth = self.d_points[-1] + self.d_points[1] - 2 * self.d_points[0]
        else:
            self.depth = depth

    def get_coord(self):
        """Método que retorna todas as coordenadas da ROI (*mesh*) no formato
        vetorizado.

        Returns
        -------
            : :class:`np.ndarray`
                Matriz :math:`M` x 3, em que :math:`M` é a quantidade de
                pontos existentes na ROI. Cada linha dessa matriz é a
                coordenada cartesiana de um ponto da ROI.

        """
        return np.array(np.meshgrid(self.w_points,
                                    self.d_points,
                                    self.h_points,
                                    indexing='ij')).reshape((3, -1)).T


class ImagingResult:
    """Classe que armazena os resultados obtidos a partir da execução de
    algoritmos de reconstrução de imagens (*imaging algorithms*)

    Parameters
    ----------
        roi : :class:`ImagingROI`
            ROI da imagem.

        description : str
            Texto com descrição do resultado.

    Attributes
    ----------
        image : :class:`np.ndarray`
            *Array* para o armazenamento da imagem reconstruída. Esse *array*
            tem duas dimensões (imagem). Por padrão, a imagem deve ser
            composta por valores *brutos*, sem qualquer tipo de
            pós-processamento. Se for necessário algum pós-processamento,
            ele deve ser realizado pelos métodos de visualização dos dados.

        roi : :class:`ImagingROI`
            ROI da imagem.

        description : str
            Texto com descrição do resultado.

        name : str
            Nome do algoritmo utilizado para a reconstrução da imagem.

    """

    def __init__(self, roi=ImagingROI(), description=""):
        # Atribuição dos atributos da instância.
        # *Array* para o armazenamento da imagem reconstruída. Esse *array*
        # tem duas dimensões (imagem). Por padrão, a imagem deve ser composta
        # por valores *bruto*, sem qualquer tipo de pós-processamento. Se for
        # necessário algum pós-processamento, ele deve ser realizado pelos
        # métodos de visualização dos dados.
        self.image = np.ndarray((roi.h_len, roi.w_len))

        # ROI da imagem.
        self.roi = roi

        # Texto com a descrição do resultado.
        self.description = description

        # Nome do algoritmo utilizado
        self.name = ''


class DataInsp:
    """Classe com todos os dados necessários de um ensaio não-destrutivo.

    Parameters
    ----------
        inspection_params : :class:`InspectionParams`
            Objeto contendo os parâmetros da inspeção.

        specimen_params : :class:`SpecimenParams`
            Objeto contendo os parâmetros da amostra.

        probe_params : :class:`ProbeParams`
            Objeto contendo os parâmetros do transdutor.

    Attributes
    ----------
        inspection_params : :class:`InspectionParams`
            Objeto contendo os parâmetros da inspeção.

        specimen_params : :class:`SpecimenParams`
            Objeto contendo os parâmetros da amostra.

        probe_params : :class:`ProbeParams`
            Objeto contendo os parâmetros do transdutor.

        ascan_data : :class:`np.ndarray`
            *Array* para armazenamento dos sinais *A-scan*. Esse é um *array*
            com quatro dimensões. A primeira dimensão representa a escala de
            tempo dos sinais *A-scan* (``time``). A segunda dimensão
            representa a sequência de disparos do transdutor (``sequence``).
            Essa dimensão será sempre unitária para transdutores do tipo
            ``mono``. A terceira dimensão representa os canais de recebimento
            do transdutor (``channel``). Essa dimensão será sempre unitária
            para transdutores do tipo ``mono``. A quarta dimensão representa
            os passos do transdutor (``step``). Cada índice dessa dimensão
            está diretamente associado a quantidade de coordenadas existentes
            na lista :attr:`InspectionParams.step_points`.

        ascan_data_sum : :class:`np.ndarray`
            *Array* para armazenamento da soma dos sinais *A-scan* recebidos em um ensaio.
            Esse é um *array* com três dimensões. A primeira dimensão representa a escala de
            tempo dos sinais *A-scan* (``time``). A segunda dimensão
            representa a sequência de disparos do transdutor (``sequence``).
            Essa dimensão será sempre unitária para transdutores do tipo
            ``mono``. A terceira dimensão representa os passos do transdutor (``step``).
            Cada índice dessa dimensão está diretamente associado a quantidade de coordenadas existentes
            na lista :attr:`InspectionParams.step_points`. Esse *array* somente irá existir se o ensaio
            de inspeção for configurado para coletar a soma dos canais. Caso contrário, terá valor ``None``.

        time_grid : :class:`np.ndarray`
            *Array* para armazenamento do *grid* de tempo para todos os sinais
            *A-scan*. Assim como os sinais *A-scan*, esse *array* é um
            vetor-coluna.

        imaging_results : :class:`ImagingResult`
            Dicionário com os objetos do tipo :class:`ImagingResult` contendo
            os resultados da execução dos algoritmos de reconstrução de imagens.

        dataset_name : str
            String contendo o *nome* do conjunto de dados. Essa é uma informação
            disponibilizada pelo Panther.

    """

    def __init__(self,
                 inspection_params=InspectionParams(),
                 specimen_params=SpecimenParams(),
                 probe_params=ProbeParams()):
        # Atribuição dos atributos da instância.
        # Objeto contendo os parâmetros da inspeção.
        self.inspection_params = inspection_params

        # Objeto contendo os parâmetros da amostra.
        self.specimen_params = specimen_params

        # Objeto contendo os parâmetros do transdutor.
        self.probe_params = probe_params

        # Cálculo das dimensões do *array* de armazenamento dos sinais
        # ``A-scan``.
        dim1 = self.inspection_params.gate_samples
        dim2 = self.probe_params.num_elem if (self.probe_params.type_probe == "circular") else \
            self.probe_params.num_elem if (self.probe_params.type_probe == "linear" and
                                           self.inspection_params.type_capt == "FMC") else \
            inspection_params.angles.shape[0] if (self.probe_params.type_probe == "linear" and
                                                  self.inspection_params.type_capt == "PWI") else 1
        dim3 = self.probe_params.num_elem if self.probe_params.type_probe == "circular" else \
            self.probe_params.num_elem if self.probe_params.type_probe == "linear" else 1
        dim4 = self.inspection_params.step_points.shape[0]

        # *Array* para armazenamento dos sinais ``A-scan``. Esse é um *array*
        # com quatro dimensões. A primeira dimensão representa a escala de
        # tempo dos sinais ``A-scan`` (``time``). A segunda dimensão
        # representa a sequência de disparos do transdutor (``sequence``).
        # Essa dimensão será sempre unitária para transdutores do tipo
        # 'mono'. A terceira dimensão representa os canais de recebimento do
        # transdutor (``channel``). Essa dimensão será sempre unitária para
        # transdutores do tipo 'mono'. A quarta dimensão representa os passos
        # do transdutor (``step``). Cada índice dessa dimensão está
        # diretamente associado a quantidade de coordenadas existentes na
        # lista :class:`InspectionParams.step_points`.
        self.ascan_data = np.zeros((dim1, dim2, dim3, dim4), dtype=np.float32)

        # *Array* para armazenamento da soma dos sinais *A-scan* recebidos em um ensaio.
        # Esse é um *array* com três dimensões. A primeira dimensão representa a escala de
        # tempo dos sinais ``A-scan`` (``time``). A segunda dimensão
        # representa a sequência de disparos do transdutor (``sequence``).
        # Essa dimensão será sempre unitária para transdutores do tipo
        # 'mono'. A terceira dimensão representa os passos
        # do transdutor (``step``). Cada índice dessa dimensão está
        # diretamente associado a quantidade de coordenadas existentes na
        # lista :class:`InspectionParams.step_points`.
        # Esse *array* somente irá existir se o ensaio de inspeção for configurado para
        # coletar a soma dos canais. Caso contrário, terá valor ``None``.
        self.ascan_data_sum = None

        # *Array* para armazenamento do *grid* de tempo para todos os sinais
        # ``A-scan``. Assim como os sinais ``A-scan``, esse *array* é um
        # vetor-coluna.
        self.time_grid = np.linspace(self.inspection_params.gate_start, self.inspection_params.gate_end,
                                     num=dim1, endpoint=False)[:, np.newaxis]

        # Lista com o armazenamento das informações dos *encoders* para cada *shot*.
        self.encoders_info = list()

        # Dicionário com os objetos do tipo ``ImagingResult`` contendo os
        # resultados da execução dos algoritmos de reconstrução de imagens.
        self.imaging_results = {}

        # String contendo o *nome* do conjunto de dados. Essa é uma informação disponibilizada pelo Panther.
        self.dataset_name = 'DataInsp'

        # Surface
        self.surf = None
