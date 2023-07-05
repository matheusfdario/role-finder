# -*- coding: utf-8 -*-
"""
Pacote ``imaging``
==================

Entre todos os métodos para análise de sinais ultrassônicos disponíveis, as técnicas baseadas em imagens são sem dúvida
as mais utilizadas. Diversos autores, tais como
:cite:`Doctor1986163,Muller1986,Bernus1993,Chiao1994,Ditchburn1996111,Spies2012vb`, indicam que a apresentação de uma
imagem melhora o desempenho dos inspetores na interpretação dos dados de inspeção por ultrassom. Assim, o problema em
questão é como criar a imagem de uma descontinuidade, a princípio desconhecida, a partir de um conjunto de sinais
*A-scan* medidos pelo sistema de medição e possivelmente distorcidos por ruído. Esse tipo de problema é definido como
*reconstrução de imagens* :cite:`Bovik2000`.

As imagens fornecidas por um sistema de inspeção por ultrassom representam a refletividade acústica dentro de um objeto
:cite:`Bernus1993`. Elas são criadas a partir da aplicação de um algoritmo de reconstrução apropriado sobre um conjunto
de sinais *A-scan*. Existem diversos algoritmos para a reconstrução de imagens em END por ultrassom. O algoritmo mais
simples, conhecido como *B-scan*, monta uma imagem como uma matriz de pontos. Nela, cada coluna representa a posição
espacial do transdutor e cada linha corresponde ao tempo de propagação das ondas ultrassônicas, desde o transdutor até
uma posição dentro do objeto inspecionado. A intensidade de cada ponto da imagem é proporcional a amplitude do sinal
*A-scan* relacionado à posição do transdutor e ao tempo de propagação. Uma imagem *B-scan* mostra a representação de
perfil (corte lateral) do objeto inspecionado. Embora o algoritmo *B-scan* seja simples e rápido na reconstrução de
imagens, ele apresenta uma baixa resolução lateral. Além disso, o diâmetro do transdutor e a profundidade da
descontinuidade afetam a qualidade da imagem reconstruída :cite:`Shieh2012,Schmitz2000`, devido aos efeitos da difração
e do espalhamento de feixe :cite:`Kino1987`.

No início dos anos 1970, a técnica de *Focalização por Abertura Sintética* (SAFT -- *Synthetic Aperture Focusing
Technique*) :cite:`Prine1972,Burckhardt1974,Seydel1982` foi desenvolvida para melhorar a resolução lateral das imagens
reconstruídas. Essa técnica foi inspirada nos conceitos de *abertura sintética* (SA -- *Synthetic Aperture*) utilizados
em sistemas de mapeamento por radar aerotransportado :cite:`Sherwin1962`. Em geral, o SAFT é implementado por operações
de soma e deslocamento diretamente nos sinais *A-scan* :cite:`Frederick1976,Corl1978,Kino1980`. Entretanto, ele também
pode ser implementado de outras formas, tais como: multiplicação matriz-vetor :cite:`Lingvall2003`; migração de Stolt
:cite:`Stolt1978` aplicada nos sinais *A-scan* no domínio da frequência (algoritmo wk)
:cite:`Mayer1990,Gough1997,Chang2000,Stepinski2007`; e utilizando processamento distribuído em unidades de processamento
gráfico (GPU -- *graphics processing unit*) :cite:`MartinArguedas2012`.

Os algoritmos *B-scan* e SAFT foram desenvolvidos inicialmente para sistemas de medição contendo transdutores
monoestáticos (com um único elemento ativo) na configuração pulso-eco. Recentemente, entretanto, houve um aumento
significativo nos sistemas de medição que utilizam transdutores com múltiplos elementos ativos, os chamados transdutores
*arrays* :cite:`Holmes2005`. Com esses transdutores, os sistemas de medição podem controlar, de forma eletrônica,
a abertura, a direção e o foco do feixe de ultrassom sobre as descontinuidades :cite:`Drinkwater2006525`. Eles também
podem emular o comportamento de um transdutor monoestático, disparando sequencialmente cada elemento do arranjo.
Isso proporciona duas vantagens: (I) evita a movimentação física do transdutor para varrer a região de interesse; e (II)
permite a aquisição de sinais *A-scan* para todas as combinações de elementos transmissores e receptores.
Esse modo de aquisição de sinais é chamado de *captura de matriz completa* (FMC -- *Full Matrix Capture*)
:cite:`Holmes2005`. O FMC permite a reconstrução de imagens por outros algoritmos, tais como: método de focalização
total (TFM  -- *Total Focusing Method*) :cite:`Holmes2005`; extrapolação do campo de onda inverso (IWEX --
*Inverse Wave Field Extrapolation*) :cite:`Portzgen2007`; uma versão do algoritmo *wk-SAFT* para FMC :cite:`Hunter2008`,
um algoritmo de retropropagação inversível :cite:`Velichko2009` e um algoritmo de *beamforming* adaptativo
:cite:`Li2011c`. Em todos esses algoritmos, a resolução da imagem de um refletor pontual é melhorada quando comparada
com o *B-scan* e com o SAFT :cite:`Holmes2005,Velichko2010,Li2011c`. Apesar das vantagens dos transdutores *array*,
sistemas de medição com transdutores monoestáticos ainda são amplamente utilizados, especialmente em sistemas portáteis
e embarcados.

Pacote :mod:`imaging` no projeto AUSPEX
---------------------------------------

O pacote :mod:`imaging` contém as implementações em ``Python`` de algoritmos para a reconstrução de imagens dentro do
projeto AUSPEX. Todos os algoritmos neste pacote devem apresentar uma mesma interface, visto que os mesmos podem ser
utilizados tanto no desenvolvimento de aplicações em *scripts*, quanto em aplicações com interfaces homem-máquina
gráficas.

O padrão de interface para os algoritmos de reconstrução de imagens adotado no projeto AUSPEX é o seguinte:

    - Cada algoritmo deve ser implementado como um módulo do pacote :mod:`imaging`.
    - Os nomes dos módulos devem identificar o algoritmo.
    - Os módulos devem obrigatoriamente conter duas funções públicas de acesso, ``xxxx_kernel`` e ``xxxx_params``, em
      que ``xxxx`` é necessariamente o nome do módulo.
    - A função ``xxxx_kernel`` contém a implementação do algoritmo, enquanto a função ``xxxx_params`` é para a
      utilização em aplicações com interfaces homem-máquina gráficas.

Todas as funções ``xxxx_kernel`` devem receber pelo menos 5 parâmetros obrigatórios:

    - Uma instância da classe :class:`framework.data_types.DataInsp`, que contém todos os dados provenientes de uma
      seção de inspeção.
    - Uma instância da classe :class:`framework.data_types.ImagingROI`, que define a região da imagem reconstruída.
    - Uma *chave de identificação* para o dicionário que armazena o resultado do algoritmo dentro do objeto
      :class:`framework.data_types.DataInsp`.
    - Uma *string* de identificação do resultado.
    - O índice seletor de *passo* (*step*), no caso de inspeções com múltiplos passos.

Além desses, cada algoritmo pode exigir outros parâmetros, tais como velocidade de propagação, parâmetros de
regularização, níveis de *threshold*, entre outros. O retorno das funções ``xxxx_kernel`` é feito inserindo uma
instância da classe :class:`framework.data_types.ImagingResult` no dicionário
:attr:`framework.data_types.DataInsp.imaging_results`. A chave desse objeto no dicionário é retornada ao chamador de
``xxxx_kernel``.

As funções ``xxxx_params`` não precisam de nenhum parâmetro. Elas retornam um dicionário em que os seus elementos são
os valores padrão dos parâmetros da função ``xxxx_kernel``. As chaves dos elementos são obrigatoriamente o nome de cada
parâmetro. Todos os parâmetros de ``xxxx_kernel`` devem ter um valor padrão, com exceção da instância de
:class:`framework.data_types.DataInsp`.


Algoritmos e exemplos
---------------------

Atualmente, os algoritmos disponíveis no pacote :mod:`imaging` são:

    - B-scan
    - SAFT
    - :math:`\omega k`-SAFT
    - E-:math:`\omega k`-SAFT
    - TFM
    - VTFM
    - Wavenumber
    - E-wavenumber
    - CPWC
    - :math:`\omega k`-CPWC
    - E-:math:`\omega k`-CPWC
    - UTSR
    - UTSR - FISTA
    - SpaRSA

Cada algoritmo está documentado separadamente, acompanhado de um exemplo de
uso. Os exemplos utilizam dados de inspeção sintéticos, oriundos do simulador
CIVA. Os algoritmos utilizam o módulo :mod:`framework.file_civa` para realizar
a leitura e processamento do arquivo gerado pelo simulador. 

A peça utilizada para a simulação pode ser vista na
:numref:`fig_imaging_specimen_example`. A peça possui 80 mm de largura, 60 mm
de altura e 25 mm de profundidade (não mostrado). Um furo de 1mm de diâmetro
está localizado à 40 mm da parte superior da peça e à 40 mm do canto esquerdo,
sendo que o furo atravessa a peça.

.. figure:: figures/imaging/params_specimen.*
    :name: fig_imaging_specimen_example
    :width: 35 %
    :align: center

    Peça utilizada para a simulação do ensaio no CIVA.

Para a simulação, foi utilizado um transdutor do tipo *array* linear, com
32 elementos e uma frequência central de 5 MHz. Os dados de inspeção foram
obtidos a partir do método de captura FMC.

.. raw:: html

    <hr>
    
"""
