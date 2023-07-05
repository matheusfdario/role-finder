# -*- coding: utf-8 -*-
r"""
Pacote ``surface``
==================


Na inspeção de objetos com superfícies arbitrárias, podem-se utilizar transdutores do tipo array flexível
:cite:`toullelan2008flexible,mackersie2009development,hunter2010autofocusing` ou realizar ensaios por imersão, em que o
meio no qual o sistema está imerso é responsável pelo acoplamento acústico entre o transdutor e o objeto. Os ensaios
que fazem parte do escopo deste projeto serão realizados utilizando-se um braço robótico, sem a presença de um operador
humano no local, o que dificulta a acomodação de sistemas de array flexível. Além disso os ensaios serão realizados em
ambiente submarino, o que torna a técnica de imersão uma escolha natural.

Num ensaio por imersão, para se ter acesso à superfície interna do objeto inspecionado, é necessário conhecer sua
superfície externa, uma vez que as ondas sonoras que se deslocam do transdutor até a superfície interna e de volta
sofrem refração e atenuação na interface da superfície externa, conforme representado na :numref:`snell` (a).
A refração nas interfaces segue a lei de Snell


.. math:: \frac{\text{sen}(\theta_2)}{\text{sen}(\theta_2)}=\frac{c_2}{c_1},


representada na :numref:`snell` (b), onde :math:`\theta_1` e :math:`\theta_2` são os ângulos a partir da normal da
superfície e :math:`c_1` e :math:`c_2` são as velocidades do som nos meios 1 e 2 respectivamente.


.. figure:: figures/surface/snell.*
    :name: snell
    :width: 60 %
    :align: center

    (a) Representação da refração sofrida na superfície externa pelas ondas sonoras que se deslocam do transdutor até a
    superfície interna do objeto inspecionado e de volta ao transdutor. Conhecer a superfície externa é necessário para
    a correta geração dos pulsos e interpretação dos dados de eco. (b) A refração nas interfaces entre dois meios se dá
    segundo a Lei de Snell.


Cálculo dos atrasos utilizando Lei de Snell e Princípio de Fermat
-----------------------------------------------------------------

Nos algoritmos baseados em atraso e soma, como o SAFT e o TFM, é crucial conhecer o tempo de propagação de um pulso
sonoro entre um determinado elemento transdutor A e uma determinada posição F na ROI, pois a diferença de tempo entre
as variadas combinações elemnto/pixel é compensada através da definição de leis focais :cite:`Schmerr2015`. Em ensaios
por contato,
tipicamente há apenas um meio a ser considerado (o próprio material), com uma única velocidade, de forma que o tempo de
propagação entre o elemento transdutor A e a posição F na ROI é dada pela distância euclidiana entre ambos, dividida
pela velocidade do som no meio. Porém, em ensaios por imersão, há dois meios a serem considerados: o meio acoplante
(e.g. água) com velocidade $c_1$ e o material com velocidade $c_2$. Nesse caso, o tempo decorrido na trajetória de um
pulso de A a F ou vice-versa é dada por


.. math:: T_{AF}=\frac{1}{c_1}\sqrt{(x_A-x_S)^2+(z_A-z_S)^2}+\frac{1}{c_2}\sqrt{(x_F-x_S)^2+(z_F-z_S)^2},


onde :math:`(x_A,z_A)` são as coordenadas (bi-dimensionais) do elemento transdutor A, :math:`(x_F,z_F)` são as
coordenadas da posição F na ROI, $(x_S,z_S)$ são as coordenadas do ponto em que o pulso atinge a superfície e é
refratado, :math:`c_1` é a velocidade do som no meio acoplante e :math:`c_2` é a velocidade do som no material.

O ponto de entrada :math:`(x_S,z_S)` é aquele em que a lei de Snell é respeitada. A :numref:`fastestray` mostra um
exemplo em que a trajetória do pulso de um elemento A até uma posição F dentro do material deve ser determinada. São
mostradas várias trajetórias candidatas, sendo que apenas uma é a verdadeira. De acordo com o princípio de Fermat, a
trajetória que respeita a lei de Snell é também a trajetória mais rápida :cite:`schuster1904introduction,Schmerr1998`.
Portanto, o problema da definição do ponto de
entrada na superfície corresponde à determinação do ponto :math:`(x_S,z_S)` pertencente à superfície que minimiza o
tempo :math:`T_{AF}` na Eq. :eq:`fermat`, ou seja,


.. math:: (\hat{x}_S,\hat{z}_S)= \underset{(x_S,z_S)}{\arg \min} \ \frac{1}{c_1}\sqrt{(x_A-x_S)^2+(z_A-z_S)^2}+\frac{1}{c_2}\sqrt{(x_F-x_S)^2+(z_F-z_S)^2}.
    :label: fermat

.. figure:: figures/surface/fastestray.*
    :name: fastestray
    :width: 50 %
    :align: center

    Aplicação do princípio de Fermat com :math:`c_2 > c_1`. (Fonte: :cite:`parrilla2007fast`)


Módulos e exemplos
---------------------------------------

O pacote :mod:`surface` contém as implementações em ``Python`` de algoritmos que executam seguintes tarefas:
    - Identificação da superfície (interface com a água) do material baseada nos dados pulso-eco ou FMC.
    - Cálculo das trajetórias e tempos de percurso entre cada elemento do transdutor e cada pixel definido na ROI.

A utilização do pacote ``surface`` é feita essencialmente através dos métodos da classe
:class:`surface.surface.Surface`.

A classe :class:`surface.nonlinearopt.NewtonMultivariate` é utilizada pela classe :class:`surface.surface.Surface` para
implementar algoritmos de busca de parâmetros de superfícies baseados no método de Newton Raphson Multivariável e
também pode ser utilizada independentemente.


.. raw:: html

    <hr>
    
"""
