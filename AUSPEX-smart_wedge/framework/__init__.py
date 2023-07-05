# -*- coding: utf-8 -*-
"""
Pacote ``framework``
====================

Um *framework* é uma ferramenta que visa facilitar e acelerar o desenvolvimento de aplicações específicas.
Considere o caso de um escritor que deseja publicar um livro. O escritor pode utilizar um editor de texto (LibreOffice,
TeXworks) que irá, de maneira automática, numerar as páginas, seções e equações. Com isso, o escritor direciona seu foco
para o desenvolvimento do conteúdo ao invés de contabilizar por todas as figuras em sua publicação.
Além disso, o escritor pode criar um modelo com o editor de texto (*framework*) utilizado e usá-lo novamente em outras
publicações, de maneira simples e rápida.

No âmbito de engenharia de *software*, um *framework* tem como objetivo proporcionar ao usuário o desenvolvimento de
aplicações específicas a partir de ferramentas prontas e  reutilizáveis. A literatura fornece diversas definições de
*framework* :cite:`Firesmith1994,Mattsson1996,Fayad1997,Johnson1988`. Um *framework* pode ser considerado
como uma arquitetura desenvolvida visando a máxima reutilização e com potencial de especialização :cite:`Mattsson1996`.
Ainda, um *framework* pode ser considerado como um projeto abstrato, desenvolvido para solucionar uma família de
problemas :cite:`Johnson1988,Fayad1997`.

Estrutura do *framework* no projeto AUSPEX
------------------------------------------

O *framework* do projeto AUSPEX conta com ferramentas que visam simplificar o desenvolvimento de novas aplicações,
facilitando o uso de dados provenientes de diferentes fontes, possibilitando uma rápida visualização de resultados.
Dessa forma, um maior foco pode ser direcionado ao desenvolvimento dos algoritmos de processamento de dados e análise de
resultados.

A :numref:`fig_framework` ilustra, em um diagrama de blocos, o *framework* como base para uma nova aplicação.
O *framework* conta com módulos para a leitura de dados oriundos de diferentes simuladores ou sistemas de inspeção.
Esses módulos são responsáveis em converter o formato de dados específico de cada fonte para uma estrutura de dados
padronizada, resultando em uma maior transparência entre aplicação e fonte de dados.

.. figure:: figures/framework/framework.png
    :name: fig_framework
    :width: 100 %
    :align: center

    Diagrama mostrando a organização do *framework* proposto para a integração dos dados de inspeção oriundos do
    simulador CIVA e dos sistemas de inspeção M2M e OmniScan.

O *framework* está organizado como um ``package`` da linguagem ``Python``, em que cada ``module`` encapsula
funcionalidades específicas. O módulo ``data_types`` contém as definições de todas as estruturas de dados utilizadas pelo
*framework*. Todos os módulos com as funções para a importação de dados de inspeção oriundos de diversas fontes são
identificados pelo prefixo ``file_``.

Outros módulos desse ``package`` encapsulam funções utilizadas na implementação dos algoritmos de processamento de
sinais do *framework*. Cada módulo contém sua documentação com informações específicas.

.. raw:: html

    <hr>

"""
