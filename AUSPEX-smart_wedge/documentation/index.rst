.. AUSPEX documentation master file, created by
   sphinx-quickstart on Fri Jul 27 11:36:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _home:

#################################################################################################
Projeto AUSPEX -- Processamento de Sinais para Inspeções de Equipamentos Submarinos por Ultrassom
#################################################################################################

******
Resumo
******

O objetivo deste projeto de pesquisa e inovação é desenvolver e avaliar minuciosamente o desempenho de algoritmos de
processamento avançado de sinais de ultrassom para auxiliar em inspeções submarinas. Os resultados esperados são um
melhor aproveitamento dos sinais adquiridos, com detecções e dimensionamentos mais precisos de defeitos, além da redução
do tempo de inspeção *off-shore*, reduzindo custos de embarques e operação de ROVs (*Remotely Operated underwater
Vehicle* - Veículo submarino Operado Remotamente).

O projeto será atacado em 4 frentes de trabalho:
   #. Reconstrução de Superfície Interna Corroída: combinar de maneira inteligente sinais de ultrassom para formar mapas
      da superfície interna de equipamentos.
   #. Identificação de Superfície Externa Arbitrária: identificar a geometria externa de equipamentos a partir dos
      sinais de ultrassom e corrigir as trajetórias sônicas.
   #. Identificação *On-Line* de Parâmetros de Inspeção: corrigir e ajustar automaticamente parâmetros da inspeção,
      como velocidade de propagação do som, de modo a melhorar a reconstrução de imagens.
   #. Organização de Dados e Interfaces com Instrumentos: estudar e propor uma organização de dados de inspeções por
      ultrassom, de modo a facilitar o armazenamento, recuperação e utilização dos dados em tratamentos e análises a
      partir de sinais provenientes de instrumentos de ultrassom e simuladores de ultrassom.

Cada frente de trabalho será, basicamente, dividida nas etapas:
   #. Levantamento bibliográfico, identificação e seleção dos trabalhos do estado-da-arte mais relevantes.
   #. Implementação dos algoritmos e reprodução dos resultados descritos nos trabalhos selecionados.
   #. Proposta de modificações e novos algoritmos para adequar os métodos à realidade de inspeções submarinas na
      indústria do petróleo.
   #. Documentação e publicação dos resultados em veículos de divulgação científica.


*********************
Fundamentação teórica
*********************

.. toctree::
   :maxdepth: 2

   fund_teorica.rst

******************
Pacotes do projeto
******************

.. toctree::
   :maxdepth: 2

   framework.rst
   guiqt.rst
   imaging.rst
   surface.rst

.. toctree::
   :hidden:

   zzzrefs.rst

