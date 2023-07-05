"""
Pacote ``guiqt``
================

Uma interface gráfica, ou GUI (*Graphical User Interface*), é uma interface que permite o usuário interagir com um
programa de forma gráfica, com o auxílio de botões e ícones. Para este *framework*, foi desenvolvida uma interface
gráfica com o intuito de integrar e facilitar o uso das diferentes funcionalidades desenvolvidas.

A interface consiste na implementação e comunicação de diferentes *widgets*, que são os elementos
básicos para interação com usuário como botões, por exemplo. *Widgets* também podem ser compostos de outros *widgets*,
como o ``estimation_widget`` que possui botões, caixas de seleção e gráficos.

Foi utilizado o ``PyQt`` e o ``PyQtGraph`` para desenvolver as classes utilizadas na interface. Além disso, o
``QtDesigner`` foi utilizado para criar o design de alguns widgets.

Sobre o pacote
--------------
Este pacote contém os pacotes com todos os arquivos relacionados à interface: aqueles criados pelo ``QtDesigner``, que
possuem o sufixo Design.ui, e seus correspondentes com a implementação em python do design das janelas e *widgets*, com
sufixo Design.py, e os arquivos com as implementações das funcionalidades e da integração das janelas e dos *widgets*.


.. raw:: html

    <hr>

"""
