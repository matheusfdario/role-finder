# -*- coding: utf-8 -*-
"""
Módulo ``gui``
==============

Possui a implementação da interface humano-computador utilizando o ``PyQt`` e ``PyQtGraph``.
Todas as classes de widgets são implementadas neste arquivo, exceto pelos widgets gerados pelo ``Designer`` do ``Qt``.
Para esses *widgets*, são feitas heranças das classes geradas pelo ``Designer``.
"""
from PyQt5.QtWidgets import QApplication
from guiqt.Windows.MainWindow import MainWindow
import sys

def main():
    # executa a aplicacao
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec()


if __name__ == "__main__":
    main()
