# -*- coding: utf-8 -*-
"""
Módulo ``Overlay``
==================

Possui a implementação de um *overlay* na janela principal.

.. raw:: html

    <hr>

"""

import math

from PyQt5 import QtGui, QtCore, QtWidgets


class Overlay(QtWidgets.QWidget):
    """ Desenha um *overlay* em uma janela, para sinalizar que ela está processando algo e o usuário não pode
    interagir com a tela. Cria um temporizador para redesenhar o *overlay*, sinalizando que a janela não está travada.
    """
    def __init__(self, parent=None):

        QtWidgets.QWidget.__init__(self, parent)
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
        self.counter = 0
        self.hide()
        self.timer = None

    def paintEvent(self, event):
        """ Desenha o *overlay*.
        """
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        for i in range(6):
            if int(self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(250, 250, 250)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(50, 50, 50)))
            painter.drawEllipse(
                int(self.width() / 2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10),
                int(self.height() / 2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10),
                20, 20)

        painter.end()

    def showEvent(self, event):
        """ Função chamada para iniciar o *overlay*.
        """
        self.setFixedSize(self.parent().size())
        self.timer = self.startTimer(50)
        self.counter = 0

    def timerEvent(self, event):
        """ Função chamada pelo temporizador.
        """
        self.counter += 1
        self.update()

    def hideEvent(self, event):
        """ Função chamada para esconder o *overlay*.
        """
        self.killTimer(self.timer)
