# -*- coding: utf-8 -*-
"""
Módulo ``Thread``
=================

.. raw:: html

    <hr>

"""

from PyQt5 import QtCore


class Thread(QtCore.QThread):
    """
    *Wrapper* para instanciar uma *thread* que executa uma função e emite um sinal da janela principal quando termina.
    É necessário passar a janela principal como parâmetro para poder emitir o sinal correto.
    """
    def __init__(self, mw):
        """ Construtor da classe.

        Parameters
        ----------
            mw : :class:`guiqt.gui.MainWindow`
                Janela principal.
        """
        super().__init__()
        self.result = None
        self.mw = mw
        self.params = dict()
        self.func = None
        self.exception = False

    def run(self):
        """ Executa o algoritmo passado, e depois emite o sinal de que terminou.
        """
        self.exception = False
        try:
            self.result = self.func(**self.params)
        except Exception as e:
            self.result = e
            self.exception = True

        self.mw.finished_sig.emit()

    def set_dict(self, d):
        """ Salva o dicionário com os parâmetros que serão utilizados na função.

        Parameters
        ----------
            d : `dict`
                Dicionário com os parâmetros da função.
        """
        self.params = d

    def set_func(self, f):
        """ Salva a função que será executada.

        Parameters
        ----------
            f : `method`
                Ponteiro da função a ser execudata.
        """
        self.func = f
