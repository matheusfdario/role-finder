"""
Módulo ``ErrorWindow``
======================

Implementa uma janela de erro.

.. raw:: html

    <hr>

"""

from PyQt5 import QtWidgets


class ErrorWindow(QtWidgets.QMessageBox):
    """
    Classe que instancia uma janela de erro quando criada.
    Recebe como argumento o texto a ser mostrado.
    O resto da interface fica bloqueado até que o usuário feche essa janela.
    """
    def __init__(self, error_msg):
        """Construtor da classe.

        Parameters
        ----------
            error_msg : `String`
                Texto a ser exibido.
        """
        super(self.__class__, self).__init__()
        self.setText(error_msg)
        self.setWindowTitle("Error")
        self.exec_()
