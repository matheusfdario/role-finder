# -*- coding: utf-8 -*-
"""
Módulo ``Console``
==================

Possui a herança de uma classe para instanciar um console do ``IPhyton``.

.. raw:: html

    <hr>

"""

from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget


class Console(RichJupyterWidget):
    """ Inicia o console e coloca por padrão o modo de renderização *inline*.
    """
    def __init__(self, **kwarg):
        super(self.__class__, self).__init__()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.shell.push(kwarg)
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        self.kernel.shell.run_cell('%matplotlib qt')
        self.kernel.shell.run_cell('%matplotlib inline')
