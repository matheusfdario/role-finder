# -*- coding: utf-8 -*-
"""
Módulo ``ArrayParameter``
=========================

Possui a implementação para adicionar um parâmetro em formato *array* no pyqtgraph.
Isso permite a visualização e o *input* de um vetor pelo usuário, que pode ser útil para editar atributos do
``DataInsp`` e parâmetros de funções.

O vetor é mostrado e lido como texto, sendo a conversão para inteiros feita apenas quando o valor é lido através do
método *get*.


.. raw:: html

    <hr>

"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QLineEdit
from pyqtgraph import parametertree


class ArrayParameterItem(pg.parametertree.parameterTypes.WidgetParameterItem):
    """ Cria um item de parâmetro do ``PyQtGraph`` para mostrar vetores do ``NumPy``. Os valores são processados como
    texto, e convertidos para `numpy.ndarray` apenas para as funcoes de *get*.
    """
    def __init__(self, param, depth):
        # se possui limites, garante que o valor padrao nao excede
        try:
            limits = param.opts['limits']
            val = param.opts['value']

            val = val[val >= limits[0]]
            val = val[val <= limits[1]]
            param.opts['value'] = val
        except KeyError:
            pass
        super().__init__(param, depth)

    def valueChanged(self, param, val, force=False):
        """ Método sobreposto. Chamado quando o valor do parâmetro é alterado.
        """
        if type(val) is np.ndarray:
            try:
                limits = self.param.opts['limits']
                val = val[val >= limits[0]]
                val = val[val <= limits[1]]
            except KeyError:
                pass
            self.widget.setText(str(val).replace('. ', ' ').replace('.]', ']'))
        elif type(val) is str:
            self.widget.setText(val).replace('. ', ' ')

        self.updateDefaultBtn()

    def makeWidget(self):
        """ Método sobreposto. Cria o *widget* a ser exibido dentro do parâmetro.
        """
        opts = self.param.opts
        w = QLineEdit()
        w.setStyleSheet('border: 0px')
        w.sigChanged = w.editingFinished
        w.value = self.get_value
        w.setValue = self.set_value
        w.setEnabled(not opts.get('readonly', False))
        self.hideWidget = False

        return w

    def set_value(self, value):
        """ Salva um array ou string no parâmetro.
        """
        if type(value) is np.ndarray:
            self.widget.setText(np.fromstring(value.replace('. ', ' '), sep=' '))
        elif type(value) is str:
            self.widget.setText(value)

        self.updateDefaultBtn()

    def get_value(self):
        """ Pega o array salvo no parâmetro.
        """
        string = self.widget.text()
        string = string[1:] if string.startswith('[') else string
        string = string[:-1] if string.endswith(']') else string
        final_string = ''
        for val in string.split(' '):
            if val.find(':') == -1:
                final_string += val + ' '
            else:
                first = int(float(val.split(':')[0]))
                last = int(float(val.split(':')[1]))
                sign = np.sign(last-first)
                sign = 1 if sign == 0 else sign
                array = np.arange(first, last + sign, sign)
                for num in array:
                    final_string += str(num) + ' '

        return np.fromstring(final_string[:-1], sep=' ')


class ArrayParameter(pg.parametertree.Parameter):
    """ Cria o parâmetro do ``PyQtGraph`` para mostrar vetores do ``NumPy``.
    """
    itemClass = ArrayParameterItem

    def valueIsDefault(self):
        """ Método sobreposto. Testa se o valor salvo no parâmetro é o valor padrão.
        """
        bool_array = self.value() == self.defaultValue()
        if bool_array is True or bool_array is False:
            return bool_array
        for boolv in bool_array:
            if boolv is False:
                return False
        return True

    def setValue(self, value, block_signal=None):
        """ Muda o valor do item.
        """
        self.opts['value'] = value
        self.sigValueChanged.emit(self, value)
        return value
