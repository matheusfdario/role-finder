# -*- coding: utf-8 -*-
"""
Módulo ``ParameterRoot``
========================

Possui a implementação de uma herança de classe para facilitar o acesso a certos parâmetros de uma ``parametertree``.

.. raw:: html

    <hr>

"""

import pyqtgraph as pg
from pyqtgraph import parametertree


class ParameterRoot(pg.parametertree.Parameter):
    """
    Classe herdada de ``PyQtGraph.parametertree.Parameter`` para instanciar a raiz da árvore de parâmetros.
    Possui métodos para facilitar o acesso aos parâmetros mantidos no ``parametertree``.
    Não é necessário nenhum argumento para o construtor, todos os ``Parameter`` filhos devem ser adicionados após a
    inicialização.
    """
    def __init__(self):
        """Construtor da classe.
        """

        super(self.__class__, self).__init__(name="Parametros")

    def get_parameters(self, param_name=None):
        """ Pega os parâmetros que são filhos do parâmetro passado.

        Parameters
        ----------
            param_name : `String`
                Nome do parâmetro desejado.
        """
        if param_name is None:
            out = {child.name(): child.value() for child in self.children()}
        else:
            parameters = self.child(param_name)
            out = {child.name(): child.value() for child in parameters}
        return out

    def get_roi_parameters(self):
        """ Pega os parâmetros da ROI.
        """
        return self.get_parameters("ROI")

    def set_roi_parameters(self, coord_ref, height, width):
        """ Muda os parâmetros da ROI.

        Parameters
        ----------
            coord_ref : :class:`numpy.ndarray`
                Coordenadas da ROI.

            height : `float`
                Altura da ROI.

            width : `float`
                Largura da ROI.
        """
        parameters = self.child("ROI")
        parameters.child("X Coordinate [mm]").setValue(coord_ref[0, 0])
        parameters.child("Y Coordinate [mm]").setValue(coord_ref[0, 1])
        parameters.child("Z Coordinate [mm]").setValue(coord_ref[0, 2])
        parameters.child("Height [mm]").setValue(height)
        parameters.child("Width [mm]").setValue(width)
