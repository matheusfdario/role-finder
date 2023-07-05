# -*- coding: utf-8 -*-
"""Módulo ``post_proc``
=======================

Neste módulo estão implementadas funções que aplicam operações de *pós-processamento* nos resultados fornecidos pelos
algoritmos de reconstrução de imagens. Todas as funções desse módulo tomam uma imagem como parâmetro principal (na forma
de um *array* bidimensional) e aplicam operações necessárias para facilitar a análise das imagens pelo usuário.

.. raw:: html

    <hr>

"""
import numpy as np
from scipy.signal import hilbert

from framework.data_types import ImagingROI


def envelope(image, axis=-2):
    """Função que calcula o envelope em uma imagem criada a partir de algum algoritmo de reconstrução.
    O envelope da imagem é calculado utilizando a Transformada de Hilbert, tomando somente um eixo da imagem.

    Parameters
    ----------
    image : :class:`np.ndarray`
        Imagem na forma de um *array* bidimensional.

    axis : int
        Eixo a ser aplicado o envelope.

    Returns
    -------
    :class:`numpy:ndarray`
        Envelope da imagem.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``image`` não for do tipo
        :class:`np.ndarray`.

    """
    # Teste dos tipos dos parâmetros.
    if type(image) is not (np.ndarray) and type(image) is not np.memmap:
        raise TypeError("O argumento ``image`` não é um objeto do tipo ``numpy.ndarray``.")

    # Verifica se é complexo, se for, retorna o seu módulo.
    if np.iscomplexobj(image) is True:
        return np.abs(image)

    # Retorna o envelope da imagem.
    return np.abs(hilbert(image, axis=axis))

    
def normalize(image, final_min=0, final_max=1, image_min=None, image_max=None):
    """Função que normaliza os valores de uma imagem, colocando-os sempre entre [final_min; final_max].
    Caso apenas a imagem seja passada como argumento, coloca todos os valores no intervalo [0; 1].

    Parameters
    ----------
    image : :class:`np.ndarray`
        Imagem na forma de um *array* bidimensional.

    final_min : :class:`float`
        Valor mínimo da imagem final.

    final_max : :class:`float`
        Valor máximo da imagem final.

    image_min : :class:`float`
        Valor mínimo a ser considerado para a imagem de entrada. Por padrão é considerado o menor valor presente na
        imagem.

    image_max : :class:`float`
        Valor máximo a ser considerado para a imagem de entrada. Por padrão é considerado o maior valor presente na
        imagem.

    Returns
    -------
    :class:`numpy:ndarray`
        Imagem normalizada.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``image`` não for do tipo
        :class:`np.ndarray`.

    """
    # Teste dos tipos dos parâmetros.
    if type(image) is not np.ndarray:
        raise TypeError("O argumento ``image`` não é um objeto do tipo ``numpy.ndarray``.")

    try:
        final_min = float(final_min)
    except Exception:
        raise TypeError("Não foi possível converter o argumento ``final_min`` para o tipo ``float``")

    try:
        final_max = float(final_max)
    except Exception:
        raise TypeError("Não foi possível converter o argumento ``final_max`` para o tipo ``float``")

    if image_min is None:
        image_min = image.min()
    else:
        try:
            image_min = float(image_min)
        except Exception:
            raise TypeError("Não foi possível converter o argumento ``image_min`` para o tipo ``float``")

    if image_max is None:
        image_max = image.max()
    else:
        try:
            image_max = float(image_max)
        except Exception:
            raise TypeError("Não foi possível converter o argumento ``image_max`` para o tipo ``float``")

    if np.isclose(image_max, image_min):
        return np.zeros_like(image)
    else:
        return ((image-image_min) / (image_max-image_min)) * (final_max-final_min) + final_min


def api(image, roi, wavelength=5900/5e6):
    """Função que calcula o índice API de uma imagem. Esse índice, definido em :cite:`Holmes2005` indica a área da
    imagem que se encontra com valores superiores a -6 dB. O patamar de 0 db é referente ao maior valor absoluto da
    imagem.

    Parameters
    ----------
    image : :class:`np.ndarray`
        Imagem na forma de um *array* bidimensional.

    roi : :class:`framework.data_types.ImagingROI`
        ROI da imagem.

    wavelength : :class:`float`
        Comprimento de onda do pulso ultrassônico utilizado no processo de inspeção.

    Returns
    -------
    :class:`float`
        Índice API.

    Raises
    ------
    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``image`` não for do tipo
        :class:`np.ndarray`.

    """
    # Teste dos tipos dos parâmetros.
    if type(image) is not np.ndarray:
        raise TypeError("O argumento ``image`` não é um objeto do tipo ``numpy.ndarray``.")

    if type(roi) is not ImagingROI:
        raise TypeError("O argumento ``roi`` não é um objeto do tipo ``ImagingROI``.")

    if type(wavelength) is not float:
        raise TypeError("O argumento ``wavelength`` não é um objeto do tipo ``float``.")

    # Calcula o nível do índice API para a imagem.
    api_level = 10 ** (-6.0/20.0) * max(abs(np.amin(image)), abs(np.amax(image)))

    # Calcula o índice API da imagem.
    return image[abs(image) >= api_level].size * roi.w_step * roi.h_step * 1e-6 / (wavelength ** 2)


def cnr(foreground, background):
    """Função que calcula o índice API de uma imagem. Esse índice, definido em :cite:`Holmes2005` indica a área da
    imagem que se encontra com valores superiores a -6 dB. O patamar de 0 db é referente ao maior valor absoluto da
    imagem.

    Parameters
    ----------
        foreground : :class:`np.ndarray`
            Imagem da frente na forma de um *array* bidimensional.

        background : :class:`np.ndarray`
            Imagem de fundo na forma de um *array* bidimensional.
    Returns
    -------
        : float
            Razão contraste ruído.

    Raises
    ------

    TypeError
        Gera exceção de ``TypeError`` se o parâmetro ``foreground`` ou ``background`` não for do tipo
        :class:`np.ndarray`.

    """

    if type(foreground) is not np.ndarray:
        raise TypeError("O argumento ``foreground`` não é um objeto do tipo ``numpy.ndarray``.")

    if type(background) is not np.ndarray:
        raise TypeError("O argumento ``background`` não é um objeto do tipo ``numpy.ndarray``.")

    c = foreground.mean() - background.mean()
    n = background.std()
    return c/n
