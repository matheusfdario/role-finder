# -*- coding: utf-8 -*-
"""
Módulo ``linterp_oper``
=======================

Na Álgebra Linear, uma *transformação linear*, ou *operador linear* é uma função que mapeia (projeta) vetores entre dois
espaços vetoriais :math:`U` e :math:`V`. Quando esses espaços vetoriais são compostos por vetores de dimensões finitas,
os operadores lineares são representados por matrizes.
O *operador adjunto* é definido como o operador que realiza a projeção contrária, ou "retroprojeção",
feita anteriormente por um operador linear :cite:`Claerbout2004`. Quando o operador linear é uma matriz real, o operador
adjunto é a matriz transposta :cite:`Barrett2003`.

A operação de interpolação linear de pontos para uma função pode ser modelada como um operador linear
:cite:`Claerbout2004`.
Este módulo implementa a operação de interpolação linear como um operador e também disponibiliza o seu operador
transposto. Essa implementação foi baseada nos pseudo-códigos disponíveis em :cite:`Margrave2003` e
:cite:`Claerbout2004`.

.. figure:: figures/linterp_oper.png
    :name: fig_linterp_oper
    :width: 40 %
    :align: center

    Representação da interpolação linear como uma multiplicação matriz-vetor. Cada linha tem exatamente dois valores
    diferentes de zero que interpolam um valor entre dois elementos da entrada :math:`x`. Esta representação somente
    esboça a aparência da matriz, usando os valores :math:`a` e :math:`b` como seus elementos. Em cada linha, :math:`a`
    e :math:`b` são diferentes numericamente, mas deve ser respeitada a relação :math:`a + b = 1`.

.. raw:: html

    <hr>

"""
import numpy as np
from numba import njit


def linterp(x, xq, y, op="d"):
    """ Função que implementa os operadores lineares direto/adjunto da interpolação linear 1-D.

    Parameters
    ----------
        x : :class:`np.ndarray`
            *Array* com os valores da abscissa com os pontos conhecidos para a interpolação.

        xq : :class:`np.ndarray`
            *Array* com os valores da abcissa em que se quer interpolar novos pontos.

        y : :class:`np.ndarray`
            *Array* com os valores da ordenada para os pontos definidos em ``x``.

        op : str
            Tipo de operador, 'd' é para *operador direto* e 'a' é para *operador adjunto*.

    Returns
    -------
        yq : :class:`np.ndarray`
            *Array* com os valores das ordenadas nos pontos ``xq`` (se ``op=d``) ou nos pontos ``x`` (se ``op=a``).

    """

    # Testa o parâmetro indicativo do tipo de operador. 'd' é direto e 'a' é adjunto.
    if op not in {"a", "d"}:
        # Qualquer outro parâmetros diferente dos permitidos gera uma exceção.
        raise TypeError("Parâmetro ``op`` inválido. Só é permitido 'a' ou 'd'.")

    # Reorienta ``x`` para um vetor-coluna.
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    elif x.shape[0] > 1 and x.shape[1] > 1:
        raise TypeError("``x`` deve ser um vetor.")

    # Reorienta ``xq`` para um vetor-coluna (esse parâmetro pode ser uma matriz de vetores-coluna).
    if len(xq.shape) == 1:
        xq = xq[:, np.newaxis]
    elif xq.shape[0] == 1 and xq.shape[1] > 1:
        xq = xq.T

    # Reorienta ``y`` para um vetor-coluna (esse parâmetro pode ser uma matriz de vetores-coluna).
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    elif y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T

    # Verifica as dimensões dos parâmetros de entrada.
    if op == "d":
        assert y.shape[0] == x.shape[0], "Dimensões de ``x`` e ``y`` não são compatíveis."
        assert y.shape[1] == xq.shape[1], "Dimensões de ``y`` e ``xq`` não são compatíveis."
    else:
        assert y.shape == xq.shape, "Dimensões de ``y`` e ``xq`` não são compatíveis."

    # Aloca espaço para a variável de saída.
    yq = np.zeros(xq.shape).astype(y.dtype) if op == "d" else \
        np.zeros((x.shape[0], xq.shape[1])).astype(y.dtype)
    dx = x[1, 0] - x[0, 0]
    n = yq.shape[0]
    y_is_complex = np.iscomplexobj(y)

    # Varre todas as colunas de ``y``.
    for c in range(y.shape[1]):
        # Calcula os deslocamentos de ``xq``.
        f = (xq[:, c] - x[0, 0]) / dx

        # Calcula os índices de ``xq`` em ``x``.
        im = np.floor(f).astype(int)
        im_min = np.searchsorted(xq[:, c], x[0, 0]) - 1
        im_max = np.searchsorted(xq[:, c], x[-1, 0]) + 1

        # Ajusta índices.
        if im_min > 0:
            im[0: im_min + 1] = 0
            im[im < 0] = 0
        else:
            im[im < 0] = 0

        if im_max < len(f):
            im[im_max - 1:] = x.shape[0] - 2
            im[im > x.shape[0] - 2] = x.shape[0] - 2
        else:
            im[im > x.shape[0] - 2] = x.shape[0] - 2

        # Calcula os fatores de interpolação.
        fx = f - im
        gx = 1 - fx

        if im_min > 0:
            fx[0: im_min] = 0.0
            gx[0: im_min] = 0.0

        if im_max < len(f):
            fx[im_max:] = 0.0
            gx[im_max:] = 0.0

        # Aplica o operador correto.
        if op == "d":
            # Interpola os valores de ``y``.
            yq[:, c] = gx * y[im, c] + fx * y[im + 1, c]

        else:
            # Calcula o adjunto da interpolação.
            gx_factor = gx * y[:, c]
            fx_factor = fx * y[:, c]
            if y_is_complex:
                p_g = np.bincount(im,
                                  weights=np.real(gx_factor), minlength=n) + np.bincount(im,
                                                                                         weights=np.imag(gx_factor),
                                                                                         minlength=n) * 1j

                p_f = np.bincount(im + 1,
                                  weights=np.real(fx_factor), minlength=n) + np.bincount(im + 1,
                                                                                         weights=np.imag(fx_factor),
                                                                                         minlength=n) * 1j
            else:
                p_g = np.bincount(im, weights=gx_factor, minlength=n)
                p_f = np.bincount(im + 1, weights=fx_factor, minlength=n)

            ampl_comp = x.shape[0] / xq.shape[0]
            yq[:, c] = (p_g + p_f) * ampl_comp

    return yq


def linterp_numba(x, xq, y, op="d"):
    """ Função que implementa os operadores lineares direto/adjunto da interpolação linear 1-D.
    Essa função é otimizada para utilizar o pacote NUMBA.

    Parameters
    ----------
        x : :class:`np.ndarray`
            *Array* com os valores da abscissa com os pontos conhecidos para a interpolação.

        xq : :class:`np.ndarray`
            *Array* com os valores da abcissa em que se quer interpolar novos pontos.

        y : :class:`np.ndarray`
            *Array* com os valores da ordenada para os pontos definidos em ``x``.

        op : str
            Tipo de operador, 'd' é para *operador direto* e 'a' é para *operador adjunto*.

    Returns
    -------
        yq : :class:`np.ndarray`
            *Array* com os valores das ordenadas nos pontos ``xq`` (se ``op=d``) ou nos pontos ``x`` (se ``op=a``).

    """

    # Testa o parâmetro indicativo do tipo de operador. 'd' é direto e 'a' é adjunto.
    if op not in {"a", "d"}:
        # Qualquer outro parâmetros diferente dos permitidos gera uma exceção.
        raise TypeError("Parâmetro ``op`` inválido. Só é permitido 'a' ou 'd'.")

    # Reorienta ``x`` para um vetor-coluna.
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    elif x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    elif x.shape[0] > 1 and x.shape[1] > 1:
        raise TypeError("``x`` deve ser um vetor.")

    # Reorienta ``xq`` para um vetor-coluna (esse parâmetro pode ser uma matriz de vetores-coluna).
    if len(xq.shape) == 1:
        xq = xq[:, np.newaxis]
    elif xq.shape[0] == 1 and xq.shape[1] > 1:
        xq = xq.T

    # Reorienta ``y`` para um vetor-coluna (esse parâmetro pode ser uma matriz de vetores-coluna).
    if len(y.shape) == 1:
        y = y[:, np.newaxis]
    elif y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T

    # Verifica as dimensões dos parâmetros de entrada.
    if op == "d":
        assert y.shape[0] == x.shape[0], "Dimensões de ``x`` e ``y`` não são compatíveis."
        assert y.shape[1] == xq.shape[1], "Dimensões de ``y`` e ``xq`` não são compatíveis."
        op_direct = True
    else:
        assert y.shape == xq.shape, "Dimensões de ``y`` e ``xq`` não são compatíveis."
        op_direct = False

    # Aloca espaço para a variável de saída.
    yq = np.zeros(xq.shape).astype(y.dtype) if op == "d" else \
        np.zeros((x.shape[0], xq.shape[1])).astype(y.dtype)
    dx = x[1, 0] - x[0, 0]
    n = yq.shape[0]

    # Executa o laço principal da intermpolação linear.
    loop_linterp(x, y, xq, yq, dx, n, op_direct)

    return yq


@njit()
def loop_linterp(x, y, xq, yq, dx, n, op_direct):
    # Varre todas as colunas de ``y``.
    for c in range(y.shape[1]):
        # Calcula os deslocamentos de ``xq``.
        f = (xq[:, c] - x[0, 0]) / dx

        # Calcula os índices de ``xq`` em ``x``.
        im = np.floor(f).astype(np.int32)
        im_min = np.searchsorted(xq[:, c], x[0, 0]) - 1
        im_max = np.searchsorted(xq[:, c], x[-1, 0]) + 1

        # Ajusta índices.
        if im_min > 0:
            im[0: im_min + 1] = 0
            im[im < 0] = 0
        else:
            im[im < 0] = 0

        if im_max < len(f):
            im[im_max - 1:] = x.shape[0] - 2
            im[im > x.shape[0] - 2] = x.shape[0] - 2
        else:
            im[im > x.shape[0] - 2] = x.shape[0] - 2

        # Calcula os fatores de interpolação.
        fx = f - im
        gx = 1 - fx

        if im_min > 0:
            fx[0: im_min] = 0.0
            gx[0: im_min] = 0.0

        if im_max < len(f):
            fx[im_max:] = 0.0
            gx[im_max:] = 0.0

        # Aplica o operador correto.
        if op_direct:
            # Interpola os valores de ``y``.
            yq[:, c] = gx * y[im, c] + fx * y[im + 1, c]

        else:
            # Calcula o adjunto da interpolação.
            gx_factor = gx * y[:, c]
            fx_factor = fx * y[:, c]
            p_g = bincount_numba(im, weights=gx_factor, minlength=n)
            p_f = bincount_numba(im + 1, weights=fx_factor, minlength=n)
            ampl_comp = x.shape[0] / xq.shape[0]
            yq[:, c] = (p_g + p_f) * ampl_comp

    return


@njit()
def bincount_numba(x, weights, minlength=0):
    result = np.zeros(max(minlength, x.max() + 1), weights.dtype)
    for i in range(len(x)):
        n = x[i]
        result[n] += weights[i]

    return result
