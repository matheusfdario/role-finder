# -*- coding: utf-8 -*-
"""
Módulo ``fk_mig``
=================

A *Transformação* ou *migração de Stolt* é um operador que faz o mapeamento de sinais que estão no domínio de
frequências :math:`k_u` e :math:`\omega` para o domínio de frequências :math:`k_x` e :math:`k_z` :cite:`Gough1997`.
Ela foi utilizada inicialmente no processamento de sinais sísmicos :cite:`Stolt1978`. O mapeamento no sentido inverso
existe e é chamado por :cite:`Claerbout2004` como *modelagem de Stolt*. A migração e a modelagem de Stolt são
operadores lineares adjuntos entre si :cite:`Claerbout2004`. Dessa forma, a modelagem de Stolt é representada por
:math:`S^{\dagger}[\cdot]`.

Segundo :cite:`Margrave2003`, a transformação de Stolt é geralmente feita por uma interpolação dos pontos espectrais que
estão no domínio :math:`(k_u,\omega)`, que formam uma grade linear, para o domínio :math:`(k_x,k_z)`, cuja grade é não
linear. Essas duas grades são mostradas na :numref:`fig_StoltMig`. A migração de Stolt é utilizada nos algoritmos de
reconstrução de imagens que trabalham como os dados no domínio da frequência, tais como *wk_saft* e *wavenumber*.

.. figure:: figures/StoltMig.png
    :name: fig_StoltMig
    :align: center

    Mapeamento de domínio da transformação de Stolt: os pontos da grade do domínio :math:`(k_u,\omega)` estão
    assinalados pelos círculos, enquanto os pontos da grade do domínio :math:`(k_x,k_z)` estão assinalados
    pelas cruzes.

Este módulo contém as funções :func:`f_k_sweep` e :func:`f_k_fmc`, que calculam os *grids* para o mapeamento da
Transformada de Stolt para um conjunto de sinais *A-scan* coletados por varredura de transdutor simples (*sweep*) ou
FMC (*Full Matrix Capture*). Essas funções são utilizadas nos algoritmos :mod:`imaging.wk_saft` e
:mod:`imaging.wavenumber`. Além delas, também faz parte do módulo a função :func:`nextpow2`, que calcula o valor inteiro potência de
2 imediatamente superior ao argumento passado.

.. raw:: html

    <hr>

"""
import numpy as np
from framework.linterp_oper import linterp


def nextpow2(a):
    """Calcula o valor inteiro que é potência de 2 imediatamente superior ao parâmetro ``a``.

    Parameters
    ----------
    a : int, float
        Valor que se deseja encontrar o número potência de 2 imediatamente superior.

    Returns
    -------
    int
        Valor inteiro que é potência de 2 imediatamente superior ao parâmetro ``a``.
    """
    
    return np.ceil(np.log(a) / np.log(2)).astype(int)


def f_k_sweep(cc, k_z, k_u):
    r"""Função que calcula a grade de frequências temporais :math:`\omega` para aplicar a Transformada de Stolt em um
    conjunto de sinais *A-scan* coletados por varredura simples de um transdutor único. Essa equação foi retirada
    de :cite:`Stepinski2007`.

    Parameters
    ----------
    cc : float
        Velocidade do som no material inspecionado, em m/s.

    k_z : :class:`np.ndarray`
        *Array* com a grade de valores para a frequência espacial :math:`k_z`.

    k_u : :class:`np.ndarray`
        *Array* com a grade de valores para a frequência espacial :math:`k_u`.

    Returns
    -------
    :class:`np.ndarray`
        *Array* com a grade de valores para a frequência temporal :math:`\omega`.
    
    """

    f_k = (cc / 2.0) * np.sign(k_z) * np.sqrt(k_u ** 2 + k_z ** 2)

    return f_k


def f_k_fmc(cc, k_z, k_x, k_u):
    r"""Função que calcula a grade de frequências temporais :math:`\omega` para aplicar a Transformada de Stolt em um
     conjunto de sinais *A-scan* coletados em FMC. Essa equação foi retirada de :cite:`Hunter2008`.

    Parameters
    ----------
    cc : float
        Velocidade do som no material inspecionado, em m/s.

    k_z : :class:`np.ndarray`
        *Array* com a grade de valores para a frequência espacial :math:`k_z`.

    k_x : :class:`np.ndarray`
        *Array* com a grade de valores para a frequência espacial :math:`k_x`.

    k_u : :class:`np.ndarray`
        *Array* com a grade de valores para a frequência espacial :math:`k_u`.

    Returns
    -------
    :class:`np.ndarray`
        *Array* com a grade de valores para a frequência temporal :math:`\omega`.

    """

    np.seterr(divide='ignore', invalid='ignore')
    f_k = (cc / 2.0) * np.sqrt(k_z ** 4 + 2 * k_z ** 2 * (k_u ** 2 + (k_x - k_u) ** 2) +
                               k_u ** 4 + (k_x - k_u) ** 4 - 2 * k_u ** 2 * (k_x - k_u) ** 2) / k_z
    f_k[np.isnan(f_k)] = 0.0
    f_k[np.isinf(f_k)] = 0.0

    return f_k


def e_wavenumber_f_shift(k, kz, ku, f, c):
    """Calcula os deslocamentos de frequência temporal necessários para
    execução do algoritmo E-wavenumber. 

    Parameters
    ----------
    k : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`x` da imagem. Deve
        possuir dimensão (n_k,).

    kz : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`z` da imagem. Deve
        possuir dimensão (n_kz,).

    ku : np.ndarray
        Frequências espacial constante para cada slice. 

    f : np.ndarray
        Vetor com o grid de frequências dos dados de aquisição. Deve possuir
        dimensão (n_f,).

    c : float, int
        Velocidade de propagação da onda.

    Returns
    -------
    f_shift : np.ndarray
        Matriz de dimensão (n_kz, n_k) com os desvios de frequências para cada
        :math:`k` and :math:`k_z`.

    i_f : np.ndarray
        Matriz de dimensão (n_kz, n_k) com os índices das frequências
        :math:`f` utilizadas para obter o desvio de frequência.
        
    """
    kz = kz.reshape(-1, 1)
    k = k.reshape(1, -1)
    
    f_mig = f_k_fmc(c, kz, k, ku)

    i_f = e_fk_bin_from_f(f_mig, f[1] - f[0], f[0], f.shape[0])

    f_shift = f[i_f] - f_mig

    return f_shift, i_f


def e_fk_f_shift(k, kz, f, c):
    """Calcula os deslocamentos de frequência temporal necessários para
    execução do algoritmo E-wk.

    Parameters
    ----------
    k : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`x` da imagem. Deve
        possuir dimensão (n_k,).

    kz : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`z` da imagem. Deve
        possuir dimensão (n_kz,).

    f : np.ndarray
        Vetor com o grid de frequências dos dados de aquisição. Deve possuir
        dimensão (n_f,).

    c : float, int
        Velocidade de propagação da onda.

    Returns
    -------
    f_shift : np.ndarray
        Matriz de dimensão (n_kz, n_k) com os desvios de frequências para cada
        :math:`k` and :math:`k_z`.

    i_f : np.ndarray
        Matriz de dimensão (n_kz, n_k) com os índices das frequências
        :math:`f` utilizadas para obter o desvio de frequência.
        
    """
    kz = kz.reshape(-1, 1)
    k = k.reshape(1, -1)

    f_mig = (c / 2) * np.sign(kz) * np.sqrt(kz ** 2 + k ** 2)

    i_f = e_fk_bin_from_f(f_mig, f[1] - f[0], f[0], f.shape[0])

    f_shift = f[i_f] - f_mig

    return f_shift, i_f


def e_fk_bin_from_f(f, df, f_min, N):
    """Retorna os índices dos valores que mais se aproximam de ``f`` em um
    grid regular.

    Parameters
    ----------
    f : np.ndarray
        Vetor com valores para se obter índices.

    df : int, float
        Espaçamento do grid.

    f_min : int, float
        Valor mínimo do grid.

    N : int
        Número de pontos máximo no grid.

    Returns
    -------
    i : np.ndarray
        Vetor com índices.
        
    """
    i = np.rint((f - f_min) / df).astype(np.int64)
    #i = ((f - f_min)/df).astype(np.int64)

    i[i < 0] = 0
    i[i >= N] = N - 1
        
    return i


def e_fk_migrate(S_kx_f, f_shift, i_f, t, k, kx):
    """Realiza a migração de Stolt por deslocamento de frequência.

    Parameters
    ----------
    S_kx_f : np.ndarray
        Espectro dos dados de aquisição, com dimensão (n_f, n_k).

    f_shift : np.ndarray
        Deslocamentos de frequência temporal, com dimensão (n_kz, n_k) se a
        migração é realizada no eixo z primeiramente ou dimensão (n_kz, n_kx)
        se a migração é realizada no eixo x primeiramente.

    i_f : np.ndarray
        Índices associados aos desvios de frequência, com dimensão (n_kz, n_k)
        se a migração é realizada no eixo z primeiramente ou dimensão
        (n_kz, n_kx) se a migração é realizada no eixo x primeiramente.

    t : np.ndarray
        Vetor com os tempos dos dados de aquisição, com dimensão (n_f,).

    k : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`x` dos dados de
        aquisição, com dimensão (n_k,).
        
    kx : np.ndarray
        Frequências espaciais relacionadas ao eixo :math:`x` da imagem. com
        dimensão (n_kx,). Só é necessário caso interpolação no eixo :math:`x`
        for utilizada.

    Returns
    -------
    S : np.ndarray
        Espectro da imagem migrada.
        
    """
    # Faz a migração no eixo z utilizando deslocamentos de frequência
    # (algoritmo E-wk). S_z é o espectro da imagem após migração do espectro
    # de aquisição no eixo z.
    S_z = e_fk_migrate_z(S_kx_f, f_shift, i_f, t)

    # Faz a migração no eixo x utilizando interpolação. S é o espectro da
    # imagem. Pula interpolação se k e kx forem iguais.
    if (k.shape == kx.shape) and (np.allclose(k, kx) is True):
        return S_z
    
    n_kz = f_shift.shape[0]
    k_mig = np.tile(kx, (n_kz, 1))
    S = linterp(k, k_mig.T, S_z.T).T
                    
    return S


def e_fk_migrate_z(S_kx_f, f_shift, i_f, t):
    """Realiza a migração de Stolt no eixo :math:`z`, com deslocamentos de
    frequência.

    Parameters
    ----------
    S_kx_f : np.ndarray
        Espectro dos dados de aquisição, com dimensão (n_f, n_k).

    f_shift : np.ndarray
        Deslocamentos de frequência temporal, com dimensão (n_kz, n_k).

    i_f : np.ndarray
        Índices associados aos desvios de frequência, com dimensão (n_kz, n_k).

    t : np.ndarray
        Vetor com os tempos dos dados de aquisição, com dimensão (n_f,).

    Returns
    -------
    S_z : np.ndarray
        Espectro dos dados migrados no eixo z.

    """
    # Dimensões
    n_kz = f_shift.shape[0]
    n_k = S_kx_f.shape[1]
    n_f = t.shape[0]

    # Padding sub-índice
    S_l_kx_z = np.zeros((n_kz, 4), dtype=complex)

    # Espectro da imagem
    S_z = np.zeros((n_kz, n_k), dtype=complex)

    _i = np.array([-2, -1, 0, 1])
    _iz = np.array([2, 1, 0, -1])
    _ik = np.arange(4)
    I_z = (np.arange(n_kz).reshape(-1, 1) + _iz) % n_kz

    # DFT
    k = np.arange(4)
    n = np.arange(4).reshape(-1, 1)
    W = np.exp(-1j * 2 * np.pi * k * n / 4)
    W_inv = 1 / 4 * np.exp(1j * 2 * np.pi * k * n / 4)

    for j in range(n_k):
        i_ff = i_f[:, j]
        ff = f_shift[:, j].reshape(-1, 1)
        
        I_f = (_i + i_ff.reshape(-1, 1)) % n_f

        S_l_kx_z[:, 2] = S_kx_f[i_ff, j]

        #s_l_kx = np.fft.ifft(S_l_kx_z, axis=1)
        s_l_kx = S_l_kx_z @ W_inv

        s_ll_kx = s_l_kx * np.exp(1j*2*np.pi*ff*t[I_f])

        #S_ll_kx = np.fft.fft(s_ll_kx, axis=1)
        S_ll_kx = s_ll_kx @ W

        #S_z[:, j] = S_ll_kx[:, 2]
        #for i in range(4):
        #    S_z[I_z[:, i], j] += S_ll_kx[:, i]
        S_z[:, j] = np.sum(S_ll_kx[I_z, _ik], axis=1)

    return S_z
