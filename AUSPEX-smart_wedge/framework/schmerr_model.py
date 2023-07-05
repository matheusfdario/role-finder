# -*- coding: utf-8 -*-
"""
Módulo ``schmerr_model``
=====================

O módulo :mod:`.schmerr_model` contém as funções de suporte para a modelagem
de sistemas de inspeção por ultrassom baseados nas referências escritas por
Schmerr :cite:`Schmerr1998,Schmerr2007,Schmerr2016`.

TODO: complementar essa documentação.

Essas funções são utilizadas em vários algoritmos de reconstrução de imagens. ``Escrever sobre eles``.

.. raw:: html

    <hr>

"""
import numpy as np
from scipy.special import j1 as besselj1
from scipy.special import struve
from scipy.special import sinc

from framework.fk_mig import nextpow2


def jinc(x):
    """Docstring da função ``jinc``.

    TODO: Escrever aqui a documentação da função ``jinc``.
    """
    np.seterr(divide='ignore', invalid='ignore')
    y = np.real(besselj1(x) / x)
    y[np.isnan(y)] = 0.5

    return y


def cossquare(f, fc, bw):
    """Docstring da função ``cossquare``.

    TODO: Escrever aqui a documentação da função ``cossquare``.
    """
    f1 = fc - bw * fc
    f4 = fc + bw * fc

    if not np.isscalar(f) and f.ndim > 2:
        raise TypeError("``f`` deve ser um escalar, um vetor ou uma matriz [2x2]")

    if np.isscalar(f):
        if (np.abs(f) >= f1) and (np.abs(f) <= f4):
            if np.abs(f) > fc:
                r = np.cos(np.pi * (fc - np.abs(f)) / (f4 - f1)) ** 2

            else:
                r = np.sin(np.pi * f / (2 * fc)) * np.cos(np.pi * (fc - np.abs(f)) / (f4 - f1)) ** 2

        else:
            r = 0.0

    elif f.ndim == 1 or f.shape[0] == 1 or f.shape[1] == 1:
        abs_f = np.abs(f)
        ss = np.sin(np.pi * f / (2.0 * fc))
        ri1 = np.logical_and(abs_f > fc, abs_f <= f4)
        ri2 = np.logical_and(abs_f <= fc, abs_f <= f4)
        ff1 = ri1 * abs_f
        ff2 = ri2 * abs_f
        r1 = ri1 * np.cos(np.pi * (fc - ff1) / (f4 - f1)) ** 2.0
        r2 = ri2 * ss * np.cos(np.pi * (fc - ff2) / (f4 - f1)) ** 2.0
        r = (r1 + np.sign(f) * r2)

    else:
        r = np.zeros(f.shape)
        for c in range(f.shape[1]):
            ff = f[:, c]
            abs_ff = np.abs(ff)
            ss = np.sin(np.pi * ff / (2.0 * fc))
            ri1 = np.logical_and(abs_ff > fc, abs_ff <= f4)
            ri2 = np.logical_and(abs_ff <= fc, abs_ff <= f4)
            ff1 = ri1 * abs_ff
            ff2 = ri2 * abs_ff
            r1 = ri1 * np.cos(np.pi * (fc - ff1) / (f4 - f1)) ** 2.0
            r2 = ri2 * ss * np.cos(np.pi * (fc - ff2) / (f4 - f1)) ** 2.0
            r[:, c] = (r1 + np.sign(ff) * r2)

    return r


def gaussian(f, fc, bw):
    """Docstring da função ``gaussian``.

    TODO: Escrever aqui a documentação da função ``gaussian``.
    """
    a = np.sqrt(np.log(2)) / (np.pi * bw * fc)

    if not np.isscalar(f) and f.ndim > 2:
        raise TypeError("``f`` deve ser um escalar, um vetor ou uma matriz [2x2]")

    if np.isscalar(f) or f.ndim == 1 or f.shape[0] == 1 or f.shape[1] == 1:
        abs_f = np.abs(f)
        gt_fc = abs_f > fc
        le_fc = abs_f <= fc
        s1 = np.exp(-(2.0 * a * np.pi * (abs_f - fc)) ** 2.0) * gt_fc
        s2 = np.exp(-(2.0 * a * np.pi * (abs_f - fc)) ** 2.0) * np.sin(np.pi * f / (2.0 * fc)) * le_fc
        r = (s1 + np.sign(f) * s2)

    else:
        r = np.zeros(f.shape)
        for c in range(f.shape[1]):
            ff = f[:, c]
            abs_ff = np.abs(ff)
            gt_fc = abs_ff > fc
            le_fc = abs_ff <= fc
            s1 = np.exp(-(2.0 * a * np.pi * (abs_ff - fc)) ** 2.0) * gt_fc
            s2 = np.exp(-(2.0 * a * np.pi * (abs_ff - fc)) ** 2.0) * np.sin(np.pi * ff / (2.0 * fc)) * le_fc
            r[:, c] = (s1 + np.sign(ff) * s2)

    return r


def calculate_matched_filter(nt, nu, dt, du, dim, fc, bw, ermv, hed_gain=1.0, type_pulse="gaussian", t1=100e-9):
    """Docstring da função ``calculate_matched_filter``.

    TODO: Escrever aqui a documentação completa da função ``calculate_matched_filter``.
    """
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    ku = np.fft.fftshift(np.fft.fftfreq(nu, d=du))
    ku_f, f_ku = np.meshgrid(ku, f)

    f_ku = f_ku + np.finfo(f_ku.dtype).eps * (f_ku == 0)
    mask = np.abs(ku_f / (f_ku / ermv)) <= 1.0
    c1 = mask * jinc(np.pi * ku_f * dim / 2.0) ** 2.0

    if type_pulse in ["gaussian"]:
        hed = gaussian(f, fc, bw) ** 2.0

    elif type_pulse in ["cossquare"]:
        hed = cossquare(f, fc, bw) ** 2.0

    else:
        raise TypeError("Tipo inválido para o modelo de pulso do transdutor")

    # Espectro do pulso de excitação do transdutor (pulso retangular com largura t1).
    if t1 != 0.0:
        exc_transd = -np.exp(-2j * np.pi * f * (t1 / 2.0))
        exc_transd = 2.0 * np.pi * sinc(2.0 * f * (t1 / 2.0)) * exc_transd
    else:
        exc_transd = 1.0

    # Eq. 12.13 de Schmerr (1998). Some o termo ``j`` em c1 pela explicação na nota da pág. 3 de Hunter (2008).
    hed = hed_gain * hed * exc_transd
    c1 = c1 * ((np.pi / 2.0) * ((2.0 * (dim / 2.0) ** 2.0 * (2.0 * np.pi) / ermv) * np.abs(f)))[:, np.newaxis]

    return c1, hed


def calculate_scattering_amplitude(nt, dt, c, flaw_type, dimmension, tilt=0.0):
    """Docstring da função ``calculate_scattering_amplitude``.

    TODO: Escrever aqui a documentação completa da função ``calculate_scattering_amplitude``.
    """
    # Cria os grids de frequência e kb.
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    kb = -(2.0 * np.pi * dimmension * f) / c
    kb2 = 2.0 * kb
    kb = kb + np.finfo(kb.dtype).eps * (kb == 0)
    kb2 = kb2 + np.finfo(kb2.dtype).eps * (kb2 == 0)

    # Calcula a amplitude de espalhamento para cada tipo de descontinuidade, utilizando as aproximações de Kirchhoff.
    if flaw_type in ['circ-crack', 'penny-shaped']:
        # Trinca circular (*circular crack*).
        tilt_rad = np.deg2rad(tilt)
        arg = np.sin(tilt_rad) * kb2
        arg = arg + np.finfo(arg.dtype).eps * (arg == 0)
        sa = 1j * kb * dimmension * np.cos(tilt_rad) * besselj1(arg) / arg

    elif flaw_type in ['side-drilled', 'sdh']:
        # Furo passante (*Side-Drilled Hole*). O fator de divisão 190.0 foi encontrado empiricamente.
        sa = - ((kb / 2.0) * (besselj1(kb2) - 1j * struve(1, kb2)) + 1j * kb / np.pi) / 190.0

    elif flaw_type in ['spherical']:
        # Cavidade esférica (*spherical cavity*).
        sa = -(dimmension / 2.0) * np.exp(-1j * kb) * (np.exp(-1j * kb) - sinc(kb))

    else:
        # Todos os outros tipos não estão implementados.
        sa = np.ones(kb.shape) / 1000.0

    return sa


def generate_model_filter(data_insp, c=5900.0, hed_gain=1500.0 * np.pi, t1=100.0e-9,
                          flaw_type=None, dimmension=0.0, tilt=0.0, filter_type="matched", ep=0.03):
    """Docstring da função ``generate_model_filter``.

    TODO: Escrever aqui a documentação completa da função ``generate_model_filter``.
    """
    dt = (data_insp.time_grid[1, 0] - data_insp.time_grid[0, 0]) * 1e-6
    nt = int(2.0 ** (nextpow2(data_insp.ascan_data.shape[0]) + 1))

    if data_insp.inspection_params.type_capt == "sweep":
        du = (data_insp.inspection_params.step_points[1, 0] - data_insp.inspection_params.step_points[0, 0]) * 1e-3
        nu = int(2.0 ** (nextpow2(data_insp.inspection_params.step_points.shape[0]) + 1))
    elif data_insp.inspection_params.type_capt == "FMC":
        du = data_insp.probe_params.pitch * 1e-3
        nu = int(2.0 ** (nextpow2(data_insp.probe_params.num_elem) + 1))
    else:
        raise NotImplementedError("Tipo de captura inválido. Só é permitido ``sweep`` e ``FMC``.")

    # Calcula o filtro casado (*matched filter*).
    c1, hed = calculate_matched_filter(nt, nu, dt, du,
                                       data_insp.probe_params.elem_dim * 1e-3,
                                       data_insp.probe_params.central_freq * 1e6,
                                       data_insp.probe_params.bw,
                                       c / 2.0,
                                       hed_gain,
                                       data_insp.probe_params.pulse_type,
                                       t1)

    # Calcula a amplitude de espalhamento, que depende do tipo de descontinuidade.
    sa = calculate_scattering_amplitude(nt, dt, c, flaw_type, dimmension, tilt)

    # Constrói o modelo.
    model = c1 * (hed * sa)[:, np.newaxis]

    # Contrói o filtro.
    if filter_type in ["wiener"]:
        # Filtro de Wiener.
        aa = np.conjugate(c1) / (np.multiply(c1, np.conjugate(c1)) + ep)
        filt = aa * hed[:, np.newaxis]
    else:
        # Filtro casado (matched filter).
        filt = np.conjugate(model)

    return model, filt, c1, hed, sa
