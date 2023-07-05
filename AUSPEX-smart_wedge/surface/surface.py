# -*- coding: utf-8 -*-
"""
Módulo ``surface``
==================




A classe :class:`surface.surface.Surface` é a principal classe do módulo e realiza as
funções de identificação da superfície e de fornecimento dos tempos :math:`T_{AF}` para um conjunto de
elementos
transdutor e um conjunto de pontos da ROI. Do ponto de vista da utilização da classe (nos algoritmos SAFT e TFM),
o processo é transparente, sendo necessário informar somente os dados de aquisição, as coordenadas dos elementos e da
ROI, e a velocidade do som no meio acoplante.

Exemplo
-------
O *script* abaixo mostra o uso da classe Surface para o cálculo dos tempos de percurso
entr o elemento central de um transdutor array e todos os pontos de uma determinada
região de interesse. Este uso tipicamente é feito por algoritmos de reconstrução de
imagens baseados em tempo de tercurso, como o SAFT e o TFM.


.. plot:: plots/surface/surface_example.py
    :include-source:
    :width: 60 %
    :align: center

.. raw:: html

    <hr>

"""

import numpy as np
import math
from scipy.signal import hilbert
from scipy.sparse.linalg import cg
from scipy.odr import Model, Data, ODR
from numba import jit, njit, prange
from enum import Enum
import scipy.spatial.distance as dist

import framework.data_types
import surface.nonlinearopt as nlo
from framework import data_types
from framework.post_proc import envelope
from imaging.cumulative_tfm import cumulative_tfm_kernel
from parameter_estimation import intsurf_estimation
import time
import matplotlib.pyplot as plt
import multiprocessing as mp


def cdist_arb_kernel(Fx, Fz, Sx, Sz, Tx, Tz, ang_critico, c1, c2, normal, tolerancia):
    resultcouplant = np.zeros([len(Tx), len(Fx)])
    resultmedium = np.zeros([len(Tx), len(Fx)])

    # distancia de cada ponto da superficie até cada ponto da roi
    ffx, ssx = np.meshgrid(Fx, Sx)
    ffz, ssz = np.meshgrid(Fz, Sz)
    dsf = np.sqrt((ffx - ssx) ** 2 + (ffz - ssz) ** 2)

    for trans in range(len(Tx)):
        # print(100 * trans / len(Tx))
        # parâmetros das retas transdutor-superfície no formato Rts: ax + bz + c = 0
        a = (Tz[trans] - Sz)
        b = (Sx - Tx[trans])
        c = (Tx[trans] * Sz - Tz[trans] * Sx)
        # normalizando para Rts: Ax + Bz + C = 0 de forma que A^2 + B^2 = 1
        norm = np.sqrt(a**2 + b**2)
        A = a / norm
        B = b / norm
        C = c / norm
        # Arts - ângulos da retas transdutor-superfície
        # sinArts = (Tz[trans] - Sz) / np.sqrt((Tz[trans] - Sz) ** 2 + (Sx - Tx[trans]) ** 2)
        # cosArts = (Tx[trans] - Sx) / np.sqrt((Tz[trans] - Sz) ** 2 + (Sx - Tx[trans]) ** 2)
        sinArts = a / norm
        cosArts = (Tx[trans] - Sx) / norm
        Arts = np.arctan2(sinArts, cosArts)
        # Ai - ângulos de incidência
        Ai = Arts - normal
        validos = np.abs(Ai) < ang_critico  # pontos da superficie que obedecem ao angulo crítico
        indexValidos = np.nonzero(validos)[0]  # indices dos pontos da superficie que obedecem ao angulo critico

        # Arr - ângulos da retas refratadas
        Arr = np.pi + normal[indexValidos] + np.arcsin(c2 * np.sin(Ai[indexValidos]) / c1)  # com culling
        sinArr = np.sin(Arr)
        cosArr = np.cos(Arr)

        # parâmetros das retas refratadas no formado Rr: Dx + Ez + F = 0
        D = np.asmatrix(sinArr)
        E = np.asmatrix(-cosArr)
        F = np.asmatrix(Sz[indexValidos] * cosArr - Sx[indexValidos] * sinArr)  # com culling

        # tempo de percurso do elemento até cada ponto valido da superficie
        dts = np.sqrt((Tx[trans] - Sx[indexValidos]) ** 2 + (Tz[trans] - Sz[indexValidos]) ** 2)

        erroFocal, penalidade = dist_kernel(D, E, F, Fx[np.newaxis, :], Fz[np.newaxis, :], tolerancia)

        # tempo total de percurso do transdutor até o ponto focal
        ttf = dsf[indexValidos, :] / c2 + dts[:, np.newaxis] / c1 + penalidade

        # encontra o índice do ponto de entrada
        try:
            indiceCandidatoMinimo = np.argmin(ttf, axis=0)
            indicePontoEntrada = indexValidos[indiceCandidatoMinimo]

            # calcula distâncias
            d1 = dts[indiceCandidatoMinimo]
            d2 = dsf[indicePontoEntrada, np.arange(len(Fx))]
        except:
            d1 = np.Inf
            d2 = np.Inf

        resultcouplant[trans, :] = d1
        resultmedium[trans, :] = d2
    # print(time.time()-t)

    return resultcouplant, resultmedium

@jit(nopython=True, fastmath=True)
def dist_kernel(D, E, F, Fx, Fz, tolerancia):
    # distâncias entre cada reta refratada e cada ponto da ROI (maior custo computacional)
    erroFocal = np.abs(D.T @ Fx + E.T @ Fz + F.T)
    candidatos = erroFocal < tolerancia
    penalidade = 1e3 * np.invert(candidatos)

    return erroFocal, penalidade


@jit(nopython=True, fastmath=True)
def vk(x_discr, z_discr, m, c1, c2, xa, za, xf, zf, k):
    da = np.sqrt((x_discr[k]-xa)**2 + (z_discr[k]-za) ** 2)
    df = np.sqrt((x_discr[k]-xf)**2 + (z_discr[k]-zf) ** 2)
    sa = (x_discr[k]-xa) + m[k]*(z_discr[k]-za)
    sf = (x_discr[k]-xf) + m[k]*(z_discr[k]-zf)
    r = (1/c1) * sa/da + (1/c2) * sf/df
    return r


@jit(nopython=True, fastmath=True)
def newtonraphsonk(x_discr, z_discr, m, c1, c2, xa, za, xf, zf, k0):
    k_old = k0
    k1 = k0
    for i in range(0, 100):

        vk0 = vk(x_discr, z_discr, m, c1, c2, xa, za, xf, zf, k0)
        vk0plus1 = vk(x_discr, z_discr, m, c1, c2, xa, za, xf, zf, k0 + 1)
        step = vk0 / (vk0plus1 - vk0)

        # This prevents a non convergence issue observed during tests of
        # the method. Without this "damping" procedure, the algorithm is
        # too sensible to the initial guess of k0 and may oscillate
        # divergently in some cases.
        if np.abs(step) > 30:
            step = step / 3
        k1 = np.int(k0 - np.round(step))

        if k1 >= x_discr.shape[0] - 1:
            k1 = x_discr.shape[0] - 2
        elif k1 < 0:
            k1 = 0

        if i > 0:
            if k0 == k1:
                #print('k0 == k1')
                break
            elif i > 2:
                if k1 == k_old:
                    #print('k1 == k_old')
                    break
        k_old = k0
        k0 = k1
    #print(k1, i)
    #if i > 50:
    #    print(xa, za)
    return k1


@jit(nopython=True, fastmath=True, parallel=False)
def newtonraphsonbatchelement(x_discr, z_discr, m, vmed, vmat, xa, za, xroi, zroi):
    x_interface = np.zeros_like(xroi) + 0.0
    z_interface = np.zeros_like(zroi) + 0.0
    k = np.int(x_discr.shape[0] / 2)
    for i in range(0, xroi.shape[0]):
        k = newtonraphsonk(x_discr, z_discr, m, vmed, vmat,
                           xa, za, xroi[i], zroi[i], k)
        x_interface[i] = x_discr[k]
        z_interface[i] = z_discr[k]
        # display(x_interface[i])
    return [x_interface, z_interface]


#@jit(nopython=True, fastmath=True, parallel=True)
def newtonraphsonbatchelement_continuous(c1, c2, xA, zA, xF_param, yF_param, zF, x0, maxit):
    # yA is not an argument because it is supposed to be zero.

    # Rotate on z to put the pixel (focus) on y=0
    # (Hello again.)
    eps = np.zeros(xF_param.shape)
    eps[xF_param == 0.] = 1e-7
    alpha = np.arctan(yF_param / (xF_param + eps))
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    xF = xF_param * cos_alpha + yF_param * sin_alpha
    # The line below is only illustrative. yF is not used here.
    # yF = -xF_param * sin_alpha + yF_param * cos_alpha

    x_current = x0
    f_values = np.array(range(maxit))
    for i in range(maxit):
        f = (x_current - xA) / (c1 * np.sqrt((x_current - xA) ** 2 + zA ** 2)) + (x_current - xF) / (
                c2 * np.sqrt((x_current - xF) ** 2 + np.power(zF, 2)))
        dfdx = -(x_current - xA) ** 2 / (
                c1 * ((x_current - xA) ** 2 + zA ** 2) ** (3 / 2)) + 1. / (
                       c1 * np.sqrt((x_current - xA) ** 2 + zA ** 2)) + 1. / (
                       c2 * np.sqrt((x_current - xF) ** 2 + zF ** 2)) - (x_current - xF) ** 2 / (
                       c2 * ((x_current - xF) ** 2 + zF ** 2) ** (3 / 2))
        x_current = x_current - f*(np.abs(dfdx) > 1e-10) / dfdx


    # Undo the rotation on z and return value
    x_unrotated = x_current * cos_alpha
    y_unrotated = x_current * sin_alpha

    z_interface = np.zeros_like(x_unrotated)
    return [x_unrotated, y_unrotated, z_interface]


@njit(parallel=True)
def kernel(f, g, rx_elements, tx_elements, nb_comb, samp_dist, gate_start, sample_freq):
    for comb in range(nb_comb):
        j = np.rint((samp_dist[rx_elements[comb], :] + samp_dist[tx_elements[comb], :]) - int(gate_start * sample_freq))
        j[j >= g.shape[0]] = -1
        for i in prange(j.size):
            ind = int(j[i])
            f[:, i] = f[:, i] + g[ind, tx_elements[comb], rx_elements[comb]]
    return f


def rotate_axis_y(xyz, alpha):
    x = xyz[:, 0]
    v = xyz[:, 1]
    z = xyz[:, 2]
    u = x * np.cos(alpha) - z * np.sin(alpha)
    w = x * np.sin(alpha) + z * np.cos(alpha)
    uvw = np.array([u, v, w])
    return uvw.transpose()

def rotate_axis_y_tensor(xyz, alpha):
    x = xyz[:, :, 0]
    v = xyz[:, :, 1]
    z = xyz[:, :, 2]
    u = x * np.cos(alpha) - z * np.sin(alpha)
    w = x * np.sin(alpha) + z * np.cos(alpha)
    uvw = np.array([u, v, w])
    return uvw.transpose(1, -1, 0)

def rotate_axis_y_inv(uvw, alpha):
    u = uvw[:, 0]
    v = uvw[:, 1]  # estava xyz, mudei para uvw já que xyz não existe aqui
    w = uvw[:, 2]
    x = u*np.cos(alpha) + w*np.sin(alpha)
    z = -u*np.sin(alpha) + w*np.cos(alpha)
    xyz = np.array([x, v, z])
    return xyz.transpose()

def movavg(signal, n=5):
    signal = np.array(signal)
    signal_size = signal.shape[0]
    output = np.zeros(signal_size)

    if n % 2 == 0:
        n = n+1

    n_ = np.int(np.floor(n/2))

    for i in range(signal.shape[0]):
        n_inff = i - n_
        if n_inff < 0:
            n_inff = 0
        n_supp = i + n_ + 1
        if n_supp > signal_size:
            n_supp = signal_size

        output[i] = np.sum(signal[n_inff:n_supp]) / (n_supp - n_inff)
    return output

class Lineparam:
    def __init__(self, a=0, b=0, SSE=np.inf, water_path=0):
        self.a = a
        self.b = b
        self.SSE = SSE
        self.water_path = water_path

class Circleparam:
    def __init__(self, x=0, z=0, r=0, SSE=np.inf, water_path=0):
        self.x = x
        self.z = z
        self.r = r
        self.SSE = SSE
        self.errors = 0
        self.speed = 0
        self.water_path = water_path

class Planeparam:
    def __init__(self, a=0, b=0, c=0, SSE=np.inf, water_path=0):
        self.a = a
        self.b = b
        self.c = c
        self.rot_x = 0  # Rotation along x axis following the right hand rule, where the thumb points to the
        # positive growth direction of the x axis.
        self.rot_y = 0  # Rotation along y axis following the right hand rule, where the thumb points to the
        # positive growth direction of the x axis.
        self.SSE = SSE
        self.water_path = water_path

class Cylinderparam:
    def __init(self, x1=0, y1=0, z1=0, x2=0, y2=0, z2=0, r=0, SSE=np.inf, water_path=0):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

        self.r = r
        self.direction_vector = np.zeros(3)
        self.translation_vector =  np.zeros(3)
        self.rot_x = 0
        self.rot_y = 0
        self.rot_z = 0
        self.SSE = SSE
        self.water_path = water_path


class Atfmparam:
    def __init__(self):
        self.xdiscr = np.zeros(0)
        self.zdiscr = np.zeros(0)

class Surface_cdist_memory:
    def __init__(self):
        self.coordelem = 0
        self.coordroi = 0
        self.returned_result = 0


class Surface:
    """Classe contendo informações sobre a superfície identificada no ensaio por imersão

        Parameters
        ----------
            data_insp : :class:`framework.DataInsp`
                Conjunto de dados de inspeção.

            xdczerototal : int
                Atraso (em número de amostras) imposto pelo transdutor aos sinais recebidos, independente das distâncias
                peroccirdas pelo pulso no meio acoplante. É a diferença entre qual amostra contém o valor máximo de um
                eco ultrassônico (considerando a envoltória) e qual amostra "deveria" conter tal valor máximo caso o
                transdutor não impusesse nenhum atraso. O valor padrão é ``0``.

            c_medium : float
                Velocidade de propagação no meio acoplante. Se não for informada ou o valor for ``0``, a classe
                utiliza o valor contido no ``data_insp``. O valor padrão é ``0``.

            keep_data_insp : bool
                Se ``True``, mantém o atributo ``data_insp`` após a execução do método construtor. Se ``False``,
                remove o atributo ``data_insp`` após a execução do método construtor de forma a reduzir o
                uso de memória. O valor padrão é ``False``.

        Returns
        -------
        None

        Attributes
        ----------
            surfacetype : :class:`surface.surface.SurfaceType`
                Tipo da superfície encontrada, bem como método de regressão utilizado para definir os parâmetros
                correspondentes.

            x_discr : :class:`numpy.array`
                Coordenadas horizontais do conjunto de pontos discretizados a partir da descrição analítica da
                superfície obtida pelos métodos de regressão.

            z_discr : :class:`numpy.array`
                Coordenadas verticais do conjunto de pontos discretizados a partir da descrição analítica da
                superfície obtida pelos métodos de regressão.

    """

    numelements = 0
    ElementPitch = 0
    CalibratedDelay = 0
    SamplingFreq = 0
    VelocityMedium = 0
    VelocityMaterial = 0
    xpivec = np.empty((1,))
    zpivec = np.empty((1,))
    surfacetype = None
    surfaceparam = None
    x_discr = np.empty((1,))
    z_discr = np.empty((1,))
    bscan = None
    pixel_in_material = np.empty((1,))
    entrypoints = np.empty((1,))
    m = np.empty((1,))
    k_interface = np.empty((1,))
    elementpos = np.empty((1,))
    gate_start = 0.0
    data_insp = None
    keep_data_insp = False

    def __del__(self):
        self.data_insp = None

    def encapsulateerrorcircle(self, theta):
        return self.sumsqudistbscancilinder(theta[0], theta[1], theta[2])

    def sumsqudistbscancircle(self, xc, zc, radius):
        center = np.zeros((1, 3))
        center[0, :] = np.asarray([xc, self.data_insp.probe_params.elem_center[0, 1], zc])
        centertoelem = dist.cdist(self.data_insp.probe_params.elem_center, center)
        distances = self.peak_distances_mm
        errors = np.subtract(np.array(distances) + radius, centertoelem.transpose())
        sse = np.sum(np.power(errors, 2))
        return sse

    def sumsqudistbscancilinder(self, xc, zc, radius):
        center = np.zeros((1, 3))
        center[0] = [xc, 0, zc]
        elem_center_xz = np.copy(self.data_insp.probe_params.elem_center)
        elem_center_xz[:, 1] = 0
        centertoelem = dist.cdist(elem_center_xz, center)
        distances = self.peak_distances_mm
        errors = np.subtract(np.array(distances) + radius, centertoelem.transpose())
        sse = np.sum(np.power(errors, 2))
        return sse

    def encapsulateerrorline(self, theta):
        return self.sumsqudistbscanline(theta[0], theta[1])

    def __encapsulateerrorplane(self, theta):
        # Where theta is the initial guess.
        return self.sumsqudistbscanplane(theta[0], theta[1], theta[2])

    def __encapsulateerrorcylinder(self, theta):
        # Where theta is the initial guess.
        return self.sumsqudistbscancylinder(theta[0], theta[1], theta[2], theta[3], theta[4])

    def sumsqudistbscanline(self, coef_a, coef_b):
        errors = np.zeros(self.numelements)
        for i in range(self.numelements):
            measured_dist = self.peak_distances_mm
            # Reference: ##############################################################
            # https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/

            # The parameters coef_a and coef_b represent the line y = coef_a*x + coef_b
            # The code below uses parameters a,b,c where a*x + b*y + c = 0
            # The conversion of parameters is as follows:
            a = -coef_a
            b = 1
            c = -coef_b
            x0 = self.data_insp.probe_params.elem_center[i, 0]
            z0 = 0
            guessed_dist = np.abs(a*x0 + b*z0 + c) / np.sqrt(a**2 + b**2)
            ###########################################################################
            errors[i] = guessed_dist - self.peak_distances_mm[i]
        sse = np.sum(np.power(errors, 2))
        return sse

    def sumsqudistbscancylinder(self,  x1, z1, x2, z2, r):
        """
                Calcula o a soma quadrática da diferença entre os pontos pertecentes à superfície externa do objeto,
                medidos pelo transdutor, e o cilindro descrito pelos parâmetros [x1, a, z1]^T, [x2, -a, z2]^T e r. O
                par de pontos [x1, a, z1]^T e [x2, -a, z2]^T irão definir uma reta que passa pelo centro do cilindro.

                    Parameters
                    ----------
                        self: surface

                        x1 : float
                            Coordenada x do vetor 1 que descreve a reta que passará pelo centro do cilindro

                        z1 : float
                            Coordenada z do vetor 1 que descreve a reta que passará pelo centro do cilindro

                        x2 : float
                            Coordenada x do vetor 2 que descreve a reta que passará pelo centro do cilindro

                        z2 : float
                            Coordenada z do vetor 2 que descreve a reta que passará pelo centro do cilindro

                        r : float
                            Raio do cilindro

                    Returns
                    -------
                    float
                        Resultado da soma quadrática dos erros.

                    """

        ai = self.data_insp.probe_params.elem_center[:]
        c1 = np.array([x1, self.surfaceparam.y1, z1], dtype=object)
        c2 = np.array([x2, self.surfaceparam.y2, z2], dtype=object)

        denominator = [np.linalg.norm(vec) for vec in (np.cross(ai - c1, ai - c2))]
        dist = denominator/np.linalg.norm(c2 - c1)
        delta = r + self.peak_distances_mm
        delta = delta.reshape(len(delta), 1)
        dist = dist.reshape(len(dist), 1)
        error = dist - delta
        sse = np.sum(np.power(error, 2))
        return sse

    def sumsqudistbscanplane(self, coef_a, coef_b, coef_c):
        """
        Calcula o a soma quadrática da diferença entre os pontos pertecentes à superfície externa do objeto,
        medidos pelo transdutor, e o plano descritor pelos parâmetros coef_a,coef_b e coef_c.

            Parameters
            ----------
                self: surface


                coef_a : float
                    Coeficiente a em : z = a*x + b*y + c

                coef_b : float
                    Coeficiente b em : z = a*x + b*y + c
                coef_c : float
                    Coeficiente c em : z = a*x + b*y + c

            Returns
            -------
            float
                Resultado da soma quadrática dos erros.

            """

        # Since we want to find A,B,C and D, but we only have the data of the points that belong to the
        # plane, which are internally represented by:
        # point(x0,y0,z0) = z0 = coef_a*x0 + coef_b*y0 + coef_c0
        # The desired parameters is present in the equation: a*x + b*y + c*z + d = 0
        #  Thus, the following conversion must be made:
        #  a = -coef_a, b = -coef_b, c = 1 and d = -coef_c
        # Source: https://mathinsight.org/distance_point_plane

        a = -coef_a
        b = -coef_b
        c = 1
        d = -coef_c

        # Coordinates of the transducer elements are:
        x0 = self.data_insp.probe_params.elem_center[:, 0]
        y0 = self.data_insp.probe_params.elem_center[:, 1]
        z0 = 0 # Since the coordinate z=0 plane always contains the transducer, z=0 for all cases.

        # The distance between a certain measured point (x0,y0,z0) is computed:
        computed_distance = np.abs(a * x0 + b * y0 + c * z0 + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        errors = computed_distance - self.peak_distances_mm[:]
        sse = np.sum(np.power(errors, 2))
        return sse

    def computedelay(self, elemindex, surfk, xf, zf):

        time1 = (1/self.VelocityMedium)*np.sqrt((self.elementpos[elemindex, 0]-self.x_discr[surfk])**2 +
                                                (self.elementpos[elemindex, 2]-self.z_discr[surfk])**2)
        time2 = (1 / self.VelocityMaterial) * np.sqrt((xf - self.x_discr[surfk]) ** 2 + (zf - self.z_discr[surfk]) ** 2)
        timetotal = time1+time2
        return timetotal

    def extract_bscan(self, shot):
        if self.data_insp.inspection_params.type_capt == "FMC":
            size_ascan = self.data_insp.ascan_data.shape[0]
            num_ascans = self.data_insp.ascan_data.shape[1]
            bscan = np.zeros((size_ascan, num_ascans), dtype=self.data_insp.ascan_data.dtype)
            i = np.arange(num_ascans)
            bscan = np.real(self.data_insp.ascan_data[:, i, i, shot])
        elif self.data_insp.inspection_params.type_capt == "sweep":
            size_ascan = self.data_insp.ascan_data.shape[0]
            num_ascans = self.data_insp.probe_params.num_elem
            bscan = self.data_insp.ascan_data[:, 0, 0, 0:num_ascans]
        self.bscan = bscan
        return bscan

    def extract_bscan3D(self, shot):
        # Método que considera o bscan como uma matriz 3D, onde a dimensões da saída são:
        # [ponto do ascan, emissor, transmissor], e para o caso de
        if self.data_insp.inspection_params.type_capt == "FMC":
            size_ascan = self.data_insp.ascan_data.shape[0]
            num_ascans = self.data_insp.ascan_data.shape[1]
            i = np.arange(num_ascans)
            bscan3D = np.real(self.data_insp.ascan_data[:, i, i, shot])
            # Extract only the b-scans where the transducer is the emitter and also the receiver.
        elif self.data_insp.inspection_params.type_capt == "sweep":
            size_ascan = self.data_insp.ascan_data.shape[0]
            num_ascans = self.data_insp.probe_params.num_elem
            bscan3D = self.data_insp.ascan_data[:, 0, 0, 0:num_ascans]
        self.bscan = bscan3D
        return bscan3D

    def __init__(self, data_insp, xdczerototal=0, c_medium=0, keep_data_insp=False,
                 surf_type=None, surf_param=None):
        self.surface_cdist_memory = Surface_cdist_memory()

        self.keep_data_insp = keep_data_insp
        self.data_insp = data_insp
        if data_insp.probe_params.type_probe == 'linear':
            self.ElementPitch = data_insp.probe_params.pitch
        # self.ElementPitch = np.max([np.abs(data_insp.probe_params.elem_center[0, 0] -
        #                                    data_insp.probe_params.elem_center[1, 0]),
        #                             np.abs(data_insp.probe_params.elem_center[0, 1] -
        #                                    data_insp.probe_params.elem_center[1, 1])])
        self.CalibratedDelay = xdczerototal
        self.SamplingFreq = data_insp.inspection_params.sample_freq * 1e3
        self.gate_start = data_insp.inspection_params.gate_start * 1e-3
        if c_medium == 0:
            self.VelocityMedium = data_insp.inspection_params.coupling_cl
        else:
            self.VelocityMedium = c_medium
        self.VelocityMaterial = data_insp.specimen_params.cl
        if self.data_insp.inspection_params.type_capt == "FMC":
            self.numelements = data_insp.probe_params.num_elem
        elif self.data_insp.inspection_params.type_capt == "sweep":
            self.numelements = data_insp.probe_params.num_elem
        self.elementpos = self.data_insp.probe_params.elem_center
        self.surfacetype = surf_type
        self.surfaceparam = surf_param
        self.fitted = False
        # if surf_type is not None:
        #     self.fit(surf_type, surf_param)
        # elif surf_param is not None:
        #     raise ValueError("O argumento surf_type não pode ser None, pois o argumento surf_param não é None")

    def fit(self, surf_type=None, surf_param=None, shot=0, roi=None, sel_shots=None):
        if surf_param is not None:
            self.surfacetype = surf_type
            self.surfaceparam = surf_param
            self.fitted = True
            return
        if isinstance(shot, int):
            self.__findpoints(self.extract_bscan(shot))
        else:
            surf_type = SurfaceType.ARBITRARY
        if surf_type is None and self.surfacetype is not None:
            surf_type = self.surfacetype

        # Ignorando o surf_param como entrada
        if surf_type is not None:
            if surf_type == SurfaceType.LINE_LS:
                self.surfaceparam = Lineparam()
                self.surfacetype = surf_type
                self.__linefit()
                self.discretize_line(2 * self.xpivec[0] - self.xpivec[-1],
                                     -self.xpivec[0] + 2 * self.xpivec[-1], 3e-3)
                self.fitted = True
            elif surf_type == SurfaceType.CIRCLE_MLS:
                self.surfaceparam = Circleparam()
                self.surfacetype = surf_type
                self.__circlefit()
                self.discretize_circle(5 * np.pi / 4, 7 * np.pi / 4, 0.001)
                self.fitted = True
            elif surf_type == SurfaceType.LINE_OLS:
                self.surfaceparam = Lineparam()
                self.surfacetype = surf_type
                self.__linefitodr()
                self.discretize_line(2 * self.xpivec[0] - self.xpivec[-1],
                                         -self.xpivec[0] + 2 * self.xpivec[-1],
                                         3e-3)
                self.fitted = True
            elif surf_type == SurfaceType.CIRCLE_QUAD:
                self.surfaceparam = Circleparam()
                self.surfacetype = surf_type
                self.circlequadfit()
                self.discretize_circlequad(5 * np.pi / 4, 7 * np.pi / 4, 0.001)
                self.fitted = True
            elif surf_type == SurfaceType.CIRCLE_NEWTON:
                self.surfaceparam = Circleparam()
                self.surfacetype = surf_type
                self.circlenewtonfit()
                self.discretize_circlenewton(5 * np.pi / 4, 7 * np.pi / 4, 0.001)
                self.fitted = True
            elif surf_type == SurfaceType.LINE_NEWTON:
                self.surfaceparam = Lineparam()
                self.surfacetype = surf_type
                self.__linefit()  # Because linenewton uses its result as starting guess
                self.linenewtonfit()
                self.discretize_linenewton(2 * self.xpivec[0] - self.xpivec[-1],
                                           -self.xpivec[0] + 2 * self.xpivec[-1], 3e-3)
                self.fitted = True


            elif surf_type == SurfaceType.ARBITRARY:
                self.surfaceparam = Atfmparam()
                self.surfacetype = surf_type
                if sel_shots is None:
                    sel_shots = shot
                self.__atfm(roi, sel_shots)
                self.discretize_atfm()
                self.fitted = True
            else:
                raise TypeError('Surface Type not recognized')

    def fit3D(self, surf_type=None, surf_param=None, shot=0, roi=None, sel_shots=None):
        """ Calcula os parâmetros relacionados à superfície externa tridimensional desejada ou reconhecida.

            Parameters
            ----------
            surf_type : :class: 'SurfaceType'
                Valor que se deseja encontrar o número potência de 2 imediatamente superior.

            surf_param : :class: 'SurfaceParam'
                Parâmetros de inicialização da superfície do tipo
            shot : int

            roi : :class:`.data_types.ImagingROI`
                Região de interesse na qual o algoritmo será executado. As
                dimensões da ROI devem estar em mm.

            sel_shots :
                Parâmetro que refere-se ao disparo caso o transdutor tenha sido
                deslocado.

            Returns
            -------
            None

            Raises
            ------
            TypeError
                Se ``surf_type`` não for de um dos tipos implementados pelo algorítmo.

            TypeError
                Se ``surf_param`` não for do tipo :class:`SurfaceType` ou :class:`NoneType`.

            TypeError
                Se ``shot`` não for do tipo int e não puder ser convertido para int.

            TypeError
                Se ``roi`` não for do tipo :class:`.data_types.ImagingROI` ou :class:`NoneType`

            TypeError
                Se ``sel_shots`` não for do tipo int e não puder ser convertido para int, ou :class:`NoneType`.

            """
        # Teste dos tipos dos parâmetros.
        if surf_type is not None:
            if type(surf_type) is not SurfaceType:
                raise TypeError("O argumento ``surf_type`` não é um objeto do tipo ``SurfaceType``.")

        if roi is not None:
            if type(roi) is not framework.data_types.ImagingROI:
                raise TypeError("O argumento ``roi`` não é um objeto do tipo ``ImagingROI``.")

        if shot != 0:
            try:
                shot = int(shot)
            except ValueError:
                raise TypeError("Não foi possível converter o argumento ``shot`` para ``int``.")

        if sel_shots is not None:
            try:
                sel_shots = int(sel_shots)
            except Exception:
                raise TypeError("Não foi possível converter o argumento ``sel_shots`` para o tipo ``int``.")


        if isinstance(shot, int):
            self.__findpoints3D(self.extract_bscan3D(shot))
        else:
            surf_type = SurfaceType.ARBITRARY
        if surf_type is None and self.surfacetype is not None:
            surf_type = self.surfacetype

        if surf_type == SurfaceType.PLANE_NEWTON:
            self.surfaceparam = Planeparam()
            self.surfacetype = surf_type
            #self.__planefitLSA()  # Compute initial guess for the a,b,c parameters.
            self.surfaceparam.a = 0
            self.surfaceparam.b = 0
            self.surfaceparam.c = np.mean(self.peak_distances_mm)
            self.planenewtonfit()  # Improves the initial estimative of a,b,c.
            self.__planerotation()  # Compute the plane rotation.
            # self.discretize_planenewton Pendente
            self.fitted = True
        elif surf_type == SurfaceType.CYLINDER_NEWTON:
            self.surfaceparam = Cylinderparam()
            self.surfacetype = surf_type
            guess_height = np.mean(self.peak_distances_mm)
            guess_y = 15.# Determinação arbitrária.
            self.surfaceparam.x1 = 0.
            self.surfaceparam.y1 = guess_y
            self.surfaceparam.z1 = guess_height

            self.surfaceparam.x2 = 0.
            self.surfaceparam.y2 = -guess_y
            self.surfaceparam.z2 = guess_height

            self.surfaceparam.r = 50 # Pendente método mais eficiente para estimativa inicial de raio.
            self.cylindernewtonfit()
            # self.discretize_cylindernewton Pendente
        else:
            raise TypeError("O argumento ``surf_param`` não é um tipo de superfície implementada.")

    def __compute_mk(self):
        self.m = np.zeros_like(self.x_discr)
        for k in range(0, self.m.shape[0] - 1):
            self.m[k] = (self.z_discr[k + 1] - self.z_discr[k]) /\
                        (self.x_discr[k + 1] - self.x_discr[k])
            self.m[-1] = self.m[-2]
        return

    def discretize_line(self, x_init, x_end, delta_x):
        n = int(np.round((x_end-x_init)/delta_x))+1
        self.x_discr = np.zeros(n)
        self.z_discr = np.zeros_like(self.x_discr)
        i = np.arange(n)
        self.x_discr = x_init + i*delta_x
        self.z_discr = self.x_discr*self.surfaceparam.a + self.surfaceparam.b
        # for i in range(0, n):
        #     self.x_discr[i] = x_init + i * delta_x
        #     self.z_discr[i] = (self.x_discr[i] * self.lineparam_a +
        #                        self.lineparam_b)
        self.__compute_mk()
        return

    def discretize_linenewton(self, x_init, x_end, delta_x):
        n = int(np.round((x_end-x_init)/delta_x))+1
        self.x_discr = np.zeros(n)
        self.z_discr = np.zeros_like(self.x_discr)
        i = np.arange(n)
        self.x_discr = x_init + i*delta_x
        self.z_discr = self.x_discr*self.surfaceparam.a + self.surfaceparam.b
        self.__compute_mk()
        return

    def discretize_atfm(self):
        self.x_discr = self.surfaceparam.xdiscr
        self.z_discr = self.surfaceparam.zdiscr
        return

    def discretize_circle(self, ang_init, ang_end, delta_ang):
        n = int(np.round((ang_end - ang_init) / delta_ang) + 1)
        self.x_discr = np.zeros(n)
        self.z_discr = np.zeros_like(self.x_discr)
        n = self.x_discr.shape[0]
        for i in range(0, n):
            angle = ang_init + i*delta_ang
            self.x_discr[i] = (self.surfaceparam.x +
                               self.surfaceparam.r*np.cos(angle))
            self.z_discr[i] = (self.surfaceparam.z +
                               self.surfaceparam.r*np.sin(angle))
        self.__compute_mk()
        return

    def discretize_circlequad(self, ang_init, ang_end, delta_ang):
        n = int(np.round((ang_end - ang_init) / delta_ang) + 1)
        self.x_discr = np.zeros(n)
        self.z_discr = np.zeros_like(self.x_discr)
        n = self.x_discr.shape[0]
        for i in range(0, n):
            angle = ang_init + i*delta_ang
            self.x_discr[i] = (self.surfaceparam.x +
                               self.surfaceparam.r*np.cos(angle))
            self.z_discr[i] = (self.surfaceparam.z +
                               self.surfaceparam.r*np.sin(angle))
        self.__compute_mk()
        return

    def discretize_circlenewton(self, ang_init, ang_end, delta_ang):
        n = int(np.round((ang_end - ang_init) / delta_ang) + 1)
        self.x_discr = np.zeros(n)
        self.z_discr = np.zeros_like(self.x_discr)
        n = self.x_discr.shape[0]
        for i in range(0, n):
            angle = ang_init + i*delta_ang
            self.x_discr[i] = (self.surfaceparam.x +
                               self.surfaceparam.r*np.cos(angle))
            self.z_discr[i] = (self.surfaceparam.z +
                               self.surfaceparam.r*np.sin(angle))
        self.__compute_mk()
        return

    def __linefitodr(self):
        mymodel = Model(f)
        mydata = Data(self.xpivec.transpose(), self.zpivec.transpose())
        myodr = ODR(mydata, mymodel, beta0=[1, 1])
        myodr.set_job(fit_type=0)  # fit_type=2 returns the same as leastsq
        out = myodr.run()
        self.surfaceparam.a = out.beta[0]
        self.surfaceparam.b = out.beta[1]

        num_points = self.xpivec.shape[0]

        a = self.surfaceparam.a / self.surfaceparam.b
        b = -1 / self.surfaceparam.b
        c = 1

        self.surfaceparam.SSE = 0
        for i in range(num_points):
            x0 = self.xpivec[i]
            z0 = self.zpivec[i]
            self.surfaceparam.SSE = (self.surfaceparam.SSE +
                                     abs(a*x0 + b*z0 + c)/np.sqrt(a**2 + b**2))
        self.surfaceparam.water_path = self.surfaceparam.b

    def __linefit(self):
        num_points = self.xpivec.shape[0]
        a = np.ones((num_points, 2))
        a[:, 0] = self.xpivec.reshape(num_points)
        x = cg(np.dot(a.T, a), np.dot(a.T, self.zpivec.reshape(num_points)))[0]
        self.surfaceparam.a = x[0]
        self.surfaceparam.b = x[1]

        a = self.surfaceparam.a/self.surfaceparam.b
        b = -1/self.surfaceparam.b
        c = 1

        self.surfaceparam.SSE = 0
        for i in range(num_points):
            x0 = self.xpivec[i]
            z0 = self.zpivec[i]
            self.surfaceparam.SSE = self.sumsqudistbscanline(self.surfaceparam.a, self.surfaceparam.b)

        self.surfaceparam.water_path = self.surfaceparam.b

    def __planefit(self):
        # This is a method which through least squares it compute a initial guess for the surface (plane) parameters.
        # In order to find the parameters a,b,c, we want to solve:
        # z = ax + by + c
        # Where we have multiple values of x,y,z (measured points) and we want to find the best possible
        # plane that fit those points.
        # In matrix form the notation will be equivalent to:
        # x = [a b c]^T
        # b = [z1 z2 ... zn]^T
        # A =
        # [ x1 y1 1]
        # [ x2 y2 1]
        # [ ... ... ...]
        # [xn yn 1]
        # So, doing A*x we will get for the first line: x1 * a + y1 * b + c = z1, a valid plane equation for each point.
        # Since we have a full column rank system, we can apply least square.
        # Applying least square in matrix form will be equals to:
        # A^T * A * x = A^T b
        # After that, we will get a rough guess of a,b,c.
        # Water path is equals to c.

        num_points = self.xpivec.shape[0]

        # A = [X Y 1] [a b c]^T = [z1 z2 ... zn]^T
        A = np.ones((num_points, 3))
        A[:, 0] = self.xpivec.reshape(num_points)
        A[:, 1] = self.ypivec.reshape(num_points)
        x = cg(np.dot(A.T, A), np.dot(A.T, self.zpivec.reshape(num_points)))[0]
        self.surfaceparam.a = x[0]
        self.surfaceparam.b = x[1]
        self.surfaceparam.c = x[2]

        self.surfaceparam.SSE = self.sumsqudistbscanplane(self.surfaceparam.a, self.surfaceparam.b, self.surfaceparam.c)

        self.surfaceparam.water_path = self.surfaceparam.c


    def __planefitLSA(self):
        # This method computes a guess for the Newton Raphson method using least square approximation.
        # Assuming that all the points are right under the transducer position, the (x,y) coordinates are the same as
        # the transducer element right above it. The only coordinate component that differs is the z.
        # The z coordinate value is approximated by the max value of the envelope.

        x0 = [position[0] for position in self.data_insp.probe_params.elem_center[:]]
        y0 = [position[1] for position in self.data_insp.probe_params.elem_center[:]]
        z0 = self.peak_distances_mm

        # The equation (in it's scalar form) that one must solve is:
        # ax + by + c = z
        # A = [x0 y0 1] # x and y are vectors
        # b = [z0]  # z is a vector
        # x = [a b c] # a,b and c are scalars
        # Or in it's matrix representation:
        # Ax = b
        # But it's impossible to solve, since there are errors in x,y and z component. Thus, LSA will be used to bypass
        # this difficulty:
        # A^T A x = A ^T b
        A = np.ones((self.numelements, 3))
        A[:, 0] = x0
        A[:, 1] = y0

        # Important note: x0 is the position vector containing the positions of the points which belongs to the surface.
        # On the other hand, x is the solution vector of the least square approximation.
        x = cg(np.dot(A.T, A), np.dot(A.T, z0.reshape(self.numelements)))[0]

        self.surfaceparam.a = x[0]
        self.surfaceparam.b = x[1]
        self.surfaceparam.c = x[2]
        self.surfaceparam.SSE = self.sumsqudistbscanplane(self.surfaceparam.a, self.surfaceparam.b, self.surfaceparam.c)
        self.surfaceparam.water_path = self.surfaceparam.c


    def linenewtonfit(self):
        theta_init = np.array([[self.surfaceparam.a], [self.surfaceparam.b]])
        newtonResult = nlo.newtonsearch(self.encapsulateerrorline, theta_init)
        theta_final = newtonResult.theta_final
        self.surfaceparam.a = theta_final[0][0]
        self.surfaceparam.b = theta_final[1][0]
        self.surfaceparam.SSE = self.sumsqudistbscanline(theta_final[0][0], theta_final[1][0])
        self.surfaceparam.resultnewton = newtonResult
        self.surfaceparam.water_path = self.surfaceparam.b

    def initial_guess_radius(self):
        n = len(self.peak_distances_mm)
        r = np.zeros(n)
        dc = self.peak_distances_mm[0]
        for i in range(1, n):
            di = self.peak_distances_mm[i]
            xi = self.elementpos[i]
            proj_x = xi
            norm_x = np.sqrt(proj_x[0]**2 + proj_x[1]**2 + proj_x[2]**2)
            r[i] = (di**2 - dc**2 + norm_x**2)/(2*(di - dc))
        r = 50
        return np.mean(r)

    def cylindernewtonfit(self):
        theta_init = np.array([[self.surfaceparam.x1], [self.surfaceparam.z1], [self.surfaceparam.x2], [self.surfaceparam.z2], [self.surfaceparam.r]])
        newtonResult = nlo.newtonsearch(self.__encapsulateerrorcylinder, theta_init)
        theta_final = newtonResult.theta_final
        self.surfaceparam.x1 = theta_final[0][0]
        self.surfaceparam.z1 = theta_final[1][0]
        self.surfaceparam.x2 = theta_final[2][0]
        self.surfaceparam.z2 = theta_final[3][0]
        self.surfaceparam.r = theta_final[4][0]
        self.__point_to_line()
        self.__cylinderrotation()
        self.surfaceparam.SSE = self.sumsqudistbscancylinder(theta_final[0][0], theta_final[1][0], theta_final[2][0], theta_final[3][0], theta_final[4][0])
        self.surfaceparam.resultnewton = newtonResult
        self.surfaceparam.water_path = self.__distance_line_to_point()

    def planenewtonfit(self):
        """Calcula os parâmetros a,b e c do da superfície.

         Através de estimativas inicias para os parâmetros a,b e c, a função (ou método), por meio do método
          Newton-Raphson, tenta minimizar a função do

         """
        # Starting guess previously computed:
        theta_init = np.array([[self.surfaceparam.a], [self.surfaceparam.b], [self.surfaceparam.c]])

        # Running Newton-Raphson method to find the best parameters a,b and c which minimizes the errors (distance
        # between the plane and the measured point):
        newtonResult = nlo.newtonsearch(self.__encapsulateerrorplane, theta_init)
        theta_final = newtonResult.theta_final

        # Surface parameters will be equal to the solution of the newton method:
        self.surfaceparam.a = theta_final[0][0]
        self.surfaceparam.b = theta_final[1][0]
        self.surfaceparam.c = theta_final[2][0]

        # The SSE (sum squared error) of the final solution:
        self.surfaceparam.SSE = self.sumsqudistbscanplane(theta_final[0][0], theta_final[1][0], theta_final[2][0])

        self.surfaceparam.resultnewton = newtonResult

        self.surfaceparam.water_path = self.surfaceparam.c


    def __planerotation(self):
        # Basis of the coordinate system.
        i_vec = np.array([1, 0, 0])
        j_vec = np.array([0, 1, 0])
        k_vec = np.array([0, 0, 1])


        # It's important to note that we ignore the translation of the plane c, since the results are the same as if the
        # plane contained the origin in it's subspace.


        # In order to discover the rotation of a plane, the following procedure will be taken place:
        # Assuming the general equation of a plane:
        # A * x + B * y + C * z = D
        # The normal vector to this plane, applied in the origin is:
        # n = [A B C]^T
        # But the parameters a,b,c that were computed in surface's methods are in the following notation:
        # z = a*x + b*y + c
        # Thus a conversion has to take place:
        # z - ax - by = c
        # A = -a
        # B = -b
        # C = 1
        # D = C
        # Therefore: n = [-a, -b, c]^T
        # Source: https://onlinemschool.com/math/assistance/cartesian_coordinate/plane_angl/

        n_x = np.array([-self.surfaceparam.a, 0, 1])  # Projection of the normal vector into xz plane
        n_y = np.array([0, -self.surfaceparam.b, 1])  # Projection of the normal vector into yz plane

        # The angle which the plane is rotated along y axis (according to the right hand rule):
        self.surfaceparam.rot_y = np.rad2deg(np.arcsin(np.dot(np.cross(k_vec, n_x)/np.linalg.norm(n_x), j_vec)))

        # The angle which the plane is rotated along x axis (according to the right hand rule):
        self.surfaceparam.rot_x = np.rad2deg(np.arcsin(np.dot(np.cross(k_vec, n_y)/np.linalg.norm(n_y), i_vec)))

    def __distance_line_to_point(self):
        # Computes parameters of a 3D line that contains two points x1 and x2.
        vec_x1 = np.array([self.surfaceparam.x1, self.surfaceparam.y1, self.surfaceparam.z1])
        vec_x2 = np.array([self.surfaceparam.x2, self.surfaceparam.y2, self.surfaceparam.z2])
        vec_x0 = np.zeros(3)
        return np.linalg.norm(np.cross((vec_x0 - vec_x1), vec_x0-vec_x2))/np.linalg.norm(vec_x2-vec_x1) - self.surfaceparam.r


    def __point_to_line(self):
        # Computes parameters of a 3D line that contains two points x1 and x2.
        vec_x1 = np.array([self.surfaceparam.x1, self.surfaceparam.y1, self.surfaceparam.z1], dtype=object)
        vec_x2 = np.array([self.surfaceparam.x2, self.surfaceparam.y2, self.surfaceparam.z2], dtype=object)
        self.surfaceparam.direction_vector = vec_x1 - vec_x2
        self.surfaceparam.translation_vector = vec_x1

    def __cylinderrotation(self):

        D = self.surfaceparam.direction_vector

        # Basis of the coordinate system.
        i_vec = np.array([1., 0, 0])
        j_vec = np.array([0, 1., 0])
        k_vec = np.array([0, 0, 1.])

        x = D[0]
        # Projections of the direction vector:
        proj_xy = np.array([D[0], D[1], 0])
        proj_zy = np.array([0, D[1], D[2]])

        alpha = np.rad2deg(np.arcsin(np.dot(np.cross(k_vec, proj_zy)/np.linalg.norm(proj_zy), i_vec)))

        self.surfaceparam.rot_x = -np.rad2deg(np.arccos(np.dot(proj_zy, k_vec)/np.linalg.norm(proj_zy)))
        # np.rad2deg(np.arcsin(np.dot(np.cross(k_vec, proj_zy)/np.linalg.norm(proj_zy), i_vec)))

        beta = np.rad2deg(np.arcsin(np.dot(np.cross(j_vec, proj_xy)/np.linalg.norm(proj_xy), k_vec)))
        self.surfaceparam.rot_z = np.rad2deg(np.arcsin(np.dot(np.cross(j_vec, proj_xy)/np.linalg.norm(proj_xy), k_vec)))



    def __atfm(self, roi, shots, resolution_ratio=10, scattering_angle=90):
        if roi is None:
            ascan_envelope = envelope(self.bscan[:, self.data_insp.probe_params.num_elem//2], 0) # alterado para envelope
            idx = int(np.argwhere(ascan_envelope <= np.median(ascan_envelope))[0][0])
            peaks = idx + np.argmax(envelope(self.bscan[idx:, :]), axis=0)

            roi_min_z = -1 + 1e-3 * self.data_insp.inspection_params.coupling_cl * .5 * \
                        (self.data_insp.inspection_params.gate_start +
                         np.min(peaks) / self.data_insp.inspection_params.sample_freq)
            if roi_min_z < 0.0:
                roi_min_z = 0.0
            roi_max_z = +1 + 1e-3 * self.data_insp.inspection_params.coupling_cl * .5 * \
                        (self.data_insp.inspection_params.gate_start +
                         np.max(peaks) / self.data_insp.inspection_params.sample_freq)
            roi_min_x = self.data_insp.probe_params.elem_center[0][0]
            roi_max_x = self.data_insp.probe_params.elem_center[-1][0]

            height = roi_max_z - roi_min_z
            width = roi_max_x - roi_min_x
            h_len = np.maximum(height * resolution_ratio, 600)
            w_len = np.maximum(width * resolution_ratio, 600)
            corner_roi = np.array([[roi_min_x, 0.0, roi_min_z]])
            roi = data_types.ImagingROI(corner_roi, height=height, width=width, h_len=h_len, w_len=w_len)
        #
        memory_type_insp = self.data_insp.inspection_params.type_insp
        memory_specimen_cl = self.data_insp.specimen_params.cl
        self.data_insp.inspection_params.type_insp = 'contact'
        self.data_insp.specimen_params.cl = self.data_insp.inspection_params.coupling_cl
        tfm_key = cumulative_tfm_kernel(self.data_insp, roi, scattering_angle=scattering_angle, sel_shots=shots)

        self.data_insp.inspection_params.type_insp = memory_type_insp
        self.data_insp.specimen_params.cl = memory_specimen_cl

        im_envelope = envelope(self.data_insp.imaging_results[tfm_key].image)
        eroi = self.data_insp.imaging_results[tfm_key].roi
        im_envelope = im_envelope/im_envelope.max()
        a = intsurf_estimation.img_line(im_envelope)
        z = eroi.h_points[a[0].astype(int)]
        w = np.diag(a[1])
        d = intsurf_estimation.matrix_d2(z, 'mirrored')
        # lamb = 1e-2  # 1/np.mean(np.abs(d@z))
        rho = 100
        lamb = 15
        # rho = np.linalg.eigvals(w.T @ w)[-1] * lamb
        print(f'Estimating Surface')
        zf, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb, x0=z, rho=rho, eta=.999, itmax=500,
                                                                max_iter_cg=1500, tol=1e-6)
        self.surfaceparam.xdiscr = eroi.w_points
        self.surfaceparam.zdiscr = zf

    def __newtonraphson(self, x, zcandidate, distance):
        iter = 200
        z0 = zcandidate
        r0 = zcandidate - distance

        #result = np.zeros(iter)
        #funcvalue = np.zeros(iter)
        #funcvalued = np.zeros(iter)

        dif = 0.01
        for i in range(iter):
            f_z0 = (self.sumsqudistbscancircle(x, z0, r0) - self.sumsqudistbscancircle(x, z0 - dif, r0 - dif)) / dif
            fl_z0 = (self.sumsqudistbscancircle(x, z0 + dif, r0 + dif) - 2 * self.sumsqudistbscancircle(x, z0, r0)
                     + self.sumsqudistbscancircle(x, z0 - dif, r0 - dif)) / (dif ** 2)
            #result[i] = z0
            #funcvalue[i] = f_z0
            #funcvalued[i] = fl_z0

            z1 = z0 - f_z0 / fl_z0
            z0 = z1
            r0 = z0 - distance

            if np.abs(f_z0) < 1e-6:
                break
        #return [result, funcvalue, funcvalued]
        #plt.figure()
        #plt.plot(result,'o-')
        #plt.legend(['Z'])
        #plt.figure()
        #plt.plot(funcvalue, 'o-')
        #plt.legend(['function'])
        #plt.figure()
        #plt.plot(funcvalued, 'o-')
        #p#lt.legend(['derivative'])

        return z1

    def circlenewtonfit(self):
        distancias = np.array(self.peak_distances_mm)

    # The code commented below provides a rough estimate of the circle parameters x,z,r to be
    # used as a starting guess on the Multivariate Newton method. Since the circlequadfit()
    # provides a quite fine estimate of the parameters, its result is currently used as
    # starting guess.
        # The roughly estimated x is the position of the closest element
        # in relation to the surface (earliest echo)
        closest_element = np.argmin(distancias)
        rough_x = self.data_insp.probe_params.elem_center[closest_element, 0]

        # The distance (z axis) from the transducer to the surface is
        mindist = np.min(distancias)

        furthest_x = self.data_insp.probe_params.elem_center[0, 0] - \
                     self.data_insp.probe_params.elem_center[closest_element, 0]
        furthest_element = 0

        ########## Geometric scheme:   #########################
        ########## https://photos.app.goo.gl/85S4TDnvdAEMMKkd9
        d = distancias[furthest_element]
        c = furthest_x
        e = mindist
        if np.abs(d-e) > 1e-10:
            a = (e**2 + c**2 - d**2) / (2 * (d-e))
        else:
            a = (e ** 2 + c ** 2 - d ** 2) / (2 * 1e-10)
        ########################################################

        # When the surface is a plane, the rough estimate might result
        # in a very large radius (value for 'a'). In this case, do not
        # perform the algorithm and return dummy values.
        if np.abs(a) > 1e6:
            self.surfaceparam.x = 0
            self.surfaceparam.z = 0
            self.surfaceparam.r = 1
            self.surfaceparam.SSE = 1e6
            return None

        rough_r = a
        rough_z = distancias[closest_element] + rough_r

        # Now refine the search with Newton-Raphson multivariate method
        theta_init = np.array([[rough_x], [rough_z], [rough_r]])
        #theta_init = np.array([[0.2], [81.2], [70.2]])
        #theta_init = np.array([[self.circlequadparam.x], [self.circlequadparam.z], [self.circlequadparam.r]])
        resultnewton = nlo.newtonsearch(self.encapsulateerrorcircle, theta_init)
        theta_final = resultnewton.theta_final

        iteration = np.argmin(resultnewton.cost_values)

        self.surfaceparam.x = resultnewton.theta_values[0, iteration]
        self.surfaceparam.z = resultnewton.theta_values[1, iteration]
        self.surfaceparam.r = resultnewton.theta_values[2, iteration]
        self.surfaceparam.SSE = self.sumsqudistbscancircle(self.surfaceparam.x, self.surfaceparam.z,
                                                            self.surfaceparam.r)
        self.surfaceparam.resultnewton = resultnewton

        # Record the individual errors
        center = np.zeros((1, 3))
        center[0, :] = np.array([self.surfaceparam.z, self.data_insp.probe_params.elem_center[0, 1],
                                 self.surfaceparam.z])
        centertoelem = dist.cdist(self.data_insp.probe_params.elem_center, center)
        distances = np.subtract(np.array(self.peak_distances_mm) + self.surfaceparam.r,
                                centertoelem.transpose())
        self.surfaceparam.errors = distances

        self.surfaceparam.water_path = self.surfaceparam.z - self.surfaceparam.r

        return None

    def circlequadfit(self):
        # Fit a parabola to the B-scan peaks through linear regression
        distancias = np.array(self.peak_distances_mm)
        N = distancias.shape[0]
        A = np.zeros((N, 3))
        A[:, 0] = np.power(np.arange(0, N), 2)
        A[:, 1] = np.power(np.arange(0, N), 1)
        A[:, 2] = np.power(np.arange(0, N), 0)
        X_LS = cg(np.dot(A.transpose(), A), np.dot(A.transpose(), distancias))

        a = X_LS[0][0]
        b = X_LS[0][1]
        c = X_LS[0][2]

        # The estimatex x is where the derivative of the parabola equals zero
        x = -b / (2 * a)

        # The distance (z axis) from the transducer to the surface is
        mindist = a * x * x + b * x + c

        x_corrected = self.data_insp.probe_params.elem_center[0, 0] + \
                      x*(self.data_insp.probe_params.elem_center[1, 0] - self.data_insp.probe_params.elem_center[0, 0])


        # Find a rough estimate for the radius with a grid search
        radiuscandidates = np.arange(-10, 500, 1)
        zcandidates = radiuscandidates + mindist

        sqerrorbscan = np.zeros(radiuscandidates.shape)

        for i in range(0, radiuscandidates.shape[0]):
            sqerrorbscan[i] = self.sumsqudistbscancircle(x_corrected, zcandidates[i], radiuscandidates[i])
        radiusmin_sq_bscan = radiuscandidates[np.argmin(sqerrorbscan)]
        zmin_sq_bscan = radiusmin_sq_bscan + mindist

        self.surfaceparam.x = x_corrected
        self.surfaceparam.z = zmin_sq_bscan
        self.surfaceparam.r = radiusmin_sq_bscan
        self.surfaceparam.SSE = sqerrorbscan[np.argmin(sqerrorbscan)]

        # Now refine the search with Newton-Raphson method
        self.surfaceparam.z = self.__newtonraphson(self.surfaceparam.x, self.surfaceparam.z, mindist)
        self.surfaceparam.r = self.surfaceparam.z - mindist
        self.surfaceparam.SSE = self.sumsqudistbscancircle(self.surfaceparam.x, self.surfaceparam.z,
                                                   self.surfaceparam.r)

        # Record the individual errors
        center = np.zeros((1, 3))
        center[0, :] = np.array([self.surfaceparam.z, self.data_insp.probe_params.elem_center[0, 1],
                                 self.surfaceparam.z])
        centertoelem = dist.cdist(self.data_insp.probe_params.elem_center, center)
        distances = np.subtract(np.array(self.peak_distances_mm) + self.surfaceparam.r,
                                centertoelem.transpose())
        self.surfaceparam.errors = distances

        self.surfaceparam.water_path = self.surfaceparam.z - self.surfaceparam.r

        return None

    def __circlefit(self):
        n = self.xpivec.shape[0]
        x = self.xpivec
        z = self.zpivec
        a = n*np.sum(self.xpivec**2) - np.sum(self.xpivec)**2
        b = n*np.sum(self.xpivec*self.zpivec) -\
            np.sum(self.xpivec)*np.sum(self.zpivec)
        c = n*np.sum(self.zpivec**2) - np.sum(self.zpivec)**2
        d = 0.5 * (n * np.sum(x * z ** 2) - np.sum(x) * np.sum(z ** 2) +
                   n * np.sum(x ** 3) - np.sum(x) * np.sum(x ** 2))
        e = 0.5 * (n * np.sum(z * x ** 2) - np.sum(z) * np.sum(x ** 2) +
                   n * np.sum(z ** 3) - np.sum(z) * np.sum(z ** 2))

        Am = (d*c-b*e)/(a*c-b**2)
        self.surfaceparam.x = Am
        Bm = (a*e-b*d)/(a*c-b**2)
        self.surfaceparam.z = Bm
        self.surfaceparam.r = np.sum(np.sqrt((x-Am)**2+(z-Bm)**2)/n)

        # Measure the quality of fit by the sum of squared errors. The error is
        # the difference between the distance from center to point and the
        # radius.
        self.surfaceparam.SSE = 0
        dists_to_center = np.zeros(n)
        for i in range(n):
            dist_to_center = np.sqrt((self.xpivec[i]-self.surfaceparam.x)**2 +
                                     (self.zpivec[i]-self.surfaceparam.z)**2)
            self.surfaceparam.SSE = (self.surfaceparam.SSE +
                                    (dist_to_center-self.surfaceparam.r)**2)
            dists_to_center[i] = dist_to_center
        self.surfaceparam.water_path = self.surfaceparam.z - self.surfaceparam.r

    def __findpoints(self, bscan):
        # foi necessário mudar o nome da variavel envelope para env, pois o pycharm estava confundindo a função
        # envelope com a variavel, causando um erro de unresolved reference.
        num_points = self.numelements - 1
        peak_distances = list()
        peak_samples = list()
        # If the signal is not rectified nor envelope...
        if np.min(bscan[:, int(self.numelements/2)]) < 0:
            env = envelope(bscan[:, int(self.numelements/2)], 0)  # alterado para envelope
        # If the signal is already rectified or envelope...
        else:
            env = bscan[:, int(self.numelements / 2)]
        # If the signal is not gated, try to remove the transducer ringing
        if self.data_insp.inspection_params.gate_start == 0.0:
            idx = int(np.argwhere(env <= np.median(env))[0][0])
        else:
            idx = 0
        self.threshold_median = np.median(env)
        for i_elem in range(self.numelements):
            env = envelope(bscan[idx:, i_elem], 0)  # aterado para envelope
            peak = idx + np.argmax(env)
            peak_samples.append(np.argmax(env))
            peak_corrected = peak - self.CalibratedDelay
            peak_time = (peak_corrected / self.SamplingFreq) + self.gate_start
            peak_distance = self.VelocityMedium * peak_time / 2
            peak_distances.append(peak_distance)

        self.xpivec = np.zeros((num_points, 1))
        self.zpivec = np.zeros((num_points, 1))

        self.sinphi = np.zeros(num_points)
        for i_elem in range(num_points):
            xai = self.elementpos[i_elem, 0]
            zai = self.elementpos[i_elem, 2]
            sin_phi = (peak_distances[i_elem] -
                       peak_distances[i_elem + 1]) / self.ElementPitch
            self.sinphi[i_elem] = sin_phi
            try:
                phi = np.arcsin(sin_phi)
            except:
                phi = np.pi/2
            xpi = xai + peak_distances[i_elem] * np.sin(phi)
            zpi = zai + peak_distances[i_elem] * np.cos(phi)
            self.xpivec[i_elem] = xpi
            self.zpivec[i_elem] = zpi

            if np.isnan(xpi) or np.isnan(zpi):
                self.xpivec[i_elem] = self.xpivec[i_elem - 1]
                self.zpivec[i_elem] = self.zpivec[i_elem - 1]

        # Sort xpivec and zpivec
        # Source: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        x = self.xpivec.reshape(self.xpivec.shape[0])
        z = self.zpivec.reshape(self.zpivec.shape[0])
        z_ordered = [i for _, i in sorted(zip(x, z))]
        x_ordered = [i for _, i in sorted(zip(x, x))]
        self.xpivec = np.array(x_ordered)
        self.zpivec = np.array(z_ordered)

        self.peak_distances_mm = peak_distances
        self.peak_distances_samples = peak_samples

        return

    def __compute_rough_surface_plane(self):
        # Using the method described by Camacho 2014, but generalizing to a matrix transducers
        # IMPORTANT NOTE:
        # this method is only valid if the following conditions are met:
        # * The matrix transducer have a rectangular  shape (m x n).
        # * The pitch (the distance between subsequent transducer elements center) is constant in x and y directions.

        # Important note: since the algorithm considers pairs of transducer elements, in order to compute the surface the
        # guess of the points which belongs to the surface, it will only be used n-1 transducer element, where n is the
        # total number of transducer elements.

        # Number of transducer elements in the x direction.
        x_dim = self.data_insp.probe_params.shape[0]

        # Number of transducer elements in the x direction.
        y_dim = self.data_insp.probe_params.shape[1]

        # Total number of transducer elements:
        n = self.data_insp.probe_params.num_elem

        # Vectors containing the estimation of the surface points organized in the vector form:
        self.xpivec = np.zeros((n-1, 1))
        self.ypivec = np.zeros((n-1, 1))
        self.zpivec = np.zeros((n-1, 1))

        for i in range(n):
            if i % y_dim == 0:
                # If the ith element is in the y upper bound (y = m in a 'n x m' rectangular matrix transducer)
                # only compute the xpi
                xai = self.elementpos[i, 0]
                zai = self.elementpos[i, 2]
                sin_phi_x = (self.peak_distances_mm[i] -
                             self.peak_distances_mm[i + 1]) / self.ElementPitch
                #self.sinphi[i] = sin_phi_x

                # Finding the phi angle in rad.
                try:
                    x_phi = np.arcsin(sin_phi_x)
                except:
                    # arcsin(pi/2) gives nan instead of pi/2, so it must be fixed:
                    x_phi = np.pi / 2

                xpi = xai + self.peak_distances_mm[i] * np.sin(x_phi)
                zpi = zai + self.peak_distances_mm[i] * np.cos(x_phi)
                self.xpivec[i] = xpi
                self.zpivec[i] = zpi
                if i + x_dim > n:
                    # If the ith element is in the x upper bound (x = n in a 'n x m' rectangular matrix transducer)
                    # only compute the ypi
                    #
                    # We must sum i + ydim due to the fact that we are considering a matrix rectangular shape
                    # transducer, but the data (peak_distances_mm) is organized in a vector form, where for example in a 4x4
                    # transducer shape, the 1th element is next to the 2th element (right under it) and the (1+4)th element
                    # it is next to it's right side.

                    yai = self.elementpos[i, 1]
                    sin_phi_y = (self.peak_distances_mm[i] -
                                 self.peak_distances_mm[i + y_dim]) / self.ElementPitch

                    # Finding the phi angle in rad.
                    try:
                        y_phi = np.arcsin(sin_phi_y)
                    except:
                        # arcsin(pi/2) gives nan instead of pi/2, so it must be fixed:
                        y_phi = np.pi / 2
                    ypi = yai + self.peak_distances_mm[i] * np.sin(y_phi)

                    self.ypivec[i] = ypi

        return

    def __findpoints3D(self, bscan3D):
        # Computes the most external measured point that belongs to the surface.
        peak_distances = np.zeros(self.numelements)
        peak_samples = np.zeros(self.numelements)

        # If the signal is not rectified nor envelope:
        if np.min(bscan3D[:, int(self.numelements / 2)]) < 0:
            env = envelope(bscan3D[:, int(self.numelements / 2)], 0)  # alterado para envelope
            # Parece que o terceiro parâmetro tem que ser mudado.
        # If the signal is already rectified or envelope...
        else:
            env = bscan3D[:, int(self.numelements / 2)]
        # If the signal is not gated, try to remove the transducer ringing
        if self.data_insp.inspection_params.gate_start == 0.0:
            idx = int(np.argwhere(env <= np.median(env))[0][0])
        else:
            idx = 0

        # Since we have already computed the beginning of the experimental signal
        for i in range(self.numelements):
            # You must only use the b-scan obtained by the pulse and echo of the same transducer element, in order
            # to the proposed geometry work properly.

            # The envelope signal (env) is equals to the old raw b-scan from the point that the pulse noise ends,
            # applying the usual envelope methods (Hilbert transform):
            env = envelope(bscan3D[idx:, i], 0)

            # The peak is the maximum value of the envelope signal (removing the pulse noise).

            peak = idx + np.argmax(env)
            peak_samples[i] = (np.argmax(env))
            peak_corrected = peak - self.CalibratedDelay
            peak_time = (peak_corrected / self.SamplingFreq) + self.gate_start

            # Considerando que a distância é calculada como o tempo de (ida + volta)/2 * velocidade
            # Onde a velocidade será a do meio acoplante, visto que estamos considerando a superfície externa.

            # The peak_distance value represents the point (right under the ith transducer element) which belongs to
            # the external surface. Since the peak_time is measured as 2 times the travel distance between the external
            # surface point and the transducer element, the peak distance is computed by dividing the coupling medium
            # velocity times the travel time.
            peak_distance = self.VelocityMedium * peak_time / 2
            peak_distances[i] = peak_distance

        self.peak_distances_mm = peak_distances
        return peak_distances

    def __newtonraphsonbatchelement(self, xa, za, xroi, zroi):
        # x_interface = np.zeros_like(xroi) + 0.0
        # z_interface = np.zeros_like(zroi) + 0.0
        # k = np.int(self.x_discr.shape[0] / 2)
        # for i in range(0, xroi.shape[0]):
        #     k = newtonraphsonk(self.x_discr, self.z_discr, self.m, self.VelocityMedium, self.VelocityMaterial,
        #                        xa, za, xroi[i], zroi[i], k)
        #     x_interface[i] = self.x_discr[k]
        #     z_interface[i] = self.z_discr[k]
        #     # display(x_interface[i])
        return newtonraphsonbatchelement(self.x_discr, self.z_discr, self.m, self.VelocityMedium, self.VelocityMaterial,
                                         xa, za, xroi, zroi)
        # return [x_interface, z_interface]

    def newtonraphsonbatchfocus(self, xroi, zroi):
        x_interface = np.zeros(self.numelements,)
        z_interface = np.zeros(self.numelements,)
        self.k_interface = np.zeros(self.numelements,)
        k = np.int(self.x_discr.shape[0] / 2)
        for i in range(0, self.numelements):
            k = newtonraphsonk(self.x_discr, self.z_discr, self.m, self.VelocityMedium, self.VelocityMaterial,
                               self.elementpos[i, 0], self.elementpos[i, 2], xroi, zroi, k)
            # k = self.gridsearch(self.elementpos[i, 0], self.elementpos[i, 2],
            #                     xroi, zroi, k)
            self.k_interface[i] = k
            x_interface[i] = self.x_discr[k]
            z_interface[i] = self.z_discr[k]
        return [x_interface, z_interface]

    def cdist(self, coordelem, coordroi):
        """ Calcula todas as distâncias entre as posições bidimensionais ``coordelem`` e as posições ``coordroi``.
        O método assume que todas as posições ``coordroi`` encontram-se no interior do material, de forma que todas
        as trajetórias passam pela superfície sofrendo refração segundo a Lei de Snell. O ponto da superfície em que a
        trajetória é refratada é calculado pelo método de Newton-Raphson discretizado :cite:`parrilla2007fast`.

        Parameters
        ----------
            coordelem : :class:`numpy.array`
                    Coordenadas espaciais (em mm) dos elementos do transdutor.
            coordroi : :class:`numpy.array`
                    Coordenadas espaciais dos pixels definidos para a região de interesse.
        Returns
        -------
            :class:`numpy.array`
                Array com dois elementos. O primeiro elemento é a matriz de distâncias entre as posições
                ``coordelem`` e as posições da superfície em que os feixes sofrem refração a caminho das posições
                ``coordroi``. O segundo elemento é a matriz de distâncias entre aquelas posições da superfície
                e as posições ``coordroi``. Ambas as matrizes possuem o número de linhas igual à quantidade de
                posições ``coordelem`` e o número de colunas igual à quantidade de posições ``coordroi``, e seguem
                o padrão das matrizes retornadas pela função :meth:`scipy.spatial.distance.cdist`.
        """
        xz_interface = np.zeros((coordelem.shape[0], coordroi.shape[0], 2))
        for i_elem in range(0, coordelem.shape[0]):
            # print('element', i_elem)
            [x_interface, z_interface] = self.__newtonraphsonbatchelement(
                coordelem[i_elem, 0], coordelem[i_elem, 2],
                coordroi[:, 0], coordroi[:, 2])
            xz_interface[i_elem, :, 0] = x_interface
            xz_interface[i_elem, :, 1] = z_interface
        resultcouplant = np.sqrt((coordelem[:, 0, np.newaxis]-xz_interface[:, :, 0])**2 +
                                 (coordelem[:, 2, np.newaxis]-xz_interface[:, :, 1])**2)
        resultmaterial = np.sqrt((coordroi[np.newaxis, :, 0]-xz_interface[:, :, 0])**2 +
                                 (coordroi[np.newaxis, :, 2]-xz_interface[:, :, 1])**2)
        return np.array([resultcouplant, resultmaterial])

    def cdist_medium(self, coordelem, coordroi, roi=None, sel_shot=0):
        """ Calcula todas as distâncias entre as posições bidimensionais ``coordelem`` e as posições ``coordroi``.
            O método verifica, para cada posição de ``coordroi``, se esta se encontra no interior do material  ou no
            meio acoplante.

            Parameters
            ----------
                coordelem : :class:`numpy.array`
                        Coordenadas espaciais (em mm) dos elementos do transdutor.
                coordroi : :class:`numpy.array`
                        Coordenadas espaciais dos pixels definidos para a região de interesse.
            Returns
            -------
                :class:`numpy.array`
                    Para as posições de ``coordroi`` localizadas no interior do material, o formato dos dados retornados
                    é o mesmo do método :meth:`surface.surface.cdist`. Para as posições de ``coordroi`` localizadas no
                    meio acoplante, o valor correspondente na primeira matriz contém a distância do elemento ao pixel e
                    o da segunda matriz é igual a zero.
            """

        # If distances are calculated twice for the same coordinates,
        # then return the same result as before
        # if np.array_equal(coordelem, self.surface_cdist_memory.coordelem) \
        #         and np.array_equal(coordroi, self.surface_cdist_memory.coordroi):
        #     return self.surface_cdist_memory.returned_result
        if not self.fitted:
            raise ValueError('Surface not initialized (Not fitted)')
        xz_interface = np.zeros((coordelem.shape[0], coordroi.shape[0], 3))

        self.pixel_in_material = np.zeros((coordroi.shape[0]))
        if self.surfacetype == SurfaceType.LINE_LS:
            self.pixel_in_material = (coordroi[:, 2] > self.surfaceparam.a*coordroi[:, 0] + self.surfaceparam.b) + 0.0
        elif self.surfacetype == SurfaceType.LINE_OLS:
            self.pixel_in_material = (coordroi[:, 2] > self.surfaceparam.a*coordroi[:, 0] + self.surfaceparam.b) + 0.0
        elif self.surfacetype == SurfaceType.LINE_NEWTON:
            self.pixel_in_material = (coordroi[:, 2] > self.surfaceparam.a*coordroi[:, 0] + self.surfaceparam.b) + 0.0
        elif self.surfacetype == SurfaceType.CIRCLE_MLS:
            self.pixel_in_material = (((coordroi[:, 0] - self.surfaceparam.x)**2 + (coordroi[:, 2] - self.surfaceparam.z)**2) <
                                 (self.surfaceparam.r**2)) + 0.0
        elif self.surfacetype == SurfaceType.CIRCLE_QUAD:
            self.pixel_in_material = (((coordroi[:, 0] - self.surfaceparam.x)**2 + (coordroi[:, 2] - self.surfaceparam.z)**2) <
                                 (self.surfaceparam.r**2)) + 0.0
        elif self.surfacetype == SurfaceType.CIRCLE_NEWTON:
            self.pixel_in_material = (((coordroi[:, 0] - self.surfaceparam.x)**2 + (coordroi[:, 2] - self.surfaceparam.z)**2) <
                                 (self.surfaceparam.r**2)) + 0.0


        # The continuous Newton-Raphson method for definition of entrance points
        # shall be used
        # only when at least one of the following conditions is met:
        #   1. 3D ROI
        #   2. 2D ROI displaced from the plane y=0
        if ((self.surfacetype == SurfaceType.LINE_LS or self.surfacetype == SurfaceType.LINE_NEWTON)):
        #and (np.any(coordroi[:, 1] != 0) or np.any(coordelem[:, 1] != 0))):
            if self.surfacetype == SurfaceType.LINE_LS:
                lineparam_a = self.surfaceparam.a
                lineparam_b = self.surfaceparam.b
            elif self.surfacetype == SurfaceType.LINE_NEWTON:
                lineparam_a = self.surfaceparam.a
                lineparam_b = self.surfaceparam.b
            coordelem_copy = np.copy(coordelem)
            coordroi_copy = np.copy(coordroi)
            result = np.asarray(self.__cdist_medium_kernel_continuous(self.VelocityMedium,
                                                                      self.VelocityMaterial, coordelem_copy,
                                                                      coordroi_copy, lineparam_a, lineparam_b))
        # For other cases, the discretized method is used
        elif self.surfacetype == SurfaceType.ARBITRARY or self.surfacetype == SurfaceType.ARBITRARY:
            # aux = coordroi[:, [0, 2]].reshape(roi.w_len, roi.h_len, 2)
            # xaux = np.linspace(aux[:, :, 0].min(), aux[:, :, 0].max(), aux.shape[0])
            # zaux = np.interp(xaux, self.x_discr, self.z_discr)[:, np.newaxis]
            # self.pixel_in_material = aux[:, :, 1] > zaux
            result = self.__cdist_medium_arbitrario(coordelem, coordroi, roi, sel_shot)
        else:
            result = np.asarray(self.__cdist_medium_kernel(self.x_discr, self.z_discr, self.m, self.VelocityMedium,
                                                           self.VelocityMaterial, xz_interface, self.pixel_in_material,
                                                           coordelem, coordroi))

        # ??
        # returned_result = result.reshape(result.shape[1:])
        returned_result = result

        # self.surface_cdist_memory.coordelem =  np.copy(coordelem)
        # self.surface_cdist_memory.coordroi = np.copy(coordroi)
        # self.surface_cdist_memory.returned_result = np.copy(returned_result)

        return returned_result

    def __cdist_medium_arbitrario(self, coordelem, coordroi, roi, sel_shot, num_points=1500, tolerancia=0.05):
        # print(f'Cdist shot {sel_shot}')
        x_interp = np.linspace(roi.w_points[0], roi.w_points[-1], num_points)
        offset = self.data_insp.inspection_params.step_points[sel_shot, 0] \
            - self.data_insp.inspection_params.step_points[0, 0]
        z_interp = np.interp(x_interp + offset,
                             self.x_discr, self.z_discr)
        aux = coordroi[:, [0, 2]].reshape(roi.w_len, roi.h_len, 2)
        xaux = np.linspace(aux[:, :, 0].min(), aux[:, :, 0].max(), aux.shape[0])
        zaux = np.interp(xaux + offset, self.x_discr, self.z_discr)[:, np.newaxis]
        self.pixel_in_material = aux[:, :, 1] > zaux
        # data.surf.z_discr = z_interp
        # data.surf.x_discr = x_interp

        Tx = coordelem[:, 0]
        Tz = coordelem[:, 2]
        Fx = coordroi[self.pixel_in_material.ravel(), 0]
        Fz = coordroi[self.pixel_in_material.ravel(), 2]
        # Sx = self.x_discr
        # Sz = self.z_discr
        Sx = x_interp
        Sz = z_interp
        c1 = self.VelocityMedium*1e3
        c2 = self.VelocityMaterial*1e3

        # calculo normal
        difSx = (Sx[2:] - Sx[:-2]) / 2
        difSz = (Sz[2:] - Sz[:-2]) / 2

        # replica derivada nas extremidades
        difSx = np.concatenate([[difSx[0]], difSx, [difSx[-1]]])
        difSz = np.concatenate([[difSz[0]], difSz, [difSz[-1]]])

        # ângulo da normal é o arcotangente da derivada girada 90 graus
        normal = np.arctan2(-difSx, difSz)

        ang_critico = np.arcsin(c1 / c2)
        resultcouplant = np.zeros((len(Tx), len(coordroi[:,0])))
        resultmedium = np.zeros((len(Tx), len(coordroi[:,0])))
        resultcouplant[:, np.invert(self.pixel_in_material.ravel())] = \
            dist.cdist(coordelem, coordroi[np.invert(self.pixel_in_material.ravel())])
        # resultcouplant, resultmedium = cdist_arb_kernel(Fx, Fz, Sx, Sz, Tx, Tz, ang_critico, c1, c2, normal, tolerancia)
        resultcouplant[:, self.pixel_in_material.ravel()], resultmedium[:, self.pixel_in_material.ravel()] = \
            cdist_arb_kernel(Fx, Fz, Sx, Sz, Tx, Tz, ang_critico, c1, c2, normal, tolerancia)
        returned_result = np.stack([resultcouplant, resultmedium])

        return returned_result

    def __cdist_medium_kernel(self, x_discr, z_discr, m, vmed, vmat, xyz_interface, inmat, coordelem, coordroi):
        n = np.int64(coordelem.shape[0])
        for i_elem in range(n):
            # print('element', i_elem)
            [x_interface, z_interface] = newtonraphsonbatchelement(x_discr, z_discr, m, vmed, vmat,
                                                                   coordelem[i_elem, 0],
                                                                   coordelem[i_elem, 2], coordroi[:, 0], coordroi[:, 2])
            xyz_interface[i_elem, :, 0] = x_interface * inmat + coordroi[:, 0] * (1 - inmat)
            xyz_interface[i_elem, :, 1] = np.zeros_like(xyz_interface[i_elem, :, 0])
            xyz_interface[i_elem, :, 2] = z_interface * inmat + coordroi[:, 2] * (1 - inmat)
        a = coordelem[:, 0].shape
        resultcouplant = np.sqrt((coordelem[:, 0].copy().reshape(*a, 1) - xyz_interface[:, :, 0]) ** 2 +
                                 (coordelem[:, 2].copy().reshape(*a, 1) - xyz_interface[:, :, 2]) ** 2)
        a = coordroi[:, 0].shape
        resultmaterial = np.sqrt((coordroi[:, 0].copy().reshape(1, *a) - xyz_interface[:, :, 0]) ** 2 +
                                 (coordroi[:, 2].copy().reshape(1, *a) - xyz_interface[:, :, 2]) ** 2)

        self.entrypoints = np.array(xyz_interface)

        return resultcouplant, resultmaterial

    def __cdist_medium_kernel_continuous(self, vmed, vmat, coordelem, coordroi, lineparam_a, lineparam_b):
        resultcouplant = np.zeros((coordelem.shape[0], coordroi.shape[0]))
        resultmaterial = np.zeros((coordelem.shape[0], coordroi.shape[0]))
        self.entrypoints = np.zeros((coordelem.shape[0], coordroi.shape[0], 3))

        alpha = -np.arctan(lineparam_a)

        n = np.int64(coordelem.shape[0])
        for i_elem in range(n):
            # Copy coordinates
            coordelem_copy = np.copy(coordelem[i_elem, :])
            coordelem_copy.shape = (1, 3)
            coordroi_copy = np.copy(coordroi)
            # Rotate on y to put the surface on z=0
            coordelem_copy[:, 2] = coordelem_copy[:, 2] - lineparam_b
            coordroi_copy[:, 2] = coordroi_copy[:, 2] - lineparam_b
            coordelem_copy = rotate_axis_y(coordelem_copy, alpha)
            coordroi_copy = rotate_axis_y(coordroi_copy, alpha)
            # Shift (x,y) to put the element on x=0
            elem_x_shift = np.copy(coordelem_copy[0, 0])
            coordelem_copy[:, 0] -= elem_x_shift
            coordroi_copy[:, 0] -= elem_x_shift
            elem_y_shift = np.copy(coordelem_copy[0, 1])
            coordelem_copy[:, 1] -= elem_y_shift
            coordroi_copy[:, 1] -= elem_y_shift
            # Rotate on z to put the pixel (focus) on y=0
            # (This will be done inside the kernel. See you there.)

            x0 = coordelem_copy[0, 0]
            newton_maxit = 10

            [x_interface, y_interface, z_interface] = newtonraphsonbatchelement_continuous(vmed, vmat,
                                                                                           coordelem_copy[0, 0],
                                                                                           coordelem_copy[0, 2],
                                                                                           coordroi_copy[:, 0],
                                                                                           coordroi_copy[:, 1],
                                                                                           coordroi_copy[:, 2],
                                                                                           x0, newton_maxit)
            self.entrypoints[i_elem, :, 0] = x_interface * self.pixel_in_material + \
                                             coordroi_copy[:, 0] * (1 - self.pixel_in_material)
            self.entrypoints[i_elem, :, 1] = y_interface * self.pixel_in_material + \
                                             coordroi_copy[:, 1] * (1 - self.pixel_in_material)
            self.entrypoints[i_elem, :, 2] = z_interface * self.pixel_in_material + \
                                             coordroi_copy[:, 2] * (1 - self.pixel_in_material)

            ###### Undo the rotations and shifts
            # Shift (x,y) to put the element on x=0
            self.entrypoints[i_elem, :, 0] += elem_x_shift
            self.entrypoints[i_elem, :, 1] += elem_y_shift
            # Rotate on y to put the surface on z=0
            self.entrypoints[i_elem, :, :] = rotate_axis_y(self.entrypoints[i_elem, :, :], -alpha)
            self.entrypoints[i_elem, :, 2] = self.entrypoints[i_elem, :, 2] + lineparam_b


            #resultcouplant[i_elem, :] = np.sqrt((coordelem_copy[0, 0] - self.entrypoints[i_elem, :, 0]) ** 2 +
            #                                    (coordelem_copy[0, 1] - self.entrypoints[i_elem, :, 1]) ** 2 +
            #                                    (coordelem_copy[0, 2] - self.entrypoints[i_elem, :, 2]) ** 2)
            #resultmaterial[i_elem, :] = np.sqrt((coordroi_copy[:, 0] - self.entrypoints[i_elem, :, 0]) ** 2 +
            #                                    (coordroi_copy[:, 1] - self.entrypoints[i_elem, :, 1]) ** 2 +
            #                                    (coordroi_copy[:, 2] - self.entrypoints[i_elem, :, 2]) ** 2)
#
            resultcouplant[i_elem, :] = np.sqrt((coordelem[i_elem, 0] - self.entrypoints[i_elem, :, 0]) ** 2 +
                                                (coordelem[i_elem, 1] - self.entrypoints[i_elem, :, 1]) ** 2 +
                                                (coordelem[i_elem, 2] - self.entrypoints[i_elem, :, 2]) ** 2)
            resultmaterial[i_elem, :] = np.sqrt((coordroi[:, 0] - self.entrypoints[i_elem, :, 0]) ** 2 +
                                                (coordroi[:, 1] - self.entrypoints[i_elem, :, 1]) ** 2 +
                                                (coordroi[:, 2] - self.entrypoints[i_elem, :, 2]) ** 2)

        return resultcouplant, resultmaterial

    def cdist_focus(self, coordelem, coordroi):
        xz_interface = np.zeros((coordelem.shape[0], coordroi.shape[0], 2))
        pixel_in_material = np.zeros((coordelem.shape[0], coordroi.shape[0]))

        for i_roi in range(0, coordroi.shape[0]):

            if self.surfacetype == SurfaceType.LINE_LS:
                if coordroi[i_roi, 2] > self.lineparam.a*coordroi[i_roi, 0] + self.lineparam.b:
                    # Pixel is in material
                    pixel_in_material[:, i_roi] = np.ones((coordelem.shape[0]))
                    [x_interface, z_interface] = self.newtonraphsonbatchfocus(
                        coordroi[i_roi, 0], coordroi[i_roi, 2])
                    xz_interface[:, i_roi, 0] = x_interface
                    xz_interface[:, i_roi, 1] = z_interface
                else:
                    # Pixel is in couplant medium (not the material)
                    [x_interface, z_interface] = [coordroi[:, 0], coordroi[:, 2]]

        resultcouplant = np.sqrt((coordelem[:, 0, np.newaxis] - xz_interface[:, :, 0]) ** 2 +
                                 (coordelem[:, 2, np.newaxis] - xz_interface[:, :, 1]) ** 2)
        resultmaterial = np.sqrt((coordroi[np.newaxis, :, 0] - xz_interface[:, :, 0]) ** 2 +
                                 (coordroi[np.newaxis, :, 2] - xz_interface[:, :, 1]) ** 2) * pixel_in_material

        return np.asarray([resultcouplant, resultmaterial])

    def tf(self, xa, za, xf, zf, k):
        da = np.sqrt((xa-self.x_discr[k])**2+(za-self.z_discr[k])**2)
        df = np.sqrt((xf-self.x_discr[k])**2+(zf-self.z_discr[k])**2)
        return (1/self.VelocityMedium)*da + (1/self.VelocityMaterial)*df

    def gridsearch(self, xa, za, xf, zf):
        mintf = 1e10
        k = 0
        for i in range(0, self.x_discr.shape[0]):
            thistf = self.tf(xa, za, xf, zf, i)
            if thistf < mintf:
                mintf = thistf
                k = i
        return k

    def get_water_path(self):
        """ Retorna o tamanho da coluna d'água estimada pelo método de estimativa que retornou o menor SSE. Se a
            superfície for uma reta descrita por z = ax+b, a coluna d'água corresponde ao coeficiente b. Se a superfície
            for um círculo com centro (x, z) e raio r, a coluna d'água corresponde à diferença z-r.

        Returns
        -------
            `int`
                Coluna d'água em mm.
        """

        if self.surfacetype == SurfaceType.LINE_NEWTON:
            return self.linenewtonparam.water_path
        elif self.surfacetype == SurfaceType.CIRCLE_NEWTON:
            return self.circlenewtonparam.water_path
        elif self.surfacetype == SurfaceType.LINE_OLS:
            return self.lineODRparam.water_path
        elif self.surfacetype == SurfaceType.LINE_LS:
            return self.lineparam.water_path
        elif self.surfacetype == SurfaceType.CIRCLE_QUAD:
            return self.circlequadparam.water_path
        elif self.surfacetype == SurfaceType.CIRCLE_MLS:
            return self.circleparam.water_path

    def get_points_in_roi(self, roi, shot, num_points=None):
        if num_points is None:
            num_points = len(roi.w_points)
        x_interp = np.linspace(roi.w_points[0], roi.w_points[-1], num_points)
        offset = self.data_insp.inspection_params.step_points[shot, 0] \
                 - self.data_insp.inspection_params.step_points[0, 0]
        z_interp = np.interp(x_interp + offset,
                             self.x_discr, self.z_discr)
        aux = roi.get_coord()[:, [0, 2]].reshape(roi.w_len, roi.h_len, 2)
        xaux = np.linspace(aux[:, :, 0].min(), aux[:, :, 0].max(), aux.shape[0])
        zaux = np.interp(xaux + offset, self.x_discr, self.z_discr)
        return xaux, zaux

class SurfaceType(Enum):
    """Classe do tipo ``Enum`` que lista os possíveis tipos de superfícies, bem como os métodos de regressão
    correspondentes:

        - CIRCLE_MLS: Círculo com parâmetros definidos pelo método Modified Least Squares.
        - CIRCLE_QUAD: Círculo com parâmetros definidos pelo método Newton-Raphson unidimensional.
        - LINE_LS: Linha com parâmetros definidos por mínimos quadrados.
        - LINE_OLS: Linha com parâmetros definidos por mínimos quadrados ortogonais.
        - CIRCLE_NEWTON: Círculo com parâmetros definidos por Newton-Raphson multivariável
        - LINE_NEWTON: Linha com parâmetros definidos por Newton-Raphson miltivariável.
        - ARBITRARY: Superfície não paramétrica determinada pelo método ARBITRARY
        - ARBITRARY: Superfície não paramétrica determinada pelo método HECTOR

    """

    CIRCLE_MLS = 1
    CIRCLE_QUAD = 2
    LINE_LS = 3
    LINE_OLS = 4
    CIRCLE_NEWTON = 5
    LINE_NEWTON = 6
    ARBITRARY = 7
    PLANE_NEWTON = 8
    CYLINDER_NEWTON = 9


def f(b, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return b[0]*x + b[1]


