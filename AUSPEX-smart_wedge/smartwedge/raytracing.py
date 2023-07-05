import numpy as np
import matplotlib.pyplot as plt

# from numba import njit, prange
def computeCriticAng(v1, v2):
    if v1 <= v2:
        # It is possible to occur total internal refletion;
        return np.arcsin(v1 / v2)
    else:
        # It is not possible to occur total internal reflecton;
        return 2  # Returning 2 means that for every value obtained by np.sin(), 2 is always bigger.


def computeNormalAng(Sx, Sz):
    difSx = (Sx[2:] - Sx[:-2]) / 2
    difSz = (Sz[2:] - Sz[:-2]) / 2

    # replica derivada nas extremidades
    difSx = np.concatenate([[difSx[0]], difSx, [difSx[-1]]])
    difSz = np.concatenate([[difSz[0]], difSz, [difSz[-1]]])

    normal = np.arctan2(-difSx, difSz)
    return normal

# @njit(fastmath=True, parallel=True)
def cdist_arb_kernel_3medium(Fx, Fz, Sx1, Sz1, Sx2, Sz2, Tx, Tz, ang_critico1, ang_critico2, v1, v2, v3, normal1,
                             normal2, tolerancia, plot_ray=False):
    resultcouplant = np.zeros([len(Tx), len(Fx)])
    resultmedium = np.zeros([len(Tx), len(Fx)])

    # distancia de cada ponto da superficie 2 até cada ponto da roi
    ffx, ssx = np.meshgrid(Fx, Sx2)
    ffz, ssz = np.meshgrid(Fz, Sz2)
    ds2f = np.sqrt((ffx - ssx) ** 2 + (ffz - ssz) ** 2)  # ds2f.shape = (número de pontos de s2, número de focos)

    for trans in range(len(Tx)):
        # Parâmetros das retas transdutor-superfície no formato Rts: ax + bz + c = 0
        # Reta esta que parte do elemento do transdutor "trans" até a superfície S1 (mais próxima dele):
        a = (Tz[trans] - Sz1)
        b = (Sx1 - Tx[trans])
        c = (Tx[trans] * Sz1 - Tz[trans] * Sx1)

        # normalizando para Rts: Ax + Bz + C = 0 de forma que A^2 + B^2 = 1
        norm = np.sqrt(a ** 2 + b ** 2)
        A = a / norm
        B = b / norm
        C = c / norm
        # Arts - ângulos da retas Transdutor p/ Superfície 1
        sinArts = a / norm  # Cateto oposto / Hipotenusa ou (Tz - Sz)/norm
        cosArts = (Tx[trans] - Sx1) / norm  # Cateto adjascente / Hipotenusa ou (Tx - Sx) / norm
        Arts = np.arctan2(sinArts, cosArts)  # tan = sin/cos Esse ângulo é no sentido horário do centro do transdutor!
        # Ai - ângulos de incidência
        Ai = Arts - normal1
        validos_s1 = np.abs(Ai) < ang_critico1  # pontos da superficie que obedecem ao angulo crítico
        indexValidos_s1 = np.nonzero(validos_s1)[0]  # indices dos pontos da superficie que obedecem ao angulo critico

        # Foram filtrados os pontos da superfície 1. Será reproduzido o mesmo processo, porém agora considerando
        # que o transmissor é o ponto da superfície 2 que é válido (considerando o critério de ângulo crítico.
        a2 = np.array([(z1 - Sz2) for z1 in Sz1])  # [Varia o ponto se S1, Varia o ponto de S2]
        b2 = np.array([(x1 - Sx2) for x1 in Sx1])  #
        c2 = np.array([x1 * Sz2 - z1 * Sx2 for x1, z1 in zip(Sx1, Sz1)])
        # normalizando para Rts: Ax + Bz + C = 0 de forma que A^2 + B^2 = 1
        norm2 = np.sqrt(a2 ** 2 + b2 ** 2)
        A2 = a2 / norm2
        B2 = b2 / norm2
        C2 = c2 / norm2
        # Arts - ângulos da retas transdutor-superfície
        sinArts2 = a2 / norm2  # (Tz - Sz)/H
        cosArts2 = np.array([(x1 - Sx2) for x1 in Sx1]) / norm2  # (Tx - Sx) / H
        Arts2 = np.arctan2(sinArts2, cosArts2)  # x1 / x2. Esse ângulo é no sentido horário do centro do transdutor!

        # Ai - ângulos de incidência
        Ai2 = Arts2 - normal2  # TEM QUE AVALIAR SE SUBTRAÇÃO É AO LONGO DAS LINHAS OU COLUNAS
        validos2 = np.abs(Ai2) < ang_critico2  # pontos da superficie que obedecem ao angulo crítico
        indexValidos_s2 = np.nonzero(validos2)  # indices dos pontos da superficie que obedecem ao angulo critico

        normal2_matrix = np.tile(normal2, reps=(len(Sx1), 1))
        # Arr - ângulos da retas refratadas
        Arr = np.pi + normal2_matrix * validos2 + np.arcsin(v3 * np.sin(Ai2 * validos2) / v2)  # Será uma matriz
        sinArr = np.sin(Arr)
        cosArr = np.cos(Arr)

        # parâmetros das retas refratadas no formado Rr: Dx + Ez + F = 0
        D = np.asmatrix(sinArr)  # Tz - Sz
        E = np.asmatrix(-cosArr)  # Tx - Sx
        Sz2_mat = np.tile(Sz2, reps=(len(Sz1), 1)) * validos2
        Sx2_mat = np.tile(Sx2, reps=(len(Sx1), 1)) * validos2  # [S2 cte, Varia S2]
        F = np.asmatrix(Sz2_mat * cosArr - Sx2_mat * sinArr)  # com culling

        # F = Sz * (Tx - Sx)/H - Sx * (Tz - Sz)/H

        # tempo de percurso do elemento até cada ponto valido da superficie
        dts1 = np.sqrt((Tx[trans] - (Sx1 + 1000 * ~validos_s1)) ** 2 + (Tz[trans] - (Sz1 + 1000 * ~validos_s1)) ** 2)
        ds1s2 = np.sqrt(
            (np.tile(Sx1 + (~validos_s1 * 1000), reps=(len(Sx2), 1)).T - (Sx2_mat + (~validos2 * 1000))) ** 2 + (
                        np.tile(Sz1 + (~validos_s1 * 1000), reps=(len(Sz2), 1)).T - (
                            Sz2_mat + (~validos2 * 1000))) ** 2)  # [Varia S1, Varia S2]
        erroFocal, penalidade = dist_kernel(D, E, F, Fx[np.newaxis, :], Fz[np.newaxis, :], tolerancia)

        # tempo total de percurso do transdutor até o ponto focal
        tts2 = np.tile(dts1, reps=(len(Sz2), 1)).T / v1 + ds1s2 / v2
        ttf = ds2f / v3 + tts2[:, :, np.newaxis] + penalidade

        # Dimensão do ttf : [Varia S1, Varia S2, Varia F]
        try:
            indiceCandidatoMinimo = [np.where(ttf[:, :, f] == np.min(ttf[:, :, f])) for f in range(len(Fx))]

            if plot_ray:
                # Para o foco n:
                n = 0

                idx_S1 = indiceCandidatoMinimo[n][0]
                idx_S2 = indiceCandidatoMinimo[n][:]
                A_inicial = A[idx_S1]
                B_inicial = B[idx_S1]
                C_inicial = C[idx_S1]

                A_intermed = A2[idx_S2]
                B_intermed = B2[idx_S2]
                C_intermed = C2[idx_S2]

                D_final = D[idx_S2]
                E_final = E[idx_S2]
                F_final = F[idx_S2]

                S1_coord = np.array([Sx1[idx_S1], Sz1[idx_S1]])
                S2_coord = np.array([Sx2[idx_S2[1]], Sz2[idx_S2[1]]])

                epsilon = 1e-2

                # Raio transdutor - superfície 1
                z1 = np.arange(Tz[trans], S1_coord[1] + epsilon, epsilon)
                x1 = (B_inicial * z1 + C_inicial) / (-A_inicial)

                # Raio superfície 1 - superfície 2:
                z2 = np.arange(S1_coord[1], S2_coord[1] + epsilon, epsilon)
                x2 = (float(B_intermed) * z2 + float(C_intermed)) / (float(A_intermed))
                x2 -= (x2[0] - x1[-1])

                # Raio superfície 2 - foco 1:
                z3 = np.arange(S2_coord[1], Fz[n] + epsilon + 15, epsilon)
                x3 = (float(E_final) * z3 + float(F_final)) / (-float(D_final))

                plt.plot(x1, z1, color='r')
                plt.plot(x2, z2, color='b')
                plt.plot(x3, z3, color='m')
                plt.plot(Sx1, Sz1, 'o', color='g')
                plt.plot(Sx2, Sz2, 'o', color='g')
                plt.plot(Tx[trans], Tz[trans], 'o', color='k')
                circ = plt.Circle((Fx[n], Fz[n]), tolerancia)
                plt.gca().add_artist(circ)
                plt.axis("Equal")
                plt.show()

            indicePontoEntrada = indexValidos_s1[indiceCandidatoMinimo[0]]

            # x = (bz + c) / -a
            # x = (ez + f) / -d

            # calcula distâncias
            d1 = dts1[indiceCandidatoMinimo]
            d2 = ds1s2[indiceCandidatoMinimo]
            d3 = ds2f[indicePontoEntrada, np.arange(len(Fx))]


        except:
            d1 = np.Inf
            d2 = np.Inf
            d3 = np.Inf

        resultcouplant[trans, :] = d1
        resultmedium[trans, :] = d2

    return resultcouplant, resultmedium


# @njit(fastmath=True, parallel=True)
def dist_kernel(D, E, F, Fx, Fz, tolerancia):
    # distâncias entre cada reta refratada e cada ponto da ROI (maior custo computacional)
    numFocus = Fx.shape[1]
    erroFocal = np.zeros(shape=(D.shape[0], D.shape[1], numFocus))
    for i in range(numFocus):
        focusX = Fx[:, i][0]
        focusY = Fz[:, i][0]
        erroFocal[:, :, i] = np.abs(D * focusX + E * focusY + F)
    candidatos = erroFocal < tolerancia
    penalidade = 1e3 * np.invert(candidatos)

    return erroFocal, penalidade


