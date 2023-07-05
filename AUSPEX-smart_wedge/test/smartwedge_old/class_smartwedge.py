import numpy as np
import scipy.optimize as optm

from geometric_auxiliaries import *

class Smartwedge:
    def __init__(self, c, r0, wc, wedge_cl, coupling_cl, Tprime, offset):
        self.c = c
        self.r0 = r0
        self.wc = wc
        self.wedge_cl = wedge_cl
        self.coupling_cl = coupling_cl
        self.A = np.array([0, 0])
        self.B = np.array([2*self.c, 0])
        self.Tprime = Tprime
        self.offset = offset
        self.__compute_time_bda()
        Nx = 53.26
        self.N = np.array([Nx, self.parametric_curve(Nx)])

        # Encontra parâmetros da smartwedge_old com base nos dados de entrada:
        self.__fit_wedge()


    def __compute_time_bda(self):
        ac = np.arcsin(self.coupling_cl / self.wedge_cl)
        self.En = np.array([
            (self.A[0] + self.B[0]) / 2,
            self.c / (1 / np.cos(ac) + np.tan(ac))
        ])
        r, h, k = findCircle(self.B, self.A, self.En)
        R = self.r0 + self.wc

        self.xint = computeIntersectionBetweenCircles(self.r0 + self.wc, r, h, k)
        self.yint = np.sqrt(R ** 2 - self.xint ** 2)

        ad = np.sqrt((self.A[0] - self.xint) ** 2 + (self.A[1] - self.yint) ** 2)
        db = np.sqrt((self.B[0] - self.xint) ** 2 + (self.B[1] - self.yint) ** 2)

        time_ad = ad / self.coupling_cl
        time_db = db / self.wedge_cl
        self.time_bda = time_db + time_ad

    def __fit_wedge(self):
        # Obtendo parâmetros da elipse:
        # Encontra qual é a reta que é tangente a circunferência de raio = norm(N) e toca no ponto Tprime:
        # Equação da CircunferÊncia centrada em (0,0); f(x) = np.sqrt(r**2 - x**2)
        # A derivada será: df(x)dx = -x/(np.sqrt(r**2 - x**2)
        # Onde é válido apenas no intervalo: [-r, r]
        # A reta então de mesma inclinação deverá ser:
        # y(x) = dfdx * x + b
        # Para o caso em que x=2*c => y(x) = Tprime
        # Tprime = an * (2*c) + b
        N = np.array([53.26, self.parametric_curve([53.26])], dtype=float)  # Ponto arbitrário na curva paramétrica;
        raio = np.linalg.norm(N)
        f_circle = lambda x: np.sqrt(raio ** 2 - x ** 2)
        dfdx_circle = lambda x: -x / (np.sqrt(raio ** 2 - x ** 2))
        b1 = lambda xt: - self.Tprime + (xt * (2 * self.c)) / (np.sqrt(raio ** 2 - xt ** 2))
        b2 = lambda xt: f_circle(xt) - xt * dfdx_circle(xt)

        cost_fun = lambda x: np.power(b1(x) - b2(x), 2)
        xT = optm.minimize_scalar(cost_fun, bounds=(0, self.r0 + self.wc), method='bounded').x
        yT = f_circle(xT)
        an = dfdx_circle(xT)
        bn = -self.Tprime - an * (2 * self.c)
        tangent_line_circle = lambda x: an * x + bn
        Fs = np.array([0, tangent_line_circle(0) + self.offset])


        # Descobrir equação da elipse que possui como foco A e B e passa pelo ponto (0, Fs):

        # a =np.sqrt( 20143.2)
        # b = np.sqrt(13040.93)

        b = lambda a: np.sqrt(a ** 2 - self.c ** 2)
        ellipse_costfun = lambda a: np.power(((Fs[0] - self.c) ** 2) / (a ** 2) + (Fs[1] ** 2) / (b(a) ** 2) - 1, 2)
        self.a = optm.minimize_scalar(ellipse_costfun, bounds=(10, 300), method='bounded').x
        self.b = np.sqrt(self.a ** 2 - self.c ** 2)

        return None

    def __parametric_curve_scalar(self, x, v1, v2, time_bda, B):
        f = (-(2 * v2 ** 4 * x * B[0] - v1 ** 4 * x ** 2 - v2 ** 4 * B[
            0] ** 2 - v2 ** 4 * x ** 2 + time_bda ** 2 * v2 ** 2 * v1 ** 4
               + time_bda ** 2 * v2 ** 4 * v1 ** 2 + 2 * v2 ** 2 * v1 ** 2 * x ** 2 + v2 ** 2 * v1 ** 2 * (B[0]) ** 2
               - 2 * time_bda * v2 ** 2 * v1 ** 2 *
               (time_bda ** 2 * v2 ** 2 * v1 ** 2 - v2 ** 2 * (B[0]) ** 2
                + 2 * x * v2 ** 2 * B[0] + v1 ** 2 * (B[0]) ** 2 - 2 * x * v1 ** 2 * B[0]) ** (1 / 2)
               - 2 * v2 ** 2 * v1 ** 2 * x * B[0]) ** (1 / 2)) \
            / (v2 ** 2 - v1 ** 2)
        return f

    def parametric_curve(self, x):
        fcn = lambda x: self.__parametric_curve_scalar(x, self.wedge_cl, self.coupling_cl, self.time_bda, self.B)
        fnc_vectorized = np.frompyfunc(fcn, nin=1, nout=1)
        return fnc_vectorized(x)

    def compute_entrypoints(self, circle_angle):
        angle_a = 0
        angle_b = np.rad2deg(np.arctan(self.N[1] / self.N[0]))
        angle_c = 90

        beta = np.abs(circle_angle)

        if angle_a <= beta <= angle_b:
            xi, yi = self.center_area_intersection(beta)
        elif angle_b < beta <= angle_c:
            xi, yi = self.side_area_intersection(beta)
        else:
            print("Ângulo não suportado")

        if circle_angle < 0:
            yi = -yi

        alpha = np.float(np.arctan(yi / (2 * self.c - xi)))

        return alpha, (xi, yi)

    def compute_planewave_angles(self, circle_angle):
        angles = list()
        for beta in circle_angle:
            ang, _ = self.compute_entrypoints(beta)
            angles.append(ang)
        return np.rad2deg(np.array(angles))

    def center_area_intersection(self, circle_angle):
        beta = np.deg2rad(circle_angle)

        line_func = lambda x, beta, offset : (x - offset) * np.tan(beta)

        phi = lambda x: np.power(line_func(x, beta, 0) - self.parametric_curve(x), 2)
        xi = optm.minimize_scalar(phi, bounds=(self.xint, 85.24284999101062), method='bounded')
        xi = xi.x
        yi = self.parametric_curve(xi)

        return xi, yi


    def side_area_intersection(self, circle_angle):
        beta = np.deg2rad(circle_angle)
        A = 2 * self.c

        sol = optm.root(lambda B: - (B**2) + A**2 + (2*self.a - B)**2 - 2*A*(2*self.a - B) * np.cos(beta), 0)
        B = sol.x[0]
        C = 2 * self.a - B

        xi = C * np.cos(beta)
        yi = C * np.sin(beta)

        return xi, yi

    def compute_total_dist(self, ang_focus_deg):
        # Encontra o ângulo e a distância do foco em relação ao referencial do transdutor.
        # r_focus é a distância do foco para o centro da tubulação
        # ang_focus_deg é posição angular da leitura em relação ao centro da tubulação
        ang_focus_deg = np.abs(ang_focus_deg)

        angle_a = 0
        angle_b = np.rad2deg(np.arctan(self.N[1] / self.N[0]))
        angle_c = 90

        radius = np.linalg.norm(self.N)
        # Checa se o foco está na direção da incidÊncia direta:
        if angle_a <= ang_focus_deg < angle_b:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            dist_bw = np.linalg.norm(
                intersect - self.B) * 1e-3  # Tempo entre centro do transdutor e incidência direta
            dist_ws = (np.linalg.norm(
                self.A - intersect) - self.r0) * 1e-3  # Tempo entre tubulação e sapata
            dist = dist_bw + dist_ws + self.r0 * 1e-3

        # Checa se o foco está na direção da incidÊncia indireta:
        elif angle_b <= ang_focus_deg <= angle_c:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            dist_bd = np.linalg.norm(intersect - self.B) * 1e-3  # Distância entre centro do transdutor e elipse
            dist_dw = (np.linalg.norm(self.A - intersect) - radius) * 1e-3  # Dist. entre elipse e borda da sapata
            dist_ws = (radius - self.r0) * 1e-3  # Distância entre sapata e tubulação
            dist = dist_bd + dist_dw + dist_ws + self.r0 * 1e-3

        # Distância em metros
        return dist

    def compensate_time_wedge(self, ang_focus_deg):
        # Encontra o ângulo e a distância do foco em relação ao referencial do transdutor.
        # r_focus é a distância do foco para o centro da tubulação
        # ang_focus_deg é posição angular da leitura em relação ao centro da tubulação
        ang_focus_deg = np.abs(ang_focus_deg)

        angle_a = 0
        angle_b = np.rad2deg(np.arctan(self.N[1] / self.N[0]))
        angle_c = 90

        radius = np.linalg.norm(self.N)
        # Checa se o foco está na direção da incidÊncia direta:
        if angle_a <= ang_focus_deg < angle_b:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            time_bw = np.linalg.norm(intersect - self.B) / (
                        self.wedge_cl * 1e3) * 1e-3  # Tempo entre centro do transdutor e incidência direta

            compesation_time = time_bw

        # Checa se o foco está na direção da incidÊncia indireta:
        elif angle_b <= ang_focus_deg <= angle_c:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            time_bd = np.linalg.norm(intersect - self.B) / (
                        self.wedge_cl * 1e3) * 1e-3  # Distância entre centro do transdutor e elipse
            time_dw = (np.linalg.norm(self.A - intersect) - radius) / (
                        self.wedge_cl * 1e3) * 1e-3  # Dist. entre elipse e borda da sapata
            compesation_time = time_bd + time_dw

        return compesation_time