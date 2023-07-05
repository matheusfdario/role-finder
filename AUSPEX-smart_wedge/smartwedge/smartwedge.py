import scipy.optimize as optm
import numpy as np
from smartwedge.geometric_utils import *


class Smartwedge:
    def __init__(self, c, r0, wc, wedge_cl, coupling_cl, Tprime, offset, criteria_1=True, multiply_factor=1.5,
                 sm_type="full"):
        # Se o critério 1 for True, isso significa que um raio partindo de qualquer elemento do transdutor (sobretudo os
        # mais nos extremos) deve ser capaz de varrer a região de incidência direta inteira.
        self.c = c  # Tamanho da metade do semieixo menor da elipse
        self.r0 = r0  # Raio da tubulação sob inspeção
        self.wc = wc  # Espaçamento mínimo entre smartwedge_old e tubulação na região de incidência indireta
        self.wedge_cl = wedge_cl  # Velocidade na sapata
        self.coupling_cl = coupling_cl  # Velocidade no acoplante
        self.A = np.array([0, 0])  # Posição do centro da tubulação
        self.B = np.array([2 * self.c, 0])  # Posição do centro do transdutor
        self.Tprime = Tprime
        self.active_aperture = Tprime / 2  # Metade do comprimento da abertura ativa do transdutor
        self.TTprime = Tprime * multiply_factor  # Região plana que o transdutor está em contato(maior do que a
        # abertura ativa do transdutor)
        self.offset = offset  #
        self.sm_type = sm_type # Tipo de smartwedge: compacta (apenas incidência direta) ou completa.

        # Calcula o tempo de voo entre o centro do transdutor "B", superfície da sapata na incidÊncia indireta "D"
        # e o centro da tubulação "A":
        self.time_bda = self.__compute_time_bda()

        # Calcula o ponto máximo em X até onde a curva paramêtrica da incidência direta chega:
        self.xmax = self._compute_xmax()

        # Calcula o ponto N onde há a transição entre a zona de incidência direta e indireta:
        self.N = self._compute_indirect2direct_transition(criteria_1)

        # Calcula os ângulos de transição entre as regiões de incidência direta e indireta (antihorário em relação ao
        # eixo X e o centro da tubulação):
        self.angle_a = 0
        self.angle_b = np.rad2deg(np.arctan(self.N[1] / self.N[0]))
        self.angle_c = 90

        # Encontra parâmetros da smartwedge_old com base nos dados de entrada:
        self.__fit_wedge()

    def _compute_indirect2direct_transition(self, criteria_1):
        if criteria_1:
            # Encontra qual é a reta que é tangente a circunferência de raio r0+wc e toca no ponto Tprime:
            # Equação da CircunferÊncia centrada em (0,0); f(x) = np.sqrt(r**2 - x**2)
            # A derivada será: df(x)dx = -x/(np.sqrt(r**2 - x**2)
            # Onde é válido apenas no intervalo: [-r, r]
            # A reta então de mesma inclinação deverá ser:
            # y(x) = dfdx * x + b ----> Isolando b teremos: b = y(x) - dfdx * x (eq. 1)
            # Para o caso em que x=2*c => y(x) = -Tprime
            # -Tprime = an * (2*c) + b ----> Isolando b teremos: b = -Tprime - dfdx * (2 * c) (eq. 2)

            fun_curve = lambda x: self.parametric_curve(x)
            epsilon = 1e-5  # Tamanho do passo da derivada
            dfdx_curve = lambda x: (fun_curve(x + epsilon) - fun_curve(x)) / ((x + epsilon) - x)
            b1 = lambda xt: - self.Tprime - dfdx_curve(xt) * (2 * self.c)
            b2 = lambda xt: fun_curve(xt) - xt * dfdx_curve(xt)

            cost_fun = lambda x: np.power(b1(x) - b2(x), 2)
            xT = optm.minimize_scalar(cost_fun, bounds=(0, self.r0 + self.wc), method='bounded').x
            yT = fun_curve(xT)
            Nx = xT  # onde em tese f_circle(xT) == self.parametric_curve(xT)

            # Plot do esquema geométrico
            # Parâmetros dessa reta:
            an = dfdx_curve(xT)
            bn = -self.Tprime - an * (2 * self.c)
            raio = self.r0 + self.wc
            r_span = np.linspace(-raio, raio, 1000)
            x_span = np.linspace(0, self.B[0], 1000)
            x_curve_span = np.linspace(0, self.xmax, 1000)

            # # Plot do círculo da incidência indireta:
            # import matplotlib.pyplot as plt
            # plt.plot(r_span, circle_equation(raio)(r_span), ':')
            # # Plot da região de incidência direta
            # plt.plot(x_curve_span, self.parametric_curve(x_curve_span), ':')
            # # Plot da reta tangente à curva parametrica que liga ao ponto inferior do transdutor:
            # plt.plot(x_span, x_span * an + bn, ':g')
            # plt.plot(xT, yT, 'o', color=[1, 0, 0])

        else:
            # O ponto de transição é onde a circunferência de raio (r0 + wc) toca a curva de incidência direta:
            circle_a = circle_equation(self.r0 + self.wc)  # Círculo da incidência indireta
            cost_fun = lambda x: np.power(circle_a(x) - self.parametric_curve(x), 2)
            Nx = optm.minimize_scalar(cost_fun, bounds=(0, self.B[0]), method='bounded').x
        return np.array([Nx, self.parametric_curve(Nx)])

    def _compute_xmax(self):
        return optm.minimize_scalar(self.parametric_curve, bounds=(0, self.B[0]), method='bounded').x

    def _scalar_is_indirect_inc(self, ang_tube_deg):

        # Checa se o foco está na direção da incidÊncia direta:
        if self.angle_a <= ang_tube_deg < self.angle_b:
            return True

        # Checa se o foco está na direção da incidÊncia indireta:
        elif self.angle_b <= ang_tube_deg <= self.angle_c:
            return False

    def is_indirect_inc(self, ang_tube_deg):
        return np.array([self._scalar_is_indirect_inc(ang) for ang in ang_tube_deg])

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
        return time_db + time_ad

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
        raio = np.linalg.norm(self.N)
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
        beta = np.abs(circle_angle)

        if self.angle_a <= beta <= self.angle_b:
            xi, yi = self.center_area_intersection(beta)
        elif self.angle_b < beta <= self.angle_c:
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

        line_func = lambda x, beta, offset: (x - offset) * np.tan(beta)

        phi = lambda x: np.power(line_func(x, beta, 0) - self.parametric_curve(x), 2)
        xi = optm.minimize_scalar(phi, bounds=(self.xint, self.xmax), method='bounded')
        xi = xi.x
        yi = self.parametric_curve(xi)

        return xi, yi

    def side_area_intersection(self, circle_angle):
        beta = np.deg2rad(circle_angle)
        A = 2 * self.c

        sol = optm.root(lambda B: - (B ** 2) + A ** 2 + (2 * self.a - B) ** 2 - 2 * A * (2 * self.a - B) * np.cos(beta),
                        0)
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

        radius = np.linalg.norm(self.N)
        # Checa se o foco está na direção da incidÊncia direta:
        if self.angle_a <= ang_focus_deg < self.angle_b:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            dist_bw = np.linalg.norm(
                intersect - self.B) * 1e-3  # Tempo entre centro do transdutor e incidência direta
            dist_ws = (np.linalg.norm(
                self.A - intersect) - self.r0) * 1e-3  # Tempo entre tubulação e sapata
            dist = dist_bw + dist_ws + self.r0 * 1e-3

        # Checa se o foco está na direção da incidÊncia indireta:
        elif self.angle_b <= ang_focus_deg <= self.angle_c:
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

        radius = np.linalg.norm(self.N)
        # Checa se o foco está na direção da incidÊncia direta:
        if self.angle_a <= ang_focus_deg < self.angle_b:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            time_bw = np.linalg.norm(intersect - self.B) / (
                    self.wedge_cl * 1e3) * 1e-3  # Tempo entre centro do transdutor e incidência direta

            compesation_time = time_bw

        # Checa se o foco está na direção da incidÊncia indireta:
        elif self.angle_b <= ang_focus_deg <= self.angle_c:
            focus_angle, intersect = self.compute_entrypoints(ang_focus_deg)
            time_bd = np.linalg.norm(intersect - self.B) / (
                    self.wedge_cl * 1e3) * 1e-3  # Distância entre centro do transdutor e elipse
            time_dw = (np.linalg.norm(self.A - intersect) - radius) / (
                    self.wedge_cl * 1e3) * 1e-3  # Dist. entre elipse e borda da sapata
            compesation_time = time_bd + time_dw

        return compesation_time

    def draw(self, plot_tube=False):
        # Descobrindo o ponto em que a elipse intercepta a circunferÊncia:
        raio = np.linalg.norm(self.N)
        y_tubulacao = circle_equation(raio)
        y_elipse = ellipse_equation(self.a, self.b, self.c)
        cost_fun_a = lambda x: np.power(y_tubulacao(x) - y_elipse(x), 2)
        x_intercept_elipse_tub = optm.minimize_scalar(cost_fun_a, bounds=(-raio, raio), method='bounded').x

        # Circunferência que liga o fim da região reta em que fica em contato com o transdutor e centro no (0,0)
        y_circ = circle_equation(np.sqrt(self.B[0] ** 2 + self.TTprime ** 2))
        cost_fun_b = lambda x: np.power(y_circ(x) - y_elipse(x), 2)
        x_intercept_ellipse_trands = optm.minimize_scalar(cost_fun_b, bounds=(0, self.B[0]), method='bounded').x

        # Importa biblioteca para plotar:
        import matplotlib.pyplot as plt

        if self.sm_type == "compact":
            pass
        else:
            # Plota incidência indireta:
            x_indirect_span = np.linspace(x_intercept_elipse_tub, self.N[0], 1000)
            plt.plot(x_indirect_span, circle_equation(np.linalg.norm(self.N))(x_indirect_span), color=[0, 0, 0])
            plt.plot(x_indirect_span, -circle_equation(np.linalg.norm(self.N))(x_indirect_span), color=[0, 0, 0])

        # Plota incidência direta:
        x_direct_span = np.linspace(self.N[0], self.xmax, 1000)
        plt.plot(x_direct_span, self.parametric_curve(x_direct_span), color=[0, 0, 0])
        plt.plot(x_direct_span, -self.parametric_curve(x_direct_span), color=[0, 0, 0])

        # Smartwedge: Elipse
        x_elipse = np.arange(x_intercept_elipse_tub, x_intercept_ellipse_trands)
        plt.plot(x_elipse, ellipse_equation(self.a, self.b, self.c)(x_elipse), color=[0, 0, 0])
        plt.plot(x_elipse, -ellipse_equation(self.a, self.b, self.c)(x_elipse), color=[0, 0, 0])

        # Região plana transdutor:
        y_transd = np.linspace(-self.TTprime, self.TTprime, num=1000)
        x_transd = np.ones_like(y_transd) * self.B[0]
        plt.plot(x_transd, y_transd, color=[0, 0, 0])

        # Círculo da sapata que liga a regiao do trandsdutor e a elipse:
        cir_span = np.linspace(x_intercept_ellipse_trands, self.B[0], num=1000)
        plt.plot(cir_span, y_circ(cir_span), color=[0, 0, 0])
        plt.plot(cir_span, -y_circ(cir_span), color=[0, 0, 0], label='Sapata')

        if plot_tube == True:
            # Tubulação:
            tube = lambda x: np.sqrt(self.r0 ** 2 - x ** 2)
            tube_span = np.linspace(-self.r0, self.r0, num=1000)
            plt.plot(tube_span, tube(tube_span), ':', color=[0, 0, 0])
            plt.plot(tube_span, -tube(tube_span), ':', color=[0, 0, 0], label='Tubulação sob inspeção.')


