import matplotlib.pyplot as plt
from smartwedge import *
from prettytable import PrettyTable
from class_smartwedge import Smartwedge

def circle_equation(x, radius, h, k):
    y = np.sqrt(radius**2 - (x - h)**2) + k
    return y



if __name__ == "__main__":
    # Parêmtros da Elipse:

    c = 84.28
    r0 = 71
    wc = 6.2
    offset = 2
    Tprime = 19.5

    # Cria um objeto do tipo smartwedge_old:
    # Ellipse parameters:
    c = 84.28
    r0 = 67.15
    wc = 6.2
    offset = 2
    Tprime = 19.5

    # Velocidade do som nos materiais em mm/us
    v1 = 6.36
    v2 = 1.43
    v_steel = 5.9

    # Espessura da parede da tubulação em milímetros:
    tube_wall_width = 16

    # Criação do objeto smartwedge_old:
    sm = Smartwedge(c, r0, wc, v1, v2, Tprime, offset)

    # Localização dos dois focos da elípse:
    A = np.array([0, 0])
    B = np.array([2 * c, 0])

    # Velocidade do som nos materiais em km/s
    v1 = 6.37
    v2 = 1.43

    # Ângulo crítico:
    ac = np.arcsin(v2 / v1)
    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])
    r, h, k = findCircle(B, A, En)


    xint = computeIntersectionBetweenCircles(r0 + wc, r, h, k)
    yint = np.sqrt((r0 + wc) ** 2 - xint ** 2)
    D = np.array([xint, yint])

    da = np.linalg.norm(A-D)
    bd = np.linalg.norm(B-D)

    time_ad = da / v2
    time_db = bd / v1
    time_bda = time_db + time_ad




    # Ângulos de leitura da circuferência:
    step = 1
    betas = np.arange(0, 45+step, step)


    # # Range em X da região da curva parametrizada:
    xmax = 85.08749525101061+.15535474
    x = np.arange(-r0, r0, 1e-5)
    t = np.arange(xint, 85.24284999101062, 1e-5)

    # Encapsulamento da curva parametrizada:
    curve = lambda x : center_part_curve(x, v1, v2, time_bda, B)
    specimen_circle = lambda t : circle_equation(t, r0, 0, 0)

    # Encontra qual é a reta que é tangente a circunferência de raio r0+wc e toca no ponto Tprime:
    # Equação da CircunferÊncia centrada em (0,0); f(x) = np.sqrt(r**2 - x**2)
    # A derivada será: df(x)dx = -x/(np.sqrt(r**2 - x**2)
    # Onde é válido apenas no intervalo: [-r, r]
    # A reta então de mesma inclinação deverá ser:
    # y(x) = dfdx * x + b
    # Para o caso em que x=2*c => y(x) = Tprime
    # Tprime = an * (2*c) + b
    N = np.array([53.26, curve([53.26])], dtype=float)  # Ponto arbitrário na curva paramétrica;
    raio = np.linalg.norm(N)
    f_circle = lambda x: np.sqrt(raio ** 2 - x ** 2)
    dfdx_circle = lambda x: -x / (np.sqrt(raio ** 2 - x ** 2))
    b1 = lambda xt: - Tprime + (xt * (2 * c)) / (np.sqrt(raio ** 2 - xt ** 2))
    b2 = lambda xt: f_circle(xt) - xt * dfdx_circle(xt)

    cost_fun = lambda x: np.power(b1(x) - b2(x), 2)
    xT = optm.minimize_scalar(cost_fun, bounds=(0, r0 + wc), method='bounded').x
    yT = f_circle(xT)
    an = dfdx_circle(xT)
    bn = -Tprime - an * (2 * c)
    tangent_line_circle = lambda x: an * x + bn
    Fs = np.array([0, tangent_line_circle(0) + offset])

    # Descobrir equação da elipse que possui como foco A e B e passa pelo ponto (0, Fs):

    # a =np.sqrt( 20143.2)
    # b = np.sqrt(13040.93)

    b = lambda a : np.sqrt(a**2 - c**2)
    ellipse_costfun = lambda a : np.power(((Fs[0] - c)**2)/(a**2) + (Fs[1]**2)/(b(a)**2) - 1, 2)
    a = optm.minimize_scalar(ellipse_costfun, bounds=(10, 300), method='bounded').x
    b = np.sqrt(a**2 - c**2)

    table = PrettyTable(["β_original", "β_calculado", "α1", "α2", "ângulo normal à circunferência (calculado)"])
    for k, beta in enumerate(betas):
        alpha, intersect = compute_entrypoints(beta, a, c, r0 + wc, v1, v2, A, B, sm.N)
        # Reta que parte do centro do transdutor (2*c, 0) no ângulo alpha:
        an = -np.tan(alpha)
        # (2*c) * a + b = 0
        bn = -(2 * c) * an

        planewave_line = lambda x: an * (x) + bn
        phi = lambda x: np.power(curve(x) - planewave_line(x), 2)
        # Encontra o ponto de intersecão da reta que liga o centro do transdutor e a superfície parametrizada:
        xintersection = optm.minimize_scalar(phi, bounds=(xint, xmax), method='bounded').x
        # Ponto de intersecção:
        plt.scatter(xintersection, curve(xintersection), color=[1, 0, 1])

        # Derivada da superfície parametrizada no ponto de refração:
        idx = np.argmin(np.power(t - xintersection, 2))
        x1 = t[idx - 1]
        x2 = t[idx]
        y1 = curve(x1)
        y2 = curve(x2)
        dfdx = (y2 - y1) / (x2 - x1)
        # Ângulo da reta tangente à curva parametrizada no ponto de intersecção:
        tang_line_angle = np.abs(np.rad2deg(np.arctan(dfdx)))

        # Plota a reta que liga o centro do transdutor e a curva parametrizada:
        x_span = np.arange(xintersection, 2 * c)
        plt.plot(x_span, an * x_span + bn, ':')

        # Reta tangente ao ponto de intersecção (x1,y1):
        tn = np.arange(x1 - 20, x1 + 20)
        an = dfdx
        bn = y1 - dfdx * x1
        plt.plot(tn, an * tn + bn)
        plt.scatter(x1, y1)

        # O ângulo de incidência na superfície é:
        inc_angle = 90 - (tang_line_angle - np.rad2deg(alpha))

        # Aplicando a lei de Snell:
        refract_angle = np.arcsin(v2 / v1 * np.sin(np.deg2rad(inc_angle)))

        np.rad2deg(refract_angle)

        # Convertendo para um ângulo que descreve uma reta:
        refract_line_angle = 90 - tang_line_angle - np.rad2deg(refract_angle)

        # Desenhando a trajetória refratada:
        an = np.tan(np.deg2rad(refract_line_angle))
        bn = y1 - an * x1
        xspan = np.arange(0, x1, 1e-3)
        refracted_line = lambda x: an * x + bn
        plt.plot(xspan, refracted_line(xspan), ":", color=[0, 1, 0])

        # Interscção entre raio refratado e circunferência:
        phi = lambda x: np.power(specimen_circle(x) - refracted_line(x), 2)
        xi = optm.minimize_scalar(phi, bounds=(0, r0), method='bounded').x
        yi = specimen_circle(xi)
        angulo_central = np.rad2deg(np.arctan(yi / xi))
        plt.scatter(xi, yi, color=[0, 0, 0])

        circle_incid_angle = 90 - (refract_line_angle - angulo_central)

        table.add_row([f"{beta:.2f}", f"{refract_line_angle:.2f}", f"{inc_angle:.2f}", f"{np.rad2deg(refract_angle):.2f}", f"{circle_incid_angle:.2f}"])

        plt.axis('equal')
        plt.ylim([-20, 100])
    print(table)
