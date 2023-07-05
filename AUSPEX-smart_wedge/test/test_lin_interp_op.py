import numpy as np

from scipy.stats import norm
from framework.linterp_oper import linterp
from framework.fk_mig import nextpow2, f_k_sweep
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # execute only if run as a script
    grid_x = np.linspace(-20.0, 20.0, 9)[:, np.newaxis]
    #y = np.random.randn(9, 1)
    y = grid_x ** 2 + 1.0
    grid_xq = np.sort(np.random.uniform(-11.0, 31.0, 200)[:, np.newaxis], axis=0)
    # grid_xq = np.linspace(-20.0, 20.10, 200)[:, np.newaxis]
    yq = np.random.randn(200, 1)

    yqn_til = linterp(grid_x, grid_xq, y)
    yn_til = linterp(grid_x, grid_xq, yq, "a")

    dot_xn = np.vdot(yn_til.flatten("F"), y.flatten("F"))
    dot_yn = np.vdot(yq.flatten("F"), yqn_til.flatten("F"))

    plt.plot(grid_x, y)
    plt.plot(grid_xq, yq_til)  # yq_til não existe, talvez yn_titl?
    plt.plot(grid_xq, yqn_til)
    plt.show()

    # np.random.seed(1)
    x = norm.ppf(np.reshape(np.random.rand(200*32), (200, 32), order="F"))
    y = norm.ppf(np.reshape(np.random.rand(200*32), (200, 32), order="F"))

    dt = 1e-8
    du = 1e-3
    c = 5900.0

    nt0, nu0 = x.shape
    nt = 2 ** (nextpow2(nt0) + 1)
    nu = nu0 * 2

    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    if nu > 1:
        ku = np.fft.fftshift(np.fft.fftfreq(nu, d=du))
    else:
        ku = -nu / du / 2.0
    kku, ff = np.meshgrid(ku, f)
    fkz = f_k_sweep(c, ff/(c / 2.0), kku)


    # ku = np.linspace(-5.0, 5.0, 90)
    # f = np.linspace(-20.0, 20.0, 81)
    # kku, ff = np.meshgrid(ku, f)
    # ermv = 0.1
    # data = np.exp((-(ff + 0.5) ** 2 / ermv ** 2 + kku ** 2) / 100.0)
    # fkz = ermv * np.sign(ff) * (np.sqrt(kku ** 2 + ff ** 2 / ermv ** 2))

    ftx = np.fft.fftshift(np.fft.fft2(x, s=(nt, nu)))
    fty = np.fft.fftshift(np.fft.fft2(y, s=(nt, nu)))

    fty_til = linterp(f, fkz, ftx, op="a")
    ftx_til = linterp(f, fkz, fty, op="d")

    y_til = np.real(np.fft.ifft2(np.fft.ifftshift(fty_til), s=(nt, nu)))
    y_til = y_til[0:nt0, 0:nu0]
    x_til = np.real(np.fft.ifft2(np.fft.ifftshift(ftx_til), s=(nt, nu)))
    x_til = x_til[0:nt0, 0:nu0]

    dot_x = np.vdot(x_til.flatten("F"), x.flatten("F"))
    dot_y = np.vdot(y.flatten("F"), y_til.flatten("F"))

    print(dot_x, dot_y, dot_x - dot_y)

