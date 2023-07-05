import matplotlib.pyplot as plt
import numpy as np


def blackman(arg, supp):
    M = 1000
    window = np.blackman(M)
    cumsum = np.cumsum(window[int(M/2):])
    cumsum = cumsum - cumsum[0]
    cumsum = cumsum / cumsum[-1]
    result = supp * np.argmin(np.abs(cumsum-arg/supp))/(M/2)
    return result

def drawrect(x, y, side0, side1):
    h0 = side0/2

    plt.fill([x - h0, x - h0, x + h0, x + h0, x - h0],
             [y - h0, y + h0, y + h0, y - h0, y - h0], 'k')


N = 128
x = np.zeros(N)
y = np.zeros_like(x)
r = np.zeros_like(x)
Rap = 1e3 * 4.72E-02/2
#Rap = 6
alpha = np.pi * (3 - np.sqrt(5))
Thetan_1 = (N - 1) * alpha

f = open('C:/f.txt', 'w')
small = 0.6
large = 0.6
for i in range(N):
    n = i + 1
    thetan = n * alpha
    rn = Rap * np.sqrt(thetan / Thetan_1)
    #rn = blackman(rn, Rap)
    x[i] = rn * np.cos(thetan)
    y[i] = rn * np.sin(thetan)
    r[i] = rn
    drawrect(x[i], y[i], .6, .6)
    f.write('0\t0.6\t0.6\t' + str(x[i]) + '\t' + str(y[i]) + '\t0\n')
f.close()

#plt.plot(x, y, 'sk')

