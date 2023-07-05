from surface.surface import Surface, SurfaceType
from framework import file_m2k, file_mat, file_civa, file_omniscan
import matplotlib.pyplot as plt
import numpy as np
import time
from framework.post_proc import envelope
from scipy.signal import hilbert
from framework.data_types import ImagingROI



def plotline(mySurf):
    hor_axis = np.array([-100, 100])
    ver_axis = hor_axis * mySurf.lineparam.a + mySurf.lineparam.b
    #hor_axis = [mySurf.xpivec[(0, 0)]-10, mySurf.xpivec[(mySurf.xpivec.shape[0] - 1, 0)]+10]
    #ver_axis = [mySurf.xpivec[(0, 0)]-10 * mySurf.lineparam_a + mySurf.lineparam_b,
    #            (mySurf.xpivec[(mySurf.xpivec.shape[0] - 1, 0)]) * mySurf.lineparam_a + mySurf.lineparam_b]
    plt.plot(np.asarray(hor_axis), -1*np.asarray(ver_axis))
    plt.legend(["Elements", "LS"], loc="upper right")
    #plt.plot(mySurf.x_discr, -1*mySurf.z_discr, 'o', markersize=2)
    #plt.axis('scaled')

def plotlinenewton(mySurf):
    hor_axis = np.array([-100, 100])
    ver_axis = hor_axis * mySurf.linenewtonparam.a + mySurf.linenewtonparam.b
    #hor_axis = [mySurf.xpivec[(0, 0)]-10, mySurf.xpivec[(mySurf.xpivec.shape[0] - 1, 0)]+10]
    #ver_axis = [mySurf.xpivec[(0, 0)]-10 * mySurf.lineparam_a + mySurf.lineparam_b,
    #            (mySurf.xpivec[(mySurf.xpivec.shape[0] - 1, 0)]) * mySurf.lineparam_a + mySurf.lineparam_b]
    plt.plot(np.asarray(hor_axis), -1*np.asarray(ver_axis))
    plt.legend(["Elements", "M-Newton"], loc="upper right")
    #plt.plot(mySurf.x_discr, -1*mySurf.z_discr, 'o', markersize=2)
    #plt.axis('scaled')

def plotcircle(mySurf):
    #a = -2.229
    #b = 80.6
    #r = 70.2
    a = mySurf.circleparam.x
    b = mySurf.circleparam.z
    r = mySurf.circleparam.r
    circle_x = list();
    circle_z = list();
    for i in range(0, 360):
        circle_x.append(a + r * np.cos(2 * np.pi * i / 360))
        circle_z.append(b + r * np.sin(2 * np.pi * i / 360))
    plt.plot(np.asarray(circle_x), -1*np.asarray(circle_z))
    plt.legend(["Elements", "MLS"], loc="upper right")
    #plt.plot(mySurf.x_discr, -1*mySurf.z_discr)
    #plt.axis('scaled')

def plotcirclenewton(mySurf):
    #a = -2.229
    #b = 80.6
    #r = 70.2
    a = mySurf.circlenewtonparam.x
    b = mySurf.circlenewtonparam.z
    r = mySurf.circlenewtonparam.r
    circle_x = list();
    circle_z = list();
    for i in range(0, 360):
        circle_x.append(a + r * np.cos(2 * np.pi * i / 360))
        circle_z.append(b + r * np.sin(2 * np.pi * i / 360))
    plt.plot(np.asarray(circle_x), -1*np.asarray(circle_z))
    plt.legend(["Elements", "M-Newton"], loc="upper right")
    #plt.plot(mySurf.x_discr, -1*mySurf.z_discr)
    #plt.axis('scaled')

    plt.plot([mySurf.circlenewtonparam.x, mySurf.circlenewtonparam.x + mySurf.circlenewtonparam.r],
             [-mySurf.circlenewtonparam.z, -mySurf.circlenewtonparam.z], 'orange')
    plt.scatter([mySurf.circlenewtonparam.x], [-mySurf.circlenewtonparam.z], marker='x', color='black')
    plt.text(mySurf.circlenewtonparam.x, 1 - mySurf.circlenewtonparam.z,
             '(' + np.str(np.around(mySurf.circlenewtonparam.x, decimals=2)) + ',' + np.str(
                 np.around(mySurf.circlenewtonparam.z, decimals=2)) + ')')
    plt.text(mySurf.circlenewtonparam.x + mySurf.circlenewtonparam.r / 2, 1 - mySurf.circlenewtonparam.z,
             'R=' + np.str(np.around(mySurf.circlenewtonparam.r, decimals=2)))
    plt.plot([mySurf.circlenewtonparam.x, mySurf.circlenewtonparam.x],
             [0, -mySurf.circlenewtonparam.z + mySurf.circlenewtonparam.r], 'orange')
    plt.text(mySurf.circlenewtonparam.x, (-mySurf.circlenewtonparam.z + mySurf.circlenewtonparam.r) / 2,
             'd=' + np.str(np.around(mySurf.circlenewtonparam.z - mySurf.circlenewtonparam.r, 2)))

def plotquadcircle(mySurf):
    #a = -2.229
    #b = 80.6
    #r = 70.2
    a = mySurf.circlequadparam.x
    b = mySurf.circlequadparam.z
    r = mySurf.circlequadparam.r
    circle_x = list();
    circle_z = list();
    for i in range(0, 360):
        circle_x.append(a + r * np.cos(2 * np.pi * i / 360))
        circle_z.append(b + r * np.sin(2 * np.pi * i / 360))
    plt.plot(np.asarray(circle_x), -1*np.asarray(circle_z))
    plt.legend(["Elements", "Quad+Newton"], loc="lower right")
    plt.plot([mySurf.circlequadparam.x, mySurf.circlequadparam.x + mySurf.circlequadparam.r],
             [-mySurf.circlequadparam.z, -mySurf.circlequadparam.z], 'orange')
    plt.scatter([mySurf.circlequadparam.x], [-mySurf.circlequadparam.z], marker='x', color='black')
    plt.text(mySurf.circlequadparam.x, 1 -mySurf.circlequadparam.z,
             '(' + np.str(np.around(mySurf.circlequadparam.x, decimals=2)) + ',' + np.str(
                 np.around(mySurf.circlequadparam.z, decimals=2)) + ')')
    plt.text(mySurf.circlequadparam.x+mySurf.circlequadparam.r/2, 1 -mySurf.circlequadparam.z,
             'R=' + np.str(np.around(mySurf.circlequadparam.r, decimals=2)))
    plt.plot([mySurf.circlequadparam.x, mySurf.circlequadparam.x],
             [0, -mySurf.circlequadparam.z + mySurf.circlequadparam.r], 'orange')
    plt.text(mySurf.circlequadparam.x, (-mySurf.circlequadparam.z + mySurf.circlequadparam.r) / 2,
             'd=' + np.str(np.around(mySurf.circlequadparam.z - mySurf.circlequadparam.r, 2)))
    #plt.plot(mySurf.x_discr, -1*mySurf.z_discr)
    #plt.axis('scaled')


def plotexamplepoint(mySurf, xf=0, zf=0):
    t_init = time.time()
    if xf==0 and zf==0:
        xf = mySurf.elementpos[0, 0] + (mySurf.elementpos[-1, 0] - mySurf.elementpos[0, 0]) * np.random.rand()
        zf = 10 + 30 * np.random.rand()
    [xxx, zzz] = mySurf.newtonraphsonbatchfocus(xf, zf)
    t_elapsed = time.time() - t_init
    display(t_elapsed)  # display não existe
    for i in range(0, xxx.shape[0]):
        plt.plot([mySurf.elementpos[i, 0], xxx[i]], -1*[mySurf.elementpos[i, 2], zzz[i]], 'g')
        plt.plot([xxx[i], xf], -1*[zzz[i], zf], 'g')

def plotexampleelement(mySurf):
    xROI = np.arange(55)+15
    zROI = np.zeros(55) + 100
    xa,za = mySurf.elementpos[10, 0], mySurf.elementpos[10, 2]
    [xxx,zzz] = mySurf.newtonraphsonbatchelement(xa, za ,xROI,zROI)
    valores = np.zeros(xxx.shape[0])
    for i in range(0, xxx.shape[0]):
        plt.plot([xa,xxx[i]], -1*[za,zzz[i]], 'g')
        plt.plot([xxx[i],xROI[i]], -1*[zzz[i],zROI[i]], 'g')
        angleMedium = np.arctan((za-zzz[i])/(xa-xxx[i])) - np.pi/2
        angleMaterial = np.arctan((zzz[i]-zROI[i])/(xxx[i]-xROI[i])) - np.pi/2
        valores[i] = (np.sin(angleMaterial)/np.sin(angleMedium)) / (mySurf.VelocityMaterial/mySurf.VelocityMedium)
        #display(np.sin(angleMaterial)/np.sin(angleMedium))
    plt.axis('scaled')
    return np.array([xxx,zzz])


def plotfindpoints(mySurf):
    #num_points = mySurf.numelements - 1
    if mySurf.data_insp.inspection_params.water_path == 0:
        env = envelope(mySurf.data_insp.ascan_data[:, int(mySurf.numelements / 2),  # alterado para envelope
                                  int(mySurf.numelements / 2), 0], -2)
        idx = int(np.argwhere(env < np.median(env))[0][0])
    else:
        idx = int((mySurf.data_insp.inspection_params.water_path / mySurf.VelocityMedium) * mySurf.SamplingFreq)
    for i_elem in range(mySurf.numelements):
        env = envelope(mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0], -2)
        peak = idx + np.argmax(env)
        # plot:
        f, [a, b] = plt.subplots(2, 1)
        a.plot(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])))
        a.plot(mySurf.data_insp.ascan_data[:, i_elem, i_elem, 0])
        a.plot(envelope(mySurf.data_insp.ascan_data[:, i_elem, i_elem, 0], -2))
        b.plot(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])))
        b.plot(envelope(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])), -2))
        # scatter:
        eixo_ver = [np.max(env)]
        eixo_hor = [peak]
        b.scatter(eixo_hor, eixo_ver, color="r", marker="o")
        plt.savefig('elemento ' + str(i_elem))
        plt.close(f)
    return


xdczerototal = -1

data = file_civa.read('C:/Users/Thiago Passarin/Desktop/block_sdh_immersion_close.civa')
corner_roi = np.array([-20.0, 0.0, 0.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=65.0, width=40.0, h_len=100, w_len=64)

mySurf = Surface(data, xdczerototal, c_medium=1498, keep_data_insp=True)
myFig = plt.figure()
plt.scatter(mySurf.xpivec, -1*mySurf.zpivec)
#for i in range(0, mySurf.xpivec.shape[0]):
#    plt.text(mySurf.xpivec[i], -1 * mySurf.zpivec[i], str(i))
plt.plot(mySurf.elementpos[:, 0], -1*mySurf.elementpos[:, 2], 'sk', markersize=5)

if mySurf.surfacetype == SurfaceType.CIRCLE_MLS:
    plotcircle(mySurf)
elif mySurf.surfacetype == SurfaceType.CIRCLE_QUAD:
    plotquadcircle(mySurf)
elif mySurf.surfacetype == SurfaceType.LINE_LS:
    plotline(mySurf)
elif mySurf.surfacetype == SurfaceType.LINE_OLS:
    plotline(mySurf)
elif mySurf.surfacetype == SurfaceType.LINE_NEWTON:
    plotlinenewton(mySurf)
elif mySurf.surfacetype == SurfaceType.CIRCLE_NEWTON:
    plotcirclenewton(mySurf)

plt.axis('scaled')

#[xxx,zzz] = plotexampleelement(mySurf)


