import matplotlib.pyplot as plt
import numpy as np
import time
from surface.surface import Surface, SurfaceType
from framework import file_m2k, file_mat, file_civa, file_omniscan
from scipy.signal import hilbert
from imaging import tfm, tfm3d
from framework.data_types import ImagingROI, ElementGeometry
from framework import file_m2k, file_civa, post_proc
import sys
import gc



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
    display(t_elapsed)
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
        env = post_proc.envelope(mySurf.data_insp.ascan_data[:, int(mySurf.numelements / 2),
                                  int(mySurf.numelements / 2), 0], -2)
        idx = int(np.argwhere(env < np.median(env))[0][0])
    else:
        idx = int((mySurf.data_insp.inspection_params.water_path / mySurf.VelocityMedium) * mySurf.SamplingFreq)
    for i_elem in range(mySurf.numelements):
        env = post_proc.envelope(mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0], -2)
        peak = idx + np.argmax(env)
        # plot:
        f, [a, b] = plt.subplots(2, 1)
        a.plot(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])))
        a.plot(mySurf.data_insp.ascan_data[:, i_elem, i_elem, 0])
        a.plot(post_proc.envelope(mySurf.data_insp.ascan_data[:, i_elem, i_elem, 0], -2))
        b.plot(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])))
        b.plot(post_proc.envelope(np.concatenate((np.zeros(idx), mySurf.data_insp.ascan_data[idx:, i_elem, i_elem, 0])),
                                  -2))
        # scatter:
        eixo_ver = [np.max(env)]
        eixo_hor = [peak]
        b.scatter(eixo_hor, eixo_ver, color="r", marker="o")
        plt.savefig('elemento ' + str(i_elem))
        plt.close(f)
    return


xdczerototal = -1

#data = file_civa.read('C:/Users/Thiago Passarin/Desktop/gate0_10deg_70MHz_cilinder.civa')
#data = file_civa.read('C:/Users/Thiago Passarin/Desktop/gate0_10deg_70MHz.civa')
#data = file_civa.read('J:/gate0_0deg_70MHz.civa')
#data = file_civa.read('J:/gate0_0deg_70MHz_spheres_shifted30.civa')
#data = file_civa.read('K:/Retangular interelem 0.1 pitch var.civa')
#data = file_civa.read('J:/gate0_0deg_70MHz_spheres_shifted.civa')
#data = file_civa.read('J:/gate0_10deg_70MHz.civa')
#data = file_civa.read('J:/gate0_10deg_70MHz_cilinder_3.civa')
#data.probe_params.elem_center[:,2] = -10.0

#data = file_civa.read('G:/Retangular interelem 0.1 pitch.civa')
#data = file_civa.read('G:/Dados 3D/RetangularRetangular.var/proc0/results/Config_[10]/model.civa')
#data = file_civa.read('J:/RetangularRetangularInteractionFrontCilinder11x8.civa')
data = file_civa.read('J:/model.civa')
#data = file_civa.read('G:/Dados 3D/Retangular Circular/RetangularCircularRadius_0.65.civa')
## Deixa as coordenadas dos elementos no mesmo formato dos transdutores lineares
#N_elem = data.probe_params.elem_center.shape[0]*data.probe_params.elem_center.shape[1]
#extended = np.zeros((N_elem, 3))
#extended[:,:-1] = data.probe_params.elem_center.reshape((N_elem, 2))
#data.probe_params.elem_center = extended
## Insere o número de elementos
#data.probe_params.num_elem = N_elem


#for y in range(0,1):#corner_roi = np.array([-12.1+4.3706/np.sqrt(2), 4.3706/np.sqrt(2), 5.0+12.6752])[np.newaxis, :]
y = -0.0
if True:
    roi_y = y+0.0
    corner_roi = np.array([-15.0, -6., 20.0])[np.newaxis, :]
    #roi = ImagingROI3D(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=1)
    roi = ImagingROI(corner_roi, height=15.0, width=30.0, h_len=40, w_len=60, depth=12.0, d_len=20)

    #corner_roi = np.array([-20.0, roi_y, 15.0])[np.newaxis, :]
    #roi = ImagingROI(corner_roi, height=10.0, width=40.0, h_len=100, w_len=64)e

    #mySurf = Surface(data, xdczerototal=xdczerototal, c_medium=1498, keep_data_insp=True, atfm_surface=False)

    t0 = time.time()

    data.ascan_data = hilbert(data.ascan_data[:, :, :, 0], axis=0)[:, :, :, np.newaxis].astype(np.complex64)
    chave = tfm3d.tfm3d_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl,
                               elem_geometry=ElementGeometry.RECTANGULAR, scattering_angle=20)
    #chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl)
    print('Elapsed time [s] '+(time.time()-t0).__str__())
    #np.save('img'+y.__str__(), data.imaging_results[chave].image)

    plt.figure()
    #plt.imshow(np.abs(data.imaging_results[chave].image))
    plt.imshow(np.abs(data.imaging_results[chave].image[:,0,:]), aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])

    result = np.array([data.imaging_results[chave].image, roi])

    np.save('result_tfm.npy', result)

    plt.show()
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('y='+roi_y.__str__()+'mm')
    #plt.savefig(y.__str__())

    print(roi_y)

    mySurf = Surface(data)