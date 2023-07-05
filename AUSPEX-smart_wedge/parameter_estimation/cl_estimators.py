import cv2
from framework import post_proc
from framework.data_types import ImagingROI
import numpy as np
from scipy.signal import hilbert
import copy


def cl_estimator_tenenbaum(image):
    kernel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img1 = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
    img2 = cv2.filter2D(src=image, kernel=kernel.T, ddepth=-1)
    return np.sum(np.square(img1)) + np.sum(np.square(img2))


def cl_estimator_normalized_variance(image):
    if np.isclose(np.sum(image), 0, atol=1e-15):
        return 0
    mean = np.mean(image)
    return (1/mean) * np.sum(np.square(image-mean))


def cl_estimator_contrast(image):
    if np.isclose(image.max(), image.min(), atol=1e-15):
        return 0
    image = (image - image.min())/abs(image.max()-image.min())
    return np.mean(image**2)/np.mean(image)**2


def gs(data, roi, sel_shot, img_func, a, b, tol, metric_func):
    tol = int(abs(tol))
    if tol == 0:
        tol = 13
    if np.iscomplexobj(data.ascan_data):
        data2 = data
    else:
        data2 = copy.deepcopy(data)
        data2.ascan_data = hilbert(data.ascan_data[:, :, :, sel_shot], axis=0)[:, :, :, np.newaxis].astype(np.complex64)
        data2.ascan_data[-1, :, :, :] = 0
    cs = np.linspace(a, b, num=tol)
    k = np.zeros_like(cs, dtype=np.int32)
    met = np.zeros_like(cs)
    vals = dict()
    for i in range(len(cs)):
        k[i] = img_func(data2, roi, c=cs[i], sel_shot=sel_shot)
        img = np.abs(data2.imaging_results[k[i]].image)
        data.imaging_results[k[i]] = copy.deepcopy(data2.imaging_results[k[i]])
        met[i] = metric_func(img)
        vals[cs[i]] = [met[i], k[i]]
    c = cs[np.argmax(met)]
    data2 = None
    return c, vals


def gss(data, roi, metric_func, img_func, a, b, tol=1e-5):

    invphi = (np.sqrt(5) - 1) / 2
    invphi2 = (3 - np.sqrt(5)) / 2

    def func(cle):
        return metric_func(data, cle, img_func, roi)

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    [yc, kc] = func(c)
    [yd, kd] = func(d)

    vals = dict()
    vals[c] = [yc, kc]
    vals[d] = [yd, kd]

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            [yc, kc] = func(c)
            vals[c] = [yc, kc]

        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            [yd, kd] = func(d)
            vals[d] = [yd, kd]

    if yc < yd:
        return [(a+d)/2, vals]
    else:
        return [(c+b)/2, vals]
