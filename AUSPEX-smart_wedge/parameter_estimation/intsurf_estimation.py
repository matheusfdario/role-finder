import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator as lo
from scipy.sparse import diags
from framework.post_proc import envelope

# Profile - Reference
def specimen(roi_0, width, shots):
    pos_0 = 497 + 10 * roi_0
    step = 50
    inc1 = 10 / 37.32
    inc2 = 10 / 5.77
    inc3 = 10 / 10
    inc4 = 10 / 17.32
    axis_y = np.zeros(2000)
    axis_y[100:473] = inc1 * np.arange(0, 373)
    axis_y[473:846] = inc1 * (373 - np.arange(0, 373))
    axis_y[946:1004] = inc2 * np.arange(0, 58)
    axis_y[1004:1062] = inc2 * (58 - np.arange(0, 58))
    axis_y[1162:1262] = inc3 * np.arange(0, 100)
    axis_y[1262:1362] = inc3 * (100 - np.arange(0, 100))
    axis_y[1462:1635] = inc4 * np.arange(0, 173)
    axis_y[1635:1808] = inc4 * (173 - np.arange(0, 173))
    x_inf = np.int(pos_0 + step * (shots[0] - 1))
    x_sup = np.int(pos_0 + 10 * width + step * (shots[-1] - 1))
    ref = (200 - axis_y[x_inf:x_sup])  # ref = (200-axis_y[x_inf:x_sup])
    return ref, x_inf, x_sup


# D matrix - first order
def matrix_d1(z, boundary='mirrored'):
    assert len(z.shape) == 1
    n = z.size + 1
    d = np.zeros(n)
    d[0] = -1
    d[1] = 1
    d = diags(d, np.arange(n), shape=(n - 1, n)).toarray()
    d = d[:, 1:]
    if boundary == 'circular':
        d[0, -1] = d[-1, 0] = 1
    elif boundary == 'mirrored':
        d[0, 0] = 1
        d[-1, -1] = 1
    return d


# D matrix - second order
def matrix_d2(z, boundary='mirrored'):
    assert len(z.shape) == 1
    n = z.size + 1
    d = np.zeros(n)
    d[0] = 1
    d[1] = -2
    d[2] = 1
    d = diags(d, np.arange(n), shape=(n - 1, n)).toarray()
    d = d[:, 1:]
    if boundary == 'circular':
        d[0, -1] = d[-1, 0] = 1
    elif boundary == 'mirrored':
        d[0, 0] += 1
        d[-1, -1] += 1
    return d


# D matrix - second order
def matrix_d3(z, boundary='mirrored'):
    assert len(z.shape) == 1
    n = z.size + 1
    d = np.zeros(n)
    d[0] = -1
    d[1] = 3
    d[2] = -3
    d[3] = 1
    d = diags(d, np.arange(n), shape=(n - 1, n)).toarray()
    d = d[:, 1:]
    if boundary == 'circular':
        d[0, -1] = d[-1, 0] = 1
    elif boundary == 'mirrored':
        d[0, 0] += 1
        d[-1, -1] += 1
    return d


def diff2_dir(x):
    return x[:-2] - 2 * x[1:-1] + x[2:]


def diff2_adj(y):
    xp = np.zeros(len(y) + 2)
    xp[0] = y[0]
    xp[1] = y[1] - 2 * y[0]
    xp[-1] = y[-1]
    xp[-2] = y[-2] - 2 * y[-1]
    xp[2:-2] = y[:-2] - 2 * y[1:-1] + y[2:]
    return xp


def shrink(z, tau):
    return np.maximum(np.abs(z) - tau, 0) * np.sign(z)


# Algorithm: Group Sparse Total Variation denoising by Selesnick2013
# An Majorization-Minimization Algorithm for Group-Sparse -TV denoising
# solve x' = min 1/2 || z-x||_2^2 + lambda*phi(Dx)
def gstv(z, w, k, d, lamb, eps=1e-15, itmax=150, tol=1e-6):
    res = np.zeros(itmax)
    n = z.size + 1
    x = np.transpose(z)
    dt = np.transpose(d)
    b = dt @ np.transpose(z)
    it = itmax
    for i in range(itmax):
        a = np.zeros((n - 1, n - 1))
        u = d @ x
        for m in range(n - 1):
            for j in range(k - 1):
                for k in range(k - 1):
                    if (m - j + k) < n - 1:
                        a[m, m] = a[m, m] + np.sqrt(abs(u[m - j + k]) ** 2)
        a_inv = np.linalg.inv(a + eps * np.eye(n - 1))
        f = (1 / lamb) * a_inv + dt @ d
        f_inv = np.linalg.inv(f)
        x = z - dt @ (f_inv @ b)
        res[i] = np.linalg.norm(w @ (x - z)) ** 2 + lamb * np.linalg.norm(d @ x, 1)
        if i > 1 and abs(res[i] - res[i - 1]) < tol:
            res[i:] = res[i]
            it = i
            break
    return x, res, it


# Algorithm: Group Sparse Total Variation denoising by Selesnick2013
# An Majorization-Minimization Algorithm for Group-Sparse -TV denoising
# solve x' = min 1/2 ||W(z-x)||_2^2 + lambda*phi(Dx)
def gstv_w(z, w, k, d, lamb, eps=1e-15, itmax=150, tol=1e-6):
    res = np.zeros(itmax)
    n = z.size + 1
    x = np.transpose(z)
    b = d.T @ np.transpose(z)
    it = itmax
    for i in range(itmax):
        a = np.zeros((n - 1, n - 1))
        u = d @ x
        for m in range(n - 1):
            for j in range(k - 1):
                for k in range(k - 1):
                    if (m - j + k) < n - 1:
                        a[m, m] = a[m, m] + np.sqrt(abs(u[m - j + k]) ** 2)
        a_inv = np.linalg.inv(a + eps * np.eye(n - 1))
        wtwi = np.linalg.inv(w.T @ w)
        f = (2 / lamb) * a_inv + d @ wtwi @ d.T
        f_inv = np.linalg.inv(f)
        x = z - wtwi @ d.T @ (f_inv @ b)
        res[i] = lamb * np.linalg.norm(w @ (x - z)) ** 2 + np.linalg.norm(d @ x, 1)
        if i > 1 and abs(res[i] - res[i - 1]) < tol:
            res[i:] = res[i]
            it = i
            break
    return x, res, it


def profile_irls(w, b, lamb, eps=0.07e-2, itmax=150, tol=1e-6, boundary='mirrored', x0=None):
    # solve for ||W(x - b)||^2_2 + lambda*||D^2x||_1 with IRLS using INV
    assert len(b.shape) == 1
    n = b.size + 1
    d = np.zeros(n)
    d[0] = 1
    d[1] = -2
    d[2] = 1
    d = diags(d, np.arange(n), shape=(n - 1, n)).toarray()
    d = d[:, 1:]
    if boundary == 'circular':
        d[0, -1] = d[-1, 0] = 1
    elif boundary == 'mirrored':
        d[0, 0] += 1
        d[-1, -1] += 1
    if x0 is None:
        x = w @ b
    else:
        x = x0
    res = np.zeros(itmax)
    # reg = np.zeros(itmax)
    wtw = w.T @ w
    wtwb = wtw @ b
    it = itmax
    for i in range(itmax):
        p = np.diag(1 / np.sqrt(np.maximum(eps, abs(d @ x))))
        ptp = p.T @ p
        x = np.linalg.inv(wtw + lamb * d.T @ p.T @ p @ d) @ wtwb
        res[i] = np.linalg.norm(w @ (x - b)) ** 2 + lamb * np.linalg.norm(d @ x, 1)
        # reg[i] = lamb * np.linalg.norm(d @ x, 1)
        if i > 2 and abs(res[i] - res[i - 1]) < tol:
            res[i:] = res[i]
            it = i
            break
    return x, res, it


def profile_irls_cg(w, b, lamb, eps=0.07e-2, itmax=150, tol=1e-6, boundary='mirrored'):
    # solve for ||W(x - b)||^2_2 + lambda*||D^2x||_1 with IRLS using CG
    assert len(b.shape) == 1
    n = len(b)
    x = w @ b
    res = np.zeros(itmax)
    # reg = np.zeros(itmax)
    wtw = w.T @ w
    wtwb = wtw @ b
    it = itmax
    for i in range(itmax):
        p = np.diag(1 / np.sqrt(np.maximum(eps, abs(diff2_dir(x)))))
        A = lambda x: wtw @ x + lamb * diff2_adj((p.T @ p @ diff2_dir(x)))  # wtw+lamb*dt@p.T@p@d
        A = lo((n, n), matvec=A)
        x, _ = cg(A, wtwb, x0=x, maxiter=500)
        # Versão antiga
        # p = np.diag(1 / np.sqrt(np.maximum(eps, abs(d @ x))))
        # x, it[i] = cg(wtw + lamb * d.T @ p.T @ p @ d, wtwb, x0=x, maxiter=500)
        # res[i] = np.linalg.norm(w @ (x - b)) ** 2 + lamb * np.linalg.norm(d @ x, 1)
        # # reg[i] = lamb * np.linalg.norm(d @ x, 1)
        res[i] = np.linalg.norm(w @ (x - b)) ** 2 + lamb * np.linalg.norm(diff2_dir(x), 1)
        # reg[i] = lamb * np.linalg.norm(d @ x, 1)
        if i > 2 and abs(res[i] - res[i - 1]) < tol:
            res[i:] = res[i]
            it = i
            break
    return x, res, it


def profile_admm(w, b, lamb, rho, itmax=150, tol=1e-3, boundary='mirrored'):
    # solve for ||W(x - b)||^2_2 + lambda*||D^2x||_1 with ADMM
    d = matrix_d2(b, boundary)
    res = np.zeros(itmax)
    # reg = np.zeros(itmax)
    wtw = w @ w
    wtwb = wtw @ b
    dtd = d @ d
    xk = np.zeros_like(b)
    zk = np.zeros_like(b)
    uk = np.zeros_like(b)
    filt = np.linalg.inv(wtw + rho * dtd)
    filtwtwb = filt @ wtwb

    it = itmax
    for i in range(itmax):
        xk1 = filtwtwb + rho * filt @ d @ (zk - uk)
        tmp = uk + d @ xk
        zk = shrink(tmp, lamb / rho)
        uk = tmp - zk
        res[i] = np.linalg.norm(w @ (xk1 - b)) ** 2 + lamb * np.linalg.norm(d @ xk1, 1)
        if i > 1 and abs(res[i - 1] - res[i]) < tol:
            res[i:] = res[i]
            it = i
            break
        # if i > 1 and np.linalg.norm(xk1-xk)/np.linalg.norm(xk) < tol:
        #     it = i
        #     res[i:] = res[i]
        #     break
        xk = xk1
    return xk, res, it


def profile_fadmm(w, b, lamb, rho, x0=None, eta=0.999, itmax=300, tol=1e-6, boundary='mirrored', max_iter_cg=200):
    # def profile_fadmm(w, b, lamb, rho, x0=None, eta=0.999, itmax=150, tol=1e-3, boundary='mirrored'):

    if len(w) > 950:
        # return profile_fadmm_cg(w, b, lamb, rho, x0, eta, itmax, tol, max_iter_cg)
        # w deve ser um vetor
        return profile_fadmm_cg(w.diagonal(), b, lamb, rho, x0, eta, itmax, tol, max_iter_cg)
    else:
        # return profile_fadmm_inv(np.diag(w.ravel()), b, lamb, rho, x0, eta, itmax, tol)
        # w dever ser uma matriz
        return profile_fadmm_inv(w, b, lamb, rho, x0, eta, itmax, tol)

def profile_fadmm_inv(w, b, lamb, rho, x0=None, eta=0.999, itmax=150, tol=1e-3, boundary='mirrored'):
    # def profile_fadmm(w, b, lamb, rho, x0=None, eta=0.999, itmax=150, tol=1e-3, boundary='mirrored'):
    # solve for ||W(x - b)||^2_2 + lambda*||D^2x||_1 with ADMM
    d = matrix_d2(b, boundary)
    res = np.zeros(itmax)
    # reg = np.zeros(itmax)
    wtw = w @ w
    wtwb = wtw @ b
    dtd = d @ d
    if x0 is None:
        xk = np.zeros_like(b)
    else:
        xk = x0
    zk = np.zeros_like(b)
    zk_hat = np.zeros_like(b)
    yk = np.zeros_like(b)
    yk_hat = np.zeros_like(b)
    filt = np.linalg.inv(wtw + rho * dtd)
    filtwtwb = filt @ wtwb
    ak = 1
    ck = 1e999
    it = itmax
    prim = np.zeros_like(res)
    sec = np.zeros_like(res)
    for i in range(itmax):
        zkm = zk
        ykm = yk
        ckm = ck
        xk = filtwtwb + filt @ d @ (rho * zk_hat + yk_hat)
        zk = shrink(d.T @ xk - yk_hat / rho, lamb / rho)
        yk = yk_hat + rho * (zk - d.T @ xk)
        ck = (1 / rho) * np.linalg.norm(yk - yk_hat) ** 2 + rho * np.linalg.norm(zk_hat - zk) ** 2
        if ck < eta * ckm:
            ak1 = (1 + np.sqrt(1 + 4 * ak ** 2)) / 2
            zk_hat = zk + (ak - 1) / ak1 * (zk - zkm)
            yk_hat = yk + (ak - 1) / ak1 * (yk - ykm)
        else:
            ak1 = 1
            zk_hat = zkm
            yk_hat = ykm
            ck = (1 / eta) * ckm
        ak = ak1
        prim = np.linalg.norm(w @ (xk - b)) ** 2
        sec = np.linalg.norm(d @ xk, 1)
        res[i] = prim + lamb * sec
        if i > 1 and abs(res[i - 1] - res[i]) < tol:
            res[i:] = res[i]
            it = i
            break

    return xk, res, it, prim, sec


def profile_fadmm_cg(w, b, lamb, rho, x0=None, eta=0.999, itmax=150, tol=1e-9, max_iter_cg=200):
    # solve for ||W(x - b)||^2_2 + lambda*||D^2x||_1 with ADMM
    res = np.zeros(itmax)
    # reg = np.zeros(itmax)
    wtw = w ** 2
    wtwb = wtw * b
    if x0 is None:
        xk = np.zeros_like(b)
    else:
        xk = x0
    zk = np.zeros(len(b) - 2)
    zk_hat = np.zeros_like(zk)
    yk = np.zeros_like(zk)
    yk_hat = np.zeros_like(zk)
    filt = lambda x: wtw * x + rho * diff2_adj(diff2_dir(x))
    A = lo((len(b), len(b)), matvec=filt)
    ak = 1
    ck = 1e999
    it = itmax
    prim = np.zeros_like(res)
    sec = np.zeros_like(res)
    for i in range(itmax):
        zkm = zk
        ykm = yk
        ckm = ck
        xk, _ = cg(A, (wtwb + diff2_adj(rho * zk_hat + yk_hat)), x0=xk, tol=1e-9, maxiter=max_iter_cg)
        zk = shrink(diff2_dir(xk) - yk_hat / rho, lamb / rho)
        yk = yk_hat + rho * (zk - diff2_dir(xk))
        ck = (1 / rho) * np.linalg.norm(yk - yk_hat) ** 2 + rho * np.linalg.norm(zk_hat - zk) ** 2
        if ck < eta * ckm:
            ak1 = (1 + np.sqrt(1 + 4 * ak ** 2)) / 2
            zk_hat = zk + (ak - 1) / ak1 * (zk - zkm)
            yk_hat = yk + (ak - 1) / ak1 * (yk - ykm)
        else:
            ak1 = 1
            zk_hat = zkm
            yk_hat = ykm
            ck = (1 / eta) * ckm
        ak = ak1
        prim = np.linalg.norm(w * (xk - b)) ** 2
        sec = np.linalg.norm(diff2_dir(xk), 1)
        res[i] = prim + lamb * sec
        if i > 1 and abs(res[i - 1] - res[i]) < tol:
            res = res[:i + 1]
            it = i
            break

    return xk, res, it, prim, sec


def profile_fama(w, b, lamb, tau, itmax=150, tol=1e-3, boundary='mirrored'):
    # novo
    d = matrix_d2(b, boundary)
    res = np.zeros(itmax)
    wtw = w @ w
    wtwb = wtw @ b
    dtd = d @ d
    xk = np.zeros_like(b)
    zk = np.zeros_like(b)
    yk = np.zeros_like(b)
    yk_hat = np.zeros_like(b)
    ykm = np.zeros_like(b)
    res = np.zeros(itmax)
    filt = np.linalg.inv(wtw)
    it = itmax
    ak = 1
    for i in range(itmax):
        xk = b - lamb * filt @ d.T @ yk_hat
        zk = shrink(d @ xk + tau * yk_hat, 1 / tau)
        yk = yk_hat + tau * (zk - d @ xk)
        ak1 = (1 + np.sqrt(1 + 4 * ak ** 2)) / 2
        yk_hat = yk + (ak - 1) / ak1 * (yk - ykm)
        res[i] = np.linalg.norm(w @ (xk - b)) ** 2 + lamb * np.linalg.norm(d @ xk, 1)
        if (i > 1) and (np.abs(res[i] - res[i - 1]) < tol):
            res[i:] = res[i]
            it = i
            break

    return xk, res, it


def img_line(image):
    aux = np.argmax(image, 0).astype(np.int32)
    w = np.max(image, 0)
    # a = np.asarray([aux, w])
    return aux, w


def mov_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def surf_est(data, key, lamb=1, rho=100, eta=0.999, itmax=500, tol=1e-6, max_iter_cg=1500):
    image = envelope(data.imaging_results[key].image)
    image /= image.max()
    roi = data.imaging_results[key].roi
    a = img_line(image)
    z0 = roi.h_points[a[0]]
    z, _, _, _, _ = profile_fadmm(a[1], z0, lamb=lamb, rho=rho, eta=eta, itmax=itmax, tol=tol, max_iter_cg=max_iter_cg)
    return [(roi.w_points[i], z[i], 0) for i in range(len(z))]
