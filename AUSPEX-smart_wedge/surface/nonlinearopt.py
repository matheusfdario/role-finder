# -*- coding: utf-8 -*-
"""
Módulo ``nonlinearopt``
=======================



TODO

.. raw:: html

    <hr>

"""

import numpy as np
import matplotlib.pyplot as plt


class NewtonMultivariateResult:
    theta_final = None
    theta_values = None
    funcs_values = None
    norm_values = None
    iterations = None


def partialderiv(func, theta, whichvar):
    N_vars = theta.shape[0]
    eps = np.float64(0.0001)
    posterior_param = np.copy(theta.reshape(N_vars))
    posterior_param[whichvar] = posterior_param[whichvar] + eps
    posterior = func(posterior_param)
    previous_param = np.copy(theta.reshape(N_vars))
    previous_param[whichvar] = previous_param[whichvar] - eps
    previous = func(previous_param)
    derivative = (posterior - previous) / (2*eps)
    return derivative


def funcs_value(funcs, theta):
    N_vars = theta.shape[0]
    f = np.zeros((N_vars, 1))
    for i in range(N_vars):
        f[i,0] = funcs[i](theta)
    return f


def jacobian(funcs, theta):
    N_vars = theta.shape[0]
    J = np.zeros((N_vars, N_vars))
    for m_func in range(N_vars):
        for n_var in range(N_vars):
            J[m_func, n_var] = partialderiv(funcs[m_func], theta, n_var)
    return J


def newtonsearch(costfun, theta_init):
    # Reference:
    # http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html

    def __func0(theta):
        return partialderiv(costfun, theta, 0)

    def __func1(theta):
        return partialderiv(costfun, theta, 1)

    def __func2(theta):
        return partialderiv(costfun, theta, 2)

    def __func3(theta):
        return partialderiv(costfun, theta, 3)

    def __func4(theta):
        return partialderiv(costfun, theta, 4)

    funcs = np.array([__func0, __func1, __func2,
                      __func3, __func4])

    N_vars = theta_init.shape[0]
    theta = np.copy(theta_init)
    maxit = 200
    theta_values = np.zeros((N_vars,maxit))
    funcs_values = np.zeros((N_vars,maxit))
    norms_values = np.ones(maxit)*1e6
    det_values = np.ones(maxit)*1e6
    cond_values = np.zeros(maxit)
    cost_values = np.ones(maxit)*1e6
    for i in range(maxit):
        cost_values[i] = costfun(theta)
        J = jacobian(funcs, theta)
        det_values[i] = np.linalg.det(J)
        cond_values[i] = np.linalg.cond(J)
        np.linalg.cond(J)
        Jinv = np.linalg.pinv(J)
        f = funcs_value(funcs, theta)
        funcs_values[:, i] = f.reshape(N_vars)
        theta[0:N_vars] = theta[0:N_vars] - np.dot(Jinv, f)
        theta_values[:,i] = theta.reshape(N_vars)
        norm_f = np.linalg.norm(f)
        norms_values[i] = norm_f
        if norm_f < 1e-10:
            break

    result = NewtonMultivariateResult
    result.cost_values = cost_values
    result.det_values = det_values
    result.cond_values = cond_values
    result.theta_values = theta_values
    result.theta_final = theta
    result.funcs_values = funcs_values
    result.norm_values = norms_values
    result.iterations = i

    #plt.figure()
    #plt.plot(norms_values[:i+1])
    #plt.legend(['Norms'])

    return result


