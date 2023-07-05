import surface.nonlinearopt as nlo
import numpy as np

class foo:
    center = None
    def __init__(self, center):
        self.center = center
        return None

    def bar(self, theta):
        norma = np.linalg.norm(np.subtract(self.center.reshape(3), theta.reshape(3)))**2
        #norma = np.linalg.norm(theta)**3
        return norma

meufoo = foo(np.array([[4.0],[8.0],[6.0]]))
mysolver = nlo.NewtonMultivariate(meufoo.bar, 3)
myresult = mysolver.newtonsearch(np.array([[3.0],[5.0],[6.0]]))
theta_final = myresult.theta_final



#def bar(theta):
#    center = np.array([[4.0],[8.0],[6.0]])
#    norma = np.linalg.norm(np.subtract(center.reshape(3), theta.reshape(3)))**2
#    #norma = np.linalg.norm(theta)**3
#    return norma
#
#mysolver = nlo.NewtonMultivariate(bar, 3)
#theta_init = np.array([[3.0],[5.0],[6.0]])
#myresult = mysolver.newtonsearch(theta_init)
#theta_final = myresult.theta_final



#def errototal(theta_arg):
#    theta = np.copy(theta_arg.reshape(np.max(theta_arg.shape)))
#    SSE = 0
#    for i_amostra in range(values_x.shape[0]):
#        x = values_x[i_amostra]
#        y = values_y[i_amostra]
#        SE_i = (modelo_theta(theta, x) - y)**2
#        SSE = SSE + SE_i
#    return SSE
#
#
#def modelo_theta(theta, x):
#    # y = θ_0*x^2 + θ_1*x + θ_2
#    return theta[0]*theta[0] * x * x + theta[1] * x + theta[2]
#
#theta_original = np.array([2., 1., 0.5])
#values_x = np.array([1.0, 2.0, 6.0, 5.0])
#values_y = modelo_theta(theta_original, values_x)
#
#mySolver = nlo.NewtonMultivariate(errototal, 3)
#
#theta_init = x = np.array([[1.5],[2.5],[-2.5]])
#result = mySolver.newtonsearch(theta_init)
#theta_final = result.theta_final