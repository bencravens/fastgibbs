import numpy as np
from numpy import linalg as la
from sklearn.covariance import EmpiricalCovariance as Ecov
import time
from matplotlib import pyplot as plt

def conj_grad(A,b):
    """perform conjugate gradient descent on Ax=b until 
    error tolerance is reached"""
    [m,n] = np.shape(A)
    #initialize state vector
    x = np.zeros([m,1])
    #initialize residual
    r = b - np.dot(A,x)
    #initialize transpose of residual
    rsold = np.dot(np.transpose(r),r)
    #initialize search direction
    p = r
    #calculate A*p vector
    A_p = np.dot(A,p)
    #normalizing constant
    d = np.dot(np.transpose(p),np.dot(A,p))
    #iterate m times (or until convergence)
    for i in range(m):
        print("iteration {}/{}".format(i,m))
        gamma = (rsold)/d
        x = x + np.dot(p,gamma)
        #store old r to calculate new beta
        r = r - gamma*A_p
        rsnew = np.dot(np.transpose(r),r)
        #is the magnitude of the residual < 1e-16?
        if np.sqrt(rsnew) < 1e-8:
            print("converged at iteration {}/{}".format(i,m))
            break
        #calculate new beta
        beta = rsnew/rsold
        #update rsold constant
        rsold = rsnew
        #calculate new search direction
        p = r + beta*p
        #calculate A*p vector
        A_p = np.dot(A,p)
        #calculate new normalization constant
        d = np.dot(np.transpose(p),A_p)
    return x

def cg_plot(A,b):
    x_a = np.matmul(la.inv(A),b)
    x_cg = conj_grad(A,b)
    x_a = np.sort(np.ravel(x_a))
    x_cg = np.sort(np.ravel(x_cg))
    rel_err = la.norm(x_a - x_cg)/la.norm(x_a)
    plt.plot(x_a,label="analytic solution")
    plt.plot(x_cg,marker='1',linestyle='none',label='CG solution')
    plt.legend()
    plt.title("analytic solution and cg solution, rel err {}".format(rel_err))
    plt.show()

if __name__=="__main__":
    A = np.loadtxt("test_A.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.ones([dims,1])
    cg_plot(A,b)
    A = np.loadtxt("2d.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.ones([dims,1])
    cg_plot(A,b)
    A = np.loadtxt("3d.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.ones([dims,1])
    cg_plot(A,b)
