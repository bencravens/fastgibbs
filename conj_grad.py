import numpy as np
from numpy import linalg as la
from sklearn.covariance import EmpiricalCovariance as Ecov
import time

def conj_grad(A,b,k):
    """perform conjugate gradient descent on Ax=b until 
    error tolerance is reached"""
    [m,n] = np.shape(A)
    #initialize state vector
    x = np.zeros([m,1])
    #initialize sample as well
    y = x
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
    #random vectors
    z = np.random.randn(k)
    #iterate k times (or until convergence)
    for i in range(k):
        gamma = (rsold)/d
        x = x + np.dot(p,gamma)
        #update sample 
        y = y + (z[i]/np.sqrt(d))*p
        #store old r to calculate new beta
        r = r - gamma*A_p
        rsnew = np.dot(np.transpose(r),r)
        #is the magnitude of the residual < 1e-16?
        if np.sqrt(rsnew) < 1e-8:
            print("converged at iteration {}/{}".format(i,k))
            break
        #calculate new beta
        beta = -rsnew/rsold
        #update rsold constant
        rsold = rsnew
        #calculate new search direction
        p = r - beta*p
        #calculate A*p vector
        A_p = np.dot(A,p)
        #calculate new normalization constant
        d = np.dot(np.transpose(p),A_p)
    return [x,y]

if __name__=="__main__":
    A = np.loadtxt("dim.txt",delimiter=',')
    dims = np.shape(A)[0]
    print(np.shape(A))
    b = np.ones([dims,1])
    k = dims 
    sample_num = int(1e4)
    samples = np.asarray([conj_grad(A,b,k)[1] for i in range(sample_num)])
    samples = np.squeeze(samples,axis=2)
    print(np.shape(samples))
    e_cov = Ecov().fit(samples).covariance_
    real_cov = la.inv(A)
    print(e_cov)
    print(real_cov)
    print("relative error is {}".format(la.norm(e_cov - real_cov)/(la.norm(real_cov))))
    t_f = time.time()
    print("{} seconds elapsed".format(t_f - t_0))
