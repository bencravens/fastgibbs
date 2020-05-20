import numpy as np
from numpy import linalg as la
from sklearn.covariance import EmpiricalCovariance as Ecov

def conj_grad(A,k,err_tol):
    """perform conjugate gradient descent on Ax=b until 
    error tolerance is reached"""
    [m,n] = np.shape(A)
    #initialize state vector
    x = np.zeros([m,1])
    #initialize sample as well
    y = x
    #initialize residual
    r = b - np.dot(A,x)
    #initialize search direction
    p = r
    #normalizing constant
    d = np.dot(np.transpose(p),np.dot(A,p))
    #iterate k times (or until convergence)
    for i in range(k):
        gamma = (np.dot(np.transpose(r),r))/d
        x = x + np.dot(p,gamma)
        #sample from N(0,1)
        z = np.random.randn()
        #update sample 
        y = y + (z/np.sqrt(d))*p
        #store old r to calculate new beta
        r_old = r
        r = r - gamma*np.dot(A,p)
        #calculate new beta
        beta_num = -np.dot(np.transpose(r),r)
        beta_denom = np.dot(np.transpose(r_old),r_old)
        beta = beta_num/beta_denom
        #calculate new search direction
        p = r - beta*p
        #calculate new normalization constant
        d = np.dot(np.transpose(p),np.dot(A,p))
        if la.norm(r) < err_tol:
            return [x,y]
    return [x,y]

if __name__=="__main__":
    #A = np.asarray([[1,0.1,0.1],[0.1,1,0.1],[0.1,0.1,1]])
    A = np.loadtxt("res.txt",delimiter=',')
    print(np.shape(A))
    b = np.ones([3,1])
    k = 3
    err_tol = 2.22e-16
    sample_num = 50000
    samples = []
    for i in range(sample_num):
        [x,y] = conj_grad(A,b,k,err_tol)
        samples.append(y)
    samples = np.squeeze(samples,axis=2)
    e_cov = Ecov().fit(samples).covariance_
    real_cov = la.inv(A)
    print(e_cov)
    print(real_cov)
    print(la.norm(e_cov - real_cov)/(la.norm(real_cov)))

    """perform conjugate gradient descent on Ax=b until 
    error tolerance is reached"""
