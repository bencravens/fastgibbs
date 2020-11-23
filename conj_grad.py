import copy
import numpy as np
from numpy import linalg as la
from sklearn.covariance import EmpiricalCovariance as Ecov
import time
from matplotlib import pyplot as plt

def conj_grad(A,b):
    """perform conjugate gradient descent on Ax=b until 
    error tolerance is reached"""
    [m,n] = np.shape(A)
    #initialize state vectors
    x = np.zeros([m,1])
    y = copy.copy(x)
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

    #########################################
    # STORE P and d for covariance analysis #
    #########################################

    P_matrix = np.array([])
    d_vector = np.array([])

    #iterate m times (or until convergence)
    for i in range(m):
        print("iteration {}/{}".format(i,m))
        gamma = (rsold)/d
        x = x + np.dot(p,gamma)
        z = np.random.randn()
        y = y + (z/np.sqrt(d)) * p
        #store old r to calculate new beta
        r = r - gamma*A_p
        rsnew = np.dot(np.transpose(r),r)
        #is the magnitude of the residual < 1e-16?
        if np.sqrt(rsnew) < 1e-8:
            print("converged at iteration {}/{}".format(i,m))
        #calculate new beta
        beta = rsnew/rsold
        #update rsold constant
        rsold = rsnew
        #store P matrix and d vector entry
        if i==0:
            P_matrix = p
        else:
            P_matrix = np.append(P_matrix,p,axis=1)
        print("p is {}, p_matrix is {}".format(p,P_matrix))
        d_vector = np.append(d_vector,d)
        print("d_vector is {}".format(d_vector))
        #calculate new search direction
        p = r + beta*p
        #calculate A*p vector
        A_p = np.dot(A,p)
        #calculate new normalization constant
        d = np.dot(np.transpose(p),A_p)
    
    #########################################
    # make analytic covariance matrix       #
    #########################################
 
    d_matrix = np.diag(np.transpose(d_vector))
    print("shape of P is {}, shape of D is {}".format(np.shape(P_matrix),np.shape(d_matrix)))
    CG_cov = np.matmul(P_matrix, np.matmul(la.inv(d_matrix),np.transpose(P_matrix)))
    cov = la.inv(A)
    rel_err = la.norm(cov - CG_cov)/la.norm(cov)
    print("relative error in analytical covariance matrix approximation is {}".format(rel_err))
    return x,y,CG_cov

def cg_plot(A,b):
    
    #########################################
    # solver solution plot                  #
    #########################################
    cov = la.inv(A) 
    x_a = np.matmul(cov,b)
    x_cg,y,CG_cov = conj_grad(A,b)
    x_a = np.sort(np.ravel(x_a))
    x_cg = np.sort(np.ravel(x_cg))
    rel_err = la.norm(x_a - x_cg)/la.norm(x_a)
    plt.plot(x_a,label="analytic solution")
    plt.plot(x_cg,marker='1',linestyle='none',label='CG solution')
    plt.legend()
    plt.title("Analytic solution and cg solution for Ax=b, rel err {}".format(rel_err))
    plt.show()
    
    #########################################
    #    COVARIANCE EIGENVALUE PLOT         #
    #########################################
    rel_err = la.norm(cov - CG_cov)/la.norm(cov)
    eigs, eigvecs = la.eigh(cov)
    plt.semilogy(eigs,label="actual eigenvalues")
    eigs_CG, eigvecs_CG = la.eigh(CG_cov)
    #masking invalid CG eigenvalues
    eigs_CG = np.ma.masked_where(eigs_CG<1e-5,eigs_CG)
    plt.semilogy(eigs_CG,marker='1',linestyle='none',label='CG empirical eigenvalues')
    plt.legend()
    plt.title("CG sampler cov evals vs actual evals. Relative error in cov matrix approximation {}".format(rel_err))
    plt.show()

if __name__=="__main__":
    A = np.loadtxt("test_A.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.random.randn(dims,1)
    cg_plot(A,b)
    A = np.loadtxt("2d.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.random.randn(dims,1)
    cg_plot(A,b)
    A = np.loadtxt("3d.txt",delimiter=',')
    [dims,dims] = np.shape(A)
    b = np.random.randn(dims,1)
    cg_plot(A,b)
