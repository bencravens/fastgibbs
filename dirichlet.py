from gibbs_cheby import gibbs_cheby
from matplotlib import pyplot as plt
import numpy as np
from math import pi as pi
from math import sin as sin
from math import cos as cos
#generate eigenvalues for the discrete laplacian operator on a uniform grid

def w_k(k,n):
    return (pi*k)/(2*n+2)

def dir_eval(k,n):
    #nxn is the size of the discrete laplacian matrix...
    #k is the kth eigenvalue (k is from [1,n])
    w = w_k(k,n)
    return (4)*(sin(3*w)**2 + sin(w)**2)

def dir_generate(n):
    evals = [dir_eval(j,n) for j in range(1,n+1)]
    evals = np.sort(evals)
    plt.plot(np.sort(evals),label='dirichlet formula')

if __name__=="__main__":
    #run chebyshev sampler first...
    test_A = np.loadtxt("2d.txt",delimiter=',')
    err_tol = 1.5e-2
    my_gibbs = gibbs_cheby(1.0,test_A,err_tol)
    my_gibbs.sample(False)
    #now generate eigenvalues from formula
    #dir_generate(16)
    #now plot
    eigs = np.loadtxt("eigs.txt",delimiter=',')
    plt.plot(np.sort(eigs),label='dirichelt formula',marker='o')
    my_gibbs.espectrum()
    plt.legend()
    plt.show()
