
# coding: utf-8

# In[1]:


#make sure we have the right packages again
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import scipy.sparse as sp
import scipy
from scipy import ndimage
from numpy.testing import assert_array_equal

#now import from files
from fastgibbs.gibbs import gibbs
from fastgibbs.gibbs_SOR import gibbs_SOR
from fastgibbs.gibbs_SSOR import gibbs_SSOR
from fastgibbs.gibbs_cheby import gibbs_cheby

if __name__=="__main__":
    
    #load constants
    ############################################
    #test_A = np.eye(dims)*alpha - 0.5*scipy.ndimage.filters.laplace(np.eye(dims)) 
    test_A = np.loadtxt("W.txt",delimiter=',')
    print("shape of A is {}".format(np.shape(test_A)))
    [m,n] = np.shape(test_A)
    test_A += (1e-4)*np.diag(np.diag(np.ones([m,n])))
    real_cov = np.linalg.inv(test_A)
    err_tol = 5e-2
    sample_num = int(5e4)
    iters = 200

    #chebyshev error (w=1)
    my_gibbs = gibbs_cheby(1.0,test_A,err_tol)
    my_gibbs.sample(False,sample_num,iters)
    my_gibbs.cholesky_error()
    my_gibbs.plot_error() 
   
    #normal gibbs error
    my_gibbs = gibbs(test_A)
    my_gibbs.sample(sample_num,iters)
    my_gibbs.plot_error() 

    """
    #SOR error (w=1)
    my_gibbs = gibbs_SOR(1,test_A)
    my_gibbs.sample(sample_num,iters)
    my_gibbs.plot_error()

    #SSOR error (w=1)
    my_gibbs = gibbs_SSOR(1,test_A)
    my_gibbs.sample(sample_num,iters)
    my_gibbs.plot_error()
    """
 

    #SHOWING PLOT
    plt.legend()
    plt.ylabel('relative error (log scale)')
    plt.xlabel('iterations')
    plt.title('Convergence in covariance of different sampling algorithms')
    plt.show()
