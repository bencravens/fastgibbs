
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
    test_A = np.loadtxt("A.txt",delimiter=',')
    real_cov = np.linalg.inv(test_A)
    err_tol = 5e-2
    sample_num = int(5e4)

    #chebyshev error
    my_gibbs = gibbs_cheby(1.0,test_A,err_tol)
    my_gibbs.sample(False,sample_num,20)
    my_gibbs.cholesky_error()
    my_gibbs.plot_error() 
   
    #normal gibbs error
    my_gibbs = gibbs(test_A)
    my_gibbs.sample(sample_num,20)
    my_gibbs.plot_error() 
 

    #SHOWING PLOT
    plt.show()
