"""plot the eigenspectrum of the real covariance matrix (W.txt) and compare
it with the eigenspectrum of the calculated covariance matrix (W_emp.txt)
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

def espectrum(filename):
    A = np.loadtxt("{}.txt".format(filename),delimiter=',')
    print(A)
    #read from csv (comma separated value) file
    #get evals
    eigs, eigvecs = LA.eig(A)
    #there should be some small imagininary part to eigenvalues due to
    #numerical error, but it should not be too large (say, greater than 1%
    #of the absolute value of the eigenvalue)
    assert (np.mean(eigs.imag)/np.mean(eigs) < 0.01), "ERROR: complex eigenvalues"
    #get rid of imaginary evals
    eigs = eigs.real
    #sort evals to plot in ascending order
    eigs = np.sort(eigs)
    print(eigs)
    eigs+= np.mean(eigs)/100
    print("{} eigenvalues for {}".format(np.shape(eigs),filename))
    #plot
    plt.plot(eigs,linestyle=":",label="{}".format(filename))
    
if __name__=="__main__":
    espectrum("3d_inv")
    espectrum("3d_emp")
    plt.title("eigenvalues in ascending order")
    plt.xlabel("eigenvalue #")
    plt.ylabel("magnitude (log scale)")
    plt.legend()
    plt.yscale('log')
    plt.show() 
