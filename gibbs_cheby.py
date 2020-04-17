#make sure we have the right packages again
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance as Ecov
import scipy.sparse as sp
import scipy
from scipy import ndimage
from numpy.testing import assert_array_equal
import seaborn as sns

#load class
class gibbs_cheby:
    """This is a python3 implementation of a chebyshev polynomial accelerated multivariate gibbs sampler 
       for sampling from distributions of the form N(mu,A^(-1)). 
    """

    def __init__(self,omega,A):
        #Initialize the properties of the gibbs sampler
        #set up dimensions and nu
        [m,n] = np.shape(A)
        assert m==n, "A matrix must be square"
        self.dims = m
        self.nu = np.ones(self.dims)
        
        #set up the symmetric succesive over relaxation parameter (S.S.O.R)
        self.omega = omega
        # omega: 0 < omega < 2
        assert 0<omega<2, "omega must be between 0 and 2."
        #set up precision matrix
        #set up precision and covariance 
        #load from txt file for well conditioned matrix
        #self.A = np.loadtxt("res.txt",delimiter=',')
        self.A = A
        self.cov = np.linalg.inv(self.A)
        self.cond = np.linalg.cond(self.A)
        print(self.cond)
        #now set up the mean
        self.mu = np.matmul(np.linalg.inv(self.A),self.nu)
        #now split matrix... A = M - N
        
        """
        Use SSOR matrix splitting!
        """
        
        # A MUST BE SYMMETRIC for SOR or SSOR splittings to work...
        # therefore test precision matrix is symmetric
        #resymmetrize the matrix due to roundoff error
        self.A = (self.A + np.transpose(self.A))/2
        #now we can test to see whether or not it is symmetric:
        np.testing.assert_array_equal(self.A,np.transpose(self.A))

        #grab upper triangle, lower triangle, and diagonal of A...
        #lower triangular of A
        self.L = np.tril(self.A,k=-1)

        #upper triangular of A
        self.U = np.triu(self.A,k=1)

        #diagonal part of A
        self.D = np.diag(np.diag(self.A))

        #now make the successive over relaxation (SOR) M and N from these
        self.M_SOR = (1/self.omega)*self.D + self.L
        self.M_SSOR = (self.omega/(2 - self.omega))*(np.matmul(np.matmul(self.M_SOR,np.linalg.inv(self.D)),np.transpose(self.M_SOR)))
        self.M_SOR_inv = np.linalg.inv(self.M_SOR)
        self.M_SOR_inv_tran = np.transpose(self.M_SOR_inv)
        self.N_SOR = self.M_SOR - self.A

        #print tested matrices
        """
        print("\n SOR MATRICES")
        print(self.M_SOR)
        print(self.N_SOR,"\n")
        print("SOR SPLITTING GIVES A: \n {} \n".format(self.M_SOR - self.N_SOR))
        print("SSOR MATRICES")
        print(self.M_SSOR)
        print(self.N_SSOR,"\n")
        print("SSOR SPLITTING GIVES A: \n {} \n".format(self.M_SSOR - self.N_SSOR))
        """
        
        #we want the min,max evals of the M^-1 * A matrix
        eigvals = np.linalg.eigvals(np.matmul(np.linalg.inv(self.M_SSOR),self.A))
        eigvals = eigvals[eigvals>0]
        #now take max and min eigenvalues
        self.l_max = np.max(eigvals)
        self.l_min = np.min(eigvals)
        print("l_max {}, l_min {}".format(self.l_max,self.l_min))
        assert self.l_min<=self.l_max, "eigenvalue calculation is wrong..."
        
        #initialize states, empty for now...     
        #state matrix storing all state vectors
        self.state = None
        self.past_state = None
        #state vector of sample being currently iterated, and its previous state
        self.cur_state = None
        self.past_state = None
        
        #initialize empirical covariance, empty for now... 
        self.e_cov = None
        #initializing error vector
        self.error_vec = []
        
        #now setting constant values as defined in SSOR algorithm...
        #finding sqrt(D_omega)
        self.sqrt_D_omega = np.sqrt((2/self.omega - 1)*np.diag(self.A))
        #now setting constants
        self.delta = ((self.l_max - self.l_min)/4)**2
        
    def sample(self,sample_num=int(5e4),k=25):
        """NOW SAMPLING FROM THIS DISTRIBUTION USING ITERATIVE MATRIX SPLITTING 
        GIBBS SAMPLER"""
        
        #want to set the total length of the array that we will be plotting
        self.sample_num = sample_num
        self.n = k

        #make initial states
        self.state = np.random.randn(sample_num,self.dims)
        self.past_state = np.random.randn(sample_num,self.dims) 
        #alpha and tau are the acceleration parameters
        #set equal to one
        self.tau = 2/(self.l_max + self.l_min)
        #self.tau = 1
        self.alpha = 1
        
        self.beta = 2*self.tau
        self.b = 2/self.alpha - 1
        self.a = (2/self.tau - 1)*self.b
        self.kappa = self.tau
            
        #iterate over k
        for j in range(k):
            #update each sample
            for i in range(sample_num):
                #want to do this for each sample, first grab current state
                self.cur_state = self.state[i,:]
                #sample z
                self.z = np.random.randn(self.dims,) + self.nu
                #now make c & x
                self.c = np.sqrt(self.b)*self.sqrt_D_omega*self.z
                self.x = self.cur_state + np.matmul(self.M_SOR_inv,(self.c - np.matmul(self.A,self.cur_state)))
                #sample z again
                self.z = np.random.randn(self.dims,) + self.nu
                #make c again
                self.c = np.sqrt(self.a)*self.sqrt_D_omega*self.z
                #make w
                self.w = self.x - self.cur_state + np.matmul(self.M_SOR_inv_tran,self.c - np.matmul(self.A,self.x))
                #special case for first iteration
                if j==0:
                    result = self.alpha*(self.cur_state + self.tau*self.w)
                else:
                    result = self.alpha*(self.cur_state - self.past_state[i,:] + self.tau*self.w) + self.past_state[i,:]
                #set past state
                self.past_state[i,:] = self.cur_state
                self.state[i,:] = result  
            
            #update constants once every iteration
            self.beta = 1/(1/self.tau - self.beta*self.delta)
            self.alpha = self.beta/self.tau
            self.b = 2*self.kappa*(1 - self.alpha)/self.beta + 1
            self.a = (2/self.tau - 1) + (self.b - 1)*(1/self.tau + 1/self.kappa - 1)
            self.kappa = self.beta + (1 - self.alpha)*self.kappa
            
            #now we can calculate the error 
            self.e_cov = Ecov().fit(self.state).covariance_
            self.error = np.linalg.norm(self.cov - self.e_cov)/np.linalg.norm(self.cov)
            self.error_vec.append(self.error)
            print("relative error at iteration {}/{} is {}".format(j,k,self.error))

            #exit condition
            if self.error<1e-2:
                print("converged at iter {}/{}".format(j,k))
                break
    
    def get_state(self):
        """
        getter function for gibbs sampler class, returns current state, and the covariance of that state
        """
        self.e_cov = Ecov().fit(self.state).covariance_
        return self.state, self.e_cov

    def plot(self):
        f, ax = plt.subplots(figsize=(10, 6))
        hm = sns.heatmap(abs(self.e_cov - self.cov)/np.linalg.norm(self.cov), annot=False, ax=ax, cmap="coolwarm",
                 linewidths=.05,fmt='.5f')
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Empirical covariance and real covariance relative error heatmap', fontsize=14)   
        plt.show()

    def cholesky_error(self):
        #want to calculate the error from cholesky sampling as an accuracy benchmark
        chol_samples = []
        print('doing cholesky now')
        for i in range(self.sample_num):
            C = np.linalg.cholesky(self.cov)
            mean = np.zeros((self.dims,))
            cov = np.eye(self.dims)
            z=np.random.multivariate_normal(mean, cov, 1).T
            y = np.matmul(C,z)
            chol_samples.append(y)
        chol_samples = np.squeeze(chol_samples,axis=2)
        print(np.shape(chol_samples))
        e_cov = EmpiricalCovariance().fit(chol_samples).covariance_
        chol_error = np.linalg.norm(self.cov-e_cov)/np.linalg.norm(self.cov)
        plt.semilogy(range(self.n),np.ones(self.n)*(chol_error),label="Cholesky decomposition sampling")

if __name__ == "__main__":
    #testing this on a simple example
    temp = []
    dims = 200
    alpha = 0.005
    test_A = np.eye(dims)*alpha - 0.5*scipy.ndimage.filters.laplace(np.eye(dims)) 
    real_cov = np.linalg.inv(test_A)
    my_gibbs = gibbs_cheby(np.ones(dims),1.0,test_A)
    my_gibbs.sample()
    state,e_cov = my_gibbs.get_state()
    print("relative error is {}".format(np.linalg.norm(real_cov-e_cov)/np.linalg.norm(real_cov)))
    my_gibbs.plot()
