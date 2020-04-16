
# coding: utf-8

# In[ ]:

import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance as Ecov
import scipy.sparse as sp
import scipy
from scipy import ndimage
import seaborn as sns

class gibbs_SSOR:
    """This is a python3 implementation of a multivariate gibbs sampler 
       for sampling from distributions of the form N(mu,A^(-1)). This uses 
       the block form gibbs sampler with the "symmetric successive over-relaxation" 
       (SSOR) relaxation scheme
    """
    def __init__(self,omega,A):
        #Initialize the properties of the gibbs sampler

        [m,n] = np.shape(A)
        assert m==n, "A matrix must be square"
        self.dims = m
        
        #set up relaxation parameter
        self.omega = omega
        
        #covariance and precision matrix
        self.A = A
        self.cov = np.linalg.inv(self.A)
        
        # A MUST BE SYMMETRIC for SOR or SSOR splittings to work...
        # therefore test precision matrix is symmetric
        np.testing.assert_array_equal(self.A,np.transpose(self.A))

        #grab upper triangle, lower triangle, and diagonal of A...
        #lower triangular of A
        self.L = np.tril(self.A,k=-1)

        #upper triangular of A
        self.U = np.triu(self.A,k=1)

        #diagonal part of A
        self.D = np.diag(np.diag(self.A))
        self.sqrt_D = np.sqrt(self.D)
        
        #Now SOR 
        self.M = (1/self.omega)*self.D + self.L 
        self.N = self.M - self.A
        
        #calculate SSOR matrices for theoretical error scaling
        self.M_SSOR = (self.omega/(2 - self.omega))*(np.matmul(np.matmul(self.M,np.linalg.inv(self.D)),np.transpose(self.M)))
        self.N_SSOR = self.M_SSOR - self.A
        
        #Now making M inverse, transpose, etc
        self.M_inv = np.linalg.inv(self.M)
        self.M_inv_tran = np.transpose(self.M_inv)
        self.N_tran = np.transpose(self.N)
        
        #build scaling parameter gamma which appears in SSOR forward and backwards updating sweeps
        self.gamma = (2/self.omega - 1)**(0.5)
            
        #set up c mean and covariance
        self.c_cov = np.transpose(self.M) + self.N
        self.c_sample = np.sqrt(np.diag(self.c_cov))
        #set up error stuff
        self.error_vec=[]
        #we want the min,max evals of the M^-1 * A matrix
        eigvals = np.linalg.eigvals(np.matmul(np.linalg.inv(self.M_SSOR),self.N_SSOR))
        #eigenvalues need to be positive
        #now take max and min eigenvalues
        self.l_max = np.max(np.abs(eigvals))

    def sample(self,sample_num=int(5e4),k=30):
        """NOW SAMPLING FROM THIS DISTRIBUTION USING ITERATIVE MATRIX SPLITTING 
        GIBBS SAMPLER"""
        self.n = k
        #make initial states
        self.state = np.random.randn(sample_num,self.dims)

        #iterate j times
        for j in range(k):
            #iterate for each sample
            for i in range(sample_num):
                self.cur_state = self.state[i,:]
                z = np.random.randn(self.dims,)
                x1 = np.matmul(np.matmul(self.M_inv,self.N),self.cur_state)
                x2 = self.gamma*np.matmul(np.matmul(self.M_inv,self.sqrt_D),z)
                x = x1 + x2
                z = np.random.randn(self.dims,)
                y1 = np.matmul(np.matmul(self.M_inv_tran,self.N_tran),x)
                y2 = self.gamma*np.matmul(np.matmul(self.M_inv_tran,self.sqrt_D),z)
                y = y1 + y2
                self.state[i,:] = y
            self.e_cov = Ecov().fit(self.state).covariance_
            self.error = np.linalg.norm(self.e_cov-self.cov)/np.linalg.norm(self.cov)
            print(self.error)
            self.error_vec.append(self.error)
            #stopping criteria
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
        #want to calculate the mean cholesky sampling accuracy to compare with
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
        plt.plot(range(self.n),np.ones(self.n)*np.log(chol_error),label="cholesky")

if __name__ == "__main__":
    #testing this on a simple example
    temp = []
    dims = 50
    alpha = 0.01
    test_A = np.eye(dims)*alpha - 0.5*scipy.ndimage.filters.laplace(np.eye(dims)) 
    real_cov = np.linalg.inv(test_A)
    my_gibbs = gibbs_SSOR(1.0,test_A)
    my_gibbs.sample()
    state,e_cov = my_gibbs.get_state()
    print("relative error is {}".format(np.linalg.norm(real_cov-e_cov)/np.linalg.norm(real_cov)))
    my_gibbs.plot()

