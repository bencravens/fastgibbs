import seaborn as sns
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance as Ecov
import scipy.sparse as sp
import scipy
from scipy import ndimage

class gibbs:
    """This is a python3 implementation of a multivariate matrix block form
       gibbs sampler (using the gauss-seidel matrix splitting) 
       for sampling from distributions of the form N(mu,A^(-1))
    """
    def __init__(self,A):
        #Initialize the properties of the gibbs sampler

        #set up dimensions
        [m,n] = np.shape(A)
        assert m==n, "A matrix must be square"
        self.dims = m
        
        #set up precision and covariance 
        #load from txt file for low condition number
        #self.A = np.loadtxt("res.txt",delimiter=',')
        self.A = A
        self.cov = np.linalg.inv(self.A)
        self.cond = np.linalg.cond(self.A)
        print("condition number is {}".format(self.cond))
        
        #now split matrix... A = M - N
        """use gauss-siedel splitting, M is lower triangular part of matrix
        (including diagonal) and N is -1*(upper triangular part of matrix)"""
        self.M = np.tril(self.A)
        self.M_inv = np.linalg.inv(self.M)
        self.N = self.M - self.A
            
        #set up c mean and covariance
        self.c_cov = np.transpose(self.M) + self.N
        self.c_sample = np.sqrt(np.diag(self.c_cov))
        #set up error stuff
        self.error_vec=[]
        #we want the min,max evals of the M^-1 * N matrix
        eigvals = np.linalg.eigvals(np.matmul(np.linalg.inv(self.M),self.N))
        #eigenvalues need to be positive
        #now take max and min eigenvalues
        self.l_max = np.max(np.abs(eigvals))

    def sample(self,sample_num=int(5e4),k=25):
        """
        Now sample from the distribution using the Gauss-Siedel matrix splitting. The sampler will halt when a certain
        relative error difference between iterations k,k+1 is reached, default 1e-3. There is an optional "error" parameter 
        which when selected allows one to view how the algorithm is converging, used for debugging purposes. 
        """
        self.sample_num = sample_num
        self.n = k
        #make initial states
        self.state = np.random.randn(sample_num,self.dims)

        #iterate j times
        for j in range(k):
            if j%5==0:
                print("iteration number {}/{}".format(j,k))
            #iterate for each sample
            for i in range(sample_num):
                cur_state = self.state[i,:]
                #make c
                c = self.c_sample*np.random.randn(self.dims)
                #iterate!
                self.state[i,:] = np.matmul(self.M_inv,np.matmul(self.N,cur_state)) + np.matmul(self.M_inv,c)
            #break if iteration has converged. Set convergence criteria to be 1e-2 relative error
            #store the error
            self.e_cov = Ecov().fit(self.state).covariance_
            self.error = np.linalg.norm(self.e_cov-self.cov)/np.linalg.norm(self.cov)
            if self.error<1e-2:
                print("converged at iter {}/{}".format(j,k))
                break
            self.error_vec.append(self.error)
            print("error is {}".format(self.error))

    def get_state(self):
        """
        getter function for gibbs sampler class, returns current state, and the covariance of that state
        """
        self.e_cov = Ecov().fit(self.state).covariance_
        return self.state, self.e_cov

    def plot(self):
        f, ax = plt.subplots(figsize=(10, 6))
        hm = sns.heatmap(abs(self.e_cov - self.cov)/np.linalg.norm(self.cov), annot=True, ax=ax, cmap="coolwarm",
                 linewidths=.05,fmt='.5f')
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Empirical covariance and real covariance relative error heatmap', fontsize=14)   
        plt.show()
 
    def cholesky_error(self):
        #want to calculate the mean cholesky sampling accuracy to compare with
        chol_samples = []
        print('doing cholesky now')
        for i in range(self.sample_num):
            c = np.linalg.cholesky(self.cov)
            mean = np.zeros((self.dims,))
            cov = np.eye(self.dims)
            z=np.random.multivariate_normal(mean, cov, 1).t
            y = np.matmul(c,z)
            chol_samples.append(y)
        chol_samples = np.squeeze(chol_samples,axis=2)
        print(np.shape(chol_samples))
        e_cov = Ecov().fit(chol_samples).covariance_
        chol_error = np.linalg.norm(self.cov-e_cov)/np.linalg.norm(self.cov)
        plt.plot(range(self.n),np.ones(self.n)*np.log(chol_error),label="cholesky")
        
if __name__ == "__main__":
    #testing this on a simple example
    temp = []
    dim = 3
    alpha = 0.005
    test_a = np.eye(dim)*alpha - 0.5*scipy.ndimage.filters.laplace(np.eye(dim)) 
    realcov = np.linalg.inv(test_a)
    my_gibbs = gibbs(test_a)
    my_gibbs.sample()
    my_gibbs.plot()
    state,empcov = my_gibbs.get_state()
    diff = np.linalg.norm(realcov-empcov)/np.linalg.norm(realcov)
    print("relative error between real and actual covariance is {}".format(diff))

