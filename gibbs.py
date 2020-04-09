import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance
import scipy.sparse as sp
import scipy
from scipy import ndimage

class gibbs:
    """This is a python3 implementation of a multivariate matrix block form
       gibbs sampler (using the gauss-seidel matrix splitting) 
       for sampling from distributions of the form N(mu,A^(-1))
    """
    def __init__(self,dims,means,A):
        #Initialize the properties of the gibbs sampler

        #set up dimension and means
        self.dims = dims
        self.means = means
        
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
        """NOW SAMPLING FROM THIS DISTRIBUTION USING ITERATIVE MATRIX SPLITTING 
        GIBBS SAMPLER"""
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
                self.state[i,:] = np.matmul(self.M_inv,np.matmul(self.N,cur_state)) + np.matmul(self.M_inv,c) + self.means
            self.e_cov = EmpiricalCovariance().fit(self.state).covariance_
            self.error = np.linalg.norm(self.e_cov-self.cov)/np.linalg.norm(self.cov)
            self.error_vec.append(self.error)

    def plot(self):
        #simply want to plot all of our errors
        #plt.plot(range(self.n),np.log(self.error_vec),linestyle=":")
        return self.error_vec,self.n,self.l_max
    
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
    dim = 100
    run_num = 2
    alpha = 0.01
    test_A = np.eye(dims)*alpha - 5.5*scipy.ndimage.filters.laplace(np.eye(dims)) 
    for i in range(run_num):
        print("executing run {}/{}".format(i+1,run_num))
        my_gibbs = gibbs(dim,np.zeros(dim),test_A)
        my_gibbs.sample()
        err,n,l_max = my_gibbs.plot()
        temp.append(err)
    my_gibbs.cholesky_error()
    print("lmax is {}".format(l_max))
    plt.plot(range(n),np.log(np.mean(temp,axis=0)),label="mean actual convergence rate")
    plt.plot(range(n),np.log(l_max**2)*range(n),label="theoretical convergence rate")
    plt.xlabel('iterations')
    plt.ylabel('relative error (log scale)')
    plt.title("Theoretical vs Actual Convergence rate for a Gibbs sampler")
    plt.legend()
    plt.show();
