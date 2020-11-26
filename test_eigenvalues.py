from sklearn.covariance import EmpiricalCovariance as Ecov
from gibbs_cheby import gibbs_cheby
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la
from conj_grad import conj_grad
import seaborn as sb

#read dimension in from file
with open("dim.txt") as f:
    lines = f.readlines()
dim = lines[0]
print(lines)
if dim=="2d":
    test_A = np.loadtxt("2d.txt",delimiter=',')
elif dim=="3d":
    test_A = np.loadtxt("3d.txt",delimiter=',')
else:
    print("ERROR, INVALID DIMENSION")
#test_A = np.loadtxt("test_A.txt",delimiter=',')
test_A = np.array([np.exp(i) for i in test_A])
print("matrix heatmap...")
f, ax = plt.subplots(figsize=(16, 16))
hm = sb.heatmap(test_A, annot=False, ax=ax, cmap='coolwarm',
       linewidths=.05,fmt='.5f', cbar_kws={"orientation": "horizontal", "shrink": 0.3})
t= f.suptitle('Matrix A', fontsize=14)   
plt.savefig("matrix_A.png",bbox_inches='tight')
plt.close()
cov = la.inv(test_A)

#now doing conjugate gradient
print("doing conjugate gradient samples")
err_tol = 3e-2
my_gibbs = gibbs_cheby(1.0,test_A,err_tol)
cg_cov, cg_cov_analytic = my_gibbs.conj_grad_sample(1e5)
cg_err = la.norm(cov - cg_cov)/la.norm(cov)
#grabbing eigenvalues of empirical covariance
conj_eigs, eigvecs = la.eigh(cg_cov)
conj_eigs = np.sort(conj_eigs)
conj_eigs = np.ma.masked_where(conj_eigs < 1e-5,conj_eigs)
plt.semilogy(conj_eigs,label="conjugate gradient emp eigs, relative error in cov {}".format(cg_err),marker='1',linestyle="none")
#eigenvalues of analytic covariance 
conj_eigs_analytic, eigvecs = la.eigh(cg_cov_analytic)
print("analytic eigs are {}".format(conj_eigs_analytic))
conj_eigs_analytic = np.sort(conj_eigs_analytic)
conj_eigs_analytic = np.ma.masked_where(conj_eigs_analytic < 1e-5, conj_eigs_analytic)
#enable latex in plot
params={'text.usetex': True}
plt.rcParams.update(params)
plt.semilogy(conj_eigs_analytic,label="analytic conjugate gradient evals of cov=$\Sigma_{0}^{k} \\frac{1}{d_{i}} p_{i}^{T} p_{i}$",marker='2',linestyle="none")

print("running chebyshev sampler")
#create figure again
my_gibbs.sample(False)
#now load eigenvalues generated by matlab file
#eigs = np.loadtxt("eigs.txt",delimiter=',')
eigs, eigvecs = la.eigh(cov)
eigs = np.sort(eigs)
#mask invalid values
#formula_eigs = np.ma.masked_where(eigs < 1e-5,eigs)
plt.semilogy(eigs,label='analytic eigenvalues')
cheby_eigs,cheby_cov = my_gibbs.espectrum()
cheby_err = la.norm(cov-cheby_cov)/la.norm(cov)
cheby_eigs = np.sort(cheby_eigs)
plt.semilogy(cheby_eigs,label='chebyshev eigenvalues, relative error in cov {}'.format(cheby_err),marker='4',linestyle="none")
plt.legend(prop={'size': 12})
plt.savefig("results.png",bbox_inches='tight')
plt.show()
