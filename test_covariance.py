#import packages
from sklearn.covariance import EmpiricalCovariance as Ecov
from gibbs_cheby import gibbs_cheby
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la
from conj_grad import conj_grad
import seaborn as sb

###################################################
################## MAKING MATRIX ##################
###################################################

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
    print("ERROR, DIMENSION NOT SPECIFIED CORRECTLY")
cov = la.inv(test_A)

print("matrix heatmap...")
f, ax = plt.subplots(figsize=(16, 16))
# color map
hm = sb.heatmap(cov, annot=False, ax=ax, cmap='coolwarm',
        linewidths=.05,fmt='.5f', cbar_kws={"orientation": "horizontal", "shrink": 0.3},vmin=-4,vmax=4)
t= f.suptitle('Matrix A^{-1} (Covariance matrix)', fontsize=14)   
plt.savefig("cov.png",bbox_inches='tight')
plt.close()

####################################################
################ RUNNING SAMPLERS ##################
####################################################

#now run chebyshev sampler and get the empirical covariance
err_tol = 3e-2
my_gibbs = gibbs_cheby(1.0,test_A,err_tol)
my_gibbs.sample(False)
cheby_eigs, cheby_cov = my_gibbs.espectrum()
cheby_diff = np.abs(cov - cheby_cov)/la.norm(cov)
cheby_rel = la.norm(cov - cheby_cov)/la.norm(cov)
print("cheby diff is {}".format(cheby_diff))
cheby_plot_bound = np.max(cheby_diff)

#now running conjugate gradient sampler, determining empirical covariance
print("doing conjugate gradient samples")
cg_cov = my_gibbs.conj_grad_sample(5e4)
print(cg_cov)
cg_diff = np.abs(cov - cg_cov)/la.norm(cov)
cg_rel = la.norm(cov - cg_cov)/la.norm(cov)
cg_plot_bound = np.max(cg_diff)

#use same scale on both graphs to compare
if (cg_plot_bound > cheby_plot_bound):
    plot_bound = cg_plot_bound
else:
    plot_bound = cheby_plot_bound

#####################################################
################# PLOTTING ##########################
#####################################################

#plot the difference between the empirical and real covariance as a heatmap
f, ax = plt.subplots(figsize=(16, 16))
hm = sb.heatmap(cheby_diff, annot=False, ax=ax, cmap='YlOrBr',
        linewidths=.05,fmt='.5f', cbar_kws={"orientation": "horizontal", "shrink": 0.3})
t= f.suptitle('relative difference between chebyshev covariance and real covariance. total relative error {} %'.format(cheby_rel*100), fontsize=14)   
plt.savefig("cheby_diff.png",bbox_inches='tight')
plt.close()

#now doing the same with conjugate gradient
f, ax = plt.subplots(figsize=(16, 16))
hm = sb.heatmap(cg_diff, annot=False, ax=ax, cmap='YlOrBr',
        linewidths=.05,fmt='.5f', cbar_kws={"orientation": "horizontal", "shrink": 0.3})
t= f.suptitle('relative difference between CG covariance and real covariance. total relative error {} %'.format(cg_rel*100), fontsize=14)   
plt.savefig("CG_diff.png",bbox_inches='tight')
plt.close()
