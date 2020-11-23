%example 1
%solver section...
mats = ["2d.txt","3d.txt","test_A.txt"]
for mat=mats
    sprintf('running on example %s',mat')
    A = readmatrix(mat);
    [dims,dims] = size(A);
    b = randn(dims,1);
    [x_emp,y,cov_emp] = conj_grad(A,b);
    x = sort(A\b);
    x_emp = sort(x_emp);
    figure();
    plot(x_emp,'o');
    hold on;
    plot(x);
    legend('CG result', 'result from inversion');
    rel_err = abs((norm(x - x_emp)/norm(x)));
    title(sprintf('Conjugate gradient solution of Ax=b vs real solution for matrix %s Rel err %d',mat,rel_err));
    hold off;

    %sampling...
    cov = inv(A)
    evals = sort(eigs(cov));
    evals_emp = sort(eigs(cov_emp));
    rel_err = abs(norm(cov - cov_emp)/norm(cov));
    figure();
    semilogy(evals);
    hold on;
    semilogy(evals_emp,'o');
    legend('real eigenvalues', 'CG eigenvalues');
    title(sprintf('eigenvalues of empirical vs real cov of matrix %s. Rel err in matrix %d',mat,rel_err));
end