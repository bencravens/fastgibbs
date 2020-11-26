%example 1
%solver section...
%mats = ["2d.txt","3d.txt","test_A.txt"]
mats = ["test_A.txt"]
for mat=mats
    sprintf('running on example %s',mat')
    A = readmatrix(mat);
    sprintf('condition number of matrix is %d',cond(A))
    cov = inv(A);
    [dims,dims] = size(A);
    %b = [+-1, +-1, ... , +-1]'
    %need to average over many cov_emps
    samples=5e2
    cov_emp_samples = zeros(dims,dims,samples);
    for i=1:samples
        if mod(i,1000)==0
            percent = (i/samples) * 100;
            sprintf('%d percent done.',percent)
        end
        %b is randomly drawn from [-1 1]
        b_rng = [-1 1];
        b = b_rng(randi(numel(b_rng),dims,1))';
        [x_emp,y,cov_emp,count] = conj_grad(A,b);
        rel_err = norm(cov_emp - cov)/norm(cov);
        sprintf('relative error of sample %d is %d. Took %d iterations',i,rel_err,count)
        %concatenate cov_emp_samples so we can average over them later
        cov_emp_samples(:,:,i) = cov_emp;
    end
    cov_emp = mean(cov_emp_samples,3);
    sz = size(cov_emp);
    cov_emp_means = zeros(1,dims);
    for i=1:samples
        %have to do mean twice to get the mean of a matrix
        cov_emp_means(i) = mean(mean(cov_emp_samples(:,:,i)));
    end
    sprintf('~~~~~~~~~Relative Average Deviation OF COV MATRICES WITH DIFFERENT B IS~~~~~~~')
    cov_std = mad(cov_emp_means)/mean(cov_emp_means)
    sprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    figure();
    plot(sort(x_emp),'o');
    hold on;
    x = sort(A\b);
    plot(x);
    legend('CG result', 'result from inversion');
    rel_err = abs((norm(x - x_emp)/norm(x)));
    title(sprintf('Conjugate gradient solution of Ax=b vs real solution for matrix %s Rel err %d',mat,rel_err));
    hold off;

    %sampling...
    evals = eigs(cov);
    evals_emp = eigs(cov_emp);
    rel_err = abs(norm(cov - cov_emp)/norm(cov));
    figure();
    semilogy(evals);
    hold on;
    semilogy(evals_emp,'o');
    legend('real eigenvalues', 'CG eigenvalues');
    title(sprintf('eigenvalues of empirical vs real cov of matrix %s. Rel err in matrix %d',mat,rel_err));
end