%% SQUARED EXPONENTIAL GAUSSIAN COVARIANCE MATRIX
clear all;
%size of grid
x = 100;
y = 100;

%generate domain
s = linspace(-3,3,x);
%size of diagonal bump
epsilon = 1e-6;

gauss_matrix = zeros(x,y);

%generate matrix according to formula in CG sampler paper
%http://www.physics.otago.ac.nz/data/fox/publications/CDSamplerAnalysis_20120112.pdf
for i=1:x
    for j=1:y
        arg = -((s(i) - s(j))^2)/(2*(1.5)^2);
        gauss_matrix(i,j) = 2*exp(arg);
    end
end

%add diagonal bump
gauss_matrix = gauss_matrix + epsilon*eye(x)

cov = gauss_matrix;
evals = eigs(cov,x);
A = inv(cov)
mat='gauss matrix';

%% RANDOM MATRIX WITH WELL SPACED EIGS
clear all;
%size of matrix
n = 100

%D is our diagonalized matrix with eigenvalues
D = zeros(n,n);

%populate the matrix
for i=1:n
    D(i,i) = 1/i^3;
end

%normalize the matrix
D_max = D(n,n);
for i=1:n
    D(i,i) = D(i,i)/D_max;
end

%now we want to create a random orthogonal matrix U, as we can generate 
%the matrix corresponding to the eigenvalues in D by going
%UDU', source: https://math.stackexchange.com/questions/54818/construct-matrix-given-eigenvalues-and-eigenvectors
%simply generate a random nxn matrix, use the QR decomposition, and set U=Q
Random_matrix = rand(n,n);
[Q,R] = qr(Random_matrix);

%now we want our result
res = Q*D*Q';
cov = res;
heatmap(cov)
evals = eigs(cov,n);
A = inv(cov);
mat = 'well spaced eigs'
cond(cov)
%% 2D LAPLACIAN
clear all;
x = input("Enter the x dimension of the grid: ");
y = input("Enter the y dimension of the grid: ");
alpha = input("Enter the size of the diagonal bump: ");
[lambda,V,A] = laplacian([x,y],{'NN' 'NN'},x*y);
%we will add a constant on the diagonal to make A invertible, 
%so we must also increase the evals by this constant
lambda = lambda + alpha*ones(x*y,1);
A = A + alpha*eye(x*y,x*y);
%evals of cov are 1/eval of A due to inversion of matrix
cov = inv(A);
evals = eigs(cov,x*y)
mat = '2d laplacian';

%% NOW SAMPLE AND COMPARE
sprintf('running on example %s',mat')
sprintf('condition number of matrix is %d',cond(A))
[dims,dims] = size(A);
%b = [+-1, +-1, ... , +-1]'
%need to average over many cov_emps
samples=1e3;
cov_emp_samples = zeros(dims,dims,samples);
x_emp_set = zeros(dims,samples);
for i=1:samples
    if mod(i,1000)==0
        percent = (i/samples) * 100;
        sprintf('%d percent done.',percent)
    end
    %b is randomly drawn from [-1 1]
    b_rng = [-1 1];
    b = b_rng(randi(numel(b_rng),dims,1))';
    [x_emp,y,cov_emp,count] = conj_grad(A,b);
    x_emp_set(:,i) = x_emp;
    rel_err = norm(cov_emp - cov)/norm(cov);
    sprintf('relative error of sample %d is %d. Took %d iterations',i,rel_err,count)
    %concatenate cov_emp_samples so we can average over them later
    cov_emp_samples(:,:,i) = cov_emp;
end
%mean x_emp
x_emp = sort(x_emp_set(:,samples))
size(x_emp)
%mean cov_emp
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
evals_emp = eigs(cov_emp,dims);
rel_err = abs(norm(cov - cov_emp)/norm(cov));
figure();
%just plotting the first 20 evals
semilogy(evals(1:20));
hold on;
semilogy(evals_emp(1:20),'o','MarkerSize',12);
legend('real eigenvalues', 'CG eigenvalues');
title(sprintf('eigenvalues of empirical vs real cov of matrix %s. Rel err in matrix %d',mat,rel_err));