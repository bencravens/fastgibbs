%making squared exponential covariance matrix

%size of grid
x = 10;
y = 10;

%generate domain
s = linspace(-3,3,x);
%size of diagonal bump
epsilon = 1e-4;

gauss_matrix = zeros(x,y);

%generate matrix according to formula in CG sampler paper
%http://www.physics.otago.ac.nz/data/fox/publications/CDSamplerAnalysis_20120112.pdf
for i=1:x
    for j=1:y
        arg = -((s(i) - s(j))^2)/(2*(1.5)^2);
        gauss_matrix(i,j) = 2*exp(arg);
        if i==j
            gauss_matrix(i,j) = gauss_matrix(i,j) + epsilon;
        end
    end
end

%write to csv file
writematrix(gauss_matrix,'test.txt');
A = readmatrix('test.txt');
gauss_matrix == A

%double check qualities of matrix
sprintf('condition number of matrix is %d',cond(gauss_matrix))
sprintf('2 norm of matrix is %d',norm(gauss_matrix))
sprintf('2 norm of matrix inverse is %d',norm(inv(gauss_matrix)))
issymmetric(gauss_matrix)
heatmap(chol(gauss_matrix))

%generate heatmap
figure();
title('heatmap of squared exponential covariance matrix');
heatmap(gauss_matrix);