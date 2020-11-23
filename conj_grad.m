function [x,y,cov,d_matrix] = conj_grad(A,b)
    %make initial guess
    [m,n] = size(A);
    x = zeros(m,1);
    %initialize sample as well
    y = zeros(m,1);
    %initial residual
    r = b - A*x;
    %initial search direction
    p = r;
    %normalizing constant
    d = p'*A*p;
    %iterate... storing P and d for covariance analysis
    P_matrix = [];
    d_vector = [];
    count=0;
    while true
        sprintf('iteration %d, residual %d',count,norm(r))
        gamma = (r'*r)/d;
        x = x + gamma*p;
        %sample z ~ N(0,1)
        z = randn;
        %update sample
        y = y + (z/sqrt(d))*p;
        %STORE INFO ABOUT P AND D
        P_matrix = [P_matrix, p];
        d_vector = [d_vector, d];
        %store old r to calculate new beta
        r_old = r;
        r = r - gamma*A*p;
        beta = -(r'*r)/(r_old'*r_old);
        p = r - beta*p;
        d = p'*A*p;
        %check for convergence
        if norm(r)<eps
            sprintf('converged at iteration %d with r=%d',count,norm(r));
            break;
        end
        count=count+1;
    end
    sprintf('exited loop')
    d_matrix = diag(d_vector);
    %cannot manually invert d matrix, so just make each nonzero diagonal
    %entry lambda = 1/lambda... these correspond to the evals in the 
    %krylov subspace...
    [dims,dims] = size(d_matrix)
    for i=1:dims
        if (d_matrix(i,i)~=0)
            d_matrix(i,i)=1/d_matrix(i,i);
        end
    end
    P_matrix_T = P_matrix';
    cov = P_matrix*d_matrix*P_matrix_T;
end