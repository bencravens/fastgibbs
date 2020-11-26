function [x,y,cov,count] = conj_grad(A,b)
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
    real_cov = inv(A);
    cov = zeros(m,m);
    min_err = 10;
    rel_err = 9;
    cov_min = zeros(m,m);
    x_min = zeros(m,1);
    y_min = zeros(m,1);
    while count<300
        %if mod(count,50)==0
        %    sprintf('iteration %d, residual %d',count,norm(r))
        %    sprintf('relative error at iteration %d is %d',count,rel_err)
        %end
        gamma = (r'*r)/d;
        %calculating state vector
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
            %sprintf('converged at iteration %d with r=%d',count,norm(r))
            %break;
        end
        %cannot manually invert d matrix, so just make each nonzero diagonal
        %entry lambda = 1/lambda... these correspond to the evals in the 
        %krylov subspace...
        %d_matrix = diag(d_vector);
        %[dims,dims] = size(d_matrix);
        %for i=1:dims
        %    if (d_matrix(i,i)~=0)
        %        d_matrix(i,i)=1/d_matrix(i,i);
        %    end
        %end
        %P_matrix_T = P_matrix';
        %storing old covariance incase we accidentially make sample worse
        %cov = P_matrix*d_matrix*P_matrix_T;
        cov_approx_part = (1/d)*(p*p');
        cov = cov + (1/d)*(p*p');
        rel_err = norm(cov - real_cov)/norm(real_cov);
        if rel_err < min_err
            min_err = rel_err;
            cov_min = cov;
            x_min = x;
            y_min = y;
        elseif rel_err > (1.1)*min_err
            sprintf('lost conjugacy, prior error %d, current error %d. breaking loop',min_err,rel_err)
            %reverting to old covariance
            cov = cov_min;
            x = x_min;
            y = y_min;
            break
        end
        count=count+1;
    end
    sprintf('exited loop at iter count');
end