function [x,y,c,count,lost_conj] = conj_grad(A,b,iters)
    %make initial guess
    [m,n] = size(A);
    x = zeros(m,1);
    %initialize sample as well
    y = zeros(m,1);
    %initialize c_k
    c = zeros(m,1);
    %initial residual
    r = b - A*x;
    %initial search direction
    p = r;
    %normalizing constant
    d = p'*A*p;
    %cov = zeros(m,m);
    %min_rel_err = 1.0;
    %min_c = zeros(m,1);
    %min_iter = 1;
    p_vec = [];
    for count=1:iters
        gamma = (r'*r)/d;
        %calculating state vector
        x = x + gamma*p;
        %sample z ~ N(0,1)
        z = randn();
        %update sample
        y = y + (z/sqrt(d))*p;
        %update c_k, storing old c_k
        %c = c + (z/sqrt(d))*(A*p);
        %store old r to calculate new beta
        r_old = r;
        r = r_old - gamma*A*p;
        %standard beta
        %beta = -(r'*r)/(r_old'*r_old);
        %polak-ribiere method
        beta = -(r'*(r - r_old))/(r_old'*r_old);
        p_old = p;
        p = r - beta*p;
        d = p'*A*p;
        %add new p vector to set 
        p_vec = [p_vec p];
        %calculate A-conjugacy of p vectors so far
        conj_vec = [];
        for i=1:(count-1)
            p_i = p_vec(:,i);
            conj = dot(p_i,A*p);
            conj_vec(i) = conj;
        end
        mean_conj = mean(conj_vec);
        if mean_conj > 1e-2
            lost_conj = count+1;
            sprintf('lost conjugacy at iteration %d',lost_conj)
            break;
        end
        sprintf('mean A conjugacy at iter %d is: %d',count+1,mean(conj_vec));
        %calculate covariance at this point
        %cov_approx_part = (A)*(1/d)*(p*p')*(A);
        %cov = cov + cov_approx_part;
        %A;
        %rel_err = norm(cov - A)/norm(A);
        %sprintf('rel error %d at iter %d, with (1/d)=%d',rel_err,count,1/d);
        %if rel_err < min_rel_err
        %    min_c = c;
        %    min_iter = count;
        %    min_rel_err = rel_err;
        %elseif rel_err > (1.5*min_rel_err)
        %    sprintf('going uphill at iter %d, min err %d current error %d ',count,min_rel_err,rel_err)
        %    c = min_c;
        %    break;
        %end
        %check for convergence
        %if d<=1e-8
        %    sprintf('d diverged at iteration %d with d=%d',count,abs(d))
        %     %c = A*y;
        %    break;
        if norm(r)<1e-4
            sprintf('r converged to %d at iteration %d',norm(r),count)
            c = A*y;
            break;
        end
    end
    c = A*y;
    sprintf('exited loop at iter count');
end