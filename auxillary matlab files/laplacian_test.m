%we want to make a laplacian matrix
dim = 100
alpha = 1e-4

A = 4*diag(diag(ones(dim,dim)));
A(1,1) = 2;
A(end,end) = 2;
A = A - diag(diag(ones(dim-1,dim-1)),1) - diag(diag(ones(dim-1,dim-1)),-1);
A = A + alpha*diag(diag(ones(dim,dim)));
heatmap(A)
writematrix(A)
uiwait(helpdlg('Examine the figures, then click OK to finish.'));
