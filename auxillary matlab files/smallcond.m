function res = smallcond(n)

%D is our diagonalized matrix with eigenvalues
D = zeros(n,n);

%populate the matrix
for i=1:n
    D(i,i) = i^(2);
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
%check the condition number
cond(res)
%check the eigenvalues
unique(eigs(res))

%store the matrix in a txt file so i can read it with python
writematrix(res)
end