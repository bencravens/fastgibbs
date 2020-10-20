%we want to make a laplacian matrix
%2d, with dirichlet boundary conditions...
%recieve user input for dimensions
str = input("enter 2d for square grid, or 3d for cubic grid: ",'s');
if str=="2d"
    x = input("Enter the x dimension of the grid: ");
    y = input("Enter the y dimension of the grid: ");
    alpha = input("Enter the size of the diagonal bump: ");
    [lambda,V,A] = laplacian([x,y],{'NN' 'NN'},x*y);
    %we will add a constant on the diagonal to make A invertible, 
    %so we must also increase the evals by this constant
    lambda = lambda + alpha*ones(x*y,1);
    A = A + alpha*eye(x*y,x*y);

    %write matrix to file
    csvwrite("2d.txt",A);
    %write eigenvalues to file
    csvwrite("eigs.txt",lambda);
    %tell python script we are working with 2d grid..
    fileID = fopen('dim.txt','w');
    fprintf(fileID,"2d");
    fclose(fileID);
elseif str=="3d"
    x = input("Enter the x dimension of the cube: ");
    y = input("Enter the y dimension of the cube: ");
    z = input("Enter the z dimension of the cube: ");
    alpha = input("Enter the size of the diagonal bump: ");
    [lambda,V,A] = laplacian([x,y,z],{'NN' 'NN' 'NN'}, x*y*z);
    %we will add a constant on the diagonal to make A invertible, 
    %so we must also increase the evals by this constant
    lambda = lambda + alpha*ones(x*y*z,1);
    A = A + alpha*eye(x*y*z,x*y*z);

    %write matrix to file
    csvwrite("3d.txt",A);
    %write eigenvalues to file
    csvwrite("eigs.txt",lambda);
    fileID = fopen('dim.txt','w');
    fprintf(fileID,"3d");
    fclose(fileID);
else
    printf("invalid input...\n")
end
