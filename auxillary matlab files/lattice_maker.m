%    
%                n_{i} if i=j
%    W_{i,j} = { -1 if i is next to j
%                0 otherwise    
%
function W = lattice_maker(m,n,bump)
    %take in as input the dimensions of the lattice...
    fprintf("Lattice is %d x %d\n",m,n)

    %build lattice
    lattice = zeros(m,n);

    %build corresponding precision matrix W
    W = zeros(m*n,m*n);

    %now populate lattice with numbers
    count=0;
    for i=1:m
        for j=1:n
            count=count+1;
            lattice(i,j)=count;
        end
    end
    
    %now detect nearest neighbours
    for i=1:m
        for j=1:n
            %row for nearest neighbours
            tempvec = zeros(1,n*m);
            %count the number of nearest neighbours
            nncount = 0;

            %is there a neighbour to the left?
            try
                neighbour = lattice(i,j-1);
                %there is one, give a -1 in the row due to a difference
                tempvec(neighbour) = -1;
                nncount = nncount + 1;
            catch
                %'no left neighbour'
            end
            %is there a neighbour above?
            try
                neighbour = lattice(i-1,j);
                tempvec(neighbour) = -1;
                nncount = nncount + 1;
            catch
                %'no neighbour above'
            end
            %is there a neighbour to the right?
            try
                neighbour = lattice(i,j+1);
                tempvec(neighbour) = -1;
                nncount = nncount + 1;
            catch
                %'no neighbour to the right'
            end
            %is there a neighbour below?
            try
                neighbour = lattice(i+1,j);
                tempvec(neighbour) = -1;
                nncount = nncount + 1;
            catch
                %'no neighbour below'
            end
            %add entry for the location we are at (i=j so this will be n_i)
            tempvec(lattice(i,j))=nncount;
            W(lattice(i,j),:) = tempvec;
        end
    end
    W = W + bump*diag(diag(ones(n*m,n*m)))
    csvwrite("2d_test.txt", W)
end