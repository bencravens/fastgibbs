%    
%                n_{i} if i=j
%    W_{i,j} = { -1 if i is next to j
%                0 otherwise    
%
function W = lattice_maker_3d(m,n,z)
    %take in as input the dimensions of the lattice...
    fprintf("Lattice is %d x %d x %d\n",m,n,z)

    %build lattice
    lattice = zeros(m,n,z);

    %build corresponding precision matrix W
    W = zeros(m*n*z,m*n*z);

    %now populate lattice with numbers
    count=0;
    for k=1:z
        for i=1:m
            for j=1:n
                count=count+1;
                lattice(i,j,k)=count;        
            end
        end
    end
    
    %now detect nearest neighbours
    for k=1:z
        for i=1:m
            for j=1:n
                %row for nearest neighbours
                tempvec = zeros(1,n*m*z);
                %count the number of nearest neighbours
                nncount = 0;

                %is there a neighbour to the left?
                try
                    neighbour = lattice(i,j-1,k);
                    %there is one, give a -1 in the row due to a difference
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no left neighbour'
                end
                %is there a neighbour up?
                try
                    neighbour = lattice(i-1,j,k);
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no neighbour up from site'
                end
                %is there a neighbour to the right?
                try
                    neighbour = lattice(i,j+1,k);
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no neighbour to the right'
                end
                %is there a neighbour down?
                try
                    neighbour = lattice(i+1,j,k);
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no neighbour down from site'
                end
                % is there a neighbour in a lattice layer above?
                try
                    neighbour = lattice(i,j,k-1);
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no neighbour above point'
                end
                % is there a neighbour in a lattice layer below?
                try 
                    neighbour = lattice(i,j,k+1);
                    tempvec(neighbour) = -1;
                    nncount = nncount + 1;
                catch
                    %'no neighbour below point'
                end
                %add entry for the location we are at (i=j so this will be n_i)
                tempvec(lattice(i,j,k))=nncount;
                W(lattice(i,j,k),:) = tempvec;
            end
        end
    end
end