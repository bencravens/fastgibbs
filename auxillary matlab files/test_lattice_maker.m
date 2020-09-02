%tests for lattice_maker and lattice maker 3d
%matrices taken from "A primer on space time modelling from a bayesian
%perspective" - David Higdon

%first a 1-d lattice with 5 sites
test_1 = [1 -1 0 0 0;-1 2 -1 0 0; 0 -1 2 -1 0; 0 0 -1 2 -1; 0 0 0 -1 1]
result = lattice_maker(1,5)
%is it the same as in the book?
fprintf("the same?\n")
test_1==result

%now a 2-d lattice with 9 sites
test_2 = [2 -1 0 -1 0 0 0 0 0; -1 3 -1 0 -1 0 0 0 0 ; 0 -1 2 0 0 -1 0 0 0; -1 0 0 3 -1 0 -1 0 0; 0 -1 0 -1 4 -1 0 -1 0; 0 0 -1 0 -1 3 0 0 -1; 0 0 0 -1 0 0 2 -1 0; 0 0 0 0 -1 0 -1 3 -1; 0 0 0 0 0 -1 0 -1 2]
result = lattice_maker(3,3)
%is it the same as in the book?
fprintf("the same?\n")
test_2==result

%%%%%%%%%%%%%%%%%%%% TEST 3D VERSION %%%%%%%%%%%%%%%%%%%%%%%%
% Use most basic 2x2x2 neighbourhood structure, with matrix computed by
% hand...

test_3 = [3 -1 -1 0 -1 0 0 0; -1 3 0 -1 0 -1 0 0; -1 0 3 -1 0 0 -1 0; 0 -1 -1 3 0 0 0 -1; -1 0 0 0 3 -1 -1 0; 0 -1 0 0 -1 3 0 -1; 0 0 -1 0 -1 0 3 -1; 0 0 0 -1 0 -1 -1 3]
result = lattice_maker_3d(2,2,2)
%is it the same?
fprintf("the same?\n")
test_3==result

%make 5x5x5 lattice
W = lattice_maker_3d(5,5,5)
writematrix(W)
size(W)
