%{
===========================================================================
                      Migration Matrix Generation Function
===========================================================================

-- HOW TO USE --
1.  Function Call:
    - This is a function and must be called from another script or from
      the MATLAB command window. It is not a standalone script.
    - It requires one numeric input, which represents the average dispersal.
    - Example: get_mig_matrix(0.5);

2.  Execution:
    - When called, the function calculates a migration matrix
      based on the provided average dispersal value.

3.  Outputs:
    - A file named 'mig.mat' is created or overwritten in the current
      working directory.
    - This file contains two variables: 'mig_matrix' (the calculated
      matrix) and 'avd' (the input value used for the calculation).

-- POTENTIAL BUGS & ISSUES --
1.  File Overwriting: The function will automatically overwrite any
    existing 'mig.mat' file in the current directory without any warning.

2.  Input Value: The function assumes the input 'avd' is a positive
    number for a physically meaningful calculation of sigma. Using a zero
    or negative value may lead to unexpected results or errors.
===========================================================================
%}
function get_mig_matrix(avd)
mu = 0;
% sigma = avd/sqrt(pi/2);
sigma = 10.437;
avd = sigma * sqrt(pi/2);
max_dist=25;

kernel_len = 2*ceil(max_dist*2/sqrt(3))+1;
mig_matrix=zeros(kernel_len);
[centerx,centery] = deal(ceil(max_dist*2/sqrt(3))+1);
for i = 1:kernel_len
    for j = 1:kernel_len
        xdist = i-centerx;
        ydist = j-centery;
        euc_dist = sqrt(xdist^2+ydist^2-xdist*ydist);
        if euc_dist<=max_dist
            mig_matrix(i,j) = normpdf(euc_dist,mu,sigma);
        end
    end
end

mig_matrix = mig_matrix / sum(mig_matrix,"all");
save("mig.mat","mig_matrix","avd")
end
