%{
===========================================================================
                      Drive Parameter Generation Script
===========================================================================

-- HOW TO USE --
1.  Configuration:
    - Adjust the core parameters at the top row as needed.
    et is transmission efficient
    ea is conversion rate
    dw_fertility is fertility of drive heterozygotes 

2.  Execution:
    - Run this script in MATLAB. It will generate gene drive parameters
      based on the configured parameters.

3.  Output:
    - The script generates a file named 'drive_params.mat'.
    - It contains all the necessary variables for a simulation run,
      including the reaction matrices, fitness costs, and genotype indices.

4.  Genotypes:dd,dw,ww,rw,dr
===========================================================================
%}
gcr=0;
dc=1;
herr=0;
ddfitness=1;
mat_dd=[1,dc/2 + 1/2,0,0,1/2;dc/2 + 1/2,dc*(dc/2 + 1/2) - (dc/4 + 1/4)*(dc + gcr - 1) + gcr*(dc/4 + 1/4),0,0,dc/4 + 1/4;0,0,0,0,0;0,0,0,0,0;1/2,dc/4 + 1/4,0,0,1/4];
mat_dw=[0,-((herr - 1)^2*(dc + gcr - 1))/2,(herr - 1)^2,(herr - 1)^2/2,0;(herr/2 - 1/2)*(dc + gcr - 1),dc*(herr/2 - 1/2)*(dc + gcr - 1) - (dc + gcr - 1)*((herr/2 - 1/2)*(dc + gcr - 1) + (dc*(herr - 1)^2)/2 - gcr*(herr/4 - 1/4)) + gcr*(herr/4 - 1/4)*(dc + gcr - 1),(herr/2 - 1/2)*(dc + gcr - 1) + dc*(herr - 1)^2 - gcr*(herr/2 - 1/2),(herr/4 - 1/4)*(dc + gcr - 1) + (dc*(herr - 1)^2)/2 - gcr*(herr/4 - 1/4),(herr/4 - 1/4)*(dc + gcr - 1);1,dc/2 + 1/2,0,0,1/2;1/2,dc/4 + 1/4,0,0,1/4;0,(herr/4 - 1/4)*(dc + gcr - 1),1/2 - herr/2,1/4 - herr/4,0];
mat_ww=[0,0,0,0,0;0,((herr - 1)^2*(dc + gcr - 1)^2)/4,-((herr - 1)^2*(dc + gcr - 1))/2,-((herr - 1)^2*(dc + gcr - 1))/4,0;0,1/2 - gcr/2 - dc/2,1,1/2,0;0,1/4 - gcr/4 - dc/4,1/2,1/4,0;0,0,0,0,0];
mat_rw=[0,0,0,0,0;0,(gcr*(herr/4 - 1/4) - (herr*(herr - 1)*(dc + gcr - 1))/2)*(dc + gcr - 1) + gcr*(herr/4 - 1/4)*(dc + gcr - 1),herr*(herr - 1)*(dc + gcr - 1) - gcr*(herr/2 - 1/2),(dc + gcr - 1)*(herr/4 + (herr*(herr - 1))/2 - 1/4) - gcr*(herr/4 - 1/4),(herr/4 - 1/4)*(dc + gcr - 1);0,gcr/2,0,1/2,1/2;0,1/4 - dc/4,1/2,1/2,1/4;0,(herr/4 - 1/4)*(dc + gcr - 1),1/2 - herr/2,1/4 - herr/4,0];
mat_dr=[0,gcr/2 + ((herr - 1)^2/2 - 1/2)*(dc + gcr - 1),1 - (herr - 1)^2,1 - (herr - 1)^2/2,1/2;gcr/2 - (herr*(dc + gcr - 1))/2,gcr*(dc/2 + gcr/2 - (herr/4 + 1/4)*(dc + gcr - 1)) + (dc + gcr - 1)*(dc*((herr - 1)^2/2 - 1/2) - gcr*(herr/4 + 1/4) + (herr*(dc + gcr - 1))/2) + dc*(gcr/2 - (herr*(dc + gcr - 1))/2),(gcr*herr)/2 - dc*((herr - 1)^2 - 1) - (herr*(dc + gcr - 1))/2,gcr*(herr/4 + 1/4) - (herr/4 + 1/4)*(dc + gcr - 1) - dc*((herr - 1)^2/2 - 1),dc/2 + gcr/2 - (herr/4 + 1/4)*(dc + gcr - 1);0,0,0,0,0;1/2,dc/4 + 1/4,0,0,1/4;1/2,dc/2 + gcr/2 - (herr/4 + 1/4)*(dc + gcr - 1),herr/2,herr/4 + 1/4,1/2];
mat_dd = mat_dd';
mat_dw = mat_dw';
mat_ww = mat_ww';
mat_rw = mat_rw';
mat_dr = mat_dr';
F = sqrt(ddfitness);
genotype_fitness=[F^2,F,1,1,F]';
fitness_cost = reshape(genotype_fitness,[1,5]);
fitness_cost = repmat(fitness_cost,[5,1]);
mat_dd = mat_dd.*fitness_cost;
mat_dw = mat_dw.*fitness_cost;
mat_ww = mat_ww.*fitness_cost;
mat_rw = mat_rw.*fitness_cost;
mat_dr = mat_dr.*fitness_cost;
mats = [mat_dd;mat_dw;mat_ww;mat_rw;mat_dr];
fitness=genotype_fitness;
gn=5;
carrier_index = [1,2,5];
cord_pure_w = 3;
cord_release_genotype = 1;
save("drive_params.mat","gn","cord_pure_w","carrier_index","fitness","mats","cord_release_genotype")
