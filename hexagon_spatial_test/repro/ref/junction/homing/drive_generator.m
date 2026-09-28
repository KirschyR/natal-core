function [drive_name, dp] = drive_generator(gcr,dc,herr,ddfitness)
drive_name = "homing";

% generate dp for each simulation
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
mat_dd = mat_dd.*fitness_cost; % for females, fitness cost affects fertility
mat_dw = mat_dw.*fitness_cost;
mat_ww = mat_ww.*fitness_cost;
mat_rw = mat_rw.*fitness_cost;
mat_dr = mat_dr.*fitness_cost;
% 1dd 2dw 3ww 4rw 5dr


dp.mats = [mat_dd;mat_dw;mat_ww;mat_rw;mat_dr]; % 25*5
dp.fitness=genotype_fitness;
dp.gn=5;
dp.carrier_index = [1,2,5];
dp.cord_pure_d = 1;
dp.cord_pure_w = 3;
dp.genotypes = {... %colormode is 255
    'drive',[1:5],[1,0.5,0,0,0.5],[147,255,189];
    'wildtype',[1:5],[0,0.5,1,0.5,0],[155,249,255];
    'resistance',[1:5],[0,0,0,0.5,0.5],[255,246,118];
    };
end