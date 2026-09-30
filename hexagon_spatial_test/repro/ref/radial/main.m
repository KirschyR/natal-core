%===========================================================================
%                      Population Dynamics Simulation
%===========================================================================
% -- HOW TO USE --
% 1.  Prerequisites & Directory Structure:
%     - This function requires three external files to be correctly placed.
%     - 'drive_params.mat': Must be in the same directory as this script.
%     - 'renew_function.m': Must be in the same directory or on the MATLAB path.
%     - 'mig.mat': Must be in the PARENT directory ('../') relative to this script.
% 2.  Function Call:
%     - This is a function and must be called from another script or from
%       the MATLAB command window. It is not a standalone script.
%     - Call with three numeric arguments: t (release duration), h (release
%       quantity), and circle_r (release radius).
%     - Example: result = main(52, 10000, 50);
%
% -- KEY PARAMETERS --
%   - t: Release duration in weeks.
%   - h: Number of individuals released per week.
%   - circle_r: Radius of the circular release area (in grid units).
%   - m, n: Dimensions of the spatial grid (800x800).
%   - F, M: Lifespan of females (8 weeks) and males (5 weeks).
%   - lambda: Population growth rate factor.
%   - eggs: Number of eggs per female per week (40).
%   - total_time: Total simulation duration (261 weeks).
%
% -- OUTPUTS --
% 1.  CSV File Generation:
%     - A directory named 'output' is created in the current folder.
%     - A CSV file is saved inside 'output', named after the input
%       parameters (e.g., '52,10000,50.csv').
% 2.  CSV Content:
%     - The file contains data logged at specified 'collect_week' intervals.
%     - Each row contains six values:
%       [time_step, h, circle_r, cover_diameter, cover_area, cumulative_released]
%
% -- POTENTIAL BUGS & ISSUES --
% 1.  Missing Files: The script will fail with an 'Unrecognized function or
%     variable' error if 'drive_params.mat', 'mig.mat', or
%     'renew_function.m' are missing or misplaced.
% 2.  Write Permissions: An error will occur if the script does not have
%     permission to create the './output' directory.
% 3.  Matrix Dimension Mismatch: Errors during the main simulation loop often
%     indicate that the dimensions of your input data do not match what
%     the script expects.
% 4.  Unused Return Value: The 'result' variable is only updated if the
%     simulation terminates early. In a complete run, it remains [0,0,0];
%     the primary output is the CSV file.
%===========================================================================
function result=main(t,h,circle_r,m,n)
% NOTE (repro patch): optional m,n appended to parameterize the arena for
% golden generation. The original hardcoded `m=800;n=800;`. No other change.
load drive_params.mat
load ../mig.mat
if nargin < 4 || isempty(m), m = 800; end
if nargin < 5 || isempty(n), n = m; end
F = 8;
M = 5;
ages_length=2+M+F;
cord_m1=2+1;
cord_f1=2+M+1;
lambda = 5;
cord_release_age = [cord_m1,cord_f1];
total_time= 261;
startdate = 1;
drop_time=[1:t];
vis_time=-1;
result=[0,0,0];
collect_week = [104,261];
mkdir("./output");
filename=['./output/',num2str(t),',',num2str(h),',',num2str(circle_r),'.csv'];
fid = fopen(filename, 'a');
fclose(fid);
vec1=zeros([n,1]);
adult_index = 3:ages_length;
midj = round((m+1)/2);
midi=round((n+1)/2);
midy=midj-1/2*midi;
midx=sqrt(3)/2*midi;
eggs=40;
sf=(F+1)/2;
ages_f = 1/sf*(1:-1/F:1/F);
ages_m = 1/sf*(1:-1/M:1/M);
age = [eggs, 4/(F+1), ages_m, ages_f];
ages=repmat(age',1,m,n);
data = zeros(gn,ages_length,m,n);
mat = zeros(2+F+M);
mat(cord_m1,2) = 0.5;
mat(cord_f1,2) = 0.5;
for i=1:(F-1)
    mat(cord_f1+i,cord_f1+i-1) = 1-1/(F-i+1);
end
for i=1:(M-1)
    mat(cord_m1+i,cord_m1+i-1) = 1-1/(M-i+1);
end
largemat = repmat(mat,[1,1,m,n]);
terrain=ones(m,n,12);
month=day2date(startdate);
terrain0=terrain(:,:,month);
terrain2 = repmat(terrain0,[1,1,ages_length]);
terrain2=reshape(terrain2,[ages_length,m,n]);
data(cord_pure_w,:,:,:) = ages.*terrain2;
release_mat=zeros(size(data));
for i=1:m
    for j=1:n
        x=sqrt(3)/2*i;
        y=j-1/2*i;
        if ((midx-x)^2+(midy-y)^2)<=circle_r^2
            release_mat(cord_release_genotype,cord_release_age,i,j)=h/2;
        end
    end
end
release_num=sum(release_mat,"all");
release_sum=0;
std_pop=sum(data(:,adult_index,midi,midj),'all');
for t = 1:round(total_time)
    day =startdate+ t*7
    month=day2date(day);
    terrain1=terrain(:,:,month);
    if ismember(t,drop_time)
        data=data+release_mat;
        release_sum=release_num+release_sum;
    end
    if sum(data(carrier_index,adult_index,:,:),"all")< 1e-5
        result=[0,0,release_sum];
        return
    end
    j1 = sum(data(:,1,:,:),1);
    j2 = sum(data(:,2,:,:),1);
    r = (j1+5*j2)/(ages(1)+5*ages(2));
    s1 = lambda/((lambda-1)*r+1)*2/sf/eggs;
    largemat(2,1,:,:) = s1;
    total_female_num = sum(sum(data(:,cord_f1:end,:,:),1),2);
    newborns = eggs * total_female_num .* renew_function(data,mats,fitness);
    data = pagemtimes(data,'none',largemat,'transpose');
    data(:,1,:,:) = newborns;
    for j = adult_index
        for i = 1:size(data,1)
            data(i,j,:,:) = imfilter(squeeze(data(i,j,:,:)),mig_matrix,'same','replicate');
        end
    end
    data(isnan(data))=0;
    if fitness==111
        if ismember(t,collect_week)
            S=1-sum(sum(data(:,adult_index,:,:),2),1)./std_pop;
            for i1=1:n
                vec1(i1)=1-sum(sum(data(:,adult_index,round(m/2),i1),2),1)/std_pop;
            end
            cover_diameter=sum(vec1>=0.9);
            cover_area=sum(S>=0.9,'all');
            fid = fopen(filename, 'a');
            fprintf(fid, '%f\n', [t,h,circle_r,cover_diameter,cover_area,release_sum]);
            fclose(fid);
        end
    else
        if ismember(t,collect_week)
            S=sum(sum(data(carrier_index,adult_index,:,:),2),1)./sum(sum(data(:,adult_index,:,:),2),1);
            for i1=1:n
                vec1(i1)=sum(sum(data(carrier_index,adult_index,round(m/2),i1),2),1)/sum(sum(data(:,adult_index,round(m/2),i1),2),1);
            end
            cover_diameter=sum(vec1>=0.9);
            cover_area=sum(S>=0.9,'all');
            fid = fopen(filename, 'a');
            fprintf(fid, '%f,', [t,h,circle_r,cover_diameter,cover_area,release_sum]);
            fclose(fid);
        end
    end
end
end
function month= day2date(daynum)
months = [31 28 31 30 31 30 31 31 30 31 30 31];
month = 1;
daynum=mod(daynum,365);
while daynum > months(month)
    daynum = daynum - months(month);
    month = month + 1;
end
end