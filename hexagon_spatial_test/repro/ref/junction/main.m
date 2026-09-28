function return_list=main(actual_length,draw,print,mode_params,drive_name,sigma)

% drive name (string) in:
% homing homingsup tare 2ltare tade10 tade11 tadesup cifab wolbachia
load(drive_name+"/dp.mat")

% 暗号：
% dp.fitness = 111: suppression (calculate speed criterion and sum1 sum2 differently)
% draw=1,print=0(正常人不会这么搞吧): test for ror mn, 每周输出有drive的最大mn

% return:
% speed: wavespeed

m = ceil(actual_length*(1+1/sqrt(3)));
n = ceil(actual_length*2/sqrt(3));
% detect_cord = round(actual_length/(2*sqrt(3)))

%genotype;m;n
% 第一个维度是y，第二个维度是x

% dp %drive_params
% drive params包含：gn, mat_dd和其他矩阵(经过fitness和其他参数运算后的), carrier index,
% cord_pure_d, cord_pure_w, genotypes


gn=dp.gn;
carrier_index = dp.carrier_index;
cord_release_genotype = dp.cord_pure_d; % for suppression, it's cord_dw
cord_pure_w = dp.cord_pure_w;
% dp.mats, dp.fitness在调用renew的时候才用到
% dp.genotypes在调用draw_3d_plot的时候才用到


return_list=999999;
speed_criterion = 0.8;
if dp.fitness(1)==111 % suppression
    speed_criterion = -0.3; % wt < 0.3
end
total_time = 1e6;
if mode_params(1) == 0 % speed
    mode = "speed";
end

if mode_params(1) == 1 % shape
    mode = "shape";
end

checkpoint1 = mode_params(2);
checkpoint2 = mode_params(3);
distance=checkpoint2-checkpoint1;
flag1 = 0; flag2 = 0;
no_move_freq = 0.3; %检测carrier freq是no_move_freq的最远的点
no_move_crit = 350; %所有m行都no_move_crit generations no move -> end
no_drive_freq = 0.05;
no_drive_crit = 20;
last_no_move = zeros(1,n); %同时检测n行

mig_matrix=get_mig_matrix25(sigma);

lambda = 5;

%% display settings
print_every = 5;
draw_every = 5;


% %init

data = zeros(gn,m,n);
for i = 1:m
    for j = 1:n
        if j>=2*i-2/5*actual_length
            data(cord_release_genotype,i,j) = 1;
        else
            data(cord_pure_w,i,j) = 1;
        end
    end
end



%主循环

for t = 0:round(total_time)
    if print
        if mod(t,print_every) == 0
            disp(['time=',num2str(t)]);
        end
    end

    % reaction
    data = data + renew_function(data,dp.mats,lambda);

    % diffusion
    newdata=zeros(size(data));
    for i = 1:gn
        newdata(i,:,:) = imfilter(squeeze(data(i,:,:)),mig_matrix,'replicate','same');
    end
    data = newdata;
    %draw
    if draw
        if mod(t,draw_every) == 0

            if draw==1
                draw_3d_plot(data,dp.genotypes);
                drawnow;

            elseif draw==2
                plot_row = round(n/2);
                plot_data = squeeze(data(:,:,plot_row)); % 25*300
                carriers = squeeze(sum(plot_data(carrier_index,:),1));
                allinds = squeeze(sum(plot_data(:,:),1));
                plot(1:m,carriers./allinds,'LineWidth',2);
                xlim([actual_length/(2*sqrt(3)),m-actual_length/(2*sqrt(3))])
                title("wave shape")
                drawnow

            end
        end

    end

    %exception monitor

   % sum at checkpoint 1
    if flag1 == 0
    if dp.fitness(1) == 111
        sum1 = 1 - data(cord_pure_w,checkpoint1,round(n/2)) / sum(data(:,checkpoint1,round(n/2)),1); %suppression, 负的因为后面是大于，criterion也是负的
    else
        sum1 = sum(data(carrier_index,checkpoint1,round(n/2)),1) / sum(data(:,checkpoint1,round(n/2)),1);
    end
    if  sum1>speed_criterion
        flag1=t-1+(speed_criterion-last_carrierfreq(1))/(sum1-last_carrierfreq(1));
    end
    last_carrierfreq(1) = sum1;
    end

    % sum at checkpoint 2
    if dp.fitness(1) == 111
        sum2 = 1 - data(cord_pure_w,checkpoint2,round(n/2)) / sum(data(:,checkpoint2,round(n/2)),1); %suppression, 负的因为后面是大于，criterion也是负的
    else
        sum2 = sum(data(carrier_index,checkpoint2,round(n/2)),1) / sum(data(:,checkpoint2,round(n/2)),1);
    end
    if  sum2>speed_criterion && flag2==0
        flag2=t-1+(speed_criterion-last_carrierfreq(2))/(sum2-last_carrierfreq(2));
        return_list=distance/(flag2-flag1);
        return
    end
    last_carrierfreq(2) = sum2;

%     detect no move
    if mod(t,no_move_crit)==0
        new_no_move = zeros(1,n);
        for i = 1:n
            if dp.fitness(1)==111 % suppression
                detect_row = 1-squeeze(data(cord_pure_w,:,i)) ./ squeeze(sum(data(:,:,i),1));
            else
                detect_row = squeeze(sum(data(carrier_index,:,i),1)) ./ squeeze(sum(data(:,:,i),1));
            end
            lastone = find(detect_row>no_move_freq,1,'last');
            if isempty(lastone)
                disp("fail to detect no move")
                return
            end
            new_no_move(i) = lastone;
        end
        if isnan(new_no_move)
            break
        end
        if all(new_no_move==last_no_move)
            return_list=0;
            disp("no move")
            return;
        else
            last_no_move = new_no_move;
        end
    end




    % detect no drive
    if mod(t,no_drive_crit)==0
        carriers = squeeze(sum(data(carrier_index,:,:),"all"));
        allinds = squeeze(sum(data,"all"));
        carrier_freq = carriers/allinds;
        if carrier_freq < no_drive_freq
            return_list = 0;
            disp("no drive")
            return
        end
    end

    

end
end



