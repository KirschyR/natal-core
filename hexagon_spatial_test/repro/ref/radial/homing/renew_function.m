function j1 = renew(ar,mats,fitness)
% fitness is 1*5 mat
% ar: g*a*m*n
% j1: g*1*m*n
% 输出包含了female的fitness cost，但是female和male总数都当1去算的，考虑了male的fitness降低求偶几率
mat_dd=mats(1:5,:);
mat_dw=mats(6:10,:);
mat_ww=mats(11:15,:);
mat_rw=mats(16:20,:);
mat_dr=mats(21:25,:);
a=size(ar,2);
m=size(ar,3);
n=size(ar,4);
F=8; M=a-2-F;
cord_m1=3;
cord_f1=2+M+1;
cord_males = cord_m1:(cord_f1-1);
cord_females = cord_f1:a;

sum_female=reshape(sum(ar(:,cord_females,:,:),[1,2]),[1 1 1 m n]);
male=ar.*fitness;
sum_male=reshape(sum(male(:,cord_males,:,:),[1,2]),[1 1 1 m n]);

new_ar = reshape(ar,5,1,a,m,n);
newmale = reshape(male,5,1,a,m,n);

rww = reshape(squeeze(pagemtimes(pagemtimes(sum(new_ar(:,:,cord_females,:,:),3)./sum_female,'transpose',mat_ww,'none'),sum(newmale(:,:,cord_males,:,:),3)./sum_male)),[1 1 m n]); % m*n
rdw = reshape(squeeze(pagemtimes(pagemtimes(sum(new_ar(:,:,cord_females,:,:),3)./sum_female,'transpose',mat_dw,'none'),sum(newmale(:,:,cord_males,:,:),3)./sum_male)),[1 1 m n]);
rdd = reshape(squeeze(pagemtimes(pagemtimes(sum(new_ar(:,:,cord_females,:,:),3)./sum_female,'transpose',mat_dd,'none'),sum(newmale(:,:,cord_males,:,:),3)./sum_male)),[1 1 m n]);
rrw = reshape(squeeze(pagemtimes(pagemtimes(sum(new_ar(:,:,cord_females,:,:),3)./sum_female,'transpose',mat_rw,'none'),sum(newmale(:,:,cord_males,:,:),3)./sum_male)),[1 1 m n]);
rdr = reshape(squeeze(pagemtimes(pagemtimes(sum(new_ar(:,:,cord_females,:,:),3)./sum_female,'transpose',mat_dr,'none'),sum(newmale(:,:,cord_males,:,:),3)./sum_male)),[1 1 m n]);
j1(1,1,:,:)=rdd;
j1(2,1,:,:)=rdw;
j1(3,1,:,:)=rww;
j1(4,1,:,:)=rrw;
j1(5,1,:,:)=rdr;

end