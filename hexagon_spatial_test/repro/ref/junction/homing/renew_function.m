function d_ar = renew(ar,mats,lambda)
% ar: g*m*n
% d_ar: g*m*n
% mat_已经乘了female的fitness cost

gn = size(ar,1);
m = size(ar,2);
n = size(ar,3);
N = reshape(sum(ar,1),[1,m,n]); % 1*m*n
new_ar = reshape(ar,gn,1,m,n);

% female' * mat * male
% 1*gn     gn*gn   gn*1
rcat = [...
    reshape(squeeze(pagemtimes(pagemtimes(new_ar,'transpose',mats(1:5,:),'none'),new_ar)),[1,m,n]);
    reshape(squeeze(pagemtimes(pagemtimes(new_ar,'transpose',mats(6:10,:),'none'),new_ar)),[1,m,n]);
    reshape(squeeze(pagemtimes(pagemtimes(new_ar,'transpose',mats(11:15,:),'none'),new_ar)),[1,m,n]);
    reshape(squeeze(pagemtimes(pagemtimes(new_ar,'transpose',mats(16:20,:),'none'),new_ar)),[1,m,n]);
    reshape(squeeze(pagemtimes(pagemtimes(new_ar,'transpose',mats(21:25,:),'none'),new_ar)),[1,m,n]);
    ];

d_ar = rcat*lambda./((lambda-1)*N+1)./N - ar.*N;

end