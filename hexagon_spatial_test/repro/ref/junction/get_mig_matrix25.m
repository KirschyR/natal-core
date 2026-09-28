function mig_matrix = hex_migmat(sigma)
mu = 0;
% sigma = 8.1381;

% 下面是scw的部分 被我改的面目全非
max_dist=25;
kernel_len = 2*max_dist+1;
mig_matrix=zeros(kernel_len);
[centerx,centery] = deal(max_dist+1);
distance = zeros(size(mig_matrix));
for i = 1:kernel_len
    for j = 1:kernel_len
        xdist = i-centerx;
        ydist = j-centery;
        euc_dist = sqrt(xdist^2+ydist^2-xdist*ydist); % 余弦定理算欧氏距离
        mig_matrix(i,j) = normpdf(euc_dist,mu,sigma);
        distance(i,j) = euc_dist;
    end
end

mig_matrix = mig_matrix / sum(mig_matrix,"all");

avd = sum(mig_matrix.*distance,"all")

% figure;
% imagesc(mig_matrix)
% colorbar
% figure

end