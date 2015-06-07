
loadStuff;
%Xsim = Xsim';
% = data';
d = 94;
n = 61878;
m = 2;

% initiallizing these values
Xsim = bsxfun(@minus, Xsim, mean(Xsim));
mu = mean(Xsim');
W = ones(d,m);
PHI = eye(d);
C = W*W' + PHI;

%variable for x - mu so I don't have to keep calculating it
x_mu = bsxfun(@minus, Xsim', mu)'; 

iter = 100;
p = zeros(1,iter); %log likelihood probability
for i = 1 : iter    
    %---- E step------------------------------------
    MUz = W'*C^-1*(x_mu); % expectation of z|x
    SIGMAz = 1 - W'*(C^-1)*W;  % covariance of z|x
    
    %---- M step------------------------------------
    % Calculate Wnew
    Wnew = x_mu*MUz'*(n*SIGMAz + MUz*MUz')^-1;%sum(bsxfun(@times, x_mu,MUz),2)*(n*SIGMAz + MUz*MUz')^-1;
    
    % Calculate sig2new
    t1 = zeros(d,d);
    for j = 1:n
        t1 = t1 + x_mu(:,j)*x_mu(:,j)';
    end
    t2 = Wnew'*x_mu*MUz';%sum(bsxfun(@times, x_mu, MUz),2);
    PHInew = (1/(n))*diag(diag(t1))-diag(diag(t2));
    W = Wnew;
    PHI = PHInew;
    C = W'*W + PHI; 
    
   % p(i) = sum(log(mvnpdf(Xsim', mu, C))); 
end

figure(7)
title('2.e Loglikelihood for FA')
plot(p)
