% This file estimates probabilities using simple naive bayes algorithm
% For each feature look probability for current class of the feature being
% the same, i.e.
% P(C|X) = P(X|C)P(C)/P(X)
% where P(X|C) = (number of rows where X = x and C = c) / (number of rows
% where X = x)

X = XTrain;
y = yTrain;

% Preprocessing
X = X > 0;

[nTrain,f] = size(XTrain);
class = unique(y);
nClass = size(class, 1);

intervals = 1 : floor(nTrain/10) : nTrain;
intervals(size(intervals, 2)) = nTrain;
perm = randperm(nTrain);

L = 0;
N = 0;
z = zeros(nTrain, 1);
for i = 1:size(intervals, 2)-9
    idx2 = perm(intervals(i):intervals(i+1));
    idx1 = setdiff(1:nTrain, idx2);
    
    XX = X(idx1, :);
    yy = y(idx1, :);
    WW = X(idx2, :);
    
    prior = zeros(nClass, 1);
    likelihood = zeros(nClass, f);
    
    for j = 1:nClass
        idx = find(y == j);
        n = size(idx, 1);
        prior(j, 1) = n / nClass;
        likelihood(j, :) = sum(X(idx, :)) / n;
    end
    
    p = zeros(size(idx2, 1), nClass);
    for k = 1:size(WW, 1)
        for j = 1:nClass
            
        end
    end
    repmat(prod, 
    
    for j = idx2
        L = L - log(z(j, y(j)));
    end
    N = N + size(idx2, 2);
end

L = L/N;

% class = unique(yTrain);
% nClass = size(class, 1);
% z = zeros(nTrain, nClass);
% for j = 1:nClass
%     ind = find(yy == class(j));
%     z(idx2, j) = (sum(K(ind, :)) ./ sum(K))';
% end