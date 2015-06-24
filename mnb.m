X = XTrain;
y = yTrain;

[nTrain,f] = size(XTrain);
class = unique(y);
nClass = size(class, 1);

intervals = 1 : floor(nTrain/10) : nTrain;
intervals(size(intervals, 2)) = nTrain;
perm = randperm(nTrain);

L = 0;
N = 0;
z = zeros(nTrain, 1);
for i = 1:size(intervals, 2)-1
    idx2 = perm(intervals(i):intervals(i+1));
    idx1 = setdiff(1:nTrain, idx2);
    
    XX = X(idx1, :);
    yy = y(idx1, :);
    WW = X(idx2, :);

    alpha = 1000;
    beta = zeros(nClass);
    theta = zeros(nClass, f);
    
    beta = log(hist(yy, unique(yy)))/size(yy,1);
    for c = 1:nClass
        ind = find(yy == class(c));
        theta(c, :) = log((sum(XX(ind, :)) + alpha * 1) ./ (sum(sum(XX(ind, :))) + alpha * 93));
    end
    
    for c = 1:nClass
        z(idx2, c) = (beta(c) + sum(bsxfun(@times, WW, theta(c, :)), 2))';
    end
    
    maxp = max(z(idx2, :), [], 2);
    z(idx2, :) = bsxfun(@minus, z(idx2, :), maxp);
    z(idx2, :) = exp(z(idx2, :));
    sump = sum(z(idx2, :), 2);
    z(idx2, :) = bsxfun(@rdivide, z(idx2, :), sump);
    
    for c = idx2
        L = L - log(z(c, y(c)));
    end
    N = N + size(idx2, 2);
end

L = L/N;
