lr = 0.1;
nIter = 100;

[nTrain,f] = size(XTrain);
class = unique(yTrain);
nClass = size(class, 1);

X = [ones(nTrain, 1) XTrain];
y = yTrain;

[nTrain,f] = size(X);
class = unique(y);
nClass = size(class, 1);

intervals = 1 : floor(nTrain/10) : nTrain;
intervals(size(intervals, 2)) = nTrain;
perm = randperm(nTrain);

L = 0;
N = 0;
z = zeros(nTrain, nClass);
for i = 1:size(intervals, 2)-9
    idx2 = perm(intervals(i):intervals(i+1));
    idx1 = setdiff(1:nTrain, idx2);
    
    XX = X(idx1, :);
    yy = y(idx1, :);
    WW = X(idx2, :);
    
    n = size(XX, 1);
    m = size(WW, 1);

    W = zeros(f, nClass);
    grad = zeros(f, nClass);
    
    for iter = 1:nIter
        p = XX * W;
        p = exp(bsxfun(@minus, p, max(p, [], 2)));
        p = bsxfun(@rdivide, p, sum(p, 2));
        
%         S = 0;
%         for d = 1:n
%             S = S - log(p(d, yy(d)));
%         end
%         S/n
        
        for c = 1:nClass
            grad(:, c) = sum(bsxfun(@times, XX, p(:,c) - (yy == c))) / n; 
        end
        W = W - lr * grad;
    end
    
    S = 0;
    for d = 1:n
        S = S - log(p(d, yy(d)));
    end
    S/n
    
    p = WW * W;
    p = exp(bsxfun(@minus, p, max(p, [], 2)));
    z(idx2, :) = bsxfun(@rdivide, p, sum(p, 2));

    for d = idx2
        L = L - log(z(d, y(d)));
    end
    N = N + m;
end

L = L/N;
