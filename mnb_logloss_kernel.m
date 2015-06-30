lr = 0.000000000000001;
nIter = 1000;
d = 2;

[nTrain,f] = size(XTrain);
class = unique(yTrain);
nClass = size(class, 1);

% preprocessing
mu = mean(XTrain);
s = std(XTrain);
X = bsxfun(@rdivide, bsxfun(@minus, XTrain, mu), s);
y = yTrain;

X = XTrain(1:100:size(X, 1),:);
nTrain = size(X,1);

X = [ones(nTrain, 1) X];
y = yTrain(1:100:size(XTrain, 1), :);

[nTrain,f] = size(X);
class = unique(y);
nClass = size(class, 1);

intervals = 1 : floor(nTrain/10) : nTrain;
intervals(size(intervals, 2)) = nTrain;
perm = randperm(nTrain);

L = 0;
N = 0;
z = zeros(nTrain, nClass);
for i = 1:size(intervals, 2)-1
    idx2 = perm(intervals(i):intervals(i+1));
    idx1 = setdiff(1:nTrain, idx2);
    
    XX = X(idx1, :);
    yy = y(idx1, :);
    WW = X(idx2, :);
    
    n = size(XX, 1);
    f = n;
    W = zeros(f, nClass);
    
    for iter = 1:nIter
        S = 0;
        grad = zeros(f, nClass);
        for d = 1:n
            K = (XX(d, :) * XX').^2;
            p = K * W;
            p = exp(bsxfun(@minus, p, max(p, [], 2)));
            p = bsxfun(@rdivide, p, sum(p, 2));

            for c = 1:nClass
                grad(:, c) = grad(:, c) + sum(bsxfun(@times, K, p(:,c) - (yy(d) == c)))'; 
            end
            
            S = S - log(p(1, yy(d)));
        end
        W = W - lr * grad / n;
        
        fprintf('Iteration %d: %f\n', iter, S/n);
    end
        
    m = size(WW,1);
    
    Q = 0;
    for d = 1:m
        K = (WW(d, :) * XX').^2;
        p = K * W;
        p = exp(bsxfun(@minus, p, max(p, [], 2)));
        z(idx2(d), :) = bsxfun(@rdivide, p, sum(p, 2));
        
        Q = Q - log(z(d, y(d)));
    end

    L = L + Q;
    N = N + m;
    
    fprintf('Fold %d: min = %f, accuracy = %f\n', i, S/n, Q/m);
end

L = L/N;
fprintf('Accuracy: %f\n', L);
