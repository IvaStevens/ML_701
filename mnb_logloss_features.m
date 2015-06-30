lr = 0.1;
nIter = 200;

[nTrain,f] = size(XTrain);
class = unique(yTrain);
nClass = size(class, 1);

% preprocessing
mu = mean(XTrain);
s = std(XTrain);
X = bsxfun(@rdivide, bsxfun(@minus, XTrain, mu), s);
y = yTrain;

X = [X ones(nTrain, 1) zeros(nTrain, f*(f-1)/2)];
y = y;
k = f+2;
for i = 1:f-1
    for j = i+1:f
        X(:,k)   = X(:,i) .* X(:,j);
        k = k+1;
    end
end

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
%         fprintf('iter %d: %f\n', iter, S/n);
        
        for c = 1:nClass
            grad(:, c) = sum(bsxfun(@times, XX, p(:,c) - (yy == c))) / n; 
        end
        W = W - lr * grad;
    end
    
    S = 0;
    for d = 1:n
        S = S - log(p(d, yy(d)));
    end
    
    p = WW * W;
    p = exp(bsxfun(@minus, p, max(p, [], 2)));
    z(idx2, :) = bsxfun(@rdivide, p, sum(p, 2));

    Q = 0;
    for d = idx2
        Q = Q - log(z(d, y(d)));
    end
    L = L + Q;
    N = N + m;
    
    fprintf('Fold %d: min = %f, accuracy = %f\n', i, S/n, Q/m);
end

L = L/N;
fprintf('Accuracy: %f\n', L);
