% nTest = size(XTest, 1);

X = XTrain;
y = yTrain;

% m = mean(XTrain);
% s = std(XTrain);


% X = [bsxfun(@rdivide, bsxfun(@minus, XTrain, m), s)];
% y = yTrain;
% W = [bsxfun(@rdivide, bsxfun(@minus, W, m), s)];

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

    K = exp(kernel(XX, WW, 2, 'rbf'));
    
    for j = 1:nClass
        ind = find(yy == class(j));
        z(idx2, j) = sum(K(ind, :)) ./ sum(K);
    end
    
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