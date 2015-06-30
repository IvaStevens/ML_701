[nTrain,f] = size(XTrain);
class = unique(yTrain);
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
    
    XX = XTrain(idx1, :);
    yy = yTrain(idx1, :);
    WW = XTrain(idx2, :);
    
    z(idx2, :) = mnb_logloss(XX, yy, WW);

    Q = 0;
    for d = idx2
        Q = Q - log(z(d, yTrain(d)));
    end
    L = L + Q;
    N = N + size(WW, 1);
    
    [m pred] = max(z(idx2, :), [], 2);
    accuracy = sum(pred == yTrain(idx2,:))/size(WW,1);
    
    fprintf('Fold %d: accuracy = %f, logloss = %f\n', i, accuracy, Q/size(WW, 1));
end

L = L/N;
[m pred] = max(z, [], 2);
accuracy = sum(pred == yTrain)/N;

fprintf('accuracy: %f, logloss = %f\n', accuracy, L);
