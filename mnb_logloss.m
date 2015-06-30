function z = mnb_logloss(XTrain, yTrain, XTest)
    lr = 10;
    nIter = 20;

    [nTrain,f] = size(XTrain);
    nTest = size(XTest, 1);
    class = unique(yTrain);
    nClass = size(class, 1);

    % preprocessing
    X = XTrain;
    W = XTest;
    % X = bsxfun(@rdivide, bsxfun(@minus, XTrain, mean(XTrain)), std(XTrain));
    % X = bsxfun(@minus, XTrain, mean(XTrain));
    % X = bsxfun(@rdivide, XTrain, sum(XTrain, 2));
    % W = bsxfun(@rdivide, XTest, sum(XTest, 2));
    y = yTrain;

    tX = [ones(nTrain, 1) X];
    tW = [ones(nTest, 1) W];
    
    X = zeros(nTrain, (f+1)^2);
    W = zeros(nTest, (f+1)^2);
    
    for i=1:f+1
        for j=1:f+1
            X(:,(f+1)*i+j) = tX(:,i) .* tX(:,j);
            W(:,(f+1)*i+j) = tW(:,i) .* tW(:,j);
        end
    end
    y = y;
    
    coef = mnb_logloss_train(X, y, lr, nIter);
    
    p = W * coef;
    p = exp(bsxfun(@minus, p, max(p, [], 2)));
    z = bsxfun(@rdivide, p, sum(p, 2));
end