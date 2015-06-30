function W = mnb_logloss_train(X, y, lr, nIter)
    [n,f] = size(X);
    class = unique(y);
    nClass = size(class, 1);

    W = zeros(f, nClass);
    grad = zeros(f, nClass);
    
    for iter = 1:nIter
        p = X * W;
        p = exp(bsxfun(@minus, p, max(p, [], 2)));
        p = bsxfun(@rdivide, p, sum(p, 2));
        
        for c = 1:nClass
            grad(:, c) = sum(bsxfun(@times, X, p(:,c) - (y == c))) / n; 
        end
        W = W - lr * grad;
        
        if mod(iter, 10) == 0
            S = 0;
            for d = 1:n
                S = S - log(p(d, y(d)));
            end
            fprintf('iter %d: %f\n', iter, S/n);
        end
    end
    
    S = 0;
    for d = 1:n
        S = S - log(p(d, y(d)));
    end
    fprintf('mnb_log_loss: %f\n', S/n);
end