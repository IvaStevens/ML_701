function K = kernel(XTrain, XTest, bw, kernel_type)
    % XTrain: training data, size of nTrain * f
    % XTest: test data, size of nTest * f
    % bw: bandwidth
    % K: kernel, sized of nTrain * nTest. 
    % K_i,j denotes the kernel value for ith training sample 
    % and jth test sample

    [nTrain,f] = size(XTrain);
    nTest = size(XTest,1);
    K = zeros(nTrain,nTest);
    if strcmp(kernel_type,'rbf')
        %% begin
        for i = 1:nTest
           K(:,i) = (2*pi)^(-f/2) * bw^(-f) * exp(- 0.5 * sum(((XTrain - repmat(XTest(i,:), nTrain, 1))/bw).^2, 2))';
        end
        %% end
    elseif strcmp(kernel_type,'boxcar')
        %% begin
        C = [0.5 1/pi]
        c = C(f);
        for i = 1:nTest
           K(:,i) = c * bw^(-f) * (sqrt(sum(((XTrain - repmat(XTest(i,:), nTrain, 1))./bw).^2, 2)) <= 1)';
        end
        %% end
    elseif strcmp(kernel_type,'boxcar_n')
        %% begin
        c = gamma(f/2+1) / pi^(f/2);
        for i = 1:nTest
           K(:,i) = c * bw^(-f) * (sqrt(sum(((XTrain - repmat(XTest(i,:), nTrain, 1))./bw).^2, 2)) <= 1)';
        end
        %% end
    elseif strcmp(kernel_type,'laplace')
        %% begin
        for i = 1:nTest
           K(:,i) = (2 * bw)^(-f) * exp(- sum(((XTrain - repmat(XTest(i,:), nTrain, 1))/bw).^2, 2))';
        end
        %% end
    elseif strcmp(kernel_type,'triangular')
        %% begin
        c = gamma(f/2+1) / pi^(f/2);
        for i = 1:nTest
           K(:,i) = c * bw^(-f) * (sqrt(sum(((XTrain - repmat(XTest(i,:), nTrain, 1))./bw).^2, 2)) <= 1)';
        end
        %% end
    elseif strcmp(kernel_tye,'my_kernel')
        
    end
end