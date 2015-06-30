Xtest = dlmread('test.csv',',',1,0);
Xtest = Xtest(:,2:end)';
Xtrain = dlmread('train2.csv',';',1,0);
Xtrain = Xtrain(:,2:end)';

[~,s4,~]=princomp(Xtest');
[~,s2,~]=princomp(Xtrain');

k = 27; %Changed depending on number of components to maintain

dim_test = s4(:,[1:k]);
dim_train = s2(:,[1:k]);

%% Perform KNN
[idx, dist] = knnsearch(dim_train,dim_test,'dist','euclidean','k',10);
class = zeros(size(idx));
for i = 1:length(idx)
    for j = 1:10
        if idx(i,j) <= 1929
            class(i,j) = 1;
        elseif 1930 <= idx(i,j) && idx(i,j) <= 18051
            class(i,j) = 2;
        elseif 18052 <= idx(i,j) && idx(i,j) <= 26055
            class(i,j) = 3;
        elseif 26056 <= idx(i,j) && idx(i,j) <= 28746
            class(i,j) = 4;
        elseif 28747 <= idx(i,j) && idx(i,j) <= 31485
            class(i,j) = 5;
        elseif 31486 <= idx(i,j) && idx(i,j) <= 45620
            class(i,j) = 6;
        elseif 45621 <= idx(i,j) && idx(i,j) <= 48459
            class(i,j) = 7;
        elseif 48460 <= idx(i,j) && idx(i,j) <= 56923
            class(i,j) = 8;
        elseif 56924 <= idx(i,j) && idx(i,j) <= 61878
            class(i,j) = 9;            
        end
    end
end

%% Calculate likelihood
an = zeros(length(idx),9);
for i = 1:length(idx)
    for j = 1:9
        an(i,j) = sum(class(i,:) == j)/10;
    end
end
 
%% Create Result File
 
mypath = 'kaggleAns27.csv';
delete(mypath);
dlmwrite(mypath,ann,'delimiter',',','precision',7);
mystring = ['!' mypath];
eval(mystring);
 
