clear;
clc;
trainfile = csvread('forestfires__1_multi_class (1).csv');%Read CSV file

labels(1:450,1) = trainfile(1:450,14);
features(1:450,1:12) = sparse(trainfile(1:450,1:12));

test_labels(:,1) = trainfile(451:end,14);
test_features(:,1:12) = sparse(trainfile(451:end,1:12));

%Write the lables and features to a file.
libsvmwrite('trainfile.train',labels,features);

[train_labels, train_features] = libsvmread('trainfile.train');

%Train all models
for k = 1:5
    model(k) = svmtrain(double(labels ==k ),features,'-c 1 -g 0.2 -b 1');
end;

%# get probability estimates of test instances using each model
prob = zeros(67,5);
for k=1:5
    [~,~,p] = svmpredict(double(test_labels==k), test_features, model(k), '-b 1');
    prob(:,k) = p(:,(model(k).Label==1));
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == test_labels) ./ numel(test_labels)    %# accuracy
C = confusionmat(test_labels, pred) 