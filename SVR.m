clear;
clc;
%SVR (Support Vector Regression)
trainfile = csvread('ff.csv');%Read CSV file

trainY(1:450,1) = log(trainfile(1:450,13)+1);
trainX(1:450,1:4) = sparse(trainfile(1:450,9:12));

%Write the lables and features to a file.
libsvmwrite('trainfile.train',trainY,trainX);

[train_output, train_features] = libsvmread('trainfile.train');

model = svmtrain(train_output,train_features, '-s 4 -t 2 -d 5 -c 20 -g 64 -p 1');

% -t 0 - Linear Kernel
% -t 2 - RBF Kernel
test_output(:,1) = log(trainfile(451:517,13)+1);
test_features(:,1:4) = sparse(trainfile(451:517,9:12));

[y_hat, accuracy, projection] = svmpredict(test_output, test_features, model);