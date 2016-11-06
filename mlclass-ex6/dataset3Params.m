function [C, sigma, error_val] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_vec = sqrt(C_vec);
Param = buildParams(C_vec, sigma_vec);
error_val = ones(size(Param,1),1);
for i = 1:size(Param,1)
    model= svmTrain(X, y, Param(i,1), @(x1, x2) gaussianKernel(x1, x2, Param(i,2)));
    pred = svmPredict(model,Xval);
    error_val(i) = mean(double(pred~=yval));
end
figure(1);
plot(1:size(Param,1), error_val);
grid;
[~,ind] = min(error_val);
C = Param(ind,1);
sigma = Param(ind,2);
error_val = error_val(ind);






% =========================================================================

end
