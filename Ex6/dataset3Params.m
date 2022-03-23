function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
C_prueba = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_prueba = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

n = size(C_prueba, 1);
m = size(sigma_prueba, 1);
error = [0, 0, 0];

for i=1 : n;
  for j=1 : n;
    
    C_test = C_prueba(i);
    sigma_test = sigma_prueba(j);
    
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    
    ee = mean(double(predictions ~= yval));
    
    if (error(2) == 0 && error(3) == 0)
      error(1) = ee;
      error(2) = i;
      error(3) = j;
      
    elseif (ee < error(1))
      error(1) = ee;
      error(2) = i;
      error(3) = j;
    endif
    
  endfor
  
endfor

 C = C_prueba(error(2));
 sigma = sigma_prueba(error(3));
  
% =========================================================================

end
