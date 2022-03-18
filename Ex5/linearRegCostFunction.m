function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;

resta = (h-y);

J = sum(resta.^2) / (2*m);

f = theta.^2;

f(1) = 0;

f = (lambda/(2*m)) * sum(f); %factor de regularizacion

J += f;

%gradiente

mult = resta' * X;

grad = mult'/m;

f = lambda * theta / m;

f(1) = 0;

grad += f;

% =========================================================================

grad = grad(:);

end
