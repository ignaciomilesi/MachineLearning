function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);

p1 = (-1*y)'*log(h);

p2 = (1-y)'*log(1-h);

resta1 = p1 - p2;

J = resta1/m;

% -- Regularizacion

factor = theta.^2;

factor(1) = 0;

factor = lambda * sum(factor) / (2*m);

J = J + factor;

% -------------------------------------------------------------

resta2 = h - y;

mult = resta2' * X;

grad = mult' / m;

% -- Regularizacion

factor = lambda * theta / m;

factor(1) = 0;

grad = grad + factor;

% =============================================================

end
