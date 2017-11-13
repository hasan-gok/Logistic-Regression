function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(length(theta), 1);

[unreg_J, unreg_grad] = costFunction(theta, X, y);
J = unreg_J + (lambda/(2*m)) * (sum(theta.^2)-theta(1).^2);

grad(1) = unreg_grad(1);
for i=2:length(theta)
grad(i) = unreg_grad(i) + lambda*theta(i)/m;
end
