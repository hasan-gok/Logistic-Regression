function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

h_x = sigmoid(X * theta);
J = sum((-y)' * log(h_x) - (ones(m,1)- y)' * log(ones(m,1)- h_x))/m;
grad = ((h_x - y)' * X)/m;

end
