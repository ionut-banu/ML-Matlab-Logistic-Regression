function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression

% Initialize some useful values
m = length(y); % number of training examples

J = (1/m) * (-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta)));

grad = (1/m) * (sigmoid(X*theta)-y)'*X;

end