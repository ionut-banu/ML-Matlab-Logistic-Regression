function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization

% Initialize some useful values
m = length(y); % number of training examples

grad = zeros(size(theta));

J = (1/m) * (-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta))) +...
        (lambda/(2*m))*sum(theta(2:end).^2);

grad(1) = (1/m) * ((sigmoid(X*theta)-y)'*X(:,1));
grad(2:end) = (1/m) * (sigmoid(X*theta)-y)'*X(:,2:end) +...
        (lambda/m)*theta(2:end)';

end
