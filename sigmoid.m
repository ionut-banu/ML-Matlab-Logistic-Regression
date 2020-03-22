function g = sigmoid(z)
%SIGMOID Compute the sigmoid of each value of z (z can be a matrix,
%        vector or scalar).

g = 1./(1 + exp(-z));

end