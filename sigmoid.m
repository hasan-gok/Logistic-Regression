function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.
temp_ones = ones(size(z));
g = temp_ones./ (temp_ones + exp(-z));
end
