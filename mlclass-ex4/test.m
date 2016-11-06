clear all; close all; clc;
 
input_layer_size=2;
hidden_layer_size = 2;
num_labels = 1;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

m = 4;
X = [0, 0; 0, 1; 1, 0; 1, 1];
y = [0; 1; 1; 0];
lambda = 0;

% J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0);
options = optimset('MaxIter', 500);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);
