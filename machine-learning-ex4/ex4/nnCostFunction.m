function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%size(Theta1)      = 25 401
%size(Theta2)      = 10 26
%size(X)           = 5000 400
%size(y)           = 5000 1
%input_layer_size  = 400
%hidden_layer_size = 25
%num_labels        = 10

% Add ones to the X data matrix
a1 = [ones(m, 1) X];      %size(a1) = 5000 401
z2 = a1 * Theta1';        %size(z2) = 5000 25
a2 = sigmoid(z2);         %size(a2) = 5000 25
a2 = [ones(m,1) a2];      %size(a2) = 5000 26
z3 = a2 * Theta2';        %size(z3) = 5000 10
a3 = sigmoid(z3);         %size(a3) = 5000 10

vy = zeros(m,num_labels); %size(vy) = 5000 10
for i = 1 : m
  vy(i,y(i)) = 1;
end

for i = 1 : m,
 J += (-1*vy(i,:)*log(a3(i,:)') - (1-vy(i,:))*log(1-a3(i,:)'));
end
J = J / m;

rTheta1 = Theta1(:,2:end);    %size(rTheta1) = 25 400
rTheta2 = Theta2(:,2:end);    %size(rTheta2) = 10 25

r1 = 0;
for j = 1 : hidden_layer_size 
  r1 += rTheta1(j,:) * rTheta1(j,:)';
end

r2 = 0;
for j = 1 : num_labels
  r2 += rTheta2(j,:) * rTheta2(j,:)';
end

J = J + lambda / (2*m) * (r1 + r2);

%--------------------------------------------------------------

DELTA2 = 0;
DELTA1 = 0;
for i = 1 : m
  delta3 = a3(i,:) - vy(i,:);                            %size(delta3) = 1 10
  delta2 = delta3 * rTheta2 .* sigmoidGradient(z2(i,:)); %size(delta2) = 1 25
  DELTA2 += delta3' * a2(i,:);                           %size(DELTA2) = 10 26
  DELTA1 += delta2' * a1(i,:);                           %size(DELTA1) = 25 401
end
Theta2_grad = DELTA2 / m + lambda / m * Theta2;           %size(DELTA2) = 10 26
Theta2_grad(:,1) = 1 / m * (DELTA2(:,1));
Theta1_grad = DELTA1 / m + lambda / m * Theta1;           %size(DELTA1) = 25 401
Theta1_grad(:,1) = 1 / m * (DELTA1(:,1));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];               %size(grad) = 10285 1

end
