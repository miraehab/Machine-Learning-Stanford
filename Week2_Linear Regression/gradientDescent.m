function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration  
    
    H = X*theta;
    sum_0 = 0;
    sum_1 = 0;

    for i = 1:m,
      sum_0 = sum_0 + (H(i)-y(i));
    end;

    for i = 1:m,
      sum_1 = sum_1 + (H(i)-y(i))*X(i, 2);
    end;
    
    J_history(iter) = computeCost(X, y, theta);

    
    tmp_theta_0 = theta(1)-alpha*(1/m)*sum_0;
    tmp_theta_1 = theta(2)-alpha*(1/m)*sum_1;
    
    theta(1) = tmp_theta_0;
    theta(2) = tmp_theta_1;
    
    if iter > 1,
      if J_history(iter) > J_history(iter-1),
        break;
      endif
    endif

end

end
