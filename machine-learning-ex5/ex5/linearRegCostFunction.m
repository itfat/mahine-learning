function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h=X*theta;
error=h-y;
sqr=error.^2;
non_reg=sum(sqr*(1/(2*m)));
theta(1)=0;
reg=sum((lambda*(theta.^2))/(2*m));
J=non_reg+reg;
%%GRADIENT
theta(1)=0;
non_reg=(X'*error)/m;
reg=(lambda*theta)/m;
grad=non_reg+reg;











% =========================================================================

grad = grad(:);

end
