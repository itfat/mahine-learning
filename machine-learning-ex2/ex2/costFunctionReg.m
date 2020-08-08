function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=1./(1+(e.^((X*theta)*-1)));
%Cost
theta_t=theta.^2;
theta_t(1)=0;
theta_new=sum(theta_t);
reg=theta_new*(lambda/(2*m));
a=-[log(h)'*y];
b=-[log(1-h)'*(1-y)];
J=(a+b)/m+reg;
%Gradient
reg_g=theta;
reg_g(1)=0;
grad=[1/m *[X'*(h-y)]]+[(lambda*reg_g)/m];



% =============================================================

end
