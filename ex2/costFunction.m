function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

function hh=h(x)
  hh=sigmoid(theta'*x)
end

% You need to return the following variables correctly 
J = 0;
%l=theta'*X(1,:);
for i=1:m
   J=J-y(i)*log(sigmoid(theta'*(X(i,:))'))-(1-y(i))*log(1-sigmoid(theta'*(X(i,:))'));
end
J=J.*(1/m);
	
grad = zeros(size(theta));
for j=1:size(theta)
   mm=0;
   for i=1:m
	     mm = mm+(h((X(i,:))')-y(i))*X(i,j);
   end
 grad(j)=mm;
end
grad=(1/m).*grad;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
