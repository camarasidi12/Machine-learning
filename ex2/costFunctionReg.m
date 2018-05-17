 %costFunction.m

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

function hh=h(x)
  hh=sigmoid(theta'*x)
end
% You need to return the following variables correctly 
J = 0;

n=size(theta,2);
[J, grad]= costFunction(theta,X,y)
sm=0;
sm=(lambda/(2*m))*sum(theta(2:size(theta)).^2);
J=J+sm;
%for i=2:n
%	sm=sm+theta(i)^2;
%end
%sm=(lambda/(2*m))*sm;
%J(2:size(J))=J(2:size(J))+sm;

	
%grad = zeros(size(theta));
%for j=1:size(theta)
  % mm=0;
 %  for i=1:m
%	     mm = mm+(h((X(i,:))')-y(i))*X(i,j);
 %  end
% grad(j)=mm;
%end
grad(2:size(theta))=grad(2:size(theta))+(lambda/m)*theta(2:size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
