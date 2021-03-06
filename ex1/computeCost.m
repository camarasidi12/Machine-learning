function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
%O=pinv(X'*X)*X'*y
function h=he(x)
h=theta'*[1;x]
end
% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
r=0;

x=X(:,2)
for i=1:m
 r=r+(he(x(i))-y(i))^2;
end

       J=r/(2*m);

% =========================================================================

end

        function JJ = computeCost2(X, y, theta)
        %COMPUTECOST Compute cost for linear regression
        %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
        %   parameter for linear regression to fit the data points in X and y
        
        % Initialize some useful values
        m = length(y); % number of training examples
        %O=pinv(X'*X)*X'*y
                function h=he(x)
                h=theta'*[1;x]
                end
                % You need to return the following variables correctly
                %J = 0;
                
                % ====================== YOUR CODE HERE ======================
                % Instructions: Compute the cost of a particular choice of theta
                %               You should set J to the cost.
                r=0;
                
                x=X(:,2)
                for i=1:m
                r=r+(he(x(i))-y(i))*x(i);
                end
                
                JJ=r/m;
                
                % =========================================================================
                
                end



                function JJJ = computeCost3(X, y, theta)
                %COMPUTECOST Compute cost for linear regression
                %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
                %   parameter for linear regression to fit the data points in X and y
                
                % Initialize some useful values
                m = length(y); % number of training examples
                %O=pinv(X'*X)*X'*y
                        function h=he(x)
                        h=theta'*[1;x]
                        end
                        % You need to return the following variables correctly
                        %J = 0;
                        
                        % ====================== YOUR CODE HERE ======================
                        % Instructions: Compute the cost of a particular choice of theta
                        %               You should set J to the cost.
                        r=0;
                        
                        x=X(:,2)
                        for i=1:m
                        r=r+(he(x(i))-y(i));
                        end
                        
                        JJJ=r/m;
                        
                        % =========================================================================
                        
                        end








