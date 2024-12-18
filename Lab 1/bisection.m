function [X,N]=bisection(Fp,a,b,tol)
% For example F = @(x) 1-x*exp(-x)
% [a,b] interval
% tol = max ratio of final to initial interval lengths
% X output matrix containing final a, b and b-a from every iteration
% N = number of function evaluations

X = [];

L1 = b - a;
left_boundary = a;
right_boundary = b;

LN = L1;

N = 0;
while LN/L1 > tol
   m = (right_boundary + left_boundary)/2;
   
   if Fp(m) > 0
       right_boundary = m;
   else
       left_boundary = m;
   end
    
   N = N + 1;
   LN = right_boundary - left_boundary
   X = [X; [left_boundary right_boundary LN]];
   
end


end