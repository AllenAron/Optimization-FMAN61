function [X,N]=dichotomous(F,a,b,tol)
% For example F = @(x) 1-x*exp(-x)
% [a,b] interval
% tol = max ratio of final to initial interval lengths
% X output matrix containing final a, b and b-a from every iteration
% N = number of function evaluations
delta = 1.e-10;
X = [];

L1 = b - a;
left_boundary = a;
right_boundary = b;

LN = L1;

N = 0;
while LN/L1 > tol
   ml = (right_boundary + left_boundary)/2 - delta;
   mr = (right_boundary + left_boundary)/2 + delta;
   
   if F(ml) < F(mr)
       right_boundary = ml;
   else
       left_boundary = mr;
   end
    
   N = N + 2;
   LN = right_boundary - left_boundary
   X = [X; [left_boundary right_boundary LN]];
   
end

end