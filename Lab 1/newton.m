function [x,N]=newton(Fp, Fb,x,tol)
% For example F = @(x) 1-x*exp(-x)
% tol = max ratio of final to initial interval lengths
% N = number of function evaluations

X = [];

N = 0;
while abs(Fp(x)) > tol
    x = x - Fp(x)/Fb(x)
    N = N + 1;
end
    
end