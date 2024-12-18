function [X,N]=goldensection(F,a,b,tol)
% For example F = @(x) 1-x*exp(-x)
% [a,b] interval
% tol = max ratio of final to initial interval lengths
% X output matrix containing final a, b and b-a from every iteration
% N = number of function evaluations
tau = (sqrt(5) - 1) / 2;
X = [];

L = b - a;
ml = b - L*tau;

Fl = F(ml);

% LN = tau^N-1 * L1
Nmax = log(tol)/log(tau) + 1;

left_boundary = a;
right_boundary = b;

mr = left_boundary + L*tau;
Fr = F(mr);

for N = 2:ceil(Nmax)
    
    if Fl < Fr
       right_boundary = mr;
       mr = ml;
       ml = right_boundary - L*tau^N;
       
       Fr = Fl;
       Fl = F(ml);
    else
        left_boundary = ml;
        ml = mr;
        mr = left_boundary + L*tau^N;
        
        Fl = Fr;
        Fr = F(mr);
    end
    
    X = [X; [left_boundary right_boundary right_boundary - left_boundary]];
    
end

N = ceil(Nmax) + 2;

end