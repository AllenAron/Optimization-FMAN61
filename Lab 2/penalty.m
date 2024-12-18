function P = penalty(x,A,b)
    P = sum(max(0, A*x-b).^2);
end