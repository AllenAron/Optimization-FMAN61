function B = barrier(x, A, b)
    if (A*x-b < 0)
        B = -sum(1./(A*x-b));
    else
        B = inf
    end
end