import numpy as np
from typing import Callable, Tuple
from grad import grad_c


def non_linear_min(f: Callable[[np.ndarray], float],
                   x0: np.ndarray,
                   method: str,
                   line_search: str,
                   tol: float,
                   restart: bool,
                   printout: bool) \
        -> Tuple[np.ndarray, int, int, float]:

    if method == 'DFP':
        def met(p, q, D): return D + (p @ p.T)/(np.dot(p.T, q)) - \
            (D @ q @ q.T @ D)/np.dot(q.T, D @ q)
    elif method == 'BFGS':
        def met(p, q, D): return D + (((1 + (np.dot(q.T, D @ q) /
                                             np.dot(p.T, q))) * p @ p.T) - D @ q @ p.T - p @ q.T @ D)/np.dot(p.T,q)
    
    if line_search == 'golden_section': search_method = lambda f: golden_section(f,0,10,tol)
    elif line_search == 'armijo': search_method = lambda f: armijo(f)

    # Initialize variables
    D = np.eye(np.size(x0))  # Initial estimate of Hessian
    N_iter = 0                   # Number of iterations
    N_evals = 0
    

    x = x0
    g = grad_c(f, x0)
    N_evals += 2
    d = -g

    #lambd, N_new_eval = golden_section(lambda y: f(x0+y*d), 0, max_lambd, 0.01)
    lambd, N_new_eval = search_method(lambda y: f(x+d*y))
    x1 = x0+lambd*d
    N_evals += N_new_eval


    x_formatted = [float(f'{xf:.8f}') for xf in x]
    if printout:
        print(f"{'iter':<15}{'x':<70}{'f(x)':<15}{'norm(grad)':<15}{'# fun evals':<20}{'lambd':<15}")
        print('-' * 150)

        print(f"{N_iter:<15}{str(x_formatted):<70}{f(x):<15.8f}{np.linalg.norm(g):<15.8f}{N_evals:<20}{lambd:<15.8f}")



    cond = True
    while (cond):
        if N_iter > 250:
            raise Exception
        if restart and N_iter % np.size(x) == 0:
            D = np.eye(np.size(x))            
       
        g1 = grad_c(f, x1)
        N_evals += 2

        q = g1 - g
        p = x1 - x

        D = met(p[:, np.newaxis], q[:, np.newaxis], D)  # new Hessian estimate
        d = -g1@D  # new search direction

        x = x1  # Reset the old values
        g = g1

        lambd, N = search_method(lambda y: f(x+d*y))
        x1 = x+lambd*d
        N_new_eval += N
        N_evals += N_new_eval

        cond = np.linalg.norm(x1-x) > tol and np.linalg.norm(g) > tol
        
        N_iter += 1

        x_formatted = [float(f'{xf:.8f}') for xf in x]
        if printout:
            print(f"{N_iter:<15}{str(x_formatted):<70}{f(x):<15.8f}{np.linalg.norm(g):<15.8f}{N_evals:<20}{lambd:<15.10f}")

    return x1, N_evals, N_iter, np.linalg.norm(g), lambd


def golden_section(func, a, b, tol):
    # Implement goldens section
    tau = (np.sqrt(5)-1)/2  # golden ratio
    L1 = b-a  # length of interval
    ml = b-tau*L1  # left end point
    Ln = L1
    left_val = func(ml)
    mr = a+tau*Ln
    right_val = func(mr)
    N = 2
    while (Ln/L1 > tol):
        N = N+1
        if (right_val > left_val):
            b = mr
            mr = ml
            right_val = left_val
            Ln = b-a
            ml = b-tau*Ln
            left_val = func(ml)
        else:
            a = ml
            ml = mr
            left_val = right_val
            Ln = b-a
            mr = a+tau*Ln
            right_val = func(mr)
    return b, N  # Fix this


def armijo(f, alpha=2, epsilon=0.2, lambd=1):
    f0 = f(0)
    fprime0 = grad_c(f, np.array([0.]))
    N = 5  # total number of function evaluations since derivative takes two and each loop needs to be checked at least once

    while (f(alpha*lambd) < f0+epsilon*fprime0*alpha*lambd):
        N += 1  # one more check for each iteration of the while loop
        lambd = alpha*lambd

    while f(lambd) > f0 + epsilon*fprime0*lambd:
        N += 1
        lambd = lambd/alpha
        
    return lambd, N