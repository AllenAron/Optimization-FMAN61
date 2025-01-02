from non_linear_min import non_linear_min
from functions import rosenbrock, f, h, penalty
import numpy as np # type: ignore

def main():
    non_linear_min(rosenbrock, np.array([-3., 3.]), 'BFGS', 'armijo', 1.e-8, restart=True, printout=True)


    nu = 1
    x1 = np.array([-2.,2.,2.,-1.,-1.])
    tol = 1.e-4
    total_evals = 0
    total_iter = 0

    while np.linalg.norm(penalty(x1, nu)) > tol:
        x1, N_evals, N_iter, g, lambd = non_linear_min(lambda x: h(x)+penalty(x,nu), x1, 'BFGS', 'armijo', 1.e-8, restart=True, printout=False)
        print()
        total_evals += N_evals
        total_iter += N_iter
        nu *= 10
        
    x1_formatted = [float(f'{x:.8f}') for x in x1]
    print(f"{'total iter':<15}{'x':<70}{'f(x)':<15}{'norm(grad)':<15}{'# total fun evals':<20}{'lambda':<15}")
    print('-' * 150)        
    print(f"{total_iter:<15}{str(x1_formatted):<70}{h(x1)+penalty(x1,nu):<15.8f}{np.linalg.norm(g):<15.8f}{total_evals:<20}{lambd:<15.8f}")

main()
