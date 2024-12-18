from non_linear_min import non_linear_min
from functions import rosenbrock, f, h, penalty
import numpy as np
import random as rd

def main():
    #---test case for q(x)=x^2+xy+0.5y^2-y---
    print(f"{'Function':<15}{'Hessian approx':<15}{'Line search':<20}{'Restart':<10}{'Initial point':<20}{'Iterations':<15}{'x':<20}{'f(x)':<15}{'norm(grad)':<15}{'# fun evals':<20}")
    print('-' * 150)

    methods = ['DFP', 'BFGS']
    search_methods = ['armijo', 'golden_section']
    functions = [rosenbrock, f]
    function_names = ['rosenbrock', 'f']
    for m in range(2):
        function = functions[m]
        function_name = function_names[m]
        for l in range(2):
            restart = True
            if l == 1:
                restart = False
            for k in range(2):
                hessian_method = methods[k]
                for j in range(2):
                    line_search_method = search_methods[j]
                    for i in range(10):
                        inital_point = np.array([rd.random()*100 - 50, rd.random()*100-50])
                        inital_point_formatted = [float(f'{ip:.2f}') for ip in inital_point]
                        try:
                            x_sol, N_evals, N_iter, grad, lambd = non_linear_min(function, inital_point, hessian_method, line_search_method, 1.e-8, restart=restart, printout=False)
                            x_sol_formatted = [float(f'{x:.4f}') for x in x_sol]
                            print(f"{function_name:<15}{hessian_method:<15}{line_search_method:<20}{bool(restart):<10}{str(inital_point_formatted):<20}{N_iter:<15}{str(x_sol_formatted):<20}{f(x_sol):<15.8f}{np.linalg.norm(grad):<15.8f}{N_evals:<20}")
                        except:
                            print(f"{function_name:<15}{hessian_method:<15}{line_search_method:<20}{bool(restart):<10}{str(inital_point_formatted):<20}{'inf':<15}{str('no solution found'):<20}{'-':<15}{'-':<15}{'inf':<20}")

main()