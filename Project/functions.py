
import numpy as np # type: ignore



def rosenbrock(x : np.ndarray) -> float:
    """
    Evaluates Rosenbrock's function for a numpy-array x
    with 1 dimension and size 2
    x = [x[0], x[1]]
    """

    val = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    return val

def f(x: np.ndarray) -> float:
    val = x[0]**2 + x[0] * x[1] + 0.5 * x[1]**2 - x[1]
    return val

def h(x):
    return np.exp(np.prod(x))

def penalty(x, mu):
    return mu*((np.dot(x.T,x) - 10)**2 + (x[1]*x[2] - 5*x[3]*x[4])**2 + (x[0]**3 + x[2]**3 + 1)**2)
