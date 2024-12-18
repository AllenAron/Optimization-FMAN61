import numpy as np
...
a = 1.23434234365
b = np.array([5.67823242, 0.002])
print(f"Value of a: {a:24.4f} {a:12.4f} {a:12.4f} {a:12.4f} {a:12.4f}")
print(f"Value of a: {a:12.8f}")
print(f"Value of a: {a:24.4f}")
print(f"Value of a: {a:12.4f}, value of b: {b}")
print(f"Value of b[0]: {b[0]:12.4f}")

print(np.prod(b), b[0]*b[1])

print(np.exp(-8))