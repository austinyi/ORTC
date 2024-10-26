from scipy.optimize import linprog
import numpy as np
import time
from .utils import get_degree_cost, stochastic_block_model

# Vertex based implementation
# Weak symmetry condition
# A1: n1 * n1
# A2: n2 * n2
# c: n1 * n2

def ortc_v1(A1, A2, c):

    # Number of nodes
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Check A1 and A2 are symmetric
    if not np.allclose(A1, A1.T, rtol=1e-05, atol=1e-08) or not np.allclose(A2, A2.T, rtol=1e-05, atol=1e-08):
        print("The adjacency matrix is not symmetric.")
        return

    # Ensure each graph has a total degree of 1
    A1 = A1 / np.sum(A1)
    A2 = A2 / np.sum(A2)

    # Compute degree function of G1 and G2 (d1, d2)
    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    # Reshape cost function c ((n1*n2)*(n1*n2))
    c = c.reshape(n1*n2)
    c = np.tile(c,(n1*n2,1)).T
    c = c.reshape((n1*n2)**2)

    A = []
    b = []

    # (1) Transition coupling condition
    for u1 in range(n1):
        for v1 in range(n2):
            r = np.zeros((n1*n2)**2)
            r[n1*n2*(u1*n2 + v1): n1*n2*(u1*n2+v1+1)] = 1
            for u2 in range(n1):
                cur = np.zeros((n1*n2)**2)
                cur[n1*n2*(u1*n2 + v1) + u2*n2: n1*n2*(u1*n2 + v1) + (u2+1)*n2] = 1
                cur = cur - r * A1[u1,u2] / d1[u1]
                A.append(list(cur))
                b.append(0)

    # (2) Transition coupling condition
    for u1 in range(n1):
        for v1 in range(n2):
            r = np.zeros((n1*n2)**2)
            r[n1*n2*(u1*n2 + v1): n1*n2*(u1*n2+v1+1)] = 1
            for v2 in range(n2):
                cur = np.zeros((n1*n2)**2)
                cur[n1*n2*(u1*n2 + v1) + v2: n1*n2*(u1*n2 + v1) + (n1-1)*n2 + v2+1 : n2] = 1
                cur = cur - r * A2[v1,v2] / d2[v1]
                A.append(list(cur))
                b.append(0)

    # (3) Normalization
    A.append([1]*(n1*n2)**2)
    b.append(1)

    # (4) Weak symmetry condition
    for u1 in range(n1):
        for v1 in range(n2):
            for u2 in range(n1):
                for v2 in range(n2):
                    cur = np.zeros((n1*n2)**2)
                    cur[n1*n2*(u1*n2 + v1) + u2*n2 + v2] = 1
                    cur[n1*n2*(u2*n2 + v2) + u1*n2 + v1] = -1
                    A.append(list(cur))
                    b.append(0)
                    
    # (5) Non-negative
    bounds = [(0, None) for _ in range((n1*n2)**2)]

    res = linprog(c, A_eq = A, b_eq = b, bounds = bounds)

    return (res.fun, res.x) if res.success else (None, None)


if __name__ == "__main__":
    m1 = 2
    m2 = 2
    A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A1 = A1 / np.sum(A1)
    A2 = A2 / np.sum(A2)
    c = get_degree_cost(A1, A2)

    start = time.time()
    exp_cost, weight = ortc_v1(A1, A2, c)
    end = time.time()
    print(exp_cost)
    print(end - start)