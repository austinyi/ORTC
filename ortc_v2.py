from scipy.optimize import linprog
import numpy as np
import time
import itertools
from utils import get_degree_cost, stochastic_block_model

# Edge based implementation
### Weak symmetry condition
### The current implementation assumes that there isn't a self-loop in the graph
# A1: (n1, n1)
# A2: (n2, n2)
# c: (n1, n2)
# [i][j]: i*ne2 + j

def ortc_v2(A1, A2, c0):

    # Number of nodes
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Compute degree function of G1 and G2 (d1, d2)
    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    # edges: list of edges
    # edge_weights: list of edge weights
    #
    # i-th element of 'edge_weights' is the weight of i-th element (edge) of 'edges'
    #
    # ne: number of edges
    # connect: a dictionary where each key is a vertex in the network.
    #           The corresponding value for each vertex key is a list of edge indices that include that vertex.

    edges1 = []
    edge_weights1 = []
    connect1 = {i: [] for i in range(n1)}
    ne1 = 0
    for i in range(n1):
        for j in range(i+1, n1):
            if A1[i][j] > 0:
                edges1.append((i, j))
                edge_weights1.append(A1[i][j])
                connect1[i].append(ne1)
                connect1[j].append(ne1)
                ne1 += 1

    edges2 = []
    edge_weights2 = []
    connect2 = {i: [] for i in range(n2)}
    ne2 = 0
    for i in range(n2):
        for j in range(i+1, n2):
            if A2[i][j] > 0:
                edges2.append((i, j))
                edge_weights2.append(A2[i][j])
                connect2[i].append(ne2)
                ne2 += 1

                edges2.append((j, i))
                edge_weights2.append(A2[j][i])
                connect2[j].append(ne2)
                ne2 += 1

    A = []
    b = []

    # (1) Transition coupling condition
    for i in range(ne1):
        for v1 in range(n2):
            u1, u2 = edges1[i]
            cur = np.zeros(ne1*ne2)
            cur[i*ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1],connect2[v1]):
                cur[k*ne2 + l] -= edge_weights1[i] / d1[u1]
            A.append(list(cur))
            b.append(0)

            u2, u1 = edges1[i]
            cur = np.zeros(ne1*ne2)
            cur[i*ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1],connect2[v1]):
                cur[k*ne2 + l] -= edge_weights1[i] / d1[u1]
            A.append(list(cur))
            b.append(0)


    # (2) Transition coupling condition
    for i in range(ne2):
        for u1 in range(n1):
            v1, v2 = edges2[i]
            cur = np.zeros(ne1*ne2)
            cur[np.array(connect1[u1])*ne2 + i] = 1
            for k, l in itertools.product(connect1[u1],connect2[v1]):
                cur[k*ne2 + l] -= edge_weights2[i] / d2[v1]
            A.append(list(cur))
            b.append(0)


    # (3) normalization
    A.append([1]*(ne1*ne2))
    b.append(1)

    # (4) non-negative
    bounds = [(0, None) for _ in range(ne1*ne2)]

    # cost function c (n1*n2)
    c = np.zeros(ne1*ne2)
    for i, (u1, u2) in enumerate(edges1):
        for j, (v1, v2) in enumerate(edges2):
            c[i*ne2 + j] = c0[u1][v1] + c0[u2][v1]

    res = linprog(c, A_eq = A, b_eq = b, bounds = bounds)

    return (res.fun, res.x) if res.success else (None, None)

if __name__ == "__main__":
    m1 = 3
    m2 = 3

    A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A1 = A1 / np.sum(A1)
    A2 = A2 / np.sum(A2)
    c = get_degree_cost(A1, A2)

    start = time.time()
    AA, BB = ortc_v2(A1, A2, c)
    print(AA)
    end = time.time()
    print(end - start)


