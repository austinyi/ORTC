import numpy as np
import itertools
from ortc.utils import get_degree_cost, stochastic_block_model
from ortools.linear_solver import pywraplp
import time

# Changes from glop_v2_2.py
# minimized the use of FOR loops and leveraged numpy's vectorized operations.

# Edge based implementation
### Weak symmetry condition (4*ne1*ne2 variables)
### The current implementation assumes that there isn't a self-loop in the graph
# A1: (n1, n1)
# A2: (n2, n2)
# c: (n1, n2)
# [i][j]: i*ne2 + j

def glop_v2(A1, A2, c):

    # Number of nodes
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Check A1 and A2 are symmetric
    if not np.allclose(A1, A1.T, rtol=1e-05, atol=1e-08) or not np.allclose(A2, A2.T, rtol=1e-05, atol=1e-08):
        print("The adjacency matrix is not symmetric.")
        return

    # Ensure each graph has a total degree of 1
    A1 /= np.sum(A1)
    A2 /= np.sum(A2)

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
                ne1 += 1

                edges1.append((j, i))
                edge_weights1.append(A1[i][j])
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


    # cost function c (n1*n2)
    # c_reshape = np.zeros(ne1*ne2)
    # for i, (u1, u2) in enumerate(edges1):
    #     for j, (v1, v2) in enumerate(edges2):
    #         c_reshape[i*ne2 + j] = c[u1][v1]

    edges1 = np.array(edges1)
    edges2 = np.array(edges2)

    # Extract u1 and v1 indices from edges
    u1_indices = edges1[:, 0]
    v1_indices = edges2[:, 0]

    # Create a grid of indices and Flatten the grid indices
    i_indices, j_indices = np.meshgrid(np.arange(ne1), np.arange(ne2), indexing='ij')
    i_indices_flat = i_indices.flatten()
    j_indices_flat = j_indices.flatten()

    # Use advanced indexing to get the corresponding u1 and v1 values from the edges
    u1_flat = u1_indices[i_indices_flat]
    v1_flat = v1_indices[j_indices_flat]

    # Create the reshaped cost function using the flattened indices
    c_reshape = c[u1_flat, v1_flat]


    # Create solver
    # solver = pywraplp.Solver.CreateSolver("PDLP")
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Create decision variables
    var_flow = np.array([solver.NumVar(0, solver.infinity(), f"var_{i}_{j}") for i in range(ne1) for j in range(ne2)])


    s= time.time()

    constraints = []

    # (1) Transition coupling condition
    for i in range(ne1):
        u1, u2 = edges1[i]
        edge_weight_ratio = edge_weights1[i] / d1[u1]
        for v1 in range(n2):
            cur = np.zeros(ne1 * ne2)
            cur[i * ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weight_ratio

            constraints.append(solver.Constraint(0, 0))
            for idx in range(ne1 * ne2):
                constraints[-1].SetCoefficient(var_flow[idx], cur[idx])
            
    # (2) Transition coupling condition
    for i in range(ne2):
        v1, v2 = edges2[i]
        edge_weight_ratio = edge_weights2[i] / d2[v1]
        for u1 in range(n1):
            cur = np.zeros(ne1 * ne2)
            cur[np.array(connect1[u1]) * ne2 + i] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weight_ratio

            constraints.append(solver.Constraint(0, 0))
            for idx in range(ne1 * ne2):
                constraints[-1].SetCoefficient(var_flow[idx], cur[idx])


    e=time.time()
    print("i am ", e-s)

    # (3) Weak symmetry condition
    for i in range(0, ne1, 2):
        for j in range(0, ne2, 2):
            constraints.append(solver.Constraint(0, 0))
            constraints[-1].SetCoefficient(var_flow[i*ne2 + j], 1)
            constraints[-1].SetCoefficient(var_flow[(i+1) * ne2 + (j+1)], -1)

            constraints.append(solver.Constraint(0, 0))
            constraints[-1].SetCoefficient(var_flow[(i+1)*ne2 + j], 1)
            constraints[-1].SetCoefficient(var_flow[i * ne2 + (j+1)], -1)

    # (4) Normalization
    solver.Add(solver.Sum(var_flow) == 1)

    # Create objective function
    solver.Minimize(solver.Sum(var_flow * c_reshape))


    s= time.time()

    # Solve
    status = solver.Solve()

    e=time.time()
    print("me too", e-s)


    opt_flow = []
    if status == pywraplp.Solver.OPTIMAL:
        #print(f"optimal obj = {solver.Objective().Value()}")
        for src_idx in range(ne1):
            opt_vals = [var_flow[src_idx * ne2 + dest_idx].solution_value()
                        for dest_idx in range(ne2)]
            opt_flow.append(opt_vals)
        return status, solver.Objective().Value(), opt_flow
    else:
        return status, None, None


