import numpy as np
import itertools
from utils import get_degree_cost, stochastic_block_model
from ortools.linear_solver import pywraplp
import time


# Edge based implementation
### Weak symmetry condition (2*ne1*ne2 variables)
### The current implementation assumes that there isn't a self-loop in the graph
# A1: (n1, n1)
# A2: (n2, n2)
# c: (n1, n2)
# [i][j]: i*ne2 + j

def glop_v2(A1, A2, c, vertex=False):
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
        for j in range(i + 1, n1):
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
        for j in range(i + 1, n2):
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
    #         c_reshape[i*ne2 + j] = c[u1][v1] + c[u2][v2]

    # Reshape cost function using vectorized operations
    edges1 = np.array(edges1)
    edges2 = np.array(edges2)

    u1_indices = edges1[:, 0].reshape(-1, 1)
    v1_indices = edges2[:, 0].reshape(1, -1)
    u2_indices = edges1[:, 1].reshape(-1, 1)
    v2_indices = edges2[:, 1].reshape(1, -1)

    c_reshape = c[u1_indices, v1_indices] + c[u2_indices, v2_indices]
    c_reshape = c_reshape.flatten()

    # Create solver
    # solver = pywraplp.Solver.CreateSolver("PDLP")
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Create decision variables
    var_flow = np.array([solver.NumVar(0, solver.infinity(), f"var_{i}_{j}") for i in range(ne1) for j in range(ne2)])

    constraints = []

    # (1) Transition coupling condition
    for i in range(ne1):
        u1, u2 = edges1[i]  # u1 < u2
        edge_weight_ratio = edge_weights1[i] / d1[u1]
        for v1 in range(n2):
            cur = np.zeros(ne1 * ne2)
            cur[i * ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                a, b = edges1[k]
                if u1 == a:
                    cur[k * ne2 + l] -= edge_weight_ratio
                else:
                    c, d = edges2[l]
                    if c < d:
                        cur[k * ne2 + l + 1] -= edge_weight_ratio
                    else:
                        cur[k * ne2 + l - 1] -= edge_weight_ratio

            constraints.append(solver.Constraint(0, 0))
            for idx in range(ne1 * ne2):
                constraints[-1].SetCoefficient(var_flow[idx], cur[idx])

        u2, u1 = edges1[i]
        edge_weight_ratio = edge_weights1[i] / d1[u1]
        for v1 in range(n2):
            cur = np.zeros(ne1 * ne2)

            # for edge_index in connect2[v1]:
            #     a, b = edges2[edge_index] # a = v1
            #     if a < b: cur[i * ne2 + edge_index + 1] = 1
            #     else: cur[i * ne2 + edge_index - 1] = 1

            connect2_v1 = np.array(connect2[v1])
            a = edges2[connect2_v1, 0]
            b = edges2[connect2_v1, 1]
            indices = np.where(a < b, i * ne2 + connect2_v1 + 1, i * ne2 + connect2_v1 - 1)
            cur[indices] = 1

            for k, l in itertools.product(connect1[u1], connect2[v1]):
                a, b = edges1[k]
                if u1 == a:
                    cur[k * ne2 + l] -= edge_weight_ratio
                else:
                    c, d = edges2[l]
                    if c < d:
                        cur[k * ne2 + l + 1] -= edge_weight_ratio
                    else:
                        cur[k * ne2 + l - 1] -= edge_weight_ratio

            constraints.append(solver.Constraint(0, 0))
            for idx in range(ne1 * ne2):
                constraints[-1].SetCoefficient(var_flow[idx], cur[idx])

    # (2) Transition coupling condition
    for i in range(ne2):
        v1, v2 = edges2[i]
        edge_weight_ratio = edge_weights2[i] / d2[v1]
        for u1 in range(n1):
            cur = np.zeros(ne1 * ne2)
            for edge_index in connect1[u1]:
                a, b = edges1[edge_index]
                if a == u1:
                    cur[edge_index * ne2 + i] = 1
                elif b == u1 and v1 < v2:
                    cur[edge_index * ne2 + i + 1] = 1
                else:
                    cur[edge_index * ne2 + i - 1] = 1

            for k, l in itertools.product(connect1[u1], connect2[v1]):
                a, b = edges1[k]
                if u1 == a:
                    cur[k * ne2 + l] -= edge_weight_ratio
                else:
                    c, d = edges2[l]
                    if c < d:
                        cur[k * ne2 + l + 1] -= edge_weight_ratio
                    else:
                        cur[k * ne2 + l - 1] -= edge_weight_ratio

            constraints.append(solver.Constraint(0, 0))
            for idx in range(ne1 * ne2):
                constraints[-1].SetCoefficient(var_flow[idx], cur[idx])

    # (3) Normalization
    solver.Add(solver.Sum(var_flow) == 1 / 2)

    # Create objective function
    solver.Minimize(solver.Sum(var_flow * c_reshape))

    # Solve
    status = solver.Solve()

    opt_flow = []
    if status == pywraplp.Solver.OPTIMAL:
        # print(f"optimal obj = {solver.Objective().Value()}")
        for src_idx in range(ne1):
            opt_vals = [var_flow[src_idx * ne2 + dest_idx].solution_value()
                        for dest_idx in range(ne2)]
            opt_flow.append(opt_vals)
        if vertex:
            weight = np.zeros((n1, n2, n1, n2))
            for i in range(ne1):
                for j in range(ne2):
                    u1, u2 = edges1[i]
                    v1, v2 = edges2[j]
                    weight[u1,v1,u2,v2] = opt_flow[i][j]
                    weight[u2,v2,u1,v1] = opt_flow[i][j]     
            return status, solver.Objective().Value(), weight
        else:
            return status, solver.Objective().Value(), opt_flow
    else:
        return status, None, None


