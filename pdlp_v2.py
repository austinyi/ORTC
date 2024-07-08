import numpy as np
import itertools
from utils import get_degree_cost, stochastic_block_model
from ortools.linear_solver import pywraplp

# Edge based implementation
### Weak symmetry condition
### The current implementation assumes that there isn't a self-loop in the graph
# A1: (n1, n1)
# A2: (n2, n2)
# c: (n1, n2)
# [i][j]: i*ne2 + j

def pdlp_v2(A1, A2, c):

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
    c_reshape = np.zeros(ne1*ne2)
    for i, (u1, u2) in enumerate(edges1):
        for j, (v1, v2) in enumerate(edges2):
            c_reshape[i*ne2 + j] = c[u1][v1]



    # Create solver
    solver = pywraplp.Solver.CreateSolver("PDLP")
    # solver = pywraplp.Solver.CreateSolver("GLOP")

    # Create decision variables
    var_flow = []
    for src_idx in range(ne1):
        for dest_idx in range(ne2):
            var_flow.append(solver.NumVar(0, solver.Infinity(),
                                          name=f"var_{src_idx}, {dest_idx}"))


    constraints = []

    # (1) Transition coupling condition
    for i in range(ne1):
        for v1 in range(n2):
            u1, u2 = edges1[i]
            cur = np.zeros(ne1 * ne2)
            cur[i * ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weights1[i] / d1[u1]

            constraints.append(solver.Constraint(0, 0))

            for src_idx in range(ne1):
                for dest_idx in range(ne2):
                    constraints[-1].SetCoefficient(var_flow[src_idx * ne2 + dest_idx], cur[src_idx * ne2 + dest_idx])

    # (2) Transition coupling condition
    for i in range(ne2):
        for u1 in range(n1):
            v1, v2 = edges2[i]
            cur = np.zeros(ne1 * ne2)
            cur[np.array(connect1[u1]) * ne2 + i] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weights2[i] / d2[v1]

            constraints.append(solver.Constraint(0, 0))

            for src_idx in range(ne1):
                for dest_idx in range(ne2):
                    constraints[-1].SetCoefficient(var_flow[src_idx * ne2 + dest_idx], cur[src_idx * ne2 + dest_idx])

    # (3) Weak symmetry condition
    for i in range(0, ne1, 2):
        for j in range(0, ne2, 2):
            cur = np.zeros(ne1 * ne2)
            cur[i*ne2 + j] = 1
            cur[(i+1) * ne2 + (j+1)] = -1

            constraints.append(solver.Constraint(0, 0))
            constraints[-1].SetCoefficient(var_flow[i*ne2 + j], 1)
            constraints[-1].SetCoefficient(var_flow[(i+1) * ne2 + (j+1)], -1)

            cur = np.zeros(ne1 * ne2)
            cur[(i+1)*ne2 + j] = 1
            cur[i * ne2 + (j+1)] = -1

            constraints.append(solver.Constraint(0, 0))
            constraints[-1].SetCoefficient(var_flow[(i+1)*ne2 + j], 1)
            constraints[-1].SetCoefficient(var_flow[i * ne2 + (j+1)], -1)

    # (4) Normalization
    solver.Add(solver.Sum(var_flow) == 1)


    # Create objective function
    obj_expr = []
    for src_idx in range(ne1):
        for dest_idx in range(ne2):
            obj_expr.append(var_flow[src_idx * ne2 + dest_idx] * c_reshape[src_idx * ne2 + dest_idx])
    solver.Minimize(solver.Sum(obj_expr))

    status = solver.Solve()

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


