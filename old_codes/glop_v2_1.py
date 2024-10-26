import numpy as np
import time
import itertools
from ortc.utils import get_degree_cost, stochastic_block_model
import pickle

from ortools.linear_solver import pywraplp

### STRONG SYMMETRY CONDITION

def glop_v2(A1, A2, c0):
    # Number of nodes
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Compute degree function of G1 and G2 (d1, d2)
    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

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
                connect2[j].append(ne2)
                ne2 += 1

    # cost function c (n1*n2)
    c = np.zeros(ne1 * ne2)
    for i, (u1, u2) in enumerate(edges1):
        for j, (v1, v2) in enumerate(edges2):
            c[i * ne2 + j] = c0[u1][v1] + c0[u1][v2] + c0[u2][v1] + c0[u2][v2]

    # num_sources = 4 # ne1
    # num_destinations = 5 # ne2
    # supplies = [58, 55, 64, 71]
    # demands = [44, 28, 36, 52, 88]
    # costs = [[8, 5, 13, 12, 12],
    #         [8, 7, 18, 6, 5],
    #         [11, 12, 5, 11, 18],
    #         [19, 13, 5, 10, 18]]

    # create solver
    # solver = pywraplp.Solver.CreateSolver("PDLP")

    # start = time.time()
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # create decision variables
    var_flow = []
    for src_idx in range(ne1):
        for dest_idx in range(ne2):
            var_flow.append(solver.NumVar(0, solver.Infinity(),
                                          name=f"var_{src_idx}, {dest_idx}"))

    #print("Number of variables =", solver.NumVariables())

    #end = time.time()
    #print(end - start)

    # create constraints
    # for src_idx in range(num_sources):
    #     expr = [var_flow[src_idx][dest_idx]
    #             for dest_idx in range(num_destinations)]
    #     solver.Add(solver.Sum(expr) == supplies[src_idx])
    #

    constraints = []

    # (1)
    for i in range(ne1):
        for v1 in range(n2):
            u1, u2 = edges1[i]
            cur = np.zeros(ne1 * ne2)
            cur[i * ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weights1[i] / d1[u1]

            # end = time.time()
            # print(end - start)

            constraints.append(solver.Constraint(0, 0))

            for src_idx in range(ne1):
                for dest_idx in range(ne2):
                    constraints[-1].SetCoefficient(var_flow[src_idx * ne2 + dest_idx], cur[src_idx * ne2 + dest_idx])
                    # expr.append(var_flow[src_idx * ne2 + dest_idx] * cur[src_idx * ne2 + dest_idx])

            # end = time.time()
            # print(end - start)

            # solver.Add(solver.Sum(expr) == 0)
            # end = time.time()
            # print(end - start)

            u2, u1 = edges1[i]
            cur = np.zeros(ne1 * ne2)
            cur[i * ne2 + np.array(connect2[v1])] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weights1[i] / d1[u1]

            # end = time.time()
            # print(end - start)

            constraints.append(solver.Constraint(0, 0))

            for src_idx in range(ne1):
                for dest_idx in range(ne2):
                    constraints[-1].SetCoefficient(var_flow[src_idx * ne2 + dest_idx], cur[src_idx * ne2 + dest_idx])

            # end = time.time()
            # print(end - start)

    #end = time.time()
    #print(end - start)
    #print("Number of constraints =", solver.NumConstraints())

    # (2)
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

            v2, v1 = edges2[i]
            cur = np.zeros(ne1 * ne2)
            cur[np.array(connect1[u1]) * ne2 + i] = 1
            for k, l in itertools.product(connect1[u1], connect2[v1]):
                cur[k * ne2 + l] -= edge_weights2[i] / d2[v1]

            constraints.append(solver.Constraint(0, 0))

            for src_idx in range(ne1):
                for dest_idx in range(ne2):
                    constraints[-1].SetCoefficient(var_flow[src_idx * ne2 + dest_idx], cur[src_idx * ne2 + dest_idx])

    # end = time.time()
    # print(end - start)
    # print("Number of constraints =", solver.NumConstraints())

    # (3)
    solver.Add(solver.Sum(var_flow) == 1)
    #print("Number of constraints =", solver.NumConstraints())

    # create objective function
    obj_expr = []
    for src_idx in range(ne1):
        for dest_idx in range(ne2):
            obj_expr.append(var_flow[src_idx * ne2 + dest_idx] * c[src_idx * ne2 + dest_idx])
    solver.Minimize(solver.Sum(obj_expr))

    # end = time.time()
    # print(end - start)

    status = solver.Solve()

    # end = time.time()
    # print(end - start)

    # end = time.time()
    # print(end - start)

    opt_flow = []
    #print(status)
    if status == pywraplp.Solver.OPTIMAL:
        #print(f"optimal obj = {solver.Objective().Value()}")
        for src_idx in range(ne1):
            opt_vals = [var_flow[src_idx * ne2 + dest_idx].solution_value()
                        for dest_idx in range(ne2)]
            opt_flow.append(opt_vals)
        return status, solver.Objective().Value(), opt_flow
    else:
        return status, None, None



if __name__ == "__main__":
    # gather data
    m1 = 3
    m2 = 3
    A1 = stochastic_block_model(np.array([m1, m1, m1, m1]), np.array(
        [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
    A2 = stochastic_block_model(np.array([m2, m2, m2, m2]), np.array(
        [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
    c0 = get_degree_cost(A1, A2)
    #
    # with open("A1", "wb") as fp:  # Pickling
    #     pickle.dump(A1, fp)
    # with open("A2", "wb") as fp:  # Pickling
    #     pickle.dump(A2, fp)

    # with open("A1", "rb") as fp:  # Unpickling
    #     A1 = pickle.load(fp)
    # with open("A2", "rb") as fp:  # Unpickling
    #     A2 = pickle.load(fp)
    # print(A1)
    # print(A2)
    # c0 = get_degree_cost(A1, A2)
