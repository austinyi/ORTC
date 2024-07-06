import numpy as np
from ortools.linear_solver import pywraplp

# Incomplete

def glop_v1(A1, A2, c):
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


    # Create solver
    # solver = pywraplp.Solver.CreateSolver("PDLP")
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Create decision variables
    var_flow = []
    for u1 in range(n1):
        for v1 in range(n2):
            for u2 in range(n1):
                for v2 in range(n2):
                    var_flow.append(solver.NumVar(0, solver.Infinity(), name=f"var_{u1},{v1},{u2},{v2}"))

    constraints = []

    # (1)
    for u1 in range(n1):
        for v1 in range(n2):
            r = np.zeros((n1 * n2) ** 2)
            r[n1 * n2 * (u1 * n2 + v1): n1 * n2 * (u1 * n2 + v1 + 1)] = 1
            for u2 in range(n1):
                cur = np.zeros((n1 * n2) ** 2)
                cur[n1 * n2 * (u1 * n2 + v1) + u2 * n2: n1 * n2 * (u1 * n2 + v1) + (u2 + 1) * n2] = 1
                cur = cur - r * A1[u1, u2] / d1[u1]

                constraints.append(solver.Constraint(0, 0))
                for a1 in range(n1):
                    for b1 in range(n2):
                        for a2 in range(n1):
                            for b2 in range(n2):
                                constraints[-1].SetCoefficient(var_flow[n1 * n2 * (a1 * n2 + b1) + a2 * n2 + b2], cur[a1 * n2 * (u1 * n2 + b1) + a2 * n2 + b2])



    # (2)
    for u1 in range(n1):
        for v1 in range(n2):
            r = np.zeros((n1 * n2) ** 2)
            r[n1 * n2 * (u1 * n2 + v1): n1 * n2 * (u1 * n2 + v1 + 1)] = 1
            for v2 in range(n2):
                cur = np.zeros((n1 * n2) ** 2)
                cur[n1 * n2 * (u1 * n2 + v1) + v2: n1 * n2 * (u1 * n2 + v1) + (n1 - 1) * n2 + v2 + 1: n2] = 1
                cur = cur - r * A2[v1, v2] / d2[v1]

                constraints.append(solver.Constraint(0, 0))

                for a1 in range(n1):
                    for b1 in range(n2):
                        for a2 in range(n1):
                            for b2 in range(n2):
                                constraints[-1].SetCoefficient(var_flow[n1 * n2 * (a1 * n2 + b1) + a2 * n2 + b2],
                                                               cur[a1 * n2 * (u1 * n2 + b1) + a2 * n2 + b2])

    # (3)
    solver.Add(solver.Sum(var_flow) == 1)

    # (4)
    for u1 in range(n1):
        for v1 in range(n2):
            for u2 in range(n1):
                for v2 in range(n2):
                    cur = np.zeros((n1*n2)**2)
                    cur[n1*n2*(u1*n2 + v1) + u2*n2 + v2] = 1
                    cur[n1*n2*(u2*n2 + v2) + u1*n2 + v1] = -1

                    constraints.append(solver.Constraint(0, 0))
                    for a1 in range(n1):
                        for b1 in range(n2):
                            for a2 in range(n1):
                                for b2 in range(n2):
                                    constraints[-1].SetCoefficient(var_flow[n1 * n2 * (a1 * n2 + b1) + a2 * n2 + b2],
                                                                   cur[a1 * n2 * (u1 * n2 + b1) + a2 * n2 + b2])



    # Create objective function
    obj_expr = []
    for u1 in range(n1):
        for v1 in range(n2):
            for u2 in range(n1):
                for v2 in range(n2):
                    obj_expr.append(var_flow[n1 * n2 * (u1 * n2 + v1) + u2 * n2 + v2] * c[u1][v1])
    solver.Minimize(solver.Sum(obj_expr))


    status = solver.Solve()

    #opt_flow = []
    if status == pywraplp.Solver.OPTIMAL:
        #print(f"optimal obj = {solver.Objective().Value()}")
        # for src_idx in range(ne1):
        #     opt_vals = [var_flow[src_idx * ne2 + dest_idx].solution_value()
        #                 for dest_idx in range(ne2)]
        #     opt_flow.append(opt_vals)
        return status, solver.Objective().Value()
    else:
        return status, None
