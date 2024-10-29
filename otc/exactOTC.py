import sys
import os
sys.path.append(os.path.abspath('../'))

import numpy as np
from numpy.linalg import pinv
import ot
import copy
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog

def get_ind_tc(Px, Py):
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape

    P_ind = np.zeros((dx*dy, dx_col*dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy*(x_row) + y_row
                    idx2 = dy*(x_col) + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
    return P_ind

def get_stat_dist(Pz):
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Pz.T)

    # Find the index of the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))

    # Get the corresponding eigenvector
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist /= np.sum(stationary_dist)  # Normalize to make it a probability distribution

    return stationary_dist


def get_best_stat_dist(P, c):
    # Set up constraints.
    n = P.shape[0]
    c = c.reshape(n, -1).flatten()  # Ensure c is a flat array of length n

    Aeq = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    beq = np.hstack([np.zeros(n), [1]])

    # Define lower bounds
    lb = np.zeros(n)
    bounds = [(lb_i, None) for lb_i in lb]  # Upper bounds are None (unbounded)

    # Define options.
    options = {'disp': False, 'presolve': False}

    # Solve linear program.
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs', options=options)

    if res.success:
        stat_dist = res.x
        exp_cost = res.fun
    else:
        stat_dist = None
        exp_cost = None

    return stat_dist, exp_cost

# def get_best_stat_dist(P, c):
#     n = P.shape[0]
#     c = c.reshape(n, -1).flatten()  
    
#     # Create the linear solver with the GLOP 
#     solver = pywraplp.Solver.CreateSolver('GLOP')
    
#     # Decision variables: stat_dist[i]
#     stat_dist = [solver.NumVar(0.0, solver.infinity(), f'stat_dist_{i}') for i in range(n)]

#     # Constraints: (P' - I) * stat_dist = 0
#     for i in range(n):
#         constraint_expr = solver.Sum((P[j, i] - (1.0 if i == j else 0.0)) * stat_dist[j] for j in range(n))
#         solver.Add(constraint_expr == 0)

#     # Constraint: sum(stat_dist) == 1
#     solver.Add(solver.Sum(stat_dist) == 1)

#     # Create objective function
#     solver.Minimize(solver.Sum(stat_dist * c))

#     # Solve the problem.
#     status = solver.Solve()

#     if status == pywraplp.Solver.OPTIMAL:
#         stat_dist_values = np.array([var.solution_value() for var in stat_dist])
#         exp_cost = solver.Objective().Value()
#         return stat_dist_values, exp_cost
#     else:
#         # In case the solver fails, try rescaling constraints.
#         alpha = 1
#         while alpha >= 1e-10:
#             alpha /= 10
#             # Create a new solver instance to reset all variables and constraints.
#             solver = pywraplp.Solver.CreateSolver('GLOP')

#             # Decision variables: stat_dist[i]
#             stat_dist = [solver.NumVar(0.0, solver.infinity(), f'stat_dist_{i}') for i in range(n)]

#             # Constraints: alpha * (P' - I) * stat_dist = 0
#             for i in range(n):
#                 constraint_expr = solver.Sum(alpha * (P[j, i] - (1.0 if i == j else 0.0)) * stat_dist[j] for j in range(n))
#                 solver.Add(constraint_expr == 0)

#             # Constraint: alpha * sum(stat_dist) == alpha
#             solver.Add(solver.Sum([alpha * var for var in stat_dist]) == alpha)

#             # Create objective function
#             solver.Minimize(solver.Sum(stat_dist * c))

#             # Solve the problem again.
#             status = solver.Solve()
#             if status == pywraplp.Solver.OPTIMAL:
#                 stat_dist_values = np.array([var.solution_value() for var in stat_dist])
#                 exp_cost = solver.Objective().Value()
#                 return stat_dist_values, exp_cost

#         # If still no solution, raise an error.
#         raise ValueError('Failed to compute stationary distribution.')


def computeot_pot(C, r, c):
    # Ensure r and c are numpy arrays
    r = np.array(r).flatten()
    c = np.array(c).flatten()

    # Compute the optimal transport plan and the cost using the ot.emd function
    lp_sol = ot.emd(r, c, C)
    lp_val = np.sum(lp_sol * C)

    return lp_sol, lp_val

def exact_tce(Pz, c):
    d = Pz.shape[0]
    #c = np.reshape(c.T, (d, -1))
    c = np.reshape(c, (d, -1))
    A = np.block([[np.eye(d) - Pz, np.zeros((d, d)), np.zeros((d, d))],
                  [np.eye(d), np.eye(d) - Pz, np.zeros((d, d))],
                  [np.zeros((d, d)), np.eye(d), np.eye(d) - Pz]])
    b = np.concatenate([np.zeros((d, 1)), c, np.zeros((d, 1))])
    try:
        sol = np.linalg.solve(A, b)
    except:
        sol = np.matmul(pinv(A), b)

    g = sol[0:d].flatten()
    h = sol[d:2*d].flatten()
    return g, h

def exact_tci(g, h, P0, Px, Py):
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx*dy, dx*dy))
    
    g_const = True
    for i in range(dx):
        for j in range(i+1, dx):
            if abs(g[i] - g[j]) > 1e-3:
                g_const = False
                break
        if not g_const:
            break
    # If g is not constant, improve transition coupling against g.
    if not g_const:
        g_mat = np.reshape(g, (dx, dy))
        for x_row in range(dx):
            for y_row in range(dy):
                dist_x = Px[x_row, :]
                dist_y = Py[y_row, :]
                # Check if either distribution is degenerate.
                if any(dist_x == 1) or any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y)
                # If not degenerate, proceed with OT.
                else:
                    sol, val = computeot_pot(g_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
                #P[idx, :] = sol
        if np.max(np.abs(np.matmul(P0, g) - np.matmul(Pz, g))) <= 1e-7:
            Pz = copy.deepcopy(P0)
        else:
            return Pz
    ## Try to improve with respect to h.
    h_mat = np.reshape(h, (dx, dy))
    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]
            # Check if either distribution is degenerate.
            if any(dist_x == 1) or any(dist_y == 1):
                sol = np.outer(dist_x, dist_y)
            # If not degenerate, proceed with OT.
            else:
                sol, val = computeot_pot(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
    return Pz

def exact_otc(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx*dy, dx*dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P-P_old)) > 1e-10:
        iter_ctr += 1
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            #stat_dist = get_stat_dist(P)
            stat_dist, exp_cost = get_best_stat_dist(P, c)
            #stat_dist = np.reshape(stat_dist, (dy, dx)).T
            stat_dist = np.reshape(stat_dist, (dx, dy))
            return iter_ctr, exp_cost, P, stat_dist

    return None, None, None, None

def exact_otc2(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx*dy, dx*dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P-P_old)) > 1e-10:
        iter_ctr += 1
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            #stat_dist = np.reshape(stat_dist, (dy, dx)).T
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return iter_ctr, exp_cost, P, stat_dist

    return None, None, None, None
