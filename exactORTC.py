import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
import copy
from utils import get_degree_cost, adj_to_trans, stochastic_block_model
import time



def exact_tce(Pz, c):
    d = Pz.shape[0]
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

def computeot_lp(C, r, c):
    nx = r.size
    ny = c.size
    Aeq = np.zeros((nx+ny, nx*ny))
    beq = np.concatenate((r.flatten(), c.flatten()))
    beq = beq.reshape(-1,1)

    # column sums correct
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row*ny)+t] = 1

    # row sums correct
    for row in range(nx, nx+ny):
        for t in range(nx):
            Aeq[row, t*ny+(row-nx)] = 1

    #lb = np.zeros(nx*ny)
    bound = [[0, None]] * (nx*ny)

    # solve OT LP using linprog
    cost = C.reshape(-1,1)
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bound, method='highs')
    lp_sol = res.x
    lp_val = res.fun
    return lp_sol, lp_val

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
                    sol, val = computeot_lp(g_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                # print(sol)
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
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
                sol, val = computeot_lp(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)

    return Pz



def get_ind_tc(A1, A2):
    dx, dx_col = A1.shape
    dy, dy_col = A2.shape

    w_ind = np.zeros((dx * dy, dx_col * dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy * (x_row) + y_row
                    idx2 = dy * (x_col) + y_col
                    w_ind[idx1, idx2] = A1[x_row, x_col] * A2[y_row, y_col]
    return w_ind

def trans_to_adj(R, d, dx, dy):
    w = np.zeros((dx * dy, dx * dy))
    for x_row in range(dx):
        for x_col in range(dx):
            for y_row in range(dy):
                for y_col in range(dy):
                    idx1 = dy * (x_row) + y_row
                    idx2 = dy * (x_col) + y_col
                    w[idx1, idx2] = R[idx1, idx2] * d[idx1]
    return w

def exact_ortc(A1, A2, c):
    P1 = adj_to_trans(A1)
    P2 = adj_to_trans(A2)
    dx = P1.shape[0]
    dy = P2.shape[0]

    w_old = np.ones((dx * dy, dx * dy))
    w = get_ind_tc(A1, A2)
    iter = 0

    while np.max(np.abs(w - w_old)) > 1e-10:
        iter += 1
        print(iter)
        print(np.max(np.abs(w - w_old)))

        w_old = np.copy(w)
        d = np.sum(w_old, axis=1)

        R = adj_to_trans(w)

        # Transition coupling evaluation.
        g, h = exact_tce(R, c)

        # Transition coupling improvement.
        R = exact_tci(g, h, R, P1, P2)

        w = trans_to_adj(R, d, dx, dy)

        for x_row in range(dx):
            for y_row in range(dy):
                idx1 = dy * (x_row) + y_row
                for x_col in range(x_row - 1, dx):
                    for y_col in range(dy):
                        idx2 = dy * (x_col) + y_col
                        w[idx1, idx2] = w[idx2, idx1] = (w[idx1, idx2] + w[idx2, idx1]) / 2

        # Check for convergence.
        if np.all(w == w_old):
            d = np.sum(w_old, axis=1)
            c = np.reshape(c, (dx * dy, -1))
            exp_cost = np.sum(d * c)
            return w, exp_cost

    return None, None




m1 = 3
m2 = 3
A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A1 = A1 / np.sum(A1)
A2 = A2 / np.sum(A2)
c = get_degree_cost(A1, A2)

start = time.time()
w, exp_cost = exact_ortc(A1, A2, c)
print(exp_cost)
end = time.time()
print(end - start)
