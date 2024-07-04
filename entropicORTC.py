# code written by Bongsoo Yi (email: bongsoo@unc.edu), and
#              Phuong N. Hoang (email: phoang3@charlotte.edu), and
#              Son Le Thanh (email: sonlt@kth.se)

import math
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
from utils import get_degree_cost, adj_to_trans, stochastic_block_model
import time
import copy

from ortc_v1 import ortc_v1


def entropic_ortc_new(A1, A2, c, eps, niter, delta):
    # A1 -> w1
    # A2 -> w2
    # u, u' -> x
    # v, v' -> y

    dx, _ = A1.shape
    dy, _ = A2.shape

    # d1: dx
    # d2: dy
    # Also convert d1, d2 to different type
    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    # Init
    # C: dx * dy * dx * dy
    # Convert C
    C = np.exp(-(c[:, :, np.newaxis, np.newaxis] + c[np.newaxis, np.newaxis, :, :]) / eps)

    # F: dx * dy * dx
    F = np.tile(A1[:, np.newaxis, :], (1, dy, 1))
   
    # G: dx * dy * dy
    G = np.tile(A2, (dx, 1, 1))

    # H: dx * dy * dx * dy
    H = np.ones((dx, dy, dx, dy))

    # K: scalar
    K = np.sum(F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H)
    K = 1 / K

    # w: dx * dx * dx * dy
    w = C * F[:, :, :, np.newaxis] * G[:, :, np.newaxis, :] * H * K

    # A mask for fast computation in the update of H
    # This mask have true everywhere, except for the ijij components
    # Probably there is a more efficient way to creat this
    mask = np.ones((dx, dy, dx, dy), dtype=bool)
    for i in range(dx):
        for j in range(dy):
            mask[i, j, i, j] = False
            
    w_old = np.ones((dx,dy,dx,dy))
    num_iter = 0
    
    #storing iterates and cost values
    iter_history = []
    cost_history = []
    
    iter_history.append(num_iter+1)
    
    for _ in range(niter):
        if np.max(np.abs(w - w_old)) > delta:
            num_iter += 1
        #print(np.max(np.abs(w - w_old)))
            w_old = np.copy(w)

            # 2: update F
            # d: dx * dy
            d = np.sum(w, axis=(2, 3))
            # Sum over v', which is the third axis
            # t: sum((dx * dy * dx * dy ) * (dx * 1 * dx * dy) * (dx * dy * dx * dy) = dx * dy * dx * dy, axis=3)
            # t: dx * dy * dx
            t = np.sum(C * G[:, :, np.newaxis, :] * H * K, axis=3)
            # F: dx * dy * dx
            # For some reason (probably due to the limit in the float representation)
            # This way of calculating F can produce small errors ( <-1.38777878e-17) comparing to the loop
            # Will need to investigate
            F = (d[:, :, np.newaxis] * A1[:, np.newaxis, :] / d1[:, np.newaxis, np.newaxis]) / t

            # 3: update G
            d = np.sum(w, axis=(2, 3))
            # Sum over u'
            # t: sum((dx * dy * dx * dy) * (dx * dy * dx * 1) * (dx * dy * dx * dy) = dx * dy * dx * dy, axis=2)
            # t: dx * dy * dy
            t = np.sum(C * F[:, :, :, np.newaxis] * H * K, axis=2)
            # G: dx * dy * dy
            # Same story with F
            G = (d[:, :, np.newaxis] * A2[np.newaxis, :, :] / d2[np.newaxis, :, np.newaxis]) / t

            # 4: update H
            # Initialize H
            H = np.ones((dx, dy, dx, dy))
            # Create mask for the condition
            # Create the condition
            F_nonzero = F > 0
            G_nonzero = G > 0
            cond = np.logical_and(F_nonzero[:, :, :, np.newaxis], G_nonzero[:, :, np.newaxis, :])

            # The indices where we should update H
            i, j, k, l = np.where(mask & cond)
            # Update H base on the mask and the condition
            H[i, j, k, l] = np.sqrt((F[k, l, i] * G[k, l, j]) / (F[i, j, k] * G[i, j, l]))

            # 5: update K
            K = np.sum(F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H)
            K = 1 / K

            # 6: update w
            w = F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H * K
            
            # 7: update cost 
            d = np.sum(w, axis=(2, 3))
            exp_cost = np.sum(d * c)
            cost_history.append(exp_cost)
            iter_history.append(num_iter+1)
        else:
            break
    
    return w, exp_cost, num_iter, iter_history, cost_history


# generate examples to test
m1 = 3
m2 = 2
A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A1 = A1 / np.sum(A1)
A2 = A2 / np.sum(A2)
c = get_degree_cost(A1, A2)
d1 = np.sum(A1, axis=1)
d2 = np.sum(A2, axis=1)
print(d1)
print(d2)


# test
start = time.time()
w, exp_cost, n, iter_his, cost_his = entropic_ortc_new(A1, A2, c, 0.0002, 10000, 1e-8)
print(exp_cost)
end = time.time()
print(end - start)
