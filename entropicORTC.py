import math

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
import copy
from utils import get_degree_cost, adj_to_trans, stochastic_block_model
import time


def entropic_ortc(A1, A2, c, eps, delta=1e-8):
    dx = A1.shape[0]
    dy = A2.shape[0]

    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    # 1: Initialization
    C = np.zeros((dx, dy, dx, dy))
    C = np.exp(-(c[:, :, np.newaxis, np.newaxis] + c[np.newaxis, np.newaxis, :, :]) / eps)

    F = np.zeros((dx, dy, dx))
    for j in range(dy):
        F[:,j,:] = A1[:,:]

    G = np.zeros((dx, dy, dy))
    for i in range(dx):
        G[i,:,:] = A2[:,:]

    H = np.ones((dx, dy, dx, dy))

    K = np.sum(F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H)
    K = 1/K

    w = F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H * K

    #w_old = np.ones((dx, dy, dx, dy))
    
    for n in range(niter):
    #while np.max(np.abs(w - w_old)) > delta:
        #print(np.max(np.abs(w - w_old)))
        #w_old = np.copy(w)

        # 2: update F
        d = np.sum(w, axis = (2,3))
        for i in range(dx):
            for j in range(dy):
                for k in range(dx):
                    t = np.sum(C[i,j,k,:] * G[i,j,:] * H[i,j,k,:] * K)
                    F[i,j,k] = (d[i,j] * A1[i,k] / d1[i]) / t

        # 3: update G
        d = np.sum(w, axis = (2,3))
        for i in range(dx):
            for j in range(dy):
                for l in range(dy):
                    t = np.sum(C[i,j,:,l]*F[i,j,:]*H[i,j,:,l]*K)
                    G[i,j,l] = (d[i,j] * A2[j,l] / d2[j]) / t

        # 4: update H
        for i in range(dx):
            for j in range(dy):
                for k in range(dx):
                    for l in range(dy):
                        if i==k and j==l:
                            H[i,j,k,l] = 1
                        else:
                            if F[i,j,k] > 0 and G[i,j,l] > 0:
                                H[i,j,k,l] = math.sqrt((F[k,l,i] * G[k,l,j]) / (F[i,j,k] * G[i,j,l]))

        # 5: update K
        K = np.sum(F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H)
        K = 1/K

        #6: update w
        w = F[:, :, :, np.newaxis] * C * G[:, :, np.newaxis, :] * H * K

        # Check for convergence.

    d = np.sum(w, axis = (2,3))
    #c = np.reshape(c, (dx * dy, -1))
    exp_cost = np.sum(d * c)
    #print(c)
    return w, exp_cost

if __name__ == "__main__":
    m1 = 3
    m2 = 3
    A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A1 = A1 / np.sum(A1)
    A2 = A2 / np.sum(A2)
    c = get_degree_cost(A1, A2)

    start = time.time()
    w, exp_cost = entropic_ortc(A1, A2, c, 0.01)
    print(exp_cost)
    end = time.time()
    print(end - start)
