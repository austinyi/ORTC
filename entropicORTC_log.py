import math
import numpy as np
from scipy.special import logsumexp
from utilsSBM import get_degree_cost, stochastic_block_model
import time
import copy


def entropic_ortc_log(A1, A2, c, eps, niter, delta):
    dx, _ = A1.shape
    dy, _ = A2.shape

    # compute d1 and d2
    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    # 1: initialization
    # C: dx*dy*dx*dy
    C = c[:, :, np.newaxis, np.newaxis] + c[np.newaxis, np.newaxis, :, :]

    # f: dx*dy*dx
    f = np.tile(A1[:, np.newaxis, :], (1, dy, 1))

    # g: dx*dy*dy
    g = np.tile(A2[np.newaxis, :, :], (dx, 1, 1))

    # h: dx*dy*dx*dy
    h = np.zeros((dx, dy, dx, dy))

    # k
    k = np.log(np.sum(f[:, :, :, np.newaxis] * np.exp(-C / eps) * g[:, :, np.newaxis, :]))
    k = (-eps) * k

    # w
    w = f[:, :, :, np.newaxis] * np.exp(-C / eps) * g[:, :, np.newaxis, :] * np.exp(k / eps)

    w_old = np.ones((dx, dy, dx, dy))
    num_iter = 0

    # store iterations and cost values
    iter_history = []
    cost_history = []
    iter_history.append(num_iter + 1)

    for _ in range(niter):  # number of iterations
        if np.max(np.abs(w - w_old)) > delta:
            num_iter += 1
            w_old = np.copy(w)

            # 2: update f
            d = np.sum(w, axis=(2, 3))
            t = logsumexp(((-C + g[:, :, np.newaxis, :] + h + k) / eps), axis=3)
            f = eps * (
                np.log(d[:, :, np.newaxis] * A1[:, np.newaxis, :] / d1[:, np.newaxis, np.newaxis] + 1e-323)) - eps * t

            # 3: update g
            t = logsumexp(((-C + f[:, :, :, np.newaxis] + h + k) / eps), axis=2)
            g = eps * (
                np.log(d[:, :, np.newaxis] * A2[np.newaxis, :, :] / d2[np.newaxis, :, np.newaxis] + 1e-323)) - eps * t

            # 4: update h
            arr = np.ones((dx, dy, dx, dy))
            i, j, k, l = np.where(arr > 0)
            h[i, j, k, l] = 1 / 2 * ((f[k, l, i] + g[k, l, j]) - (f[i, j, k] + g[i, j, l]))

            # 5: update k
            k = logsumexp((f[:, :, :, np.newaxis] - C + g[:, :, np.newaxis, :] + h) / eps)
            k = (-eps) * k

            # 6: update w
            w = np.exp((-C + f[:, :, :, np.newaxis] + g[:, :, np.newaxis, :] + h + k) / eps)

            # 7: update cost
            d = np.sum(w, axis=(2, 3))
            exp_cost = np.sum(d * c)
            cost_history.append(exp_cost)
            iter_history.append(num_iter + 1)
        else:
            break

    return w, exp_cost, num_iter, iter_history, cost_history

# generate examples
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
w, exp_cost, n, iter_his, cost_his = entropic_ortc_log(A1, A2, c, 0.0001, 10, 1e-10)
end = time.time()
print(exp_cost)
print(n)
print(end - start)