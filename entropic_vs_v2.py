from scipy.optimize import linprog
import numpy as np
import time
import itertools
from utils import get_degree_cost, stochastic_block_model


from entropicORTC import entropic_ortc
from ortc_v2 import ortc_v2
from glop_v2 import glop_v2

def check_weightTC(w, A1, A2):
    dx = A1.shape[0]
    dy = A2.shape[0]

    d1 = np.sum(A1, axis=1)
    d2 = np.sum(A2, axis=1)

    d = np.sum(w, axis=(2, 3))

    r1 = 0
    c1 = 0
    for i in range(dx):
        for j in range(dy):
            for k in range(dx):
                v = 0
                for l in range(dy):
                    v += w[i,j,k,l]
                if A1[i,k] > 0:
                    c1 += 1
                    r1 += np.abs(v / (d[i,j]*A1[i,k]/d1[i])- 1)

    r2 = 0
    c2 = 0
    for i in range(dx):
        for j in range(dy):
            for l in range(dy):
                v = 0
                for k in range(dx):
                    v += w[i,j,k,l]
                if A2[j,l] > 0:
                    c2 += 1
                    r2 += np.abs(v / (d[i,j]*A2[j,l]/d2[j])- 1)

    r3 = 0
    c3 = 0
    for i in range(dx):
        for j in range(dy):
            for k in range(dx):
                for l in range(dy):
                    c3 += 1
                    if round(w[i,j,k,l], 10) != round(w[k,l,i,j], 10):
                        r3 += 1


    return r1/c1, r2/c2, c3, r3





m1 = 7
m2 = 7

A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
A1 = A1 / np.sum(A1)
A2 = A2 / np.sum(A2)
c = get_degree_cost(A1, A2)

start = time.time()
AA, BB = ortc_v2(A1, A2, c)
print("ortc_v2 cost: ", AA)
# print(check_weightTC(BB, A1, A2))
end = time.time()
print(end - start)

# start = time.time()
# status, val, ww = glop_v2(A1, A2, c)
# print("glop_v2 cost: ", val)
# # print(check_weightTC(BB, A1, A2))
# end = time.time()
# print(end - start)

start = time.time()
w, exp_cost = entropic_ortc(A1, A2, c, 0.01, delta=1e-7)
print("entropic cost: ", exp_cost)
#print(check_weightTC(w, A1, A2))
end = time.time()
print(end - start)

start = time.time()
w, exp_cost = entropic_ortc(A1, A2, c, 0.0001, delta=1e-7)
print("entropic cost: ", exp_cost)
#print(check_weightTC(w, A1, A2))
end = time.time()
print(end - start)

# start = time.time()
# w, exp_cost = entropic_ortc(A1, A2, c, 0.01, delta=1e-8)
# print("entropic cost: ", exp_cost)
# end = time.time()
# print(end - start)
# start = time.time()
# w, exp_cost = entropic_ortc(A1, A2, c, 0.01, delta=1e-7)
# print("entropic cost: ", exp_cost)
# end = time.time()
# print(end - start)


