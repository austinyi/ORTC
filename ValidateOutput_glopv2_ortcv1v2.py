from scipy.optimize import linprog
import numpy as np
import time
from utils import get_degree_cost, stochastic_block_model

from ortc_v1 import ortc_v1
from ortc_v2 import ortc_v2
from glop_v2 import glop_v2

ortc_v1_time = 0
ortc_v2_time = 0
glop_v2_time = 0
total = 0
wrong = 0

for _ in range(50):
    try:
        m1 = 3
        m2 = 3
        A1 = stochastic_block_model(np.array([m1, m1, m1, m1]), np.array(
            [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
        A2 = stochastic_block_model(np.array([m2, m2, m2, m2]), np.array(
            [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
        A1 = A1 / np.sum(A1)
        A2 = A2 / np.sum(A2)
        c = get_degree_cost(A1, A2)

        start = time.time()
        exp_cost1, _ = ortc_v1(A1, A2, c)
        end = time.time()
        ortc_v1_time += end - start

        start = time.time()
        exp_cost2, _ = ortc_v2(A1, A2, c)
        end = time.time()
        ortc_v2_time += end - start

        start = time.time()
        _, exp_cost3, _ = glop_v2(A1, A2, c)
        end = time.time()
        glop_v2_time += end - start

        if round(exp_cost1, 10) != round(exp_cost2, 10) or round(exp_cost1, 10) != round(exp_cost3, 10):
            wrong += 1
        total += 1
    except:
        print('error')

print(total, wrong)
print(ortc_v1_time / total)
print(ortc_v2_time / total)
print(glop_v2_time / total)


ortc_v1_time = 0
ortc_v2_time = 0
glop_v2_time = 0
total = 0
wrong = 0

for _ in range(50):
    try:
        m1 = 4
        m2 = 3
        A1 = stochastic_block_model(np.array([m1, m1, m1, m1]), np.array(
            [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
        A2 = stochastic_block_model(np.array([m2, m2, m2, m2]), np.array(
            [[1, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.1, 0.8, 0.1], [0.1, 0.1, 0.1, 0.7]]))
        A1 = A1 / np.sum(A1)
        A2 = A2 / np.sum(A2)
        c = get_degree_cost(A1, A2)

        start = time.time()
        exp_cost1, _ = ortc_v1(A1, A2, c)
        end = time.time()
        ortc_v1_time += end - start

        start = time.time()
        exp_cost2, _ = ortc_v2(A1, A2, c)
        end = time.time()
        ortc_v2_time += end - start

        start = time.time()
        _, exp_cost3, _ = glop_v2(A1, A2, c)
        end = time.time()
        glop_v2_time += end - start

        if round(exp_cost1, 10) != round(exp_cost2, 10) or round(exp_cost1, 10) != round(exp_cost3, 10):
            wrong += 1
        total += 1
    except:
        print('error')


print(total, wrong)
print(ortc_v1_time / total)
print(ortc_v2_time / total)
print(glop_v2_time / total)