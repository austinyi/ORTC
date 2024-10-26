# code written by Bongsoo Yi (email: bongsoo@unc.edu), and
#              Phuong N. Hoang (email: phoang3@charlotte.edu), and
#              Son Le Thanh (email: sonlt@kth.se)

import numpy as np
from .utils import get_degree_cost, stochastic_block_model
import time

def entropic_ortc_edge(A1, A2, c, eps, niter, delta):
    dx, _ = A1.shape
    dy, _ = A2.shape

    # d1: dx
    # d2: dy
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
    connect1 = {i: [] for i in range(dx)}
    ne1 = 0
    for i in range(dx):
        for j in range(i + 1, dx):
            if A1[i][j] > 0:
                edges1.append((i, j))
                edge_weights1.append(A1[i][j])
                connect1[i].append(ne1)
                ne1 += 1
                
                edges1.append((j, i))
                edge_weights1.append(A1[j][i])
                connect1[j].append(ne1)
                ne1 += 1

    edges2 = []
    edge_weights2 = []
    connect2 = {i: [] for i in range(dy)}
    ne2 = 0
    for i in range(dy):
        for j in range(i + 1, dy):
            if A2[i][j] > 0:
                edges2.append((i, j))
                edge_weights2.append(A2[i][j])
                connect2[i].append(ne2)
                ne2 += 1

                edges2.append((j, i))
                edge_weights2.append(A2[j][i])
                connect2[j].append(ne2)
                ne2 += 1
                
    edges1 = np.array(edges1)
    edges2 = np.array(edges2)
    edge_weights1 = np.array(edge_weights1)
    edge_weights2 = np.array(edge_weights2)
    
    # Init
    # C: ne1 * ne2
    u1_indices = edges1[:, 0].reshape(-1, 1)
    v1_indices = edges2[:, 0].reshape(1, -1)
    u2_indices = edges1[:, 1].reshape(-1, 1)
    v2_indices = edges2[:, 1].reshape(1, -1)

    C = np.exp(-(c[u1_indices, v1_indices] + c[u2_indices, v2_indices]) / eps)
    
    # F: ne1 * dy
    F = np.tile(edge_weights1[:, np.newaxis], (1, dy))
    
    # G: ne2 * dx
    G = np.tile(edge_weights2[:, np.newaxis], (1, dx))

    # H: ne1 * ne2
    H = np.ones((ne1, ne2))

    # K: scalar
    F = F[:, v1_indices.flatten()]
    G = G[:, u1_indices.flatten()]

    K = 1 / np.sum(F * C * G.T * H)

    # w: ne1 * ne2
    w = F * C * G.T * H * K
            
    w_old = np.ones((ne1, ne2))
    num_iter = 0
    
    # d: dx * dy                            
    d = np.zeros((dx, dy))
    for u in range(dx):
        for v in range(dy):
            d[u, v] = np.sum(w[np.ix_(connect1[u], connect2[v])])
    
    #storing iterates and cost values
    iter_history = []
    cost_history = []
    
    iter_history.append(num_iter+1)
    
    for _ in range(niter):
        if np.max(np.abs(w - w_old)) > delta:
            num_iter += 1
            w_old = np.copy(w)

            # 2: update F
            # t: ne1 * dy  
            t = np.zeros((ne1, dy))
            for i in range(ne1):
                for j in range(dy):
                    t[i, j] = np.sum(C[i, connect2[j]] * G[connect2[j], edges1[i][0]] * H[i, connect2[j]] * K)
            
            # F: ne1 * dy     
            F = d[u1_indices.flatten(),:] * np.tile(edge_weights1[:, np.newaxis], (1, dy)) / np.tile(d1[u1_indices], (1, dy)) / t

            # 3: update G
            # t: ne2 * dx
            t = np.zeros((ne2, dx))
            for i in range(ne2):
                for j in range(dx):
                    t[i, j] = np.sum(C[connect1[j], i] * F[connect1[j], edges2[i][0]] * H[connect1[j], i] * K)
            # G: ne2 * dx
            G = d[:, v1_indices.flatten()].T * np.tile(edge_weights2[:, np.newaxis], (1, dx)) / np.tile(d2[v1_indices.T], (1, dx)) / t

            # 4: update H
            # Create k and l using numpy operations
            k = np.where(np.arange(ne1) % 2 == 0, np.arange(ne1) + 1, np.arange(ne1) - 1)
            l = np.where(np.arange(ne2) % 2 == 0, np.arange(ne2) + 1, np.arange(ne2) - 1)

            # Get the relevant elements from the arrays using broadcasting
            F_k_edges2_j1 = F[k.reshape(-1, 1), v2_indices]
            G_l_edges1_i1 = G[l.reshape(-1, 1), u2_indices.T].T

            F_i_edges2_j0 = F[np.arange(ne1).reshape(-1, 1), v1_indices]
            G_j_edges1_i0 = G[np.arange(ne2).reshape(-1, 1), u1_indices.T].T

            # Calculate H using element-wise operations
            H = np.sqrt((F_k_edges2_j1 * G_l_edges1_i1) / (F_i_edges2_j0 * G_j_edges1_i0))
            print(H)

            # 5: update K
            F = F[:, v1_indices.flatten()]
            G = G[:, u1_indices.flatten()]

            K = 1 / np.sum(F * C * G.T * H)

            # 6: update w
            w = F * C * G.T * H * K
            
            # 7: update cost 
            # d: dx * dy                            
            d = np.zeros((dx, dy))
            for u in range(dx):
                for v in range(dy):
                    d[u, v] = np.sum(w[np.ix_(connect1[u], connect2[v])])
            exp_cost = np.sum(d * c)
            cost_history.append(exp_cost)
            iter_history.append(num_iter+1)
        else:
            break
    
    return w, exp_cost, num_iter, iter_history, cost_history

if __name__ == "__main__":
    # generate examples
    m1 = 3
    m2 = 2
    A1 = stochastic_block_model(np.array([m1,m1,m1,m1]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A2 = stochastic_block_model(np.array([m2,m2,m2,m2]), np.array([[1,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.8,0.1],[0.1,0.1,0.1,0.7]]))
    A1 = A1 / np.sum(A1)
    A2 = A2 / np.sum(A2)
    c = get_degree_cost(A1, A2)

    # test with entropicORTC
    start = time.time()
    w, exp_cost, n, iter_his, cost_his = entropic_ortc_edge(A1, A2, c, 0.0001, 10000, 1e-10)
    print(exp_cost)
    end = time.time()
    print(end - start)
