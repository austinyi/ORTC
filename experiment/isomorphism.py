import sys
import os
import argparse
sys.path.append(os.path.abspath('../'))

import numpy as np
from random import randint
import random
import networkx as nx
import math

from ortc.utils import *
from ortc.glop_v2 import glop_v2
from otc.exactOTC import exact_otc

def isomorphism_experiment(network_name, n_iter, params):
    network_func = network_map.get(network_name)
    if network_func is None:
        raise ValueError(f"Unknown network function: {network_name}")
    
    n_ortc_detect = 0
    n_otc_detect = 0
    n_ortc_identify = 0
    n_otc_identify = 0
    n_total = 0
    
    for i in range(n_iter):
        print(i)
        try:
            A1 = network_func(*params)
            A1 /= np.sum(A1)
            
            # Check success
            ortc_detect, otc_detect, ortc_identify, otc_identify = permute_isomorphism(A1)
            n_ortc_detect += ortc_detect
            n_otc_detect += otc_detect
            n_ortc_identify += ortc_identify
            n_otc_identify += otc_identify
            n_total += 1
        except:
            continue
    
    if n_total == 0:
        return None, None, None, None, None
    else:
        return n_ortc_detect / n_total, n_otc_detect / n_total, n_ortc_identify / n_total, n_otc_identify / n_total, n_total

def random_erdos_renyi(a, b, p):
    n = randint(a, b)
    A = stochastic_block_model([n], np.array([[p]]))
    return A

def random_sbm(i):
    if i == 0:
        return stochastic_block_model([7,7,7], np.array([[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]]))
    elif i == 1:
        return stochastic_block_model([7,7,7,7], np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]))
    elif i == 2:
        return stochastic_block_model([10,8,6], np.array([[0.7, 0.1, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.7]]))
    
def random_weighted_adjacency_matrix(a, b):
    n = randint(a, b)
    A = np.random.randint(0, 3, (n, n))  # Generate random integers between 0 and 2
    return np.triu(A)+ np.triu(A, 1).T  # Construct the desired matrix

def random_tree(a, b):
    n = randint(a, b)
    if n <= 1:
        raise ValueError("The number of nodes must be greater than 1.")
    
    # Step 1: Generate a random Prufer sequence of length n-2
    prufer_sequence = [random.randint(1, n) for _ in range(n - 2)]
    
    # Step 2: Calculate the degree of each node
    node_degree = [1] * n  # Initialize all nodes with degree 1 (leaf)
    for node in prufer_sequence:
        node_degree[node - 1] += 1  
    
    # Step 3: Build the tree using the Prufer sequence
    edges = []
    for node in prufer_sequence:
        # Find the smallest leaf node (degree 1)
        for i in range(n):
            if node_degree[i] == 1:
                edges.append((i + 1, node))
                node_degree[i] -= 1  
                node_degree[node - 1] -= 1  
                break
    
    # Connect the last two remaining leaf nodes
    remaining_nodes = [i + 1 for i in range(n) if node_degree[i] == 1]
    edges.append((remaining_nodes[0], remaining_nodes[1]))
    
    # Create a graph from the edges and plot it
    G = nx.Graph()
    G.add_edges_from(edges)

    return nx.to_numpy_array(G, nodelist=list(range(1, n+1)))
    
    
def random_lollipop(a, b):
    n1 = np.random.randint(a, b)  # number of nodes in the candy part
    n2 = np.random.randint(a, b)  # number of nodes in the stick part
    n = n1 + n2  # total number of nodes

    # Construct adjacency matrix for candy part
    A1 = np.zeros((n1, n1))

    # Ensure each node is connected to the next node, if not already connected
    for row in range(n1 - 1):
        A1[row, row + 1] = random.random()
    A1[0, n1 - 1] = random.random()
    A1 = A1 + A1.T  

    # Construct adjacency matrix for stick part
    A2 = np.zeros((n2, n2))
    for row in range(n2 - 1):
        A2[row, row + 1] = random.random()
    A2 = A2 + A2.T  # Make the stick part matrix symmetric

    # Construct the full adjacency matrix
    A = np.zeros((n, n))
    A[:n1, :n1] = A1  # Add candy part to the full matrix
    A[n1:, n1:] = A2  # Add stick part to the full matrix

    # Connect the last node of the candy part to the first node of the stick part
    A[n1 - 1, n1] = random.random()
    A[n1, n1 - 1] = A[n1 - 1, n1]
    
    A /= np.sum(A)

    return A
    
    
def random_lollipop_fill(a, b):
    n1 = np.random.randint(a, b)  # number of nodes in the candy part
    n2 = np.random.randint(a, b)  # number of nodes in the stick part
    n = n1 + n2  # total number of nodes
    p = 0.5  # probability for the stochastic block model in the candy part

    # Construct adjacency matrix for candy part
    A1 = stochastic_block_model([n1], np.array([[p]]))
    A1 = np.triu(A1, 1)  # Get upper triangular part excluding diagonal

    # Ensure each node is connected to the next node, if not already connected
    for row in range(n1 - 1):
        A1[row, row + 1] = 1
    A1[0, n1 - 1] = 1
    
    A1 = A1 + A1.T  

    # Construct adjacency matrix for stick part
    A2 = np.zeros((n2, n2))
    for row in range(n2 - 1):
        A2[row, row + 1] = 1
    A2 = A2 + A2.T  # Make the stick part matrix symmetric

    # Construct the full adjacency matrix
    A = np.zeros((n, n))
    A[:n1, :n1] = A1  # Add candy part to the full matrix
    A[n1:, n1:] = A2  # Add stick part to the full matrix

    # Connect the last node of the candy part to the first node of the stick part
    A[n1 - 1, n1] = 1
    A[n1, n1 - 1] = 1

    return A
    
    
def permute_isomorphism(A1):
    n = A1.shape[0]

    # Random permutation
    perm = np.random.permutation(n)
    A2 = A1[np.ix_(perm, perm)]

    # Get transition matrices
    P1 = adj_to_trans(A1)
    P2 = adj_to_trans(A2)

    # Get cost function
    c = get_degree_cost(A1, A2)

    # Run algorithm
    _, ortc_cost, ortc_weight = glop_v2(A1, A2, c, vertex=True)
    ortc_alignment = np.sum(ortc_weight, axis=(2, 3))
    _, otc_cost, _, otc_alignment = exact_otc(P1, P2, c)

    # Get alignment
    idx_ortc = np.argmax(ortc_alignment, axis=1)
    idx_otc = np.argmax(otc_alignment, axis=1)
    
    return math.isclose(ortc_cost, 0, abs_tol=1e-9), math.isclose(otc_cost, 0, abs_tol=1e-9), check_isomorphism(idx_ortc, A1, A2), check_isomorphism(idx_otc, A1, A2)
    
    
network_map = {
    'random_erdos_renyi': random_erdos_renyi,
    'random_sbm': random_sbm,
    'random_weighted_adjacency_matrix': random_weighted_adjacency_matrix,
    'random_tree': random_tree,
    'random_lollipop_fill': random_lollipop_fill,
    'random_lollipop': random_lollipop,
}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Isomorphism experiment')
    parser.add_argument('--network', choices=["random_erdos_renyi", "random_sbm", "random_weighted_adjacency_matrix", "random_tree", "random_lollipop", "random_lollipop_fill"], default="random_erdos_renyi")
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--params', nargs='+', type=float, help="Multiple float values (e.g., --params 0.5 1.0 2.0)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    args = parser.parse_args()
    print(args)

    ortc_detect_ratio, otc_detect_ratio, ortc_identify_ratio, otc_identify_ratio, n_total = isomorphism_experiment(args.network, args.n, tuple(args.params))
    print(ortc_detect_ratio, otc_detect_ratio, ortc_identify_ratio, otc_identify_ratio, n_total)
    