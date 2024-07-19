from scipy.optimize import linprog
import numpy as np

def generate_symmetric_connected_graph(n):
    # Step 1: Generate a random symmetric matrix
    adj_matrix = np.random.randint(0, 2, size=(n, n))
    adj_matrix = (adj_matrix + adj_matrix.T) // 2  # Ensure symmetry

    # Step 2: Ensure the graph is connected
    # Create a random permutation of nodes
    permuted_nodes = np.random.permutation(n)

    # Randomly connect nodes until the graph is connected
    connected_nodes = set([permuted_nodes[0]])
    while len(connected_nodes) < n:
        node = np.random.choice(list(connected_nodes))
        new_node = np.random.choice(np.delete(permuted_nodes, np.where(permuted_nodes == node)))
        adj_matrix[node, new_node] = 1
        adj_matrix[new_node, node] = 1
        connected_nodes.add(new_node)

    return np.array(adj_matrix)


def get_degree_cost(A1, A2):
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Compute node degrees.
    degrees1 = np.sum(A1, axis=1)
    degrees2 = np.sum(A2, axis=1)

    # Construct matrix.
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = (degrees1[i] - degrees2[j])**2
    return cost_mat


# Return a random adjacency matrix of stochastic block model
def stochastic_block_model(sizes, probs):
    
    # Check input type
    if not isinstance(sizes, np.ndarray) or len(sizes.shape) != 1:
        raise ValueError("'sizes' must be a 1D numpy array.")
    elif not isinstance(probs, np.ndarray) or probs.shape[0] != probs.shape[1]:
        raise ValueError("'probs' must be a square numpy array.")
    elif not np.allclose(probs, probs.T):
        raise ValueError("'probs' must be a symmetric matrix.")
    elif len(sizes) != probs.shape[0]:
        raise ValueError("'sizes' and 'probs' dimensions do not match.")
    
    n = np.sum(sizes)  # Total number of nodes
    n_b = len(sizes)   # Total number of blocks
    A = np.zeros((n, n))
    
    # Column index of each block's start
    start = [0] + list(np.cumsum(sizes))
    
    # Generating Adjacency Matrix (upper)
    # Generate diagonal blocks 
    for i in range(n_b):
        p = probs[i,i]
        for j in range(start[i], start[i+1]):
            for k in range(j+1, start[i+1]):
                A[j, k] = np.random.choice([0, 1], p=[1 - p, p])
    
    # Generate Nondiagonal blocks
    for i in range(n_b - 1):
        for j in range(i+1, n_b):
            A[start[i]:start[i+1], start[j]:start[j+1]] = np.random.choice([0, 1], size=(sizes[i], sizes[j]), p=[1 - probs[i,j], probs[i,j]])
    
    # Fill lower triangular matrix
    A = A + A.T
    
    return A


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def adj_to_trans(A):
    nrow = A.shape[0]
    T = np.copy(A).astype(float)
    for i in range(nrow):
        row = A[i, :]
        k = np.where(row != 0)[0]
        vals = softmax(row[k])
        for idx in range(len(k)):
            T[i, k[idx]] = vals[idx]

    return T


def check_isomorphism(idx, A1, A2):
    if A1.shape[0] != A2.shape[0]:
        return False

    if not np.array_equal(np.unique(idx), np.arange(1, A1.shape[0] + 1)):
        return False

    row, col = np.where(np.tril(A1) != 0)
    n = len(row)
    for i in range(n):
        if A2[idx[row[i]] - 1, idx[col[i]] - 1] != A1[row[i], col[i]]:
            return False

    row, col = np.where(np.tril(A2) != 0)
    inv_idx = np.zeros(A1.shape[0], dtype=int)
    for i in range(A1.shape[0]):
        inv_idx[i] = np.where(idx == i + 1)[0][0]
    n = len(row)
    for i in range(n):
        if A1[inv_idx[row[i]], inv_idx[col[i]]] != A2[row[i], col[i]]:
            return False

    return True

def get_zero_one_cost(A1, A2):
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    # Compute node degrees.
    degrees1 = np.sum(A1, axis=1)
    degrees2 = np.sum(A2, axis=1)

    # Construct matrix.
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if degrees1[i] != degrees2[j]:
                cost_mat[i,j] = 1
    return cost_mat


def is_connected(adj_matrix):
    def dfs(v, visited):
        visited[v] = True
        for i, weight in enumerate(adj_matrix[v]):
            if weight > 0 and not visited[i]:
                dfs(i, visited)

    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)

    # Start DFS from the first vertex (index 0)
    dfs(0, visited)

    return np.all(visited)