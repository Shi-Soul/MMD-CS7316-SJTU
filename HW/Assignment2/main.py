import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# Parameters
damping = 0.85
max_iter = 100
tol = 1e-6

# Step 1: Load data and build sparse adjacency matrix
def load_graph(path, num_nodes=None):
    """
    Load edge list from CSV and build CSR sparse matrix M where M[j,i] = 1/out_degree(i)
    representing links from node i to j (column-stochastic).
    """
    # Read in chunks to save memory
    chunksize = 10_000_000
    edges = []
    out_deg = {}
    max_id = 0
    for chunk in pd.read_csv(path, names=['From','To'], skiprows=1, chunksize=chunksize):
        for src, dst in zip(chunk['From'], chunk['To']):
            edges.append((src, dst))
            out_deg[src] = out_deg.get(src, 0) + 1
            if src > max_id: max_id = src
            if dst > max_id: max_id = dst
    n = (num_nodes or (max_id + 1))
    # Build index lists
    rows, cols, data = [], [], []
    for src, dst in edges:
        rows.append(dst)
        cols.append(src)
        data.append(1.0 / out_deg[src])
    M = csr_matrix((data, (rows, cols)), shape=(n, n))
    return M

# Step 2: PageRank via power iteration on sparse matrix
def pagerank(M, damping=0.85, max_iter=100, tol=1e-6):
    n = M.shape[0]
    # Initialize rank vector
    r = np.ones(n) / n
    teleport = (1.0 - damping) / n
    for it in range(max_iter):
        r_new = damping * (M @ r) + teleport
        err = np.linalg.norm(r_new - r, 1)
        if err < tol:
            print(f"Converged at iteration {it+1} with error {err:.2e}")
            break
        r = r_new
    return r

if __name__ == '__main__':
    # Load and compute PageRank
    graph_file = 'web_links.csv'
    print('Loading graph...')
    M = load_graph(graph_file)
    print('Computing PageRank...')
    ranks = pagerank(M, damping, max_iter, tol)

    # Extract top 1000 nodes
    print('Selecting top 1000 nodes...')
    idx = np.argsort(-ranks)[:1000]
    top_nodes = pd.DataFrame({
        'NodeId': idx,
        'PageRank_Value': ranks[idx]
    })
    # Save to CSV with header
    top_nodes.to_csv('test_prediction.csv', index=False)
    print('Saved test_prediction.csv')
