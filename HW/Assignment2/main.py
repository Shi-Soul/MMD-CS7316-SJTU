import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import time
import psutil
import os

# Parameters
damping = 0.85
max_iter = 100
tol = 1e-6
chunksize = 10_000_000


def load_graph_sparse_v2(path, num_nodes=None):
    """
    Two-pass CSR builder without edge list:
    1) First pass counts out-degrees, max_id, and total edges.
    2) Preallocate arrays; second pass fills rows, cols, data.
    Uses float32 for memory efficiency.
    """
    out_deg = {}
    max_id = 0
    total_edges = 0
    # First pass: count
    for chunk in pd.read_csv(path, names=['From','To'], skiprows=1, chunksize=chunksize):
        total_edges += len(chunk)
        for src, dst in zip(chunk['From'], chunk['To']):
            out_deg[src] = out_deg.get(src, 0) + 1
            if src > max_id: max_id = src
            if dst > max_id: max_id = dst
    n = num_nodes or (max_id + 1)
    # Preallocate
    rows = np.empty(total_edges, dtype=np.int32)
    cols = np.empty(total_edges, dtype=np.int32)
    data = np.empty(total_edges, dtype=np.float32)
    # Second pass: fill
    idx = 0
    for chunk in pd.read_csv(path, names=['From','To'], skiprows=1, chunksize=chunksize):
        for src, dst in zip(chunk['From'], chunk['To']):
            rows[idx] = dst
            cols[idx] = src
            data[idx] = 1.0 / out_deg[src]
            idx += 1
    # Build CSR
    M = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return M


def load_graph_sparse_v1(path, num_nodes=None):
    """
    Load edge list from CSV and build CSR sparse matrix M where M[j,i] = 1/out_degree(i)
    representing links from node i to j (column-stochastic).
    """
    edges = []
    out_deg = {}
    max_id = 0
    for chunk in pd.read_csv(path, names=['From','To'], skiprows=1, chunksize=chunksize):
        for src, dst in zip(chunk['From'], chunk['To']):
            edges.append((src, dst))
            out_deg[src] = out_deg.get(src, 0) + 1
            max_id = max(max_id, src, dst)
    n = num_nodes or (max_id + 1)
    rows, cols, data = [], [], []
    for src, dst in edges:
        rows.append(dst)
        cols.append(src)
        data.append(1.0 / out_deg[src])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def pagerank(M, damping=0.85, max_iter=100, tol=1e-6):
    n = M.shape[0]
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
    proc = psutil.Process(os.getpid())
    start_time = time.time()
    mem_start = proc.memory_info().rss / (1024 * 1024)  # in MB

    # Load graph
    graph_file = 'web_links.csv'
    print('Loading graph...')
    load_start = time.time()
    M = load_graph_sparse_v2(graph_file)
    load_end = time.time()
    mem_after_load = proc.memory_info().rss / (1024 * 1024)
    print(f"Graph loaded in {load_end - load_start:.2f}s; memory usage: {mem_after_load - mem_start:.2f} MB")
    print(f"M.shape = ",M.shape)

    # Compute PageRank
    print('Computing PageRank...')
    pr_start = time.time()
    ranks = pagerank(M, damping, max_iter, tol)
    pr_end = time.time()
    mem_after_pr = proc.memory_info().rss / (1024 * 1024)
    print(f"PageRank computed in {pr_end - pr_start:.2f}s; memory usage increase: {mem_after_pr - mem_after_load:.2f} MB")

    # Extract top 1000
    print('Selecting top 1000 nodes...')
    idx = np.argsort(-ranks)[:1000]
    top_nodes = pd.DataFrame({'NodeId': idx, 'PageRank_Value': ranks[idx]})
    top_nodes.to_csv('test_prediction.csv', index=False)

    # Final stats
    end_time = time.time()
    mem_end = proc.memory_info().rss / (1024 * 1024)
    print(f"Total time: {end_time - start_time:.2f}s; total memory increase: {mem_end - mem_start:.2f} MB")
    print('Saved test_prediction.csv')