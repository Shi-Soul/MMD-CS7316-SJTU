#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import time
import os
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False


def compute_item_similarity(R):
    """
    Compute cosine similarity matrix for items using full matrix operations.
    R: (n_users, n_items) normalized and centered matrix (zeros for masked).
    Returns: similarity matrix (n_items, n_items).
    """
    dot = R.T @ R                                 # (n_items, n_items)
    norms = np.linalg.norm(R, axis=0)             # (n_items,)
    denom = np.outer(norms, norms)                # (n_items, n_items)
    sim = np.zeros_like(dot)
    mask = denom > 0
    sim[mask] = dot[mask] / denom[mask]
    return sim


def parse_args():
    parser = argparse.ArgumentParser(description='Item-CF with validation and performance metrics')
    parser.add_argument('--no-val', action='store_true', help='Disable validation split')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fraction for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--K', type=int, default=10, help='Number of top-K neighbors')
    parser.add_argument('--test-output', type=str, default='test_prediction.csv', help='Output file for test predictions')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    start_time = time.time()

    # 1. Load data
    data = pd.read_csv('col_matrix.csv', header=None).values  # shape (6040,3952)
    n_users, n_items = data.shape

    # 2. Mask test region
    test_u, test_i = 4100, 2700
    mask_test = np.zeros_like(data, dtype=bool)
    mask_test[test_u:, test_i:] = True

    # 3. Create validation split
    if not args.no_val:
        known = (data > 0) & (~mask_test)
        idx = np.argwhere(known)
        n_val = int(len(idx) * args.val_ratio)
        sel = np.random.choice(len(idx), n_val, replace=False)
        val_idx = idx[sel]
        mask_val = np.zeros_like(data, dtype=bool)
        mask_val[val_idx[:,0], val_idx[:,1]] = True
        print(f"Validation enabled: {n_val} entries")
    else:
        mask_val = np.zeros_like(data, dtype=bool)
        val_idx = np.empty((0,2), dtype=int)
        print("Validation disabled")

    # 4. Build training matrix and mask matrix
    train = data.copy()
    train[mask_test | mask_val] = 0
    mask_mat = (train > 0).astype(float)

    # 5. Center by item average, masked remain zero
    item_sum = (train * mask_mat).sum(axis=0)
    item_count = mask_mat.sum(axis=0)
    item_avg = np.zeros(n_items)
    nonzero = item_count > 0
    item_avg[nonzero] = item_sum[nonzero] / item_count[nonzero]
    R = (train - item_avg[None, :]) * mask_mat

    # 6. Compute similarity
    sim_start = time.time()
    S = compute_item_similarity(R)
    # Keep top-K neighbors only
    K = args.K
    idx_topk = np.argpartition(-np.abs(S), K, axis=1)[:, :K]
    rows = np.repeat(np.arange(n_items)[:,None], K, axis=1)
    S_k = np.zeros_like(S)
    S_k[rows, idx_topk] = S[rows, idx_topk]
    sim_time = time.time() - sim_start
    print(f"Similarity computed in {sim_time:.2f}s")

    # 7. Predict normalized ratings for all users and items
    pred_norm = R @ S_k.T
    denom = np.sum(np.abs(S_k), axis=1)
    denom[denom==0] = 1e-8
    pred_norm /= denom[None, :]
    pred_all = np.clip(np.rint(pred_norm + item_avg[None, :]), 0, 5).astype(int)

    # 8. Validation accuracy
    if val_idx.size:
        u_val = val_idx[:,0]; i_val = val_idx[:,1]
        preds_val = pred_all[u_val, i_val]
        actual = data[u_val, i_val].astype(int)
        print(actual[:100],'\n', preds_val[:100])
        correct = np.sum(preds_val == actual)
        mse = np.mean((preds_val - actual) ** 2)
        acc = correct / len(actual)
        print(f"Validation accuracy: {acc:.4f} ({correct}/{len(actual)})")
        print(f"Validation MSE: {mse:.4f}")

    # 9. Test predictions
    pred_test = pred_all[test_u:, test_i:]
    np.savetxt(args.test_output, pred_test, fmt='%d', delimiter=',')
    print(f"Test predictions saved to {args.test_output} with shape {pred_test.shape}")

    # 10. Performance metrics
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    if has_psutil:
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info().rss / (1024**2)
        print(f"Memory usage: {mem:.2f} MB")
    else:
        print("psutil not installed, skipping memory usage report.")

if __name__ == '__main__':
    main()
