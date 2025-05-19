import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple
from numba import njit, prange
from numba.typed import List as NumbaList
from collections import deque  # Ensure deque is imported

from .BaseModel import BaseModel


class KDTreeNode:
    __slots__ = ('point', 'label', 'split_dim', 'left', 'right', 'idx')

    def __init__(self, point: np.ndarray, label: Any, split_dim: int,
                 left: Optional['KDTreeNode'], right: Optional['KDTreeNode'], idx: int):
        self.point = point
        self.label = label
        self.split_dim = split_dim
        self.left = left
        self.right = right
        self.idx = idx


@njit(cache=True, fastmath=True)
def _knn_search_core(points_arr: np.ndarray, labels_arr: np.ndarray, split_dims_arr: np.ndarray,
                     k: int, query_point: np.ndarray, n_nodes: int,
                     tree_depth_unused: int) -> np.ndarray:
    best_dist2s = np.full(k, np.inf, dtype=np.float64)
    best_labels = np.full(k, -1.0, dtype=np.float64)
    node_stack = NumbaList()
    if n_nodes > 0 and split_dims_arr[0] != -1:
        node_stack.append(0)
    while len(node_stack) > 0:
        current_node_idx = node_stack.pop()
        if split_dims_arr[current_node_idx] == -1:
            continue
        point = points_arr[current_node_idx]
        label = labels_arr[current_node_idx]
        split_dim = split_dims_arr[current_node_idx]
        diff = query_point - point
        d2 = diff.dot(diff)
        idx_max_dist = 0
        for i in range(1, k):
            if best_dist2s[i] > best_dist2s[idx_max_dist]:
                idx_max_dist = i
        current_max_k_dist2 = best_dist2s[idx_max_dist]
        if d2 < current_max_k_dist2:
            best_dist2s[idx_max_dist] = d2
            best_labels[idx_max_dist] = label
            new_idx_max_dist = 0
            for i in range(1, k):
                if best_dist2s[i] > best_dist2s[new_idx_max_dist]:
                    new_idx_max_dist = i
            current_max_k_dist2 = best_dist2s[new_idx_max_dist]
        query_val_at_dim = query_point[split_dim]
        point_val_at_dim = point[split_dim]
        left_child_idx = 2 * current_node_idx + 1
        right_child_idx = 2 * current_node_idx + 2
        if query_val_at_dim <= point_val_at_dim:
            near_child_idx, far_child_idx = left_child_idx, right_child_idx
        else:
            near_child_idx, far_child_idx = right_child_idx, left_child_idx
        if near_child_idx < n_nodes and split_dims_arr[near_child_idx] != -1:
            node_stack.append(near_child_idx)
        dist_to_plane_sq = (query_val_at_dim - point_val_at_dim) ** 2
        if far_child_idx < n_nodes and split_dims_arr[far_child_idx] != -1 and \
                dist_to_plane_sq < current_max_k_dist2:
            node_stack.append(far_child_idx)
    return best_labels


# MODULE-LEVEL HELPER FUNCTION (MOVED FROM BEING NESTED)
def _build_kdtree_recursive_nodes(
        current_indices: np.ndarray,
        depth: int,
        X_data: np.ndarray,
        y_data_numeric_float64: np.ndarray,
        dataset_dim: int
) -> Tuple[Optional[KDTreeNode], int]:
    """
    Recursive helper to build KDTreeNode structure.
    Returns the node and the max depth reached in this branch.
    """
    current_branch_max_depth = depth
    n = len(current_indices)
    if n == 0:
        return None, current_branch_max_depth

    axis = depth % dataset_dim

    current_X_subset = X_data[current_indices]  # Data for points in current_indices

    mid_idx_in_subset = n // 2
    # `partitioned_indices_in_subset` are indices relative to `current_X_subset` (and thus `current_indices`)
    partitioned_indices_in_subset = np.argpartition(
        current_X_subset[:, axis], mid_idx_in_subset, kind='introselect'
    )

    # Index of median within `current_indices` array
    median_idx_in_current_indices_array = partitioned_indices_in_subset[mid_idx_in_subset]
    # Actual index in the original full X_data, y_data_numeric_float64
    median_original_overall_idx = current_indices[median_idx_in_current_indices_array]

    # Get subsets of `current_indices` for left and right children
    left_original_indices = current_indices[partitioned_indices_in_subset[:mid_idx_in_subset]]
    right_original_indices = current_indices[partitioned_indices_in_subset[mid_idx_in_subset + 1:]]

    left_node, left_max_depth = _build_kdtree_recursive_nodes(
        left_original_indices, depth + 1, X_data, y_data_numeric_float64, dataset_dim
    )
    current_branch_max_depth = max(current_branch_max_depth, left_max_depth)

    right_node, right_max_depth = _build_kdtree_recursive_nodes(
        right_original_indices, depth + 1, X_data, y_data_numeric_float64, dataset_dim
    )
    current_branch_max_depth = max(current_branch_max_depth, right_max_depth)

    node = KDTreeNode(
        point=X_data[median_original_overall_idx],
        label=y_data_numeric_float64[median_original_overall_idx],
        split_dim=axis,
        left=left_node,
        right=right_node,
        idx=-1  # Placeholder, will be set during BFS flattening
    )
    return node, current_branch_max_depth


def build_kdtree_iterative_flat(X: np.ndarray, y_numeric_float64: np.ndarray, verbose: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, int, int]:
    n_samples, dim = X.shape
    if n_samples == 0:
        return (np.empty((0, dim), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                0, 0)

    if verbose: print("Building KD-Tree node structure...")
    # Call the module-level recursive helper
    root_node, max_tree_depth = _build_kdtree_recursive_nodes(
        np.arange(n_samples), 0, X, y_numeric_float64, dim
    )

    if root_node is None:  # Should only happen if n_samples was 0, already handled.
        return (np.empty((0, dim), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                0, 0)  # max_tree_depth would be 0

    if verbose: print(f"KDTreeNode structure built. Max depth: {max_tree_depth}. Starting flattening...")

    # Flatten tree via BFS into numpy arrays
    flat_idx_to_kdtreenode = {}
    bfs_queue = deque()

    if root_node:
        bfs_queue.append((root_node, 0))
        flat_idx_to_kdtreenode[0] = root_node

    max_flat_idx_reached = 0

    while bfs_queue:
        current_kdtree_node, current_flat_idx = bfs_queue.popleft()
        max_flat_idx_reached = max(max_flat_idx_reached, current_flat_idx)

        if current_kdtree_node.left:
            left_flat_idx = 2 * current_flat_idx + 1
            flat_idx_to_kdtreenode[left_flat_idx] = current_kdtree_node.left
            bfs_queue.append((current_kdtree_node.left, left_flat_idx))
        if current_kdtree_node.right:
            right_flat_idx = 2 * current_flat_idx + 2
            flat_idx_to_kdtreenode[right_flat_idx] = current_kdtree_node.right
            bfs_queue.append((current_kdtree_node.right, right_flat_idx))

    num_flat_array_elements = max_flat_idx_reached + 1
    actual_num_nodes = len(flat_idx_to_kdtreenode)

    if verbose:
        print(f"Flattening complete: {actual_num_nodes} actual nodes. Flat array size: {num_flat_array_elements}.")

    points_arr = np.full((num_flat_array_elements, dim), np.nan, dtype=np.float64)
    labels_arr = np.full(num_flat_array_elements, np.nan, dtype=np.float64)
    split_dims_arr = np.full(num_flat_array_elements, -1, dtype=np.int32)

    for flat_idx, node_obj in flat_idx_to_kdtreenode.items():
        points_arr[flat_idx] = node_obj.point
        labels_arr[flat_idx] = node_obj.label
        split_dims_arr[flat_idx] = node_obj.split_dim

    return points_arr, labels_arr, split_dims_arr, num_flat_array_elements, max_tree_depth


class KNN(BaseModel):
    def __init__(self, k: int = 5, task: str = 'classification', verbose: bool = False):
        super().__init__()
        if task not in ('classification', 'regression'):
            raise ValueError("Task must be 'classification' or 'regression'.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k
        self.task = task
        self.verbose = verbose

        self.tree_points: Optional[np.ndarray] = None
        self.tree_labels: Optional[np.ndarray] = None
        self.tree_split_dims: Optional[np.ndarray] = None
        self.n_tree_nodes_allocated: int = 0
        self.tree_depth: int = 0

        self.categories_: Optional[pd.Index] = None
        self.label_mapper_: Optional[pd.Index] = None

    def fit(self, X: np.ndarray, y: Any) -> None:
        X_arr = np.ascontiguousarray(X, dtype=np.float64)

        y_np: np.ndarray
        self.categories_ = None
        self.label_mapper_ = None

        if isinstance(y, pd.Series) and isinstance(y.dtype, pd.CategoricalDtype):
            if self.verbose: print("Input y is pandas Series with CategoricalDtype.")
            self.categories_ = y.cat.categories
            y_np = y.to_numpy()
        elif hasattr(y, 'to_numpy'):
            if self.verbose: print("Input y is pandas Series or similar.")
            y_np = y.to_numpy()
        else:
            if self.verbose: print("Input y is NumPy array or list.")
            y_np = np.asarray(y)

        y_arr: np.ndarray
        if self.task == 'regression':
            if not np.issubdtype(y_np.dtype, np.number):
                raise ValueError(f"Regression task requires numeric labels. Got dtype {y_np.dtype}.")
            y_arr = y_np.astype(np.float64)
        else:
            if np.issubdtype(y_np.dtype, np.number):
                if self.verbose: print(f"Numeric labels detected for classification (dtype: {y_np.dtype}).")
                y_arr = y_np.astype(np.float64)
            else:
                if self.verbose: print(f"Non-numeric labels (dtype: {y_np.dtype}) for classification. Factorizing.")
                integer_codes, uniques = pd.factorize(y_np, sort=True)
                self.label_mapper_ = uniques
                y_arr = integer_codes.astype(np.float64)

        y_arr = np.ascontiguousarray(y_arr)

        if X_arr.shape[0] == 0:
            raise ValueError("Cannot fit on empty data X.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. X has {X_arr.shape[0]}, y processed to {y_arr.shape[0]}.")

        if self.k > X_arr.shape[0]:
            if self.verbose:
                print(
                    f"Warning: k ({self.k}) is greater than number of samples ({X_arr.shape[0]}). Setting k to {X_arr.shape[0]}.")
            self.k = X_arr.shape[0]

        if X_arr.shape[0] > 0 and self.k == 0:
            self.k = 1

        self.tree_points, self.tree_labels, self.tree_split_dims, \
            self.n_tree_nodes_allocated, self.tree_depth = \
            build_kdtree_iterative_flat(X_arr, y_arr, self.verbose)

        if self.verbose:
            actual_nodes_count = np.sum(
                self.tree_split_dims != -1) if self.tree_split_dims is not None and self.tree_split_dims.size > 0 else 0
            print(
                f"KD-Tree built. Flat array size: {self.n_tree_nodes_allocated}. Actual nodes: {actual_nodes_count}. Dimensions={X_arr.shape[1]}. Max depth: {self.tree_depth}.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_points is None or self.n_tree_nodes_allocated == 0:
            n_queries = X.shape[0]
            if self.task == 'regression':
                return np.full(n_queries, np.nan, dtype=np.float64)
            else:
                return np.full(n_queries, None, dtype=object)

        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        n_queries = X_arr.shape[0]

        k_for_predict = max(1, self.k) if self.n_tree_nodes_allocated > 0 else self.k
        if k_for_predict == 0:
            if self.task == 'regression':
                return np.full(n_queries, np.nan, dtype=np.float64)
            else:
                return np.full(n_queries, None, dtype=object)

        preds_numeric = np.empty(n_queries, dtype=np.float64)

        for i in prange(n_queries):
            query_point = X_arr[i]
            neighbor_labels = _knn_search_core(
                self.tree_points, self.tree_labels, self.tree_split_dims,
                k_for_predict, query_point, self.n_tree_nodes_allocated, self.tree_depth
            )
            valid_neighbor_labels = neighbor_labels[neighbor_labels != -1.0]
            if np.isnan(valid_neighbor_labels).any():
                valid_neighbor_labels = valid_neighbor_labels[~np.isnan(valid_neighbor_labels)]

            if len(valid_neighbor_labels) == 0:
                preds_numeric[i] = np.nan
                continue

            if self.task == 'classification':
                u_labels, u_counts = np.unique(valid_neighbor_labels, return_counts=True)
                preds_numeric[i] = u_labels[np.argmax(u_counts)]
            else:
                preds_numeric[i] = np.mean(valid_neighbor_labels)

        if self.verbose and n_queries > 0:
            print(f"Numeric predictions generated. Processed {n_queries}/{n_queries}.")

        if self.task == 'classification':
            output_dtype = object
            if self.label_mapper_ is not None:
                output_dtype = self.label_mapper_.dtype
            elif self.categories_ is not None:
                output_dtype = self.categories_.dtype

            final_preds = np.empty(n_queries, dtype=output_dtype)

            for i in range(n_queries):
                numeric_pred_val = preds_numeric[i]
                if np.isnan(numeric_pred_val):
                    final_preds[i] = np.nan if pd.api.types.is_numeric_dtype(final_preds.dtype) else None
                    continue

                if self.label_mapper_ is not None:
                    int_code = int(round(numeric_pred_val))
                    if 0 <= int_code < len(self.label_mapper_):
                        final_preds[i] = self.label_mapper_[int_code]
                    else:
                        final_preds[i] = np.nan if pd.api.types.is_numeric_dtype(final_preds.dtype) else None
                elif self.categories_ is not None:
                    if pd.api.types.is_integer_dtype(self.categories_.dtype):
                        final_preds[i] = int(round(numeric_pred_val))
                    elif pd.api.types.is_float_dtype(self.categories_.dtype):
                        final_preds[i] = float(numeric_pred_val)
                    elif pd.api.types.is_bool_dtype(self.categories_.dtype):
                        final_preds[i] = bool(round(numeric_pred_val))
                    else:
                        final_preds[i] = numeric_pred_val
                else:
                    if numeric_pred_val == round(numeric_pred_val):
                        final_preds[i] = int(round(numeric_pred_val))
                    else:
                        final_preds[i] = float(numeric_pred_val)
            if self.verbose and n_queries > 0: print("Classification labels mapped back to original types.")
            return final_preds
        else:
            if self.verbose and n_queries > 0: print("Regression predictions complete.")
            return preds_numeric