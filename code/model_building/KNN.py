import numpy as np
from typing import Any, List, Optional, Union
from numba import njit
from numba.typed import List as NumbaList

class KDTreeNode:
    """
    Node in a KD-Tree for KNN acceleration.

    Attributes:
        point: Coordinates of the data point.
        label: Corresponding target value.
        split_dim: Dimension to split on.
        left, right: Subtrees.
    """
    __slots__ = ('point', 'label', 'split_dim', 'left', 'right')

    def __init__(self, point: np.ndarray, label: Any, split_dim: int,
                 left: Optional['KDTreeNode'], right: Optional['KDTreeNode']):
        self.point = point
        self.label = label
        self.split_dim = split_dim
        self.left = left
        self.right = right


def build_kdtree(data: np.ndarray, labels: np.ndarray, depth: int = 0) -> Optional[KDTreeNode]:
    """
    Build KD-Tree recursively using argpartition for median selection.
    Complexity: O(n log n) average.
    """
    n, dim = data.shape
    if n == 0:
        return None
    axis = depth % dim
    idx = np.argpartition(data[:, axis], n // 2)
    mid = idx[n // 2]
    return KDTreeNode(
        data[mid],
        labels[mid],
        axis,
        build_kdtree(data[idx[:n//2]], labels[idx[:n//2]], depth + 1),
        build_kdtree(data[idx[n//2+1:]], labels[idx[n//2+1:]], depth + 1)
    )

@njit
def knn_query_numba(points_arr, labels_arr, split_dims, k, query_point):
    """
    Numba-accelerated KD-Tree search using flat arrays.
    Returns: array of k nearest labels.
    """
    n_nodes = points_arr.shape[0]
    dist2s = np.full(k, np.inf)
    labels_out = np.full(k, -1.0)
    stack = NumbaList()
    stack.append(0)
    while stack:
        idx = stack.pop()
        pt = points_arr[idx]
        diff = query_point - pt
        d2 = diff.dot(diff)
        max_i = 0
        max_val = dist2s[0]
        for j in range(1, k):
            if dist2s[j] > max_val:
                max_val = dist2s[j]
                max_i = j
        if d2 < dist2s[max_i]:
            dist2s[max_i] = d2
            labels_out[max_i] = labels_arr[idx]
        axis = split_dims[idx]
        diff_axis = query_point[axis] - pt[axis]
        near = idx * 2 + 1 if diff_axis <= 0 else idx * 2 + 2
        far = idx * 2 + 2 if diff_axis <= 0 else idx * 2 + 1
        if far < n_nodes:
            stack.append(far)
        if near < n_nodes:
            stack.append(near)
    return labels_out

class KNN:
    """
    K-Nearest Neighbors using custom KD-Tree with optional Numba acceleration.

    Args:
        k: Number of neighbors.
        task: 'classification' or 'regression'.
        verbose: If True, prints progress logs.
    """
    def __init__(self, k: int = 5, task: str = 'classification', verbose: bool = False):
        if task not in ('classification', 'regression'):
            raise ValueError("task must be 'classification' or 'regression'")
        self.k = k
        self.task = task
        self.verbose = verbose
        self.tree = None
        self.points_arr = None
        self.labels_arr = None
        self.split_dims = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build KD-Tree and flatten arrays for Numba query."""
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        self.tree = build_kdtree(X_arr, y_arr)
        # Flatten tree via BFS into numpy arrays
        nodes = []
        queue = [(self.tree, 0)]
        while queue:
            node, idx = queue.pop(0)
            if node is None:
                continue
            if len(nodes) <= idx:
                nodes.extend([None] * (idx + 1 - len(nodes)))
            nodes[idx] = node
            queue.append((node.left, idx*2+1))
            queue.append((node.right, idx*2+2))
        # Initialize flat arrays
        n_nodes = len(nodes)
        dim = X_arr.shape[1]
        self.points_arr = np.zeros((n_nodes, dim), dtype=np.float64)
        self.labels_arr = np.zeros(n_nodes, dtype=np.float64)
        self.split_dims = np.zeros(n_nodes, dtype=np.int32)
        for idx, node in enumerate(nodes):
            if node:
                self.points_arr[idx] = node.point
                self.labels_arr[idx] = node.label
                self.split_dims[idx] = node.split_dim
        if self.verbose:
            print(f"KD-Tree built with {n_nodes} nodes, dimensions={dim}")

    def _knn_query(self, root: KDTreeNode, point: np.ndarray) -> np.ndarray:
        """
        Iterative KD-Tree search using fixed-size numpy arrays for heap.
        Returns k neighbor labels.
        """
        k = self.k
        dist2s = np.full(k, np.inf)
        labels = np.empty(k, dtype=object)
        stack = [root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            diff = point - node.point
            d2 = diff.dot(diff)
            max_idx = np.argmax(dist2s)
            if d2 < dist2s[max_idx]:
                dist2s[max_idx] = d2
                labels[max_idx] = node.label
            axis = node.split_dim
            if point[axis] <= node.point[axis]:
                stack.append(node.right)
                stack.append(node.left)
            else:
                stack.append(node.left)
                stack.append(node.right)
        return labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for data X."""
        if self.tree is None:
            raise RuntimeError("Model not fitted.")
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        n = X_arr.shape[0]
        preds = np.empty(n, dtype=object if self.task=='classification' else float)
        for i in range(n):
            if self.verbose and i and i % 1000 == 0:
                print(f"Processed {i}/{n}")
            # use Numba-accelerated query
            neighbors = knn_query_numba(self.points_arr, self.labels_arr, self.split_dims, self.k, X_arr[i])
            if self.task == 'classification':
                classes, counts = np.unique(neighbors, return_counts=True)
                preds[i] = classes[np.argmax(counts)]
            else:
                preds[i] = np.mean(neighbors)
        if self.verbose:
            print("Prediction complete")
        return preds

if __name__ == '__main__':
    # Quick test
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 3, 100)
    model = KNN(k=5, task='classification', verbose=True)
    model.fit(X, y)
    print(model.predict(X[:5]))
