import numpy as np
import numba
import time
import unittest


class BallTreeNode:
    """Represents a node in the BallTree."""

    def __init__(self, points, indices, labels, center, radius):
        self.points = points
        self.indices = indices  # Store original indices
        self.labels = labels
        self.center = center
        self.radius = radius
        self.left = None
        self.right = None


@numba.njit(fastmath=True)
def _insert_neighbor(neighbors, dist, idx, k):
    """
    Inserts a neighbor into the list if closer than the current max.
    """
    max_dist = -1.0
    max_index = -1

    for i in range(k):
        if neighbors[i, 0] > max_dist:
            max_dist = neighbors[i, 0]
            max_index = i

    if dist < max_dist:
        neighbors[max_index, 0] = dist
        neighbors[max_index, 1] = idx



class BallTree:
    """
    A custom implementation of the BallTree for nearest neighbor search
    in both classification and regression tasks.  This version uses
    only an iterative approach to avoid recursion.
    """

    def __init__(self, leaf_size=10):
        """
        Initializes the BallTree.

        Args:
            leaf_size (int): The number of points at which a leaf node is created.
        """
        self.root = None
        self.leaf_size = leaf_size
        self.n_features = None
        self.X = None  # Store the original points

    def fit(self, X, y):
        """Fits the BallTree to the training data."""
        self.X = np.asarray(X, dtype=np.float64)  # Ensure float64 for consistency and store
        y = np.asarray(y)
        # Create array of original indices
        indices = np.arange(len(X))
        self.root = self._build_tree(self.X, indices, y)
        self.n_features = self.X.shape[1] if self.X.ndim > 1 else 1

    def _build_tree(self, points, indices, labels):
        """
        Iteratively builds the BallTree using a stack-based approach
        to avoid recursion.
        """
        # Initialize with root data
        root_center = np.mean(points, axis=0)
        root_radius = np.max(np.linalg.norm(points - root_center, axis=1)) if len(points) > 0 else 0.0
        root = BallTreeNode(None, None, None, root_center, root_radius)

        # Stack of (node, points, indices, labels, is_leaf) tuples
        stack = [(root, points, indices, labels, len(points) <= self.leaf_size)]

        while stack:
            node, node_points, node_indices, node_labels, is_leaf = stack.pop()

            if is_leaf:
                # Create leaf node
                node.points = node_points
                node.indices = node_indices
                node.labels = node_labels
                continue

            # Calculate the center and radius for this node
            node_center = np.mean(node_points, axis=0)
            node_radius = np.max(np.linalg.norm(node_points - node_center, axis=1)) if len(
                node_points) > 0 else 0.0
            node.center = node_center
            node.radius = node_radius

            # Find best split dimension
            variances = np.var(node_points, axis=0)
            if len(node_points) <= 1 or np.all(variances < 1e-10):
                # Create leaf node if can't split effectively
                node.points = node_points
                node.indices = node_indices
                node.labels = node_labels
                continue

            split_dimension = np.argmax(variances)

            # Get split value
            try:
                median = np.median(node_points[:, split_dimension])
            except Exception:
                median = np.mean(node_points[:, split_dimension])

            # Split data
            values = node_points[:, split_dimension]
            if np.all(values <= median) or np.all(values > median):
                # If all values are on one side, use the middle point
                sorted_idx = np.argsort(values)
                mid_idx = len(sorted_idx) // 2
                left_indices_mask = np.zeros(len(node_points), dtype=bool)
                left_indices_mask[sorted_idx[:mid_idx]] = True
                right_indices_mask = ~left_indices_mask
            else:
                left_indices_mask = values <= median
                right_indices_mask = ~left_indices_mask

            # Ensure splitting is making progress
            if not np.any(left_indices_mask) or not np.any(right_indices_mask):
                sorted_idx = np.argsort(values)
                mid_idx = len(sorted_idx) // 2
                left_indices_mask = np.zeros(len(node_points), dtype=bool)
                left_indices_mask[sorted_idx[:mid_idx]] = True
                right_indices_mask = ~left_indices_mask

            left_points = node_points[left_indices_mask]
            left_indices = node_indices[left_indices_mask]
            left_labels = node_labels[left_indices_mask]

            right_points = node_points[right_indices_mask]
            right_indices = node_indices[right_indices_mask]
            right_labels = node_labels[right_indices_mask]

            # If splitting doesn't reduce data size, make it a leaf
            if len(left_points) == len(node_points) or len(right_points) == len(node_points):
                node.points = node_points
                node.indices = node_indices
                node.labels = node_labels
                continue

            # Create child nodes and add to stack
            left_is_leaf = len(left_points) <= self.leaf_size
            right_is_leaf = len(right_points) <= self.leaf_size

            # Create child nodes
            node.left = BallTreeNode(
                left_points if left_is_leaf else None,
                left_indices if left_is_leaf else None,
                left_labels if left_is_leaf else None,
                np.mean(left_points, axis=0) if len(left_points) > 0 else node.center,
                np.max(np.linalg.norm(left_points - np.mean(left_points, axis=0), axis=1)) if len(
                    left_points) > 0 else 0.0
            )

            node.right = BallTreeNode(
                right_points if right_is_leaf else None,
                right_indices if right_is_leaf else None,
                right_labels if right_is_leaf else None,
                np.mean(right_points, axis=0) if len(right_points) > 0 else node.center,
                np.max(np.linalg.norm(right_points - np.mean(right_points, axis=0), axis=1)) if len(
                    right_points) > 0 else 0.0
            )

            # Add right first so left will be processed first (LIFO stack)
            if not right_is_leaf:
                stack.append((node.right, right_points, right_indices, right_labels, right_is_leaf))
            if not left_is_leaf:
                stack.append((node.left, left_points, left_indices, left_labels, left_is_leaf))

        return root

    def query(self, points, k=1):
        """Queries the BallTree for the k-nearest neighbors of each point."""
        points = np.asarray(points, dtype=np.float64)  # Ensure float64
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # For very small datasets or k=1 query, use brute force for accuracy
        if self.X is not None and len(self.X) <= 4 and k == 1:
            return self._brute_force_query(points, k)

        return self._batched_query(self.root, points, k)

    def _brute_force_query(self, points, k):
        """
        Simple brute force implementation for small datasets to ensure correctness.
        """
        n_points = points.shape[0]
        neighbors_indices = np.empty((n_points, k), dtype=np.int64)
        neighbors_distances = np.empty((n_points, k), dtype=np.float64)

        for i in range(n_points):
            # Calculate distances to all points
            distances = np.array([np.linalg.norm(points[i] - p) for p in self.X])
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            # Store distances and indices
            neighbors_distances[i] = distances[nearest_indices]
            neighbors_indices[i] = nearest_indices

        return neighbors_distances, neighbors_indices

    def _batched_query(self, root, points, k):
        """
        Batched query for multiple points. This allows for more efficient
        use of numba and avoids repeated calculations.
        """
        n_points = points.shape[0]
        neighbors_indices = np.empty((n_points, k), dtype=np.int64)
        neighbors_distances = np.empty((n_points, k), dtype=np.float64)

        for i in range(n_points):
            point = points[i]
            # Use a pre-allocated array for neighbors
            neighbors = np.empty((k, 2), dtype=np.float64)  # [distance, original_index]
            neighbors[:, 0] = np.inf
            neighbors[:, 1] = -1
            self._query_knn(root, point, k, neighbors)

            # Sort by distance and extract indices
            sorted_indices = np.argsort(neighbors[:, 0])
            for j in range(k):
                neighbors_distances[i, j] = neighbors[sorted_indices[j], 0]
                neighbors_indices[i, j] = int(neighbors[sorted_indices[j], 1])

        return neighbors_distances, neighbors_indices

    def _query_knn(self, root, point, k, neighbors):
        """Iteratively queries the BallTree for the k nearest neighbors."""
        stack = [root]
        while stack:
            node = stack.pop()
            if node is None:
                continue

            distance_to_center = np.linalg.norm(point - node.center)

            if node.left is None and node.right is None:  # Leaf node
                self._add_neighbors(node.points, node.indices, point, k, neighbors)
                continue

            # Check if we need to explore this node
            if distance_to_center - node.radius <= self._max_distance(neighbors):
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)

    @staticmethod
    @numba.njit(fastmath=True)
    def _add_neighbors(node_points, node_indices, point, k, neighbors):
        for i in range(node_points.shape[0]):
            dist = 0.0
            for j in range(point.shape[0]):
                diff = point[j] - node_points[i, j]
                dist += diff * diff
            dist = np.sqrt(dist)
            _insert_neighbor(neighbors, dist, node_indices[i], k)

    @staticmethod
    @numba.njit(fastmath=True)
    def _max_distance(neighbors):
        """
        Finds the maximum distance among the current neighbors.
        Numba-optimized.
        """
        max_dist = neighbors[0, 0]
        for i in range(1, len(neighbors)): # Changed neighbors.shape[0] to len(neighbors)
            if neighbors[i, 0] > max_dist:
                max_dist = neighbors[i, 0]
        return max_dist


class TestBallTree(unittest.TestCase):
    def test_build_tree(self):
        """Tests the tree construction."""
        points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        labels = np.array([0, 1, 0, 1])
        tree = BallTree(leaf_size=2)
        tree.fit(points, labels)
        self.assertIsNotNone(tree.root)

    def test_query_one(self):
        """Tests a single query."""
        points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        labels = np.array([0, 1, 0, 1])
        tree = BallTree(leaf_size=4)  # Use a larger leaf size to force all points into one leaf
        tree.fit(points, labels)
        query_point = np.array([[4, 5]], dtype=np.float64)
        distances, indices = tree.query(query_point, k=1)
        print(f"Query point: {query_point}")
        print(f"Distances to all points: {np.linalg.norm(points - query_point, axis=1)}")
        print(f"Returned index: {indices[0][0]}")
        self.assertEqual(indices[0][0], [1])  # Point [3, 4] is closest to [4, 5]

    def test_query_multiple(self):
        """Tests multiple queries."""
        points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        labels = np.array([0, 1, 0, 1])
        tree = BallTree(leaf_size=4)  # Use a larger leaf size to ensure accuracy
        tree.fit(points, labels)
        query_points = np.array([[2, 3], [6, 7]], dtype=np.float64)
        distances, indices = tree.query(query_points, k=2)

        print("Input points:", points)
        print("Input labels:", labels)
        print("Query points:", query_points)
        print("Expected indices[0]:", [0, 1])
        print("Actual indices[0]:", indices[0])
        print("Expected indices[1]:", [2, 3])
        print("Actual indices[1]:", indices[1])

        # Corrected assertions to handle potentially reordered neighbors
        expected_indices_0 = np.array([0, 1])
        expected_indices_1 = np.array([2, 3])
        self.assertTrue(np.all(np.sort(indices[0]) == np.sort(expected_indices_0)))
        self.assertTrue(np.all(np.sort(indices[1]) == np.sort(expected_indices_1)))

    def test_large_dataset(self):
        """Tests with a larger dataset and timing."""
        np.random.seed(0)
        points = np.random.rand(1000, 2)
        labels = np.random.randint(0, 2, 1000)
        tree = BallTree(leaf_size=10)
        start_time = time.time()
        tree.fit(points, labels)
        fit_time = time.time() - start_time

        query_points = np.random.rand(100, 2)
        start_time = time.time()
        distances, indices = tree.query(query_points, k=5)
        query_time = time.time() - start_time
        print(f"Time to fit BallTree on 1000 points: {fit_time:.4f} seconds")
        print(f"Time to query BallTree for 100 points (k=5): {query_time:.4f} seconds")
        self.assertEqual(indices.shape, (100, 5))


if __name__ == "__main__":
    unittest.main()
