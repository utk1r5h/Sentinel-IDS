import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=15, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features  # Number of features to consider per split
        self.tree = None

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = self._gini(y)
        best_idx, best_thr = None, None

    
        if self.max_features is not None and self.max_features < n:
            feature_indices = np.random.choice(n, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n)

        for idx in feature_indices:

            col_vals = X[:, idx]
            thresholds = np.percentile(col_vals, np.linspace(5, 95, 20))
            thresholds = np.unique(thresholds)  # Remove duplicates

            for thr in thresholds:
                left_mask = col_vals <= thr
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                g = (left_mask.sum() * self._gini(y[left_mask]) +
                     right_mask.sum() * self._gini(y[right_mask])) / m

                if g < best_gini:
                    best_gini = g
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def fit(self, X, y, depth=0):
        classes, counts = np.unique(y, return_counts=True)
        predicted_class = classes[np.argmax(counts)]

        if depth >= self.max_depth or len(classes) == 1:
            return predicted_class

        idx, thr = self._best_split(X, y)
        if idx is None:
            return predicted_class

        left_mask = X[:, idx] <= thr
        right_mask = ~left_mask

        node = {
            'index': idx,
            'threshold': thr,
            'left': self.fit(X[left_mask], y[left_mask], depth + 1),
            'right': self.fit(X[right_mask], y[right_mask], depth + 1),
        }
        return node

    def predict_row(self, row, node):
        if isinstance(node, dict):
            if row[node['index']] <= node['threshold']:
                return self.predict_row(row, node['left'])
            else:
                return self.predict_row(row, node['right'])
        return node  #


class RandomForestFromScratch:


    def __init__(self, n_trees=20, max_depth=15, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features  
        self.trees = []
        self.classes_ = None

    def _resolve_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def fit(self, X, y):
    
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        mf = self._resolve_max_features(n_features)

        self.trees = []
        for i in range(self.n_trees):
            
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTree(max_depth=self.max_depth, max_features=mf)
            tree.tree = tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if (i + 1) % 5 == 0:
                print(f"  Trees trained: {i + 1}/{self.n_trees}")

    def predict(self, X):

      
        tree_preds = np.array([
            [tree.predict_row(row, tree.tree) for row in X]
            for tree in self.trees
        ])
        
        return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

    def predict_proba(self, X):
 
        tree_preds = np.array([
            [tree.predict_row(row, tree.tree) for row in X]
            for tree in self.trees
        ])

        result = []
        for i in range(X.shape[0]):
            votes = Counter(tree_preds[:, i])
            total = sum(votes.values())
            proba = {c: votes.get(c, 0) / total for c in self.classes_}
            result.append(proba)
        return result
