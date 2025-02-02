import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../Regression-Trees---From-Scratch"))
from regression_trees import RegressionTree

class GradientBoosting:
    def __init__(self, X, y, learning_rate=0.1, n_trees = 100, max_depth = 4, verbose = False):
        self.X = np.array(X)
        self.y = np.array(y)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.verbose = verbose
        self.initial_prediction = np.mean(self.y)

    def fit(self):
        self._tree_list = []
        current_prediction = self.initial_prediction

        for i in range(self.n_trees):
            # Calculate residuals (errors)
            residuals = self.y - current_prediction

            # Train a regression tree on the residuals
            tree = RegressionTree(self.X, residuals, max_depth=self.max_depth)
            tree.fit()
            self._tree_list.append(tree)

            # Make predictions using the current tree
            tree_pred = tree.predict(self.X)

            # Update current predictions with the scaled predictions from the current tree
            current_prediction = current_prediction + self.learning_rate*tree_pred

            # Calculate Mean Squared Error (MSE) for the current model
            if self.verbose:
                print(f"Tree {i+1}/{self.n_trees}, MSE:",  (np.mean((self.y - current_prediction) ** 2)))


    def predict(self,X):
        predictions = np.mean(self.y)
         # Add the contributions from each tree
        for tree in self._tree_list:
            predictions += self.learning_rate*tree.predict(X)
        return predictions