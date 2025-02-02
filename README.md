# Gradient Boosting Model

This project implements the Gradient Boosting algorithm for regression tasks using Regression Trees as weak learners. Gradient Boosting is an ensemble learning technique that iteratively improves the model by fitting new learners on the residual errors of the previous predictions.

## Features:
- Gradient Boosting Algorithm: Iteratively improves the model by training regression trees on the residuals.
- Weak Learners: Utilizes Regression Trees as base models to capture patterns in the data.
- Residual Learning: In each iteration, the model fits a new tree to the difference between the actual target values and the current predictions.
- Learning Rate: A configurable parameter to control the contribution of each tree to the overall prediction
- Performance Monitoring: Provides an option to monitor the training process by calculating and printing the Mean Squared Error (MSE) after each iteration.

## Dependencies
This project depends on the RegressionTree class, which is implemented in another repository. To use the AdaBoost, you need to download the Regression-Trees---From-Scratch repository and place its code in the appropriate folder.
