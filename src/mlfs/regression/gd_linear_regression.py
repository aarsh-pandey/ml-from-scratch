import numpy as np


class LinearRegressionGD:
    def __init__(self):
        self.coefficients = None

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha=0.01, epochs=100) -> None:
        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)

        # Input validation
        if X.size == 0 or Y.size == 0:
            raise ValueError("Input arrays X and Y must not be empty.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")
        if alpha <= 0:
            raise ValueError("Learning rate alpha must be positive.")

        n, m = X.shape
        X = np.concatenate([np.ones(n).reshape(-1, 1), X], axis=1)
        self.coefficients = np.zeros(m + 1).reshape(-1, 1)

        for epoch in range(epochs):
            err = (X @ self.coefficients) - Y
            gradient = (X.T @ err) / n
            self.coefficients = self.coefficients - alpha * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)
        return np.dot(X, self.coefficients).ravel()


