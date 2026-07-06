import numpy as np
from .base import TrainableModel


class LinearRegression(TrainableModel):
    """Linear regression model."""

    def __init__(self, w, b, optimizer):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}

    def decision_function(self, X):
        return X @ self.w + self.b

    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return self.decision_function(X)


class LassoRegression(TrainableModel):
    """
    Lasso regression model implemented inside the courselib model structure.

    The model solves

        min_w 1/(2n) ||Xw - y||_2^2 + lam ||w||_1

    Supported optimizers:
    - "subgradient"
    - "ista"
    - "fista"
    - "coordinate_descent"

    Notes
    -----
    This class follows the courselib model interface by providing fit,
    decision_function, __call__, loss_grad, and _get_params. The proximal
    optimizers are implemented inside the model because the standard
    GDOptimizer in courselib only supports ordinary gradient updates, while
    Lasso requires soft-thresholding or coordinate-wise updates.
    """

    def __init__(
        self,
        lam=0.1,
        optimizer_name="ista",
        learning_rate=0.01,
        max_iter=500,
        fit_intercept=False,
        tol=None,
    ):
        super().__init__(optimizer=None)
        self.lam = float(lam)
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol

        self.w = None
        self.b = 0.0
        self.coef_ = None
        self.intercept_ = 0.0
        self.objective_history_ = []

    def objective(self, X, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        n = X.shape[0]
        residual = X @ w + b - y
        loss = (1 / (2 * n)) * np.linalg.norm(residual) ** 2
        penalty = self.lam * np.linalg.norm(w, 1)
        return loss + penalty

    def gradient(self, X, y, w, b=0.0):
        n = X.shape[0]
        residual = X @ w + b - y
        grad_w = (1 / n) * X.T @ residual
        grad_b = np.mean(residual)
        return grad_w, grad_b

    @staticmethod
    def soft_thresholding(z, alpha):
        return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)

    def loss_grad(self, X, y):
        """
        Subgradient of the Lasso objective.

        This method is included for compatibility with the courselib
        TrainableModel interface. The main fit method uses optimizer-specific
        updates, especially proximal and coordinate descent updates.
        """
        grad_w, grad_b = self.gradient(X, y, self.w, self.b)
        grad_w = grad_w + self.lam * np.sign(self.w)

        if self.fit_intercept:
            return {"w": grad_w, "b": grad_b}
        return {"w": grad_w}

    def fit(self, X, y, w0=None, b0=0.0):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        d = X.shape[1]

        if w0 is None:
            w = np.zeros(d)
        else:
            w = np.asarray(w0, dtype=float).copy()

        b = float(b0) if self.fit_intercept else 0.0
        self.objective_history_ = []

        if self.optimizer_name == "subgradient":
            w, b = self._fit_subgradient(X, y, w, b)
        elif self.optimizer_name == "ista":
            w, b = self._fit_ista(X, y, w, b)
        elif self.optimizer_name == "fista":
            w, b = self._fit_fista(X, y, w, b)
        elif self.optimizer_name == "coordinate_descent":
            w, b = self._fit_coordinate_descent(X, y, w, b)
        else:
            raise ValueError(
                "Unknown optimizer_name. Use one of: "
                "'subgradient', 'ista', 'fista', 'coordinate_descent'."
            )

        self.w = w
        self.b = b
        self.coef_ = w
        self.intercept_ = b
        return self

    def _append_objective_and_check_convergence(self, X, y, w, b):
        obj = self.objective(X, y, w, b)
        self.objective_history_.append(obj)

        if self.tol is not None and len(self.objective_history_) > 1:
            improvement = abs(self.objective_history_[-2] - self.objective_history_[-1])
            return improvement < self.tol
        return False

    def _fit_subgradient(self, X, y, w, b):
        for _ in range(self.max_iter):
            grad_w, grad_b = self.gradient(X, y, w, b)
            subgrad_w = grad_w + self.lam * np.sign(w)

            w = w - self.learning_rate * subgrad_w
            if self.fit_intercept:
                b = b - self.learning_rate * grad_b

            if self._append_objective_and_check_convergence(X, y, w, b):
                break

        return w, b

    def _fit_ista(self, X, y, w, b):
        for _ in range(self.max_iter):
            grad_w, grad_b = self.gradient(X, y, w, b)

            w = self.soft_thresholding(
                w - self.learning_rate * grad_w,
                self.learning_rate * self.lam,
            )
            if self.fit_intercept:
                b = b - self.learning_rate * grad_b

            if self._append_objective_and_check_convergence(X, y, w, b):
                break

        return w, b

    def _fit_fista(self, X, y, w, b):
        z = w.copy()
        z_b = b
        t = 1.0

        for _ in range(self.max_iter):
            w_old = w.copy()
            b_old = b

            grad_w, grad_b = self.gradient(X, y, z, z_b)

            w = self.soft_thresholding(
                z - self.learning_rate * grad_w,
                self.learning_rate * self.lam,
            )
            if self.fit_intercept:
                b = z_b - self.learning_rate * grad_b

            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            momentum = (t - 1) / t_new

            z = w + momentum * (w - w_old)
            if self.fit_intercept:
                z_b = b + momentum * (b - b_old)

            t = t_new

            if self._append_objective_and_check_convergence(X, y, w, b):
                break

        return w, b

    def _fit_coordinate_descent(self, X, y, w, b):
        n, d = X.shape
        r = y - (X @ w + b)

        for _ in range(self.max_iter):
            for j in range(d):
                r = r + X[:, j] * w[j]

                rho_j = (1 / n) * X[:, j].T @ r
                z_j = (1 / n) * np.sum(X[:, j] ** 2)

                if z_j == 0:
                    w_new_j = 0.0
                else:
                    w_new_j = self.soft_thresholding(rho_j, self.lam) / z_j

                r = r - X[:, j] * w_new_j
                w[j] = w_new_j

            if self.fit_intercept:
                b_update = np.mean(r)
                b = b + b_update
                r = r - b_update

            if self._append_objective_and_check_convergence(X, y, w, b):
                break

        return w, b

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return self.decision_function(X)

    def _get_params(self):
        if self.fit_intercept:
            return {"w": self.w, "b": np.array(self.b)}
        return {"w": self.w}

    def __call__(self, X):
        return self.decision_function(X)


class LinearBinaryClassification(TrainableModel):
    """Linear binary classification model."""

    def __init__(self, w, b, optimizer, class_labels=[-1, 1]):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.class_labels = [min(class_labels), max(class_labels)]
        self.threshold = self.class_labels[0] + (self.class_labels[1] - self.class_labels[0]) / 2.0

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}

    def decision_function(self, X):
        return X @ self.w + self.b

    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(
            self.decision_function(X) >= self.threshold,
            self.class_labels[1],
            self.class_labels[0],
        )


class RidgeClassifier(LinearBinaryClassification):
    """Ridge binary classifier."""

    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1, 1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X, Y):
        residual = self.decision_function(X) - Y
        grad_w = X.T @ residual / X.shape[0] + 2 * self.lam * self.w
        grad_b = np.mean(residual, axis=0)
        return {"w": grad_w, "b": grad_b}


class LassoClassifier(LinearBinaryClassification):
    """Lasso binary classifier."""

    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1, 1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X, Y):
        residual = self.decision_function(X) - Y
        grad_w = X.T @ residual / X.shape[0] + self.lam * np.sign(self.w)
        grad_b = np.mean(residual, axis=0)
        return {"w": grad_w, "b": grad_b}


class LinearMulticlassClassification(TrainableModel):
    """
    Linear multiclass classifier with least-squares multioutput loss.
    Trained with gradient-based optimization.
    """

    def __init__(self, w, b, optimizer):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float).reshape(1, -1)

    def decision_function(self, X):
        return X @ self.w + self.b

    def loss_grad(self, X, Y):
        residual = self.decision_function(X) - Y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual, axis=0, keepdims=True)
        return {"w": grad_w, "b": grad_b}

    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.argmax(self.decision_function(X), axis=-1)
