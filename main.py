import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from courselib.models.linear_models import LassoRegression
from courselib.utils.normalization import standardize


# Save all generated figures in one folder
save_dir = Path("figures")
save_dir.mkdir(exist_ok=True)


# =========================
# Helper functions
# =========================

def count_nonzero(w, threshold=1e-6):
    return int(np.sum(np.abs(w) > threshold))


def lasso_objective(X, y, w, lam):
    n = X.shape[0]
    loss = (1 / (2 * n)) * np.linalg.norm(X @ w - y) ** 2
    penalty = lam * np.linalg.norm(w, 1)
    return loss + penalty


def prediction_mse(X, y, w):
    return np.mean((X @ w - y) ** 2)


def gradient(X, y, w):
    n = X.shape[0]
    return (1 / n) * X.T @ (X @ w - y)


def compute_lambda_max(X, y):
    return np.max(np.abs(X.T @ y)) / X.shape[0]


def fit_lasso_model(X, y, lam, optimizer_name, learning_rate=0.01, max_iter=500, w0=None):
    """Fit the courselib LassoRegression model with a chosen optimizer."""
    model = LassoRegression(
        lam=lam,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    model.fit(X, y, w0=w0)
    return model.w, model.objective_history_, model


def run_solver(solver_name, X, y, lam, w0, max_iter=500, max_cd_sweeps=100):
    """Wrapper used by robustness and real-data experiments."""
    if solver_name == "Subgradient":
        optimizer_name = "subgradient"
        num_iter = max_iter
    elif solver_name == "ISTA":
        optimizer_name = "ista"
        num_iter = max_iter
    elif solver_name == "FISTA":
        optimizer_name = "fista"
        num_iter = max_iter
    elif solver_name in ["Coordinate", "CoordinateDescent"]:
        optimizer_name = "coordinate_descent"
        num_iter = max_cd_sweeps
    else:
        raise ValueError("Unknown solver name")

    w_hat, objectives, _ = fit_lasso_model(
        X=X,
        y=y,
        lam=lam,
        optimizer_name=optimizer_name,
        learning_rate=0.01,
        max_iter=num_iter,
        w0=w0,
    )

    return {"w": w_hat, "objectives": objectives}


def make_initialization(d, init_type, seed=42):
    rng = np.random.RandomState(seed)
    if init_type == "zero":
        return np.zeros(d)
    if init_type == "small_random":
        return 0.01 * rng.randn(d)
    if init_type == "large_random":
        return rng.randn(d)
    raise ValueError("Unknown initialization type")


def summarize_result(X, y, w_hat, w_true, lam):
    return {
        "final_objective": lasso_objective(X, y, w_hat, lam),
        "mse": prediction_mse(X, y, w_hat),
        "sparsity": count_nonzero(w_hat),
        "distance_to_true_w": np.linalg.norm(w_hat - w_true),
    }


def support_recovery(w_est, w_true, threshold=1e-6):
    true_support = set(np.where(np.abs(w_true) > threshold)[0])
    estimated_support = set(np.where(np.abs(w_est) > threshold)[0])

    true_positive = len(true_support & estimated_support)
    false_positive = len(estimated_support - true_support)
    false_negative = len(true_support - estimated_support)

    precision = true_positive / len(estimated_support) if len(estimated_support) > 0 else 0
    recall = true_positive / len(true_support) if len(true_support) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return true_positive, false_positive, false_negative, precision, recall, f1


# =========================
# 1. Generate synthetic data
# =========================

np.random.seed(42)

n = 100
d = 200
sparsity = 10
sigma = 0.5

X = np.random.randn(n, d)

w_true = np.zeros(d)
nonzero_idx = np.random.choice(d, sparsity, replace=False)
w_true[nonzero_idx] = np.random.randn(sparsity)

noise = sigma * np.random.randn(n)
y = X @ w_true + noise

# Standardize X using the course library normalization utility
X = standardize(X)


# =========================
# 2. Test the setup
# =========================

lam = 0.1
w0 = np.zeros(d)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("w_true nonzero:", np.sum(w_true != 0))
print("Initial objective:", lasso_objective(X, y, w0, lam))
print("Gradient shape:", gradient(X, y, w0).shape)


# =========================
# 3. Run Lasso solvers from courselib
# =========================

w_sub, loss_sub, model_sub = fit_lasso_model(
    X, y, lam, optimizer_name="subgradient", learning_rate=0.01, max_iter=500
)
print("Subgradient final objective:", loss_sub[-1])
print("Subgradient nonzero coefficients:", count_nonzero(w_sub))

w_ista, loss_ista, model_ista = fit_lasso_model(
    X, y, lam, optimizer_name="ista", learning_rate=0.01, max_iter=500
)
print("ISTA final objective:", loss_ista[-1])
print("ISTA nonzero coefficients:", count_nonzero(w_ista))

w_fista, loss_fista, model_fista = fit_lasso_model(
    X, y, lam, optimizer_name="fista", learning_rate=0.01, max_iter=500
)
print("FISTA final objective:", loss_fista[-1])
print("FISTA nonzero coefficients:", count_nonzero(w_fista))

w_cd, loss_cd, model_cd = fit_lasso_model(
    X, y, lam, optimizer_name="coordinate_descent", max_iter=100
)
print("Coordinate Descent final objective:", loss_cd[-1])
print("Coordinate Descent nonzero coefficients:", count_nonzero(w_cd))


# =========================
# 4. Plot Loss vs Iteration
# =========================

plt.figure(figsize=(8, 5))
plt.plot(loss_sub, label="Subgradient Descent", linewidth=2, linestyle="--")
plt.plot(loss_ista, label="ISTA", linewidth=2, linestyle="-")
plt.plot(loss_fista, label="FISTA", linewidth=2, linestyle=":")
plt.plot(loss_cd, label="Coordinate Descent", linewidth=2)
plt.xlabel("Iteration / Epoch")
plt.ylabel("Objective Value")
plt.title("Convergence Comparison: Loss vs Iteration")
plt.legend()
plt.grid(True)
plt.savefig(save_dir / "convergence.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 5. Sparsity vs Lambda
# =========================

lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
sparsity_sub, sparsity_ista, sparsity_fista, sparsity_cd = [], [], [], []

for lam_val in lambda_values:
    w_sub_tmp, _, _ = fit_lasso_model(X, y, lam_val, "subgradient", max_iter=500)
    w_ista_tmp, _, _ = fit_lasso_model(X, y, lam_val, "ista", max_iter=500)
    w_fista_tmp, _, _ = fit_lasso_model(X, y, lam_val, "fista", max_iter=500)
    w_cd_tmp, _, _ = fit_lasso_model(X, y, lam_val, "coordinate_descent", max_iter=100)

    sparsity_sub.append(count_nonzero(w_sub_tmp))
    sparsity_ista.append(count_nonzero(w_ista_tmp))
    sparsity_fista.append(count_nonzero(w_fista_tmp))
    sparsity_cd.append(count_nonzero(w_cd_tmp))

plt.figure(figsize=(8, 5))
plt.plot(lambda_values, sparsity_sub, marker="o", label="Subgradient Descent")
plt.plot(lambda_values, sparsity_ista, marker="o", label="ISTA")
plt.plot(lambda_values, sparsity_fista, marker="o", label="FISTA")
plt.plot(lambda_values, sparsity_cd, marker="o", label="Coordinate Descent")
plt.xlabel("Lambda")
plt.ylabel("Number of Nonzero Coefficients")
plt.title("Sparsity vs Lambda")
plt.legend()
plt.grid(True)
plt.savefig(save_dir / "sparsity_lambda.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 6. Recovery Error and Support Recovery
# =========================

error_sub = np.linalg.norm(w_sub - w_true)
error_ista = np.linalg.norm(w_ista - w_true)
error_fista = np.linalg.norm(w_fista - w_true)
error_cd = np.linalg.norm(w_cd - w_true)

print("Recovery Error")
print("Subgradient:", error_sub)
print("ISTA:", error_ista)
print("FISTA:", error_fista)
print("Coordinate Descent:", error_cd)

errors = [error_sub, error_ista, error_fista, error_cd]
labels = ["Subgradient", "ISTA", "FISTA", "Coordinate"]
plt.figure(figsize=(7, 4))
plt.bar(labels, errors)
plt.ylabel("L2 Recovery Error")
plt.title("Recovery Error to True Sparse Vector")
plt.savefig(save_dir / "recovery_error.png", dpi=300, bbox_inches="tight")
plt.show()

methods = {
    "Subgradient": w_sub,
    "ISTA": w_ista,
    "FISTA": w_fista,
    "Coordinate": w_cd,
}

print("\nSupport Recovery Results")
for name, w_est in methods.items():
    tp, fp, fn, precision, recall, f1 = support_recovery(w_est, w_true)
    print(name)
    print("  True positives:", tp)
    print("  False positives:", fp)
    print("  False negatives:", fn)
    print("  Precision:", precision)
    print("  Recall:", recall)
    print("  F1-score:", f1)


# =========================
# 7. Lasso Path as lambda -> 0
# =========================

lambda_max = compute_lambda_max(X, y)
lambda_path = lambda_max * np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])

coef_path_cd = []
nonzero_path_cd = []

for lam_val in lambda_path:
    w_tmp, _, _ = fit_lasso_model(X, y, lam_val, "coordinate_descent", max_iter=100)
    coef_path_cd.append(w_tmp)
    nonzero_path_cd.append(count_nonzero(w_tmp))

coef_path_cd = np.array(coef_path_cd)

plt.figure(figsize=(8, 5))
plt.plot(lambda_path, nonzero_path_cd, marker="o")
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("Lambda")
plt.ylabel("Number of Nonzero Coefficients")
plt.title("Lasso Path: Sparsity as Lambda Decreases")
plt.grid(True)
plt.savefig(save_dir / "lasso_path.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 8. Compare with sklearn Lasso
# =========================

sklearn_lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-8)
sklearn_lasso.fit(X, y)
w_sklearn = sklearn_lasso.coef_

print("\nComparison with sklearn Lasso")
print("sklearn Lasso objective:", lasso_objective(X, y, w_sklearn, lam))
print("sklearn Lasso nonzero coefficients:", count_nonzero(w_sklearn))
print("sklearn Lasso recovery error:", np.linalg.norm(w_sklearn - w_true))
print("Coordinate Descent objective:", lasso_objective(X, y, w_cd, lam))
print("Coordinate Descent nonzero coefficients:", count_nonzero(w_cd))
print("Coordinate Descent recovery error:", np.linalg.norm(w_cd - w_true))

plt.figure(figsize=(6, 4))
objectives = [lasso_objective(X, y, w_cd, lam), lasso_objective(X, y, w_sklearn, lam)]
labels = ["Our CD", "sklearn Lasso"]
plt.bar(labels, objectives)
plt.ylabel("Objective Value")
plt.title("Our Coordinate Descent vs sklearn Lasso")
plt.savefig(save_dir / "sklearn_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 9. Robustness to Initialization
# =========================

seed = 42
fixed_lambda = lam
max_iter = 500
max_cd_sweeps = 100
solver_names = ["Subgradient", "ISTA", "FISTA", "CoordinateDescent"]
init_types = ["zero", "small_random", "large_random"]

print("\nRunning Experiment A: Robustness to initialization...")

init_objective_results = {}
init_summary_rows = []

for solver_name in solver_names:
    init_objective_results[solver_name] = {}

    for init_type in init_types:
        w0 = make_initialization(d, init_type, seed=seed)
        result = run_solver(
            solver_name=solver_name,
            X=X,
            y=y,
            lam=fixed_lambda,
            w0=w0,
            max_iter=max_iter,
            max_cd_sweeps=max_cd_sweeps,
        )
        w_hat = result["w"]
        objectives_hist = result["objectives"]
        metrics = summarize_result(X, y, w_hat, w_true, fixed_lambda)
        init_objective_results[solver_name][init_type] = objectives_hist

        row = {
            "solver": solver_name,
            "initialization": init_type,
            "final_objective": metrics["final_objective"],
            "mse": metrics["mse"],
            "sparsity": metrics["sparsity"],
            "distance_to_true_w": metrics["distance_to_true_w"],
        }
        init_summary_rows.append(row)

        print(
            f"{solver_name:18s} | {init_type:14s} | "
            f"obj={metrics['final_objective']:.4f} | "
            f"sparsity={metrics['sparsity']:3d} | "
            f"dist={metrics['distance_to_true_w']:.4f}"
        )

init_df = pd.DataFrame(init_summary_rows)
print("\nInitialization robustness summary")
print(init_df)

# Initialization robustness: one plot for each solver
for solver_name in solver_names:
    plt.figure(figsize=(7, 5))
    for init_name, objectives_hist in init_objective_results[solver_name].items():
        plt.plot(objectives_hist, label=init_name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title(f"Initialization Robustness: {solver_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_name = f"init_{solver_name}.png".replace(" ", "_")
    plt.savefig(save_dir / file_name, dpi=300, bbox_inches="tight")
    plt.show()

# Initialization Robustness: Final Objective
init_order = ["zero", "small_random", "large_random"]
x = np.arange(len(init_order))
width = 0.18
offsets = {
    "Subgradient": -1.5 * width,
    "ISTA": -0.5 * width,
    "FISTA": 0.5 * width,
    "CoordinateDescent": 1.5 * width,
}

plt.figure(figsize=(9, 5))
for solver_name in solver_names:
    values = []
    for init in init_order:
        value = init_df[
            (init_df["solver"] == solver_name) &
            (init_df["initialization"] == init)
        ]["final_objective"].values[0]
        values.append(value)
    plt.bar(x + offsets[solver_name], values, width=width, label=solver_name)

plt.xticks(x, init_order)
plt.xlabel("Initialization")
plt.ylabel("Final objective")
plt.title("Initialization Robustness: Final Objective")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(save_dir / "initialization_final_objective.png", dpi=300, bbox_inches="tight")
plt.show()

# Short-run initialization sensitivity
max_iter_short = 50
max_cd_sweeps_short = 5
short_init_summary_rows = []

for solver_name in solver_names:
    for init_type in init_types:
        w0 = make_initialization(d, init_type, seed=seed)
        result = run_solver(
            solver_name=solver_name,
            X=X,
            y=y,
            lam=fixed_lambda,
            w0=w0,
            max_iter=max_iter_short,
            max_cd_sweeps=max_cd_sweeps_short,
        )
        w_hat = result["w"]
        metrics = summarize_result(X, y, w_hat, w_true, fixed_lambda)
        short_init_summary_rows.append({
            "solver": solver_name,
            "initialization": init_type,
            "final_objective": metrics["final_objective"],
            "mse": metrics["mse"],
            "sparsity": metrics["sparsity"],
            "distance_to_true_w": metrics["distance_to_true_w"],
        })

short_init_df = pd.DataFrame(short_init_summary_rows)
print("\nShort-run initialization results")
print(short_init_df)

short_init_sensitivity_rows = []
for solver_name in solver_names:
    subset = short_init_df[short_init_df["solver"] == solver_name]
    short_init_sensitivity_rows.append({
        "solver": solver_name,
        "std_final_objective": subset["final_objective"].std(),
        "std_distance_to_true_w": subset["distance_to_true_w"].std(),
        "std_sparsity": subset["sparsity"].std(),
        "mean_final_objective": subset["final_objective"].mean(),
        "mean_distance_to_true_w": subset["distance_to_true_w"].mean(),
        "mean_sparsity": subset["sparsity"].mean(),
    })

short_init_sensitivity_df = pd.DataFrame(short_init_sensitivity_rows)
print("\nShort-run initialization sensitivity")
print(short_init_sensitivity_df)

plt.figure(figsize=(7, 5))
plt.bar(short_init_sensitivity_df["solver"], short_init_sensitivity_df["std_distance_to_true_w"])
plt.xlabel("Solver")
plt.ylabel(r"Std. of $||\hat{w} - w^*||_2$ over initializations")
plt.title("Short-run Initialization Sensitivity")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(save_dir / "short_run_initialization_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 10. Robustness to Regularization Strength
# =========================

print("\nRunning Experiment B: Regularization strength robustness...")

lambda_max = compute_lambda_max(X, y)
lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_max * 0.01), 10)
lambda_ratios = lambdas / lambda_max

lambda_results = {}
lambda_summary_rows = []

for solver_name in solver_names:
    lambda_results[solver_name] = {
        "objectives": [],
        "mse": [],
        "sparsities": [],
        "distances": [],
    }

    for lam_val, lam_ratio in zip(lambdas, lambda_ratios):
        w0 = make_initialization(d, "zero", seed=seed)
        result = run_solver(
            solver_name=solver_name,
            X=X,
            y=y,
            lam=lam_val,
            w0=w0,
            max_iter=max_iter,
            max_cd_sweeps=max_cd_sweeps,
        )
        w_hat = result["w"]
        metrics = summarize_result(X, y, w_hat, w_true, lam_val)

        lambda_results[solver_name]["objectives"].append(metrics["final_objective"])
        lambda_results[solver_name]["mse"].append(metrics["mse"])
        lambda_results[solver_name]["sparsities"].append(metrics["sparsity"])
        lambda_results[solver_name]["distances"].append(metrics["distance_to_true_w"])

        lambda_summary_rows.append({
            "solver": solver_name,
            "lambda": lam_val,
            "lambda_ratio": lam_ratio,
            "final_objective": metrics["final_objective"],
            "mse": metrics["mse"],
            "sparsity": metrics["sparsity"],
            "distance_to_true_w": metrics["distance_to_true_w"],
        })

    print(f"{solver_name:18s} done.")

lambda_df = pd.DataFrame(lambda_summary_rows)
print("\nRegularization strength results")
print(lambda_df)

plt.figure(figsize=(7, 5))
for solver_name, result in lambda_results.items():
    plt.plot(lambda_ratios, result["distances"], marker="o", label=solver_name)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel(r"$||\hat{w} - w^*||_2$")
plt.title("Regularization Robustness: Distance to True w")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "regularization_distance_to_true_w.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
for solver_name in solver_names:
    subset = lambda_df[lambda_df["solver"] == solver_name]
    plt.plot(subset["lambda_ratio"], subset["distance_to_true_w"], marker="o", label=solver_name)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel(r"$||\hat{w} - w^*||_2$")
plt.title("Regularization Strength Robustness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "regularization_strength_robustness.png", dpi=300, bbox_inches="tight")
plt.show()

lambda_sensitivity_rows = []
for solver_name in solver_names:
    subset = lambda_df[lambda_df["solver"] == solver_name]
    best_idx = subset["distance_to_true_w"].idxmin()
    lambda_sensitivity_rows.append({
        "solver": solver_name,
        "std_distance_over_lambdas": subset["distance_to_true_w"].std(),
        "min_distance": subset["distance_to_true_w"].min(),
        "max_distance": subset["distance_to_true_w"].max(),
        "best_lambda_ratio": subset.loc[best_idx, "lambda_ratio"],
        "std_final_objective_over_lambdas": subset["final_objective"].std(),
        "std_sparsity_over_lambdas": subset["sparsity"].std(),
    })

lambda_sensitivity_df = pd.DataFrame(lambda_sensitivity_rows)
print("\nLambda sensitivity summary")
print(lambda_sensitivity_df)

init_sensitivity_rows = []
for solver_name in solver_names:
    subset = init_df[init_df["solver"] == solver_name]
    init_sensitivity_rows.append({
        "solver": solver_name,
        "std_final_objective": subset["final_objective"].std(),
        "std_distance_to_true_w": subset["distance_to_true_w"].std(),
        "std_sparsity": subset["sparsity"].std(),
        "mean_final_objective": subset["final_objective"].mean(),
        "mean_distance_to_true_w": subset["distance_to_true_w"].mean(),
        "mean_sparsity": subset["sparsity"].mean(),
    })

init_sensitivity_df = pd.DataFrame(init_sensitivity_rows)
robustness_summary_df = init_sensitivity_df.merge(lambda_sensitivity_df, on="solver", how="inner")
print("\nRobustness summary")
print(robustness_summary_df[[
    "solver",
    "std_distance_to_true_w",
    "std_final_objective",
    "std_sparsity",
    "std_distance_over_lambdas",
    "best_lambda_ratio",
    "min_distance",
]])

# Regularization robustness heatmap
lambda_heatmap_df = (
    lambda_df
    .pivot_table(index="solver", columns="lambda_ratio", values="distance_to_true_w")
    .loc[solver_names]
)
lambda_heatmap_df = lambda_heatmap_df[sorted(lambda_heatmap_df.columns, reverse=True)]

fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(lambda_heatmap_df.values, aspect="auto", cmap="YlOrRd")
ax.set_yticks(np.arange(len(lambda_heatmap_df.index)))
ax.set_yticklabels(lambda_heatmap_df.index)
ax.set_xticks(np.arange(len(lambda_heatmap_df.columns)))
ax.set_xticklabels([f"{x:.3f}" for x in lambda_heatmap_df.columns], rotation=45, ha="right")

for i in range(lambda_heatmap_df.shape[0]):
    for j in range(lambda_heatmap_df.shape[1]):
        value = lambda_heatmap_df.values[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

ax.set_title("Regularization Robustness Across Lambda Values")
ax.set_xlabel(r"$\lambda / \lambda_{\max}$")
ax.set_ylabel("Solver")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$||\hat{w} - w^*||_2$")
plt.tight_layout()
plt.savefig(save_dir / "lambda_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 11. Validation on Diabetes Dataset
# =========================

print("\nRunning Experiment C: Diabetes dataset validation...")

X_real, y_real = load_diabetes(return_X_y=True)
print("Original Diabetes data")
print("X_real shape:", X_real.shape)
print("y_real shape:", y_real.shape)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_real,
    y_real,
    test_size=0.2,
    random_state=42,
)

# For train/test data, use training-set statistics to avoid data leakage.
scaler_real = StandardScaler()
X_train = scaler_real.fit_transform(X_train_raw)
X_test = scaler_real.transform(X_test_raw)

y_mean = y_train_raw.mean()
y_train = y_train_raw - y_mean
y_test = y_test_raw - y_mean

print("After preprocessing")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

lambda_max_real = compute_lambda_max(X_train, y_train)
lambda_values_real = np.logspace(np.log10(lambda_max_real), np.log10(lambda_max_real * 0.01), 10)
lambda_ratios_real = lambda_values_real / lambda_max_real

print("lambda_max_real:", lambda_max_real)
print("lambda ratios:")
print(lambda_ratios_real)

max_iter_real = 1000
max_cd_sweeps_real = 100
real_result_rows = []

for solver_name in solver_names:
    print(f"Running {solver_name}...")
    for lam_val, lam_ratio in zip(lambda_values_real, lambda_ratios_real):
        w0 = np.zeros(X_train.shape[1])
        result = run_solver(
            solver_name=solver_name,
            X=X_train,
            y=y_train,
            lam=lam_val,
            w0=w0,
            max_iter=max_iter_real,
            max_cd_sweeps=max_cd_sweeps_real,
        )
        w_hat = result["w"]
        real_result_rows.append({
            "solver": solver_name,
            "lambda": lam_val,
            "lambda_ratio": lam_ratio,
            "train_objective": lasso_objective(X_train, y_train, w_hat, lam_val),
            "train_mse": prediction_mse(X_train, y_train, w_hat),
            "test_mse": prediction_mse(X_test, y_test, w_hat),
            "sparsity": count_nonzero(w_hat),
        })

real_df = pd.DataFrame(real_result_rows)

# Compare with sklearn Lasso
sklearn_rows = []
for lam_val, lam_ratio in zip(lambda_values_real, lambda_ratios_real):
    model = Lasso(alpha=lam_val, fit_intercept=False, max_iter=10000, tol=1e-8)
    model.fit(X_train, y_train)
    w_sklearn_real = model.coef_
    sklearn_rows.append({
        "solver": "sklearn Lasso",
        "lambda": lam_val,
        "lambda_ratio": lam_ratio,
        "train_objective": lasso_objective(X_train, y_train, w_sklearn_real, lam_val),
        "train_mse": prediction_mse(X_train, y_train, w_sklearn_real),
        "test_mse": prediction_mse(X_test, y_test, w_sklearn_real),
        "sparsity": count_nonzero(w_sklearn_real),
    })

sklearn_df = pd.DataFrame(sklearn_rows)
real_all_df = pd.concat([real_df, sklearn_df], ignore_index=True)

print("\nDiabetes dataset all results")
print(real_all_df)

plt.figure(figsize=(8, 5))
for solver_name in real_all_df["solver"].unique():
    subset = real_all_df[real_all_df["solver"] == solver_name]
    plt.plot(subset["lambda_ratio"], subset["test_mse"], marker="o", label=solver_name)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel("Test MSE")
plt.title("Diabetes Dataset: Test MSE vs Regularization Strength")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / "diabetes_test_mse_vs_regularization.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
for solver_name in real_all_df["solver"].unique():
    subset = real_all_df[real_all_df["solver"] == solver_name]
    plt.plot(subset["lambda_ratio"], subset["sparsity"], marker="o", label=solver_name)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel("Number of nonzero coefficients")
plt.title("Diabetes Dataset: Sparsity vs Regularization Strength")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / "diabetes_sparsity_vs_regularization.png", dpi=300, bbox_inches="tight")
plt.show()

# Convergence speed on Diabetes
fixed_lambda_real = 0.1 * lambda_max_real
real_convergence_results = {}
for solver_name in solver_names:
    w0 = np.zeros(X_train.shape[1])
    result = run_solver(
        solver_name=solver_name,
        X=X_train,
        y=y_train,
        lam=fixed_lambda_real,
        w0=w0,
        max_iter=max_iter_real,
        max_cd_sweeps=max_cd_sweeps_real,
    )
    real_convergence_results[solver_name] = result["objectives"]

plt.figure(figsize=(8, 5))
for solver_name, objectives_hist in real_convergence_results.items():
    plt.plot(objectives_hist, label=solver_name)
plt.xlabel("Iteration / Sweep")
plt.ylabel("Training objective")
plt.title("Diabetes Dataset: Convergence Speed")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(save_dir / "diabetes_convergence_speed.png", dpi=300, bbox_inches="tight")
plt.show()

# Initialization robustness on Diabetes
init_types_real = ["zero", "small_random", "large_random"]
real_init_rows = []

for solver_name in solver_names:
    for init_type in init_types_real:
        w0 = make_initialization(d=X_train.shape[1], init_type=init_type, seed=42)
        result = run_solver(
            solver_name=solver_name,
            X=X_train,
            y=y_train,
            lam=fixed_lambda_real,
            w0=w0,
            max_iter=max_iter_real,
            max_cd_sweeps=max_cd_sweeps_real,
        )
        w_hat = result["w"]
        real_init_rows.append({
            "solver": solver_name,
            "initialization": init_type,
            "train_objective": lasso_objective(X_train, y_train, w_hat, fixed_lambda_real),
            "train_mse": prediction_mse(X_train, y_train, w_hat),
            "test_mse": prediction_mse(X_test, y_test, w_hat),
            "sparsity": count_nonzero(w_hat),
        })

real_init_df = pd.DataFrame(real_init_rows)
print("\nDiabetes initialization robustness")
print(real_init_df)

# Best performance by solver
best_rows = []
for solver_name in real_all_df["solver"].unique():
    subset = real_all_df[real_all_df["solver"] == solver_name]
    best_idx = subset["test_mse"].idxmin()
    best_row = subset.loc[best_idx]
    best_rows.append({
        "solver": solver_name,
        "best_test_mse": best_row["test_mse"],
        "best_lambda_ratio": best_row["lambda_ratio"],
        "sparsity_at_best_lambda": best_row["sparsity"],
        "train_objective_at_best_lambda": best_row["train_objective"],
    })

real_best_df = pd.DataFrame(best_rows)
print("\nBest performance on Diabetes")
print(real_best_df)

# Selected features
feature_names = load_diabetes().feature_names
print("\nDiabetes feature names")
print(feature_names)

all_methods = solver_names + ["sklearn Lasso"]
threshold = 1e-5
all_result_rows = []
all_coef_rows = []

for method in all_methods:
    print(f"Running {method} for selected feature analysis...")
    for lam_val, lam_ratio in zip(lambda_values_real, lambda_ratios_real):
        if method == "sklearn Lasso":
            model = Lasso(alpha=lam_val, fit_intercept=False, max_iter=10000, tol=1e-8)
            model.fit(X_train, y_train)
            w_hat = model.coef_
        else:
            w0 = np.zeros(X_train.shape[1])
            result = run_solver(
                solver_name=method,
                X=X_train,
                y=y_train,
                lam=lam_val,
                w0=w0,
                max_iter=1000,
                max_cd_sweeps=100,
            )
            w_hat = result["w"]

        train_mse = prediction_mse(X_train, y_train, w_hat)
        test_mse = prediction_mse(X_test, y_test, w_hat)
        train_objective = lasso_objective(X_train, y_train, w_hat, lam_val)
        sparsity = np.sum(np.abs(w_hat) > threshold)

        all_result_rows.append({
            "solver": method,
            "lambda": lam_val,
            "lambda_ratio": lam_ratio,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_objective": train_objective,
            "sparsity": sparsity,
        })

        for feature, coef in zip(feature_names, w_hat):
            all_coef_rows.append({
                "solver": method,
                "lambda": lam_val,
                "lambda_ratio": lam_ratio,
                "feature": feature,
                "coefficient": coef,
                "abs_coefficient": abs(coef),
                "selected": abs(coef) > threshold,
            })

all_result_df = pd.DataFrame(all_result_rows)
all_coef_df = pd.DataFrame(all_coef_rows)

best_rows = []
for method in all_methods:
    subset = all_result_df[all_result_df["solver"] == method]
    best_idx = subset["test_mse"].idxmin()
    best_row = subset.loc[best_idx]
    best_rows.append({
        "solver": method,
        "best_lambda": best_row["lambda"],
        "best_lambda_ratio": best_row["lambda_ratio"],
        "best_test_mse": best_row["test_mse"],
        "sparsity_at_best_lambda": best_row["sparsity"],
    })

best_per_solver_df = pd.DataFrame(best_rows)
print("\nBest results for selected feature analysis")
print(best_per_solver_df)

selected_rows = []
for _, row in best_per_solver_df.iterrows():
    method = row["solver"]
    best_lambda_ratio = row["best_lambda_ratio"]
    subset = all_coef_df[
        (all_coef_df["solver"] == method) &
        (np.isclose(all_coef_df["lambda_ratio"], best_lambda_ratio))
    ].copy()
    selected_subset = subset[subset["selected"]].copy()

    for _, feature_row in selected_subset.iterrows():
        selected_rows.append({
            "solver": method,
            "best_lambda_ratio": best_lambda_ratio,
            "feature": feature_row["feature"],
            "coefficient": feature_row["coefficient"],
            "abs_coefficient": feature_row["abs_coefficient"],
            "direction": "positive" if feature_row["coefficient"] > 0 else "negative",
        })

selected_features_each_solver_df = pd.DataFrame(selected_rows)
selected_features_each_solver_df = selected_features_each_solver_df.sort_values(
    by=["solver", "abs_coefficient"],
    ascending=[True, False],
)

print("\nSelected features for each solver")
print(selected_features_each_solver_df)

for method in all_methods:
    subset = selected_features_each_solver_df[selected_features_each_solver_df["solver"] == method]
    print("=" * 60)
    print(method)
    best_info = best_per_solver_df[best_per_solver_df["solver"] == method].iloc[0]
    print(f"Best lambda ratio: {best_info['best_lambda_ratio']:.4f}")
    print(f"Best test MSE: {best_info['best_test_mse']:.4f}")
    print(f"Number of selected features: {len(subset)}")

    if len(subset) == 0:
        print("Selected features: None")
    else:
        print("Selected features:")
        for _, row in subset.iterrows():
            print(f"  {row['feature']:>4s} | coef = {row['coefficient']:>8.4f} | {row['direction']}")

summary_selected_features = (
    selected_features_each_solver_df
    .groupby("solver")["feature"]
    .apply(lambda x: ", ".join(x))
    .reset_index()
)

summary_selected_features = summary_selected_features.merge(best_per_solver_df, on="solver", how="left")
summary_selected_features = summary_selected_features[
    ["solver", "best_lambda_ratio", "best_test_mse", "sparsity_at_best_lambda", "feature"]
]
summary_selected_features = summary_selected_features.rename(columns={"feature": "selected_features"})

print("\nSummary of selected features")
print(summary_selected_features)

# Coefficient heatmap at each solver's best lambda
coef_rows_for_matrix = []
for _, best_row in best_per_solver_df.iterrows():
    method = best_row["solver"]
    best_lambda_ratio = best_row["best_lambda_ratio"]
    subset = all_coef_df[
        (all_coef_df["solver"] == method) &
        (np.isclose(all_coef_df["lambda_ratio"], best_lambda_ratio))
    ]
    for _, coef_row in subset.iterrows():
        coef_rows_for_matrix.append({
            "solver": method,
            "feature": coef_row["feature"],
            "coefficient": coef_row["coefficient"],
        })

coef_matrix_df = pd.DataFrame(coef_rows_for_matrix)
coef_matrix = coef_matrix_df.pivot_table(
    index="solver",
    columns="feature",
    values="coefficient",
).loc[all_methods]

fig, ax = plt.subplots(figsize=(10, 4.5))
max_abs_coef = np.max(np.abs(coef_matrix.values))
im = ax.imshow(
    coef_matrix.values,
    aspect="auto",
    cmap="coolwarm",
    vmin=-max_abs_coef,
    vmax=max_abs_coef,
)

ax.set_xticks(np.arange(len(coef_matrix.columns)))
ax.set_xticklabels(coef_matrix.columns)
ax.set_yticks(np.arange(len(coef_matrix.index)))
ax.set_yticklabels(coef_matrix.index)

for i in range(coef_matrix.shape[0]):
    for j in range(coef_matrix.shape[1]):
        value = coef_matrix.values[i, j]
        text = f"{value:.1f}" if abs(value) > threshold else "0"
        ax.text(j, i, text, ha="center", va="center", fontsize=8)

ax.set_xlabel("Feature")
ax.set_ylabel("Solver")
ax.set_title("Coefficient Values at Each Solver's Best λ")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Coefficient value")
plt.tight_layout()
plt.savefig(save_dir / "diabetes_coefficient_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
