# Efficient Algorithms for Lasso Regression

This repository contains the code and experimental results for the course project ...


This repository contains the code and experimental results for the course project **Efficient Algorithms for Lasso Regression** for *Applied Machine Learning in Python – LMU*.

The project studies the Lasso regression problem

```math
\min_w \frac{1}{2n}\|Xw-y\|_2^2 + \lambda\|w\|_1
```
and compares four optimization algorithms:

- Subgradient Descent
- ISTA
- FISTA
- Coordinate Descent

The experiments evaluate convergence speed, sparsity, sparse recovery, robustness to initialization, robustness to regularization strength, and validation on the Diabetes dataset. The implementation is also compared against `sklearn.linear_model.Lasso`.

## Project Structure

```text
.
├── README.md
├── main.py
├── Efficient Algorithms for Lasso Regression Report.pdf
└── figures/
    ├── convergence.png
    ├── sparsity_lambda.png
    ├── lasso_path.png
    ├── recovery_error.png
    ├── sklearn_comparison.png
    ├── init_Subgradient.png
    ├── init_ISTA.png
    ├── init_FISTA.png
    ├── init_CoordinateDescent.png
    ├── initialization_final_objective.png
    ├── short_run_initialization_sensitivity.png
    ├── regularization_distance_to_true_w.png
    ├── regularization_strength_robustness.png
    ├── lambda_heatmap.png
    ├── diabetes_convergence_speed.png
    ├── diabetes_sparsity_vs_regularization.png
    ├── diabetes_test_mse_vs_regularization.png
    └── diabetes_coefficient_heatmap.png
```

If the Python file is still named `main(1).py`, rename it to `main.py` before submitting the project.

## Requirements

The code was written in Python and uses the following packages:

```text
numpy
pandas
matplotlib
scikit-learn
```

Install the requirements with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Alternatively, if you use conda:

```bash
conda install numpy pandas matplotlib scikit-learn
```

## Running the Experiments

To reproduce all experiments and figures, run:

```bash
python main.py
```

The script automatically creates a folder called `figures/` and saves all generated plots there.

You can also run the script inside Jupyter Notebook or JupyterLab with:

```python
%run main.py
```

## Experiments

### 1. Synthetic Lasso Regression

The synthetic experiment uses:

| Parameter | Value |
|---|---:|
| Number of samples `n` | 100 |
| Number of features `d` | 200 |
| True nonzero coefficients | 10 |
| Noise level `sigma` | 0.5 |
| Random seed | 42 |

The synthetic data is used to compare the four solvers under a known sparse ground-truth coefficient vector.

### 2. Solver Comparison

The following solvers are implemented from scratch:

| Solver | Main idea |
|---|---|
| Subgradient Descent | Baseline method for the non-smooth Lasso objective |
| ISTA | Proximal gradient descent with soft-thresholding |
| FISTA | Accelerated ISTA using Nesterov momentum |
| Coordinate Descent | Updates one coefficient at a time using soft-thresholding |

The Coordinate Descent implementation is additionally compared with `sklearn.linear_model.Lasso`.

### 3. Robustness Experiments

The code evaluates robustness with respect to:

- zero initialization
- small random initialization
- large random initialization
- different values of \(\lambda / \lambda_{\max}\)
- short-run optimization with limited iteration budgets

### 4. Diabetes Dataset Validation

The experiments are also validated on the scikit-learn Diabetes dataset. The evaluation uses:

- training objective
- test MSE
- number of selected features
- coefficient heatmap at the best regularization strength

## Main Results

### Convergence

![Convergence comparison](figures/convergence.png)

FISTA and Coordinate Descent converge faster than Subgradient Descent and ISTA. Coordinate Descent reaches a low objective value within only a few sweeps.

| Solver | Final objective |
|---|---:|
| Subgradient Descent | 1.393 |
| ISTA | 1.387 |
| FISTA | 1.298 |
| Coordinate Descent | 1.298 |

### Sparsity and Lasso Path

![Sparsity vs Lambda](figures/sparsity_lambda.png)

As the regularization strength \(\lambda\) increases, the number of nonzero coefficients decreases. Subgradient Descent remains dense because it rarely sets coefficients exactly to zero, while ISTA, FISTA, and Coordinate Descent produce sparse solutions through soft-thresholding.

![Lasso path](figures/lasso_path.png)

The Lasso path confirms the expected behavior: as \(\lambda\) decreases, regularization becomes weaker and more coefficients enter the model.

### Sparse Recovery

![Recovery error](figures/recovery_error.png)

| Method | \( \|\hat{w} - w^*\|_2 \) |
|---|---:|
| Subgradient Descent | 1.486 |
| ISTA | 1.483 |
| FISTA | 0.748 |
| Coordinate Descent | 0.749 |

FISTA and Coordinate Descent recover the true sparse vector more accurately than Subgradient Descent and ISTA.

### Support Recovery

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Subgradient Descent | 0.051 | 1.000 | 0.096 |
| ISTA | 0.185 | 1.000 | 0.313 |
| FISTA | 0.345 | 1.000 | 0.513 |
| Coordinate Descent | 0.357 | 1.000 | 0.526 |

All methods recover the true nonzero coefficients, so recall is 1.000 for all solvers. However, Subgradient Descent selects many false positives, while FISTA and Coordinate Descent achieve much higher precision.

### Comparison with sklearn Lasso

![Coordinate Descent vs sklearn Lasso](figures/sklearn_comparison.png)

Our Coordinate Descent implementation matches `sklearn.linear_model.Lasso` almost exactly on the synthetic experiment.

| Method | Objective value | Nonzero coefficients | Recovery error |
|---|---:|---:|---:|
| Our Coordinate Descent | 1.2976 | 28 | 0.7485 |
| sklearn Lasso | 1.2976 | 28 | 0.7485 |

### Robustness to Initialization

![Initialization final objective](figures/initialization_final_objective.png)

After sufficient iterations, all solvers reach nearly the same final objective under zero, small random, and large random initialization. This indicates that the final optimization result is robust to initialization.

The short-run experiment shows that initialization matters more when the number of iterations is limited:

![Short-run initialization sensitivity](figures/short_run_initialization_sensitivity.png)

### Robustness to Regularization Strength

![Regularization heatmap](figures/lambda_heatmap.png)

FISTA and Coordinate Descent are more robust across different regularization strengths. Their recovery errors decrease steadily as \(\lambda\) becomes smaller. Subgradient Descent is more sensitive to \(\lambda\) and performs best only at an intermediate regularization level.

## Diabetes Dataset Results

### Test MSE

![Diabetes test MSE](figures/diabetes_test_mse_vs_regularization.png)

The best validation results are obtained at the same regularization ratio for all solvers.

| Solver | Best \(\lambda / \lambda_{\max}\) | Best test MSE | Sparsity at best \(\lambda\) | Selected features |
|---|---:|---:|---:|---|
| Subgradient Descent | 0.046416 | 2801.133037 | 10 | bmi, s5, bp, s3, sex, s1, s6, s2, s4, age |
| ISTA | 0.046416 | 2801.078318 | 8 | bmi, s5, bp, s3, sex, s1, s6, s2 |
| FISTA | 0.046416 | 2798.609930 | 7 | bmi, s5, bp, s3, sex, s1, s6 |
| Coordinate Descent | 0.046416 | 2798.601225 | 7 | bmi, s5, bp, s3, sex, s1, s6 |
| sklearn Lasso | 0.046416 | 2798.601223 | 7 | bmi, s5, bp, s3, sex, s1, s6 |

### Feature Selection

![Diabetes coefficient heatmap](figures/diabetes_coefficient_heatmap.png)

FISTA, Coordinate Descent, and sklearn Lasso select the same seven features on the Diabetes dataset. ISTA selects one additional feature, while Subgradient Descent keeps all ten features.

## Reproducibility Notes

- The synthetic experiment uses `np.random.seed(42)`.
- The Diabetes train-test split uses `random_state=42`.
- All figures are saved automatically to the `figures/` folder.
- Results may differ slightly across machines because of floating-point precision, but the qualitative conclusions should remain the same.

## Pre-trained Models

This project does not use pre-trained models. All solvers are implemented directly for Lasso regression and are run from scratch.

## Report

The full project report is included as:

```text
Efficient Algorithms for Lasso Regression Report.pdf
```

## Authors

- Yuehangsha Huang
- Mengying Li

## Contributing

This repository is intended for a course project submission. External contributions are not expected.

## License

This project is for educational use in the LMU *Applied Machine Learning in Python* course. No external open-source license is specified.
