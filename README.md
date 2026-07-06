Efficient Algorithms for Lasso Regression
This repository contains the code and experimental results for the course project Efficient Algorithms for Lasso Regression for Applied Machine Learning in Python - LMU.
The project studies the Lasso regression problem:
$$\min_w \frac{1}{2n}\lVert Xw-y\rVert_2^2 + \lambda\lVert w\rVert_1$$
where $X \in \mathbb{R}^{n \times d}$ is the data matrix, $y \in \mathbb{R}^n$ is the response vector, $w \in \mathbb{R}^d$ is the coefficient vector, and $\lambda > 0$ controls the strength of the $\ell_1$ regularization.
The $\ell_1$ penalty encourages sparsity by shrinking many coefficients exactly to zero. Therefore, Lasso is useful not only for prediction, but also for feature selection and sparse recovery.
We implemented and evaluated four optimization methods for Lasso regression:
Subgradient Descent
ISTA
FISTA
Coordinate Descent
The experiments evaluate both optimization behavior and statistical performance. We analyze convergence speed using objective-value curves, sparsity using the number of nonzero coefficients, sparse recovery using recovery error and support metrics, robustness under different initializations and regularization strengths, and real-data performance on the scikit-learn Diabetes dataset. We also compare our Coordinate Descent implementation with `sklearn.linear_model.Lasso` as a reference implementation.
---
Courselib Integration
This project extends the required courselib structure instead of implementing Lasso only as a standalone script.
The main courselib-related modifications are:
`courselib/models/linear_models.py`
A new `LassoRegression` model is implemented in `courselib/models/linear_models.py`.
The class follows the courselib model structure and provides:
`fit(X, y)` for training,
`predict(X)` and `decision_function(X)` for prediction,
`loss_grad(X, y)` for compatibility with the courselib training interface,
`_get_params()` for parameter access,
`coef_` and `intercept_` attributes,
`objective_history_` for tracking convergence,
selectable optimization methods through `optimizer_name`.
The model can be initialized with different optimizers:
```python
from courselib.models.linear_models import LassoRegression

model = LassoRegression(
    lam=0.1,
    optimizer_name="fista",
    learning_rate=0.01,
    max_iter=500
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
The supported optimizer names are:
```python
"subgradient"
"ista"
"fista"
"coordinate_descent"
```
`courselib/optimizers.py`
The file `courselib/optimizers.py` contains optimizer classes and labels for the Lasso methods:
`SubgradientOptimizer`
`ISTAOptimizer`
`FISTAOptimizer`
`CoordinateDescentOptimizer`
The Lasso-specific update rules are implemented inside `LassoRegression`, because Lasso requires nonsmooth optimization steps such as soft-thresholding and coordinate-wise updates. These operations are more specific than standard gradient descent and therefore are handled directly inside the Lasso model.
`main.py`
The file `main.py` builds the complete experimental pipeline using the courselib-based `LassoRegression` implementation.
It automatically runs all experiments, generates the figures, and saves them into the `figures/` folder.
---
Project Structure
```text
.
├── courselib/
│   ├── models/
│   │   ├── base.py
│   │   ├── glm.py
│   │   ├── linear_models.py
│   │   ├── nn.py
│   │   ├── svm.py
│   │   └── tree.py
│   ├── utils/
│   └── optimizers.py
├── figures/
│   ├── convergence.png
│   ├── sparsity_lambda.png
│   ├── lasso_path.png
│   ├── recovery_error.png
│   ├── sklearn_comparison.png
│   ├── init_Subgradient.png
│   ├── init_ISTA.png
│   ├── init_FISTA.png
│   ├── init_CoordinateDescent.png
│   ├── initialization_final_objective.png
│   ├── short_run_initialization_sensitivity.png
│   ├── regularization_distance_to_true_w.png
│   ├── regularization_strength_robustness.png
│   ├── lambda_heatmap.png
│   ├── diabetes_convergence_speed.png
│   ├── diabetes_sparsity_vs_regularization.png
│   ├── diabetes_test_mse_vs_regularization.png
│   └── diabetes_coefficient_heatmap.png
├── .gitignore
├── main.py
├── README.md
└── Efficient Algorithms for Lasso Regression Report.pdf
```
---
Mathematical Background
The Lasso objective can be written as:
$$F(w)=g(w)+\lambda\lVert w\rVert_1$$
where the smooth least-squares part is:
$$g(w)=\frac{1}{2n}\lVert Xw-y\rVert_2^2$$
The gradient of the smooth part is:
$$\nabla g(w)=\frac{1}{n}X^\top(Xw-y)$$
The main difficulty is the nonsmooth $\ell_1$ penalty. The absolute value function is not differentiable at zero, so ordinary gradient descent is not ideal for Lasso. Proximal methods and coordinate-wise methods are more suitable because they can apply soft-thresholding and produce exact zero coefficients.
The soft-thresholding operator is:
$$S_\tau(z)=\mathrm{sign}(z)\cdot\mathrm{max}(|z|-\tau,0)$$
This operator shrinks small coefficients to zero and is the key reason why ISTA, FISTA, and Coordinate Descent can produce sparse solutions.
---
Optimization Methods
Subgradient Descent
Subgradient Descent directly applies a subgradient update to the full Lasso objective:
$$w^{(k+1)}=w^{(k)}-\eta_k\left(\frac{1}{n}X^\top(Xw^{(k)}-y)+\lambda s^{(k)}\right)$$
where $s_j^{(k)} \in \partial |w_j^{(k)}|$.
For each coordinate, the subgradient is defined as follows:
$$s_j^{(k)}=1\text{ if }w_j^{(k)}>0,\quad s_j^{(k)}=-1\text{ if }w_j^{(k)}<0,\quad s_j^{(k)}\in[-1,1]\text{ if }w_j^{(k)}=0$$
Subgradient Descent is simple and useful as a baseline, but it usually converges slowly for Lasso and does not naturally produce exact zero coefficients.
ISTA
ISTA, or Iterative Shrinkage-Thresholding Algorithm, is a proximal-gradient method.
It first takes a gradient step on the smooth least-squares part:
$$z^{(k)}=w^{(k)}-\frac{1}{L}\nabla g(w^{(k)})$$
Then it applies soft-thresholding:
$$w^{(k+1)}=S_{\lambda/L}(z^{(k)})$$
ISTA is better suited for Lasso than Subgradient Descent because the soft-thresholding step can set coefficients exactly to zero.
FISTA
FISTA is an accelerated version of ISTA. It adds a momentum term to improve convergence speed.
The main update is:
$$w^{(k+1)}=S_{\lambda/L}\left(z^{(k)}-\frac{1}{L}\nabla g(z^{(k)})\right)$$
The acceleration parameter is updated by:
$$t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}$$
The extrapolated point is updated by:
$$z^{(k+1)}=w^{(k+1)}+\frac{t_k-1}{t_{k+1}}\left(w^{(k+1)}-w^{(k)}\right)$$
FISTA usually converges faster than ISTA while keeping the same sparsity-inducing soft-thresholding step.
Coordinate Descent
Coordinate Descent updates one coefficient at a time while keeping all other coefficients fixed.
For coordinate $j$, the update is based on a one-dimensional Lasso subproblem:
$$w_j\leftarrow\frac{S_\lambda\left(\frac{1}{n}X_j^\top r_j\right)}{\frac{1}{n}\lVert X_j\rVert_2^2}$$
where $r_j$ is the partial residual excluding the current contribution of feature $j$.
Coordinate Descent is especially effective for Lasso because each coordinate update has a closed-form soft-thresholding solution.
---
Requirements
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
---
Running the Experiments
To reproduce all experiments and figures, run:
```bash
python main.py
```
The script automatically creates a folder called `figures/` and saves all generated plots there.
You can also run the script inside Jupyter Notebook or JupyterLab with:
```python
%run main.py
```
---
Metric Notes
The experiments use several evaluation metrics. In all objective-value, recovery-error, and test-MSE plots, lower values indicate better performance.
Metric	Meaning	Interpretation
Objective value	Value of the Lasso objective function	Lower means the optimizer has found a better solution to the optimization problem
Nonzero coefficients	Number of coefficients that are not exactly zero	Smaller values indicate a sparser model and stronger feature selection
L2 recovery error	Distance between the estimated coefficient vector and the true coefficient vector	Lower means the estimated coefficients are closer to the ground truth
Precision	Fraction of selected features that are truly relevant	Higher means fewer false-positive features
Recall	Fraction of truly relevant features that are successfully selected	Higher means fewer missed true features
F1-score	Harmonic mean of precision and recall	Higher means better overall support recovery
Test MSE	Mean squared error on the test set	Lower means better predictive performance
For support recovery, true positives are correctly selected nonzero features, false positives are irrelevant features incorrectly selected as nonzero, and false negatives are truly relevant features that were missed. Precision, recall, and F1-score are computed as:
$$Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}$$
---
Experiments and Results
3.1 Convergence
This experiment compares the convergence behavior of Subgradient Descent, ISTA, FISTA, and Coordinate Descent on the synthetic Lasso problem. The figure reports the objective value over iterations or coordinate-descent sweeps.
<table>
  <tr>
    <td align="center" width="65%">
      <img src="figures/convergence.png" width="520">
      <br>
      <em>Figure 1. Objective value versus iteration.</em>
    </td>
    <td align="center" width="35%">
      <strong>Final Objective Values</strong>
      <br><br>
      <table>
        <tr>
          <th>Solver</th>
          <th>Final objective</th>
        </tr>
        <tr>
          <td>Subgradient Descent</td>
          <td align="right">1.393</td>
        </tr>
        <tr>
          <td>ISTA</td>
          <td align="right">1.387</td>
        </tr>
        <tr>
          <td>FISTA</td>
          <td align="right">1.298</td>
        </tr>
        <tr>
          <td>Coordinate Descent</td>
          <td align="right">1.298</td>
        </tr>
      </table>
    </td>
  </tr>
</table>
The convergence figure should be read from top to bottom: a faster downward curve means that the method reduces the Lasso objective more quickly. The final-objective table gives a compact numerical comparison at the end of training. Since the objective value is the quantity being minimized, smaller values are better.
Subgradient Descent and ISTA decrease the objective slowly, and their curves almost overlap under the chosen step size. Although both methods reduce the objective, they require many iterations to approach a good solution. ISTA has the advantage of using soft-thresholding, which can set coefficients exactly to zero, while Subgradient Descent only uses the subgradient of the $\ell_1$ penalty.
FISTA reaches a substantially lower objective value because the acceleration step improves the proximal-gradient update. Coordinate Descent achieves almost the same final objective as FISTA, but it does so by repeatedly solving one-dimensional coordinate-wise Lasso subproblems. The final objective values also show that FISTA and Coordinate Descent converge to the best solutions among the tested methods, with final objectives around 1.298 compared with around 1.39 for Subgradient Descent and ISTA.
Overall, Figure 1 and the table indicate that FISTA and Coordinate Descent are the most efficient solvers in this experiment. Subgradient Descent is useful as a baseline, but it is less suitable for Lasso because it lacks an explicit sparsity-producing thresholding step.
---
3.2 Sparsity and Lasso Path
The sparsity experiment studies how the number of nonzero coefficients changes as the regularization strength $\lambda$ changes. Larger $\lambda$ values put more weight on the $\ell_1$ penalty, so the solution is expected to become sparser.
<table>
  <tr>
    <td align="center">
      <img src="figures/sparsity_lambda.png" width="420">
      <br>
      <em>Figure 2. Sparsity versus lambda.</em>
    </td>
    <td align="center">
      <img src="figures/lasso_path.png" width="420">
      <br>
      <em>Figure 3. Lasso path as lambda decreases.</em>
    </td>
  </tr>
</table>
In the sparsity plot, the y-axis represents the number of active coefficients. A lower value means that the fitted model uses fewer features. This is important because one of the main purposes of Lasso is feature selection.
The sparsity plot confirms that increasing $\lambda$ leads to fewer nonzero coefficients for ISTA, FISTA, and Coordinate Descent. These methods use soft-thresholding, which can shrink small coefficients exactly to zero. Therefore, they are effective for producing sparse Lasso solutions.
Subgradient Descent behaves differently. Even when $\lambda$ increases, it keeps many coefficients nonzero. This happens because the subgradient update does not include an explicit thresholding step. Coefficients may become small, but they are rarely set exactly to zero. As a result, Subgradient Descent produces denser solutions and is less suitable for feature selection.
The Lasso path shows how individual coefficients enter or leave the model as the regularization strength changes. When $\lambda$ is large, regularization is strong and only a few coefficients remain active. As $\lambda$ decreases, regularization becomes weaker and more coefficients enter the model. This illustrates the bias-sparsity trade-off: a larger $\lambda$ gives a simpler and sparser model, while a smaller $\lambda$ allows the model to fit more features.
---
3.3 Sparse Recovery
Sparse recovery is evaluated on the synthetic dataset because the true coefficient vector is known. The goal is not only to minimize the objective value, but also to recover the original sparse structure of the ground-truth vector.
The first metric is the L2 recovery error, $\lVert \hat{w} - w^* \rVert_2$, where $\hat{w}$ is the estimated coefficient vector and $w^*$ is the true coefficient vector. A smaller value means that the estimated coefficients are closer to the ground truth. Precision, recall, and F1 evaluate whether the algorithm correctly identifies the nonzero features.
Method	L2 recovery error	Precision	Recall	F1
Subgradient Descent	1.486	0.051	1.000	0.096

ISTA	1.483	0.185	1.000	0.313
FISTA	0.748	0.345	1.000	0.513
Coordinate Descent	0.749	0.357	1.000	0.526
This table reports both coefficient recovery and support recovery. FISTA and Coordinate Descent achieve much smaller L2 recovery errors than Subgradient Descent and ISTA. This shows that they estimate the true coefficient vector more accurately.
All methods achieve recall equal to 1.000, meaning that all 10 true nonzero coefficients are recovered. However, precision differs strongly across methods. Subgradient Descent has very low precision because it selects many false-positive features. In other words, it keeps many coefficients nonzero even though they should be zero. ISTA improves precision because soft-thresholding removes more irrelevant coefficients. FISTA and Coordinate Descent achieve the best support recovery, with higher precision and higher F1-scores.
Method	Objective	Nonzero coefficients	L2 recovery error
Our Coordinate Descent	1.2976	28	0.7485
sklearn Lasso	1.2976	28	0.7485
This comparison is used as an implementation check. The objective value measures optimization quality, the number of nonzero coefficients measures sparsity, and the L2 recovery error measures distance from the true coefficient vector. Our Coordinate Descent implementation matches `sklearn.linear_model.Lasso` almost exactly across all three values. This validates the correctness of the courselib-based implementation.
---
3.4 Robustness
The robustness experiments test whether the solvers behave reliably under different initialization schemes and different regularization strengths. A good Lasso solver should perform well not only in one fixed setting, but also under reasonable changes in the experimental setup.
3.4.1 Robustness to Initialization
We compare three starting points: zero initialization, small random initialization, and large random initialization. The goal is to test whether the final solution depends strongly on the initial coefficient vector.
<table>
  <tr>
    <td align="center">
      <img src="figures/init_Subgradient.png" width="420">
      <br>
      <em>Subgradient Descent</em>
    </td>
    <td align="center">
      <img src="figures/init_ISTA.png" width="420">
      <br>
      <em>ISTA</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="figures/init_FISTA.png" width="420">
      <br>
      <em>FISTA</em>
    </td>
    <td align="center">
      <img src="figures/init_CoordinateDescent.png" width="420">
      <br>
      <em>Coordinate Descent</em>
    </td>
  </tr>
</table>
These four plots compare the same solver under different starting coefficient vectors. If the curves end at almost the same objective value, then the solver is robust to initialization in the long run. If the curves differ strongly at the beginning, then initialization affects the early optimization path.
For Subgradient Descent and ISTA, different initializations lead to slightly different early-stage trajectories, and convergence remains relatively slow. FISTA is much less affected after the first few iterations because acceleration rapidly reduces the objective value. Coordinate Descent also reaches nearly the same final objective across all initializations, since each coordinate update quickly corrects the coefficient values.
<table>
  <tr>
    <td align="center">
      <img src="figures/initialization_final_objective.png" width="420">
      <br>
      <em>Figure 5. Final objective under different initializations.</em>
    </td>
    <td align="center">
      <img src="figures/short_run_initialization_sensitivity.png" width="420">
      <br>
      <em>Figure 6. Short-run initialization sensitivity.</em>
    </td>
  </tr>
</table>
The final-objective comparison shows that all solvers are robust to initialization after sufficient iterations. The final values are close within each solver, which means that the final solution is not strongly determined by the initial coefficient vector.
However, the short-run experiment reveals more visible solver differences when the iteration budget is limited. This is important because practical optimization often uses a finite iteration budget. FISTA and Coordinate Descent reach good solutions quickly, while Subgradient Descent and ISTA remain farther from convergence. Therefore, initialization is not a major problem in the long run, but solver choice matters strongly in the short run.
3.4.2 Robustness to Regularization Strength
We vary $\lambda / \lambda_{\max}$ and measure the L2 distance between the estimated coefficient vector and the true sparse vector, $\lVert \hat{w} - w^* \rVert_2$. This metric is available only for synthetic data because the true coefficient vector is known.
<table>
  <tr>
    <td align="center">
      <img src="figures/regularization_distance_to_true_w.png" width="420">
      <br>
      <em>Figure 7. Distance to true coefficients across lambda values.</em>
    </td>
    <td align="center">
      <img src="figures/lambda_heatmap.png" width="420">
      <br>
      <em>Figure 8. Heatmap of recovery error across lambda values.</em>
    </td>
  </tr>
</table>
The ratio $\lambda/\lambda_{\max}$ normalizes the regularization strength, making different values easier to compare. A large ratio means strong regularization and therefore stronger shrinkage. A small ratio means weaker regularization and allows more coefficients to remain active.
For ISTA, FISTA, and Coordinate Descent, the recovery error decreases steadily as $\lambda$ becomes smaller. This means that weaker regularization allows these methods to recover the true coefficient vector more accurately in this synthetic setting. Subgradient Descent behaves less consistently and is more sensitive to the choice of $\lambda$.
The heatmap provides a compact comparison across solvers and regularization values. Darker or lower-error regions indicate better recovery. ISTA, FISTA, and Coordinate Descent show similar and stable patterns, while Subgradient Descent is less predictable. Overall, proximal and coordinate-wise methods are more robust to regularization strength and recover the sparse ground-truth vector more accurately.
---
3.5 Validation on the Diabetes Dataset
The synthetic experiments are useful because the true sparse vector is known. To evaluate the solvers on real data, we also test them on the scikit-learn Diabetes dataset. The evaluation uses test MSE, sparsity, selected features, coefficient heatmaps, and comparison with `sklearn.linear_model.Lasso`.
<table>
  <tr>
    <td align="center">
      <img src="figures/diabetes_convergence_speed.png" width="420">
      <br>
      <em>Figure 9. Convergence speed on the Diabetes dataset.</em>
    </td>
    <td align="center">
      <img src="figures/diabetes_test_mse_vs_regularization.png" width="420">
      <br>
      <em>Figure 10. Test MSE versus regularization strength.</em>
    </td>
  </tr>
</table>
The Diabetes convergence plot confirms the pattern observed on the synthetic dataset. FISTA and Coordinate Descent reduce the training objective much faster than Subgradient Descent and ISTA. This suggests that the convergence advantage of proximal acceleration and coordinate-wise updates also appears on real data.
The test MSE plot evaluates predictive performance rather than only optimization quality. Lower test MSE means better prediction on unseen data. The plot shows that FISTA, Coordinate Descent, and sklearn Lasso achieve nearly identical prediction performance across $\lambda / \lambda_{\max}$ values, which indicates that the implemented solvers are competitive with the standard sklearn implementation.
<table>
  <tr>
    <td align="center">
      <img src="figures/diabetes_sparsity_vs_regularization.png" width="420">
      <br>
      <em>Figure 11. Sparsity versus regularization strength.</em>
    </td>
    <td align="center">
      <img src="figures/diabetes_coefficient_heatmap.png" width="420">
      <br>
      <em>Figure 12. Coefficients at each solver's best lambda.</em>
    </td>
  </tr>
</table>
The sparsity plot shows how many diabetes features remain active after fitting the model. Subgradient Descent keeps more features active and gives the densest solution. ISTA is sparser, while FISTA, Coordinate Descent, and sklearn Lasso produce very similar sparsity patterns.
The coefficient heatmap shows both feature selection and coefficient direction. A zero or near-zero value means that the feature is not selected by the model. Positive and negative values show the direction of association with the target after preprocessing. The heatmap further confirms that FISTA, Coordinate Descent, and sklearn Lasso select almost identical features and coefficient values.
Solver	Best lambda ratio	Best test MSE	Sparsity at best lambda	Selected features
Subgradient Descent	0.046416	2801.133037	10	bmi, s5, bp, s3, sex, s1, s6, s2, s4, age
ISTA	0.046416	2801.078318	8	bmi, s5, bp, s3, sex, s1, s6, s2
FISTA	0.046416	2798.609930	7	bmi, s5, bp, s3, sex, s1, s6
Coordinate Descent	0.046416	2798.601225	7	bmi, s5, bp, s3, sex, s1, s6
sklearn Lasso	0.046416	2798.601223	7	bmi, s5, bp, s3, sex, s1, s6
The table summarizes the best test performance for each solver. The best lambda ratio is selected according to the lowest test MSE. The sparsity column reports the number of nonzero coefficients at that best lambda value.
All methods obtain their best test MSE at the same lambda ratio, 0.046416. FISTA, Coordinate Descent, and sklearn Lasso achieve almost identical best test MSE values and select exactly the same seven features: `bmi`, `s5`, `bp`, `s3`, `sex`, `s1`, and `s6`. ISTA selects one additional feature, `s2`, while Subgradient Descent keeps all ten features.
This result is important because it shows a difference between prediction and sparsity. Subgradient Descent obtains a test MSE close to the other methods, but it does so with a much less sparse model. FISTA, Coordinate Descent, and sklearn Lasso achieve slightly better prediction while selecting fewer features. Overall, the Diabetes validation supports the conclusions from the synthetic experiments: Coordinate Descent performs best overall, FISTA is also highly effective, and Subgradient Descent is useful as a baseline but less suitable for sparse Lasso optimization.
---
Main Findings
The main findings of the project are:
The project now uses the required `courselib` structure through `courselib/models/linear_models.py` and `courselib/optimizers.py`.
Subgradient Descent is simple and useful as a baseline, but it is inefficient for sparse Lasso optimization.
ISTA is better suited for Lasso because soft-thresholding can create exact zero coefficients.
FISTA improves convergence speed through acceleration.
Coordinate Descent performs very well because each coordinate update has a closed-form soft-thresholding solution.
Larger regularization strength produces sparser models.
The implemented Coordinate Descent method matches `sklearn.linear_model.Lasso` closely.
The courselib-based implementation gives consistent results on both synthetic data and the Diabetes dataset.
---
Reproducibility Notes
The synthetic experiment uses `np.random.seed(42)`.
The Diabetes train-test split uses `random_state=42`.
All figures are saved automatically to the `figures/` folder.
Results may differ slightly across machines because of floating-point precision, but the qualitative conclusions should remain the same.
---
Pre-trained Models
This project does not use pre-trained models. All solvers are implemented directly for Lasso regression and are run from scratch.
---
Report
The full project report is included as:
```text
Efficient Algorithms for Lasso Regression Report.pdf
```
---
Authors
Mengying Li
Yuehangsha Huang
---
Contributing
This repository is intended for a course project submission. External contributions are not expected.
---
License
This project is for educational use in the LMU Applied Machine Learning in Python course. No external open-source license is specified.
