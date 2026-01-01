"""
Step 1: NSGA-II Multi-Objective Optimization - Core Framework

This is the core optimization framework. You need to:
1. Prepare your own trained surrogate models (XGBoost, RF, NN, etc.)
2. Define your decision variables and their bounds
3. Implement your specific constraint functions
4. Set optimization directions (maximize/minimize) for each objective

Repository: [https://github.com/HarryZhang386/paper_Adaptive-multi-objective-optimization.git]
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM


# ==================================================================
# CORE: Multi-Objective Problem Definition
# ==================================================================
class OptimizationProblem(Problem):
    """
    Define your multi-objective optimization problem here.

    Key components to customize:
    - self.models: Your trained surrogate models (must have .predict() method)
    - self.directions: [1, -1, 1, ...] where 1=maximize, -1=minimize
    - _is_feasible(): Implement your domain-specific constraints
    """

    def __init__(self, models, base_values, bounds, directions):
        """
        Parameters:
            models: List of trained prediction models
            base_values: Baseline values for decision variables (numpy array)
            bounds: (lower_bounds, upper_bounds) tuple
            directions: Optimization direction for each objective
                       1 = maximize, -1 = minimize
        """
        self.models = models
        self.base_values = base_values
        self.directions = np.array(directions)

        lower_bounds, upper_bounds = bounds

        super().__init__(
            n_var=len(base_values),
            n_obj=len(models),
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        CORE EVALUATION: Use surrogate models to predict objectives.

        This is the heart of the optimization - it:
        1. Takes candidate solutions (X)
        2. Checks feasibility using your constraints
        3. Predicts objectives using trained models
        4. Applies optimization directions (max/min)
        """
        F = np.zeros((X.shape[0], self.n_obj))

        for i, x in enumerate(X):
            # Check if solution satisfies your constraints
            if not self._is_feasible(x):
                F[i] = 1e6  # Large penalty for infeasible solutions
                continue

            try:
                # Predict objectives using your surrogate models
                predictions = np.array([
                    model.predict(x.reshape(1, -1))[0]
                    for model in self.models
                ])

                # Apply direction: multiply by -1 for maximization
                # (pymoo minimizes by default)
                F[i] = predictions * self.directions

            except Exception:
                F[i] = 1e6

        out["F"] = F

    def _is_feasible(self, x):
        """
        CUSTOMIZE THIS: Implement your domain-specific constraints.

        Examples of constraints you might add:
        - Sum constraints: e.g., sum(x[0:4]) must be between [0.3, 0.9]
        - Product constraints: e.g., x[0] * x[1] < 0.5
        - Non-negativity: e.g., x[i] >= baseline[i] (no decrease allowed)
        - Physical limits: e.g., density, coverage, balance requirements

        Return:
            True if solution is feasible, False otherwise
        """
        # Example constraint (customize for your research):
        # if not (0.3 <= np.sum(x[[0,1,3]]) <= 0.9):
        #     return False

        return True  # Placeholder - implement your constraints


# ==================================================================
# CORE: NSGA-II Optimization Engine
# ==================================================================
def optimize_with_nsga2(models, x_baseline, bounds, directions,
                        pop_size=100, n_gen=100, seed=42):
    """
    Run NSGA-II multi-objective optimization.

    This is the main optimization function that:
    1. Sets up the NSGA-II algorithm with genetic operators
    2. Runs the evolutionary algorithm for specified generations
    3. Returns Pareto-optimal solutions

    Parameters:
        models: Your trained surrogate models (list)
        x_baseline: Baseline decision variables (numpy array)
        bounds: (lower, upper) bounds for variables
        directions: [1/-1] for each objective (1=max, -1=min)
        pop_size: Population size per generation
        n_gen: Number of generations
        seed: Random seed for reproducibility

    Returns:
        X_pareto: Optimized decision variables (Pareto front)
        F_pareto: Objective values (Pareto front)

    Recommended parameter tuning:
    - For quick exploration: pop_size=50, n_gen=50
    - For thorough search: pop_size=100-200, n_gen=100-200
    - For complex problems: increase both proportionally
    """
    np.random.seed(seed)

    # Define the optimization problem
    problem = OptimizationProblem(models, x_baseline, bounds, directions)

    # Configure NSGA-II algorithm
    # Key genetic operators:
    # - SBX (Simulated Binary Crossover): combines parent solutions
    # - PM (Polynomial Mutation): introduces variation
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),  # 90% crossover probability
        mutation=PM(prob=0.2, eta=20),  # 20% mutation probability
        eliminate_duplicates=True  # Remove duplicate solutions
    )

    # Run optimization
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False
    )

    if result.X is None:
        return None, None

    # Extract Pareto-optimal solutions
    X_pareto = result.opt.get("X")  # Decision variables
    F_pareto = result.opt.get("F")  # Objective values

    return X_pareto, F_pareto


# ==================================================================
# Usage Example
# ==================================================================
"""
WORKFLOW TO USE THIS FRAMEWORK:

1. PREPARE YOUR DATA AND MODELS
   - Train surrogate models (e.g., XGBoost) on your data
   - Each model predicts one objective (UTCI, restorative potential, etc.)

2. DEFINE YOUR PROBLEM
   - Set baseline values: x_baseline = np.array([...])
   - Set bounds: lower = x_baseline * 0.7, upper = x_baseline * 1.3
   - Set directions: [1, -1, 1] for [max, min, max]

3. IMPLEMENT CONSTRAINTS
   - Edit _is_feasible() method in OptimizationProblem class
   - Add your domain-specific constraints

4. RUN OPTIMIZATION
   X_opt, F_opt = optimize_with_nsga2(
       models=your_models,
       x_baseline=your_baseline,
       bounds=(lower, upper),
       directions=[1, -1, 1],
       pop_size=100,
       n_gen=100
   )

5. POST-PROCESS RESULTS
   - Convert objectives back: F_original = F_opt * directions
   - Select solutions from Pareto front based on preferences
   - Analyze trade-offs between objectives

Example code:

    import xgboost as xgb

    # Load your trained models
    models = [xgb.XGBRegressor() for _ in range(3)]
    for i, model in enumerate(models):
        model.load_model(f"model_{i}.json")

    # Define baseline and bounds
    x_baseline = np.array([0.5, 0.3, 0.2, 0.4, 0.1])
    lower = np.maximum(0.0, x_baseline * 0.7)
    upper = np.minimum(1.0, x_baseline * 1.3)

    # Run optimization (max, min, max)
    X_opt, F_opt = optimize_with_nsga2(
        models=models,
        x_baseline=x_baseline,
        bounds=(lower, upper),
        directions=[1, -1, 1],
        pop_size=100,
        n_gen=100
    )

    # Results
    print(f"Found {len(X_opt)} Pareto solutions")
    F_original = F_opt * np.array([1, -1, 1])
    print(f"Objective ranges: {F_original.min(axis=0)} to {F_original.max(axis=0)}")
"""
