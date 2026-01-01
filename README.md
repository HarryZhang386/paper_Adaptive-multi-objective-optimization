# Adaptive Multi-Objective Optimization Framework

Core implementation of the adaptive multi-objective optimization methodology presented in our paper.

## Overview

This repository contains the core algorithms for adaptive multi-objective optimization using:
- **NSGA-II** for Pareto-optimal solution generation
- **KDE-based adaptive weighting** for priority assignment
- **Weighted Pareto solution selection** for context-aware optimization

## Framework Structure

The optimization framework consists of three sequential steps:

```
Input Data
    ↓
Step 1: NSGA-II Optimization → Pareto-optimal solutions
    ↓
Step 2: Adaptive Weight Calculation → Priority weights
    ↓
Step 3: Weighted Solution Selection → Best solutions
    ↓
Final Recommendations
```

## Files

- **Step 1_NSGA-II.py** - Multi-objective optimization using NSGA-II
- **Step2 Weight Assignment.py** - KDE-based priority weight assignment
- **Step 3_WPSS.py** - Adaptive solution selection from Pareto sets

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scipy pymoo xgboost
```

### Step 1: Multi-Objective Optimization

Generate Pareto-optimal solutions using NSGA-II:

```python
from step1_nsga2_optimization import optimize_with_nsga2

# Define your problem
directions = [1, -1, 1]  # [maximize RP, minimize UTCI, maximize TQ]

# Run optimization
X_opt, F_opt = optimize_with_nsga2(
    models=your_trained_models,
    x_baseline=baseline_values,
    bounds=(lower_bounds, upper_bounds),
    directions=directions,
    pop_size=100,
    n_gen=100
)
```

### Step 2: Calculate Adaptive Weights

Assign priority weights based on performance indicators:

```python
from step2_adaptive_weight_calculation import PriorityWeightCalculator

#k-value = 0.56
calculator = PriorityWeightCalculator(steepness=0.56)

# Define indicators and their directions
indicators = {
    'UTCI': {'data': utci_values, 'direction': 'negative'},
    'RP': {'data': rp_values, 'direction': 'positive'},
    'TQ': {'data': tq_values, 'direction': 'positive'}
}

# Calculate weights
weights = calculator.calculate_combined_weight(indicators)
```

### Step 3: Select Best Solutions

Choose optimal solutions using adaptive weights:

```python
from step3_weighted_solution_selection import WeightedParetoSelector

selector = WeightedParetoSelector(objective_directions=directions_dict)

# Select best solution for each sample
selected = selector.select_for_all_samples(pareto_sets, weights_list)
```

## Customization Guide

### For Your Research

You need to customize:

1. **Decision Variables**: Define your optimization parameters
2. **Surrogate Models**: Train models for your objectives (XGBoost, RF, NN, etc.)
3. **Constraints**: Implement domain-specific feasibility rules in `_is_feasible()`
4. **Indicators**: Specify performance metrics and their directions (positive/negative)
5. **Parameters**: Tune NSGA-II parameters and weight steepness for your problem

### Key Parameters

**NSGA-II (Step 1)**:
- `pop_size`: Population size (50-200 recommended)
- `n_gen`: Number of generations (50-200 recommended)
- `directions`: [1, -1, 1] for [max, min, max] objectives

**Weight Calculation (Step 2)**:
- `steepness`: Controls weight adaptiveness (0.3-0.8, default 0.56)
  - Lower values → more focused on worst areas
  - Higher values → more uniform distribution

**Solution Selection (Step 3)**:
- `objective_directions`: Must match Step 1 directions

## Citation

If you use this code in your research, please cite:

```
[Paper citation to be added upon publication]
```

## License

This code is provided for research purposes. Please adapt it to your specific use case.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This repository contains the core algorithmic framework. Specific implementation details (data preprocessing, model training, constraint definitions) should be adapted to your research context.
