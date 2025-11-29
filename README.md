# Influence of Discount Factor on Deep Q-Learning in FrozenLake

## 📌 Overview

This project provides a comprehensive experimental study on how the discount factor ($\gamma$) influences the behavior, learning stability, and convergence speed of a Deep Q-Network (DQN) agent in the FrozenLake-v1 environment.

The study is conducted in three scientific phases:

- **Hyperparameter Optimization**: Grid Search to establish optimal baselines for both Deterministic and Stochastic environments.
- **Behavioral Analysis (Static $\gamma$)**: Investigating the transition from "short-sighted" ($\gamma=0.1$) to "far-sighted" ($\gamma=0.99$) behavior using policy heatmaps and Q-value analysis.
- **Curriculum Learning (Adaptive $\gamma$)**: Implementing an adaptive schedule where $\gamma$ increases linearly from $0.8$ to $0.99$. This approach stabilizes early training variance while allowing for optimal long-term planning, based on the findings of François-Lavet et al. (2016).

## 📂 Project Structure
```bash
drl-gamma-experiments/
├── src/ # Source Code
│ ├── agents/ # DQN Agent implementation
│ ├── environments/ # FrozenLake wrapper and config
│ └── utils/ # Helper functions (one-hot encoding)
│
├── experiments/ # Experiment Scripts
│ ├── grid_search.py # Phase 1: Hyperparameter Grid Search
│ ├── run_gamma_study.py # Phase 2: Static Gamma Analysis (Training)
│ ├── visualize_metrics.py # Phase 2: Generate Learning Curves & Metrics
│ ├── visualize_policy.py # Phase 2: Generate Policy Heatmaps
│ ├── run_adaptive_study.py # Phase 3: Adaptive Gamma (Curriculum)
│ └── visualize_adaptive.py # Phase 3: Generate Comparison Plots
│
└── results/ # Generated Artifacts
├── grid_search_*/ # JSON/CSV summaries of grid search
├── gamma_study_analysis/ # Models, Logs, and Plots for Static Study
└── adaptive_gamma_study/ # Models, Logs, and Plots for Adaptive Study
```

## ⚙️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install gymnasium torch matplotlib numpy pandas tensorboard seaborn pyyaml
```

## Usage Guide
### Phase 1: Hyperparameter Search

Run a grid search to find the best hyperparameters for both deterministic and stochastic environments.

```bash
python experiments/grid_search.py
```


### Phase 2: Static Gamma Study (Behavioral Analysis)

Train agents with fixed Gamma values (0.1, 0.5, 0.9, 0.99) to observe the transition from myopic to far-sighted behavior.

#### 1. Run Training:
```bash 
python experiments/run_gamma_study.py
```

#### 2. Generate Visualizations:
```bash
# Generate Learning Curves, Step Counts, and Q-Value plots
python experiments/visualize_metrics.py  

# Generate Policy Heatmaps (Arrows showing agent strategy)
python experiments/visualize_policy.py
```

### Phase 3: Adaptive Gamma (Curriculum Learning)

Compare a Standard DQN ($\gamma=0.99$) against an Adaptive DQN ($\gamma \to 0.8 \dots 0.99$).

1. Run Comparison:
```bash
python experiments/run_adaptive_study.py
```

2. Generate Dual-Axis Plot:
```bash
python experiments/visualize_adaptive.py
```

### 📈 Viewing Logs (TensorBoard)
To monitor training progress in real-time for any experiment, run TensorBoard from the root directory:
```bash
# For Static Study
tensorboard --logdir results/gamma_study_analysis/tensorboard

# For Adaptive Study
tensorboard --logdir results/adaptive_gamma_study/tensorboard
```

### 🔑 Key Findings

Stochastic Environments: Require a higher discount factor ($\gamma \approx 0.99$) and slower exploration decay to overcome variance and learn safe paths.

Short-Sightedness: Low $\gamma$ agents ($<0.5$) fail to solve the stochastic map, often getting stuck in local optima or wandering aimlessly.

Curriculum Learning: Starting with a lower $\gamma$ ($0.8$) and increasing it to $0.99$ reduces initial variance in Q-value targets, leading to faster initial convergence compared to a fixed $\gamma=0.99$.