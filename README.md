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

## 🚀 Usage Guide
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



### 🗺️ 8x8 Environment

This section outlines the changes required to switch the Deep Q-Network experiments from the standard 4x4 FrozenLake map to the larger 8x8 map.

#### 1. Dimensionality Change

Switching maps changes the input dimension of the Neural Network:

| Map Size | Discrete States | Input Vector Size |
|----------|-----------------|-------------------|
| 4x4      | 16              | 16                |
| 8x8      | 64              | 64                |

> ⚠️ **Important:** If `state_size` is not updated dynamically, the Neural Network will fail to load or process states, throwing `RuntimeError: shape mismatch`.

#### 2. Configuration Changes

The global constant `MAP_NAME` in `experiments/run_adaptive_study.py` controls the environment size.

To switch to 8x8, change the configuration line at the top of the script:

```python
MAP_NAME = "8x8"  # Changed from "4x4"
```

> 💡 **Recommendation:** Since the 8x8 map is significantly harder (sparser rewards), consider increasing the episode count from `2000` to `3000`.

#### 3. Running the Experiment

1. Open `experiments/run_adaptive_study_8x8.py`
2. Ensure `MAP_NAME = "8x8"` is set
3. Run the script:
   ```bash
   python experiments/run_adaptive_study_8x8.py
   ```
4. Results will be saved in `results/adaptive_gamma_study/8x8/`