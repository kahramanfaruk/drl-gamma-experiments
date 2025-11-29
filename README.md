# **Influence of Discount Factor on Deep Q-Learning in FrozenLake**

## **Overview**

This project investigates how the discount factor (**$\gamma$**) influences agent behavior in the **FrozenLake-v1** environment using **Deep Q-Learning**.

- Lower $\gamma$ values encourage **short-sighted** behavior focused on immediate rewards.  
- Higher $\gamma$ values promote **far-sighted** planning and long-term value estimation.  

The project also implements a **Curriculum Learning** strategy (Adaptive $\gamma$) to stabilize training across varying levels of environment difficulty.

---

## **Project Structure**

src/  
environments/ — FrozenLake environment wrapper  
agents/ — Deep Q-Learning agent implementation  
utils/ — Helper functions (e.g., one-hot encoding)  

experiments/ — Scripts for training + visualization across all study phases  
results/ — Logs, trained models, plots  

---

## **Usage**

### **1. Install Dependencies**

```bash
pip install gymnasium torch matplotlib numpy pandas tensorboard

2. Run Experiments

Phase 1: Hyperparameter Search
Find the best parameters for Deterministic and Stochastic maps:

python experiments/grid_search.py

Phase 2: Static Gamma Study (Behavioral Analysis)
Compare agents with low vs. high fixed discount factors.
Train agents (≈ 20 minutes):

python experiments/run_gamma_study.py

Generate Analysis Plots:

python experiments/visualize_metrics.py
python experiments/visualize_policy.py

Phase 3: Adaptive Gamma (Curriculum Learning)
Compare fixed γ = 0.99 vs. adaptive γ → 0.99 scheduling.
Train comparison:

python experiments/run_adaptive_study.py

Generate Dual-Axis Plot:

python experiments/visualize_adaptive.py

Viewing Logs
Monitor training progress in real time:

tensorboard --logdir results/adaptive_gamma_study/tensorboard