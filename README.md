# 🔥 FireHawk: Intelligent UAV Path Planning for Forest Fire Prediction and Rescue

> A Reinforcement Learning-based system using DQN+ABC and DDPG+GA for smart wildfire surveillance, fire risk prediction, and UAV Search-and-Rescue optimization.

---

## 📌 Project Overview

**FireHawk** is a hybrid system that combines:
- Forest fire risk prediction using environmental data and XGBoost.
- UAV navigation via Deep Reinforcement Learning:
  - **DQN + Artificial Bee Colony (ABC)** for discrete path optimization.
  - **DQN + Grey Wolf Optimizer (GWO)** for intelligent task sequencing and coordination.
  - **DDPG + Genetic Algorithm (GA)** for continuous, evolved UAV behavior.
- A custom OpenAI Gym environment simulating forest zones, fire risk, UAV battery, and coverage.

This system is built to support disaster management, surveillance, and intelligent decision-making in wildfire-prone regions.

---

## 🚀 Features

✅ Predict fire risk zones from simulated or real data  
✅ Simulate UAV flight with customizable environments  
✅ Learn optimal UAV patrol policies using RL  
✅ Optimize UAV task sequences with nature-inspired algorithms  
✅ Visualize heatmaps, trajectories, convergence plots, and metrics  

---

## 📂 Directory Structure

```
forest-fire-uav-project/
├── data/                      # Dataset CSVs for fire risk
├── env/                      # Custom OpenAI Gym environments
├── fire_prediction/          # XGBoost model & dataset generation
├── models/                   # DDPG+GA, DQN+ABC implementations
├── utils/                    # Visualizations, helper functions
├── results/                  # Logs, training curves, metrics
├── main.py                   # Main training runner
├── requirements.txt
├── README.md
```

---

## 📊 Visuals

### 🔥 Fire Risk Heatmap
Shows predicted risk zones generated by the ML model.

### 🧭 Path Comparison
Overlay of paths taken by different algorithms (DQN vs DDPG).

### 📈 Training Curves
Plots of cumulative reward, coverage %, and convergence across episodes.

---

## ⚙️ Installation

```bash
# Clone the repository
$ git clone https://github.com/your-username/firehawk.git
$ cd firehawk

# Install dependencies
$ pip install -r requirements.txt
```

**Recommended:** Use a virtual environment.

---

## 🧠 How It Works

1. **Fire Risk Dataset** → Simulated using temperature, humidity, wind, NDVI, slope.
2. **XGBoost Model** → Predicts risk values for each forest grid cell.
3. **Environment** → UAV flies over grid, avoiding high-risk or revisiting zones.
4. **RL Models**:
   - DQN learns discrete policies with ABC and GWO optimizing task assignments.
   - DDPG evolves continuous flight paths with GA-tuned hyperparameters.
5. **Evaluation** → Compare coverage %, convergence rate, and safety.

---

## 🧪 Example Run

```bash
$ python main.py
```

Progress will be logged and visualizations saved under `/results/`.

---

## 📚 References

- Reinforcement Learning: Sutton & Barto
- OpenAI Gym
- XGBoost Fire Risk Prediction papers
- Nature-inspired Optimization: ABC, GA and GWO

---

## 🧑‍💻 Authors
- Chirayu Badgujar

---
