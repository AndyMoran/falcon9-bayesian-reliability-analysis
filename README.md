# Falcon 9 Bayesian Reliability and Reuse Economics
This project uses hierarchical Bayesian modelling to estimate the landing reliability of SpaceX Falcon 9 boosters and to evaluate when booster recovery is economically worthwhile. The analysis combines engineering‑driven reliability modelling with a simple expected‑value framework to show how mission difficulty shapes the economics of reuse.

All code and figures are in the accompanying Jupyter notebook.

## 1. Motivation
Reusable rockets only make sense when landing reliability is high enough to justify the risk. Falcon 9 boosters fly a wide range of missions, from easy LEO deployments to demanding GTO and high‑energy trajectories. These missions differ in difficulty, and that difficulty drives recovery outcomes.

This project asks three questions:

How reliable are Falcon 9 landings across Blocks, orbits, and mission profiles

How much does mission difficulty affect recovery success

Under what conditions is reuse economically favourable

The goal is clarity: separate hardware capability from operational difficulty, and show how each contributes to the economics of reuse.

## 2. Data and Method
Hierarchical Bayesian Model
A multilevel logistic model estimates landing probability using:

Block generation (intrinsic hardware capability)

Orbit class (mission difficulty)

Launch site

Payload mass

This structure allows the model to share information across missions while still capturing real differences between Blocks and orbits. It also avoids misleading comparisons by separating intrinsic reliability from the operational environment.

Inference is performed with PyMC.

## 3. Results
### 3.1 Block × Orbit Reliability
Landing reliability varies sharply by orbit class.
LEO missions show high posterior means.
High‑energy missions show much lower recovery probability.


### 3.2 Mission‑Level Reliability
Each mission’s posterior mean and 90% credible interval shows the uncertainty inherent in operational recovery. Some missions are well‑constrained; others have wide intervals due to limited data.

### 3.3 Expected Value by Orbit Class
Using a simple payoff structure:

+30 M for a successful landing

–50 M for a failed recovery

the breakeven landing probability is:

𝑝
breakeven
=
50
80
≈
0.625
Low‑energy missions often approach or exceed this threshold.
High‑energy missions generally do not.

### 3.4 Catastrophic Loss Sensitivity
Rare, high‑impact failures dominate the downside.
Even small increases in catastrophic‑loss probability can push expected value sharply negative.

## 4. Interpretation
Three conclusions stand out:

Reuse economics are orbit‑dependent.  
Mission profile, not Block generation, is the primary driver of expected value.

Block 5 is intrinsically reliable, but its operational expected value varies because it flies the most demanding missions.

Low‑energy missions offer the strongest economic case for reuse, while high‑energy missions remain challenging under the assumed payoff structure.

The distinction between intrinsic hardware capability and orbit‑specific operational difficulty is essential for interpreting the economics of reuse.

## 5. Repository Structure

├── Falcon9_bayesian_analysis.ipynb   # Full analysis
├── figures/                          # All generated plots
│   ├── block_orbit_reliability.png
│   ├── mission_reliability.png
│   ├── ev_by_orbit_heatmap.png
│   ├── catastrophic_loss_sensitivity.png
│   └── ...
└── README.md

## 6. How to Run
Install dependencies:

Code
pymc
arviz
pandas
numpy
matplotlib
seaborn
Then open the notebook:

jupyter lab

## 7. Future Work
Add refurbishment‑cost modelling

Extend to Falcon Heavy

Posterior predictive checks for future missions

Model landing‑burn fuel margins explicitly
