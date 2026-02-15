# Falcon 9 Bayesian Reliability and Reuse Economics

This project uses Bayesian modelling to estimate how reliably Falcon 9 boosters land and when recovery makes economic sense. The analysis separates intrinsic hardware capability from mission difficulty, then uses a simple expected‑value model to show how those two forces shape the economics of reuse.

All code and figures are in the accompanying Jupyter notebook.

## 1. Motivation

Reusable rockets only pay off when landing reliability is high enough to justify the risk. Falcon 9 boosters fly everything from easy LEO missions to demanding GTO and high‑energy trajectories. These missions differ in difficulty, and that difficulty drives recovery outcomes.

This project asks three straightforward questions:

How reliable are Falcon 9 landings across Blocks, orbits, and mission profiles?

How much does mission difficulty affect recovery success?

Under what conditions is reuse economically worthwhile?

The goal is to: show what the hardware can do, show what the missions demand, and explain how the two interact.

## 2. Data and Method

**Hierarchical Bayesian Model**

A multilevel logistic model estimates landing probability using:

- Block generation (intrinsic hardware capability)

- Orbit class (mission difficulty)

- Launch site

- Payload mass

This structure lets the model share information across missions while still capturing real differences between Blocks and orbits. It avoids misleading comparisons by separating hardware reliability from operational environment.

Inference is performed with PyMC.

## 2.1 Convergence Diagnostics

The model’s convergence was checked using standard Bayesian diagnostics:

- r̂ values were all at or very near 1.00, showing stable chains.

- Effective sample sizes (ESS) were high for both bulk and tail estimates.

- Trace plots showed good mixing with no drift or sticking.

- Energy plots showed no pathological behaviour.

- No divergences were reported after tuning.

These checks give confidence that the posterior summaries are reliable and that the hierarchical structure is well‑behaved.

## 3. Results

### 3.1 Block × Orbit Reliability

Landing reliability varies sharply by orbit class.

- LEO missions show high posterior means.

- High‑energy missions show much lower recovery probability.

Block differences exist, but orbit class dominates.

### 3.2 Mission‑Level Reliability

Each mission’s posterior mean and 90% credible interval shows the uncertainty inherent in recovery.

Well‑sampled missions are tightly constrained.

Rare or high‑energy missions have wide intervals.

### 3.3 Expected Value by Orbit Class

Using a simple payoff structure:

- +30 M for a successful landing

- –50 M for a failed recovery

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

## 3.4 Catastrophic‑Loss Sensitivity

Rare, high‑impact failures dominate the downside.
Even small increases in catastrophic‑loss probability push expected value sharply negative.

## 4. Interpretation

Three conclusions stand out:

Reuse economics are orbit‑dependent.  
Mission profile, not Block generation, is the primary driver of expected value.

Block 5 is intrinsically reliable, but its operational expected value varies because it flies the most demanding missions.

Low‑energy missions offer the strongest economic case for reuse, while high‑energy missions remain challenging under the assumed payoff structure.

The key distinction is between intrinsic hardware capability and orbit‑specific operational difficulty. Mixing the two leads to bad conclusions.

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

- Add refurbishment‑cost modelling
- Extend to Falcon Heavy
- Posterior predictive checks for future missions
- Model landing‑burn fuel margins explicitly
