# Neural Network Quantum State Tomography

A PyTorch implementation of neural-network-based quantum state tomography (QST), following the approach of [Koutný et al. (arXiv:2206.06736)](https://arxiv.org/abs/2206.06736). The network reconstructs density matrices from simulated POVM measurement statistics and is benchmarked against Linear Inversion and Maximum Likelihood Estimation (MLE).

---

## Method

Density matrices are parameterized via their Cholesky decomposition: ρ = AA†/Tr(AA†), where A is lower-triangular complex. This guarantees every network output maps to a physically valid state (positive semidefinite, trace-1) without any post-processing projection.

The network takes a POVM frequency vector as input and predicts the Cholesky parameter vector as output. A tanh output activation matches the soft-clipped (−1, 1) range of the encoded parameters.

---

## Architecture

- **Network**: Feedforward (QSTNet) — `[Linear → ReLU] × 4 → Linear → Tanh`
- **Hidden layers**: 4, width 512
- **Loss**: MSE on Cholesky parameters
- **Optimizer**: Adam with cosine annealing LR schedule
- **POVM**: Pauli-basis IC-POVM (6 elements for qubit; SIC-like for higher dimensions)

---

## Dataset

Haar-random mixed states generated via the Ginibre ensemble, with simulated multinomial measurement noise.

| Split      | Samples | Shots |
|------------|---------|-------|
| Train      | 8,000   | 1,000 |
| Validation | 1,000   | 1,000 |
| Test       | 200     | 1,000 |

---

## Baselines

- **Linear Inversion (LI)**: pseudoinverse reconstruction; fast but not guaranteed to be physical.
- **MLE**: iterative R-ρ-R algorithm; enforces physicality but slow per state.

---

## Evaluation Metrics

- **Hilbert-Schmidt distance**: d_HS(ρ₁, ρ₂) = Tr[(ρ₁ − ρ₂)²]
- **Fidelity**: F(ρ₁, ρ₂) = (Tr[√(√ρ₁ ρ₂ √ρ₁)])²


---

## Requirements

```
pip install qutip torch numpy matplotlib scipy
```

---



Trains the model, evaluates against baselines, generates plots, and saves the trained model to `qst_model.pt`.

---

## Reference

Koutný et al., *Neural-network quantum state tomography*, arXiv:2206.06736
