# SP-OCP-RSE

Code and data for the paper

**Structure-Preserving Fourth-Order Lie-Group Integration and Noise-Metric-Aware Optimal Control for Random Schrödinger Equations**
S. Vijayaram and P. Balasubramaniam, submitted to *SciPost Physics Core* (2026).

The code integrates the random Schrödinger equation (RSE) pathwise with the fourth-order commutator-free Magnus scheme (CFM4). It also optimizes noise-metric-aware controls and computes their gradients with the discrete adjoint method. All numerical results, figures and tables of the paper can be reproduced with it.

## Contents

| Path | Description |
|---|---|
| `notebooks/RSE_OCP_SciPostCore_reproducibility.ipynb` | Single notebook that reproduces every result, figure and table of the paper, block by block. This is the recommended starting point. |
| `rse_experiments_corrected.py` | Single-qubit experiments: CFM4 propagator, Matérn noise, training of the four control strategies, Monte Carlo validation, sweeps (Secs. 6.1–6.4, 6.6, 6.7, 6.9, 6.10). |
| `rse_adjoint.py` | Discrete adjoint (GRAPE-type) gradients for the CFM4 update, with the gradient check, timing and re-optimization study (Secs. 5.4 and 6.5). |
| `cz_experiment.py` | Two-qubit controlled-Z gate under correlated dephasing with JAX and L-BFGS-B (Sec. 6.8). |
| `data/cz_coeffs_published.npy` | The 40 optimized Fourier coefficients of the CZ controls reported in the paper. |
| `reference_results/` | Result files (JSON) of the runs reported in the paper, for comparison. |

## Installation

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

JAX is needed only for the two-qubit example. On Windows and macOS, `pip install "jax[cpu]"` installs the CPU version.

## Quick start

Open the notebook and run all cells (*Kernel → Restart & Run All*):

```bash
jupyter notebook notebooks/RSE_OCP_SciPostCore_reproducibility.ipynb
```

Outputs are written to `notebooks/rse_outputs/`. A full run takes about 20 to 40 minutes on a laptop. Optimizations are checkpointed, so a second run is fast.

## Running the scripts

Single-qubit experiments. Run the stages in this order:

```bash
python rse_experiments_corrected.py opts        # noise-blind, isotropic and anisotropic controls
python rse_experiments_corrected.py optD        # ensemble baseline (repeat until it prints D_DONE)
python rse_experiments_corrected.py eval        # Monte Carlo validation, Table 1, Figs. 4-6
python rse_experiments_corrected.py unitarity   # Fig. 1
python rse_experiments_corrected.py pareto      # Figs. 3 and 8
python rse_experiments_corrected.py sweeps      # Figs. 9 and 10
python rse_experiments_corrected.py aniso       # Figs. 12 and 13
```

Add `--dephasing` to the `opts`, `optD` and `eval` stages for the additive dephasing scenario of Sec. 6.10 (Fig. 14). Outputs are written to `out/` and `out_dephasing/`.

Adjoint gradients (the single-qubit script must be in the same folder):

```bash
python rse_adjoint.py all        # or: check | timing | reopt
```

Outputs are written to `out_adjoint/`.

Two-qubit CZ gate:

```bash
python cz_experiment.py published   # evaluates the published coefficients (about 10 s)
python cz_experiment.py             # re-runs the L-BFGS-B optimization, then evaluates it
```

Outputs are written to `out_cz_published/` and `out_cz/`. The `published` mode reproduces the values of Sec. 6.8 exactly: noiseless fidelity 0.99977, mean fidelity 0.99766 and worst-case fidelity 0.98910 over 1000 noise realizations. L-BFGS trajectories are sensitive to round-off. A re-optimization on other hardware or software versions can therefore return a slightly different local optimum, for example a noiseless fidelity of 0.99971 and a mean fidelity of 0.99729.

## Figure files and figure numbers in the paper

The figure files keep the names used in the LaTeX source.

| File name | Figure in the paper |
|---|---|
| `figure1_structure_preservation` | Fig. 1 |
| `figure2_convergence` (notebook only) | Fig. 2 |
| `figure3_pareto_frontier` | Fig. 3 |
| `figure4_histogram` | Fig. 4 |
| `figure8a_controls_blind`, `figure8b_controls_aware` | Fig. 5 |
| `figure11_baseline_comparison` | Fig. 6 |
| `figure13_adjoint` | Fig. 7 |
| `figure5_theorem_verification` | Fig. 8 |
| `figure6_noise_amplitude` | Fig. 9 |
| `figure7_correlation_length` | Fig. 10 |
| `figure9_CZ_histogram` | Fig. 11 |
| `figure10a_anisotropy_infidelity` | Fig. 12 |
| `figure10b_anisotropy_noise` | Fig. 13 |
| `figure12_dephasing_baseline` (`figure11_baseline_comparison` in `out_dephasing/`) | Fig. 14 |

## Main settings

- Single qubit: T = 5, N = 200 steps, ω₀ = 1, target R_x(π/2), control bound 2. Matérn noise with ν = 3/2 and ℓ = 0.4, truncated at 3σ. Control-coupled noise σ_x = 0.10 and σ_y = 0.20; dephasing σ = 0.05. Projected Adam with 450 iterations (300 for the ensemble). Seeds: training 12345, validation 777.
- Two qubits: T = 10, N = 150 steps, 5 Fourier harmonics per control line (40 coefficients), λ = 1e-4, l_y = 4, 5 training realizations, L-BFGS-B with bounds ±8. Seeds: training 4242, validation 777, initial guesses 999 + 7a.

## Citation

If you use this code, please cite the paper above. The full reference will be added after publication.
