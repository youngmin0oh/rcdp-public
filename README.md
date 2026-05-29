# RCDP-UCB: Robust Linear Dueling Bandits with Post-serving Context

[![arXiv](https://img.shields.io/badge/arXiv-2605.01752-b31b1b.svg)](https://arxiv.org/abs/2605.01752)
[![ICML 2026](https://img.shields.io/badge/ICML-2026%20Poster-4b8bbe.svg)](https://icml.cc/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official reference implementation for the ICML 2026 paper

> **Robust Linear Dueling Bandits with Post-serving Context under Unknown Delays and Adversarial Corruptions**
>
> 📄 **Paper (arXiv):** https://arxiv.org/abs/2605.01752
> 🏆 **Accepted at ICML 2026 (Poster)**

RCDP-UCB is a robust linear dueling-bandit algorithm for settings where
post-serving contexts are observed after serving, preference feedback may be
delayed, and outcomes may be adversarially corrupted.

This repository reproduces the synthetic robustness figures of the paper
(**fig0**, **fig1**, **fig2**). It provides the proposed algorithm **RCDP-UCB**,
the baselines, the contextual dueling bandit environment, the delay / corruption
models, and the simulation driver.

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `torch`, `matplotlib`. A GPU is **not** required.

## 2. Files

| File | Description |
|------|-------------|
| `rcdp_ucb.py` | Core library: environment, delay/corruption models, RCDP-UCB, baselines (X-only and +PS), and the `run_simulation` driver. |
| `run_experiment.py` | Reproduces fig0 / fig1 / fig2 (plots + console regret summary). |

## 3. Reproducing the figures

Quick smoke run:

```bash
python run_experiment.py --figure fig0 --n_runs 1 --delay stochastic
```

Full synthetic reproductions:

```bash
# fig0: no post-serving context (isolates the adaptive-weighting mechanism)
python run_experiment.py --figure fig0 --n_runs 10

# fig1: with post-serving context; only RCDP-UCB exploits the learned mapping,
#       baselines use pre-serving contexts only (mapping: absolute)
python run_experiment.py --figure fig1 --n_runs 10

# fig2: with post-serving context; ALL algorithms learn the mapping (+PS),
#       across three mappings (sinusoidal, polynomial, absolute)
python run_experiment.py --figure fig2 --n_runs 10
```

Each command sweeps both delay regimes (stochastic and strategic), printing the
final cumulative regret per method. fig0/fig1 write `figures/<figure>_<delay>.pdf`;
fig2 (multi-mapping) writes `figures/fig2_<mapping>_<delay>.pdf`. Use
`--delay stochastic` (or `strategic`) to run a single regime, and `--n_runs` to
set the number of averaged seeds. Use `--output_dir <dir>` to write plots to a
different directory.

## 4. Algorithm and setup

`rcdp_ucb.py` maps the paper's names to classes:

| Paper name | Class |
|------------|-------|
| **RCDP-UCB (Ours)** | `DuelingGLMLearner` |
| RCDB | `RCDBLearner` |
| ColSTIM | `ColSTIMLearner` |
| MaxInP | `MaxInPLearner` |
| MaxPairUCB | `MaxPairUCBLearner` |

For fig2, the baselines also learn the post-serving mapping; their post-serving
variants are `RCDBPostServingLearner`, `ColSTIMPostServingLearner`,
`MaxInPPostServingLearner`, and `MaxPairUCBPostServingLearner`.

RCDP-UCB selects arms from predicted features `z_hat = (x, phi_hat(x))` using the
full-information matrix `V`; after the played pair's true post-serving contexts
are observed, it updates the a-priori robust weight `omega_t` and `V` from the
observed complete-feature difference, and re-solves the weighted regularized MLE
over arrived outcomes for the preference parameter `Theta`. Only the preference
estimation is gated by feedback delay.

**Fixed configuration** (all figures): `d_x = 10`, `K = 10`, `T = 2000`,
worst-case error budget `C + D = 125` (corruption `C = 25`, delay scale
`D = max(sqrt(Lambda), mu_tau) = 100`), `lambda = 1.0`, `kappa = 0.25`. The
post-serving mapping is approximated by a 2x64 ReLU MLP (Adam, lr `1e-3`,
2 epochs/round). Two delay regimes are evaluated: stochastic
`tau ~ N(100, 100^2)` and strategic (adversarial starvation, budget `1e4`).
Under the prioritized interference protocol, corrupted outcomes are delivered
immediately and delays affect only uncorrupted outcomes.

## 5. Notes

- Runs are seeded per repetition; curves report the mean over `--n_runs` seeds
  with `±1 std` error bars. Increasing `--n_runs` tightens the bands.
- Arm pairs are constrained to be distinct. The environment uses separate random
  streams for contexts, preference outcomes, and stochastic delays, so methods
  see the same context stream even when their feedback/delay paths differ.
- The comparison baselines use the same regularized logistic MLE convention as
  RCDP-UCB. In fig2, +PS baselines select arms using predicted post-serving
  features and update preference estimates using the observed complete features
  of the served pair.
- The exploration coefficient `c_t` is a tuned constant (the paper's
  theoretically-motivated `c_t = 2 beta_t` is deferred to the analysis).

## 6. Citation

Paper: [arXiv:2605.01752](https://arxiv.org/abs/2605.01752) — accepted at ICML 2026 (Poster).

```bibtex
@inproceedings{oh2026rcdpucb,
  title     = {Robust Linear Dueling Bandits with Post-serving Context under
               Unknown Delays and Adversarial Corruptions},
  author    = {Oh, Youngmin},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {To appear},
  eprint    = {2605.01752},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2605.01752}
}
```

## 7. License

Released under the MIT License (see `LICENSE`).
