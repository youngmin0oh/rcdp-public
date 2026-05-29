"""
Reproduce the synthetic robustness figures of the paper.

  fig0 : no post-serving context (e = 0; standard linear dueling). Isolates the
         effect of the adaptive weighting mechanism. Mapping: polynomial
         (irrelevant since e = 0).                      -> figures/fig0_*.pdf
  fig1 : with post-serving context (e > 0). Only RCDP-UCB learns and uses the
         pre -> post mapping phi_*; baselines use pre-serving contexts only.
         Mapping: absolute.                             -> figures/fig1_*.pdf
  fig2 : with post-serving context (e > 0). ALL algorithms (incl. baselines)
         learn phi_*. Mappings: sinusoidal, polynomial, absolute.
                                                        -> figures/fig2_<phi>_*.pdf

All figures fix the worst-case error budget C + D = 125 (C = 25, D = 100) and
sweep the two delay regimes (stochastic / strategic).

Usage:
    python run_experiment.py --figure fig0 --n_runs 10
    python run_experiment.py --figure fig1 --n_runs 10 --delay stochastic
    python run_experiment.py --figure fig2 --n_runs 10
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from rcdp_ucb import (
    ContextualDuelingBanditEnv,
    StochasticGaussianDelay, StrategicDelay, StrategicOutcomeCorruption,
    DuelingGLMLearner,
    RCDBLearner, ColSTIMLearner, MaxInPLearner, MaxPairUCBLearner,
    RCDBPostServingLearner, ColSTIMPostServingLearner,
    MaxInPPostServingLearner, MaxPairUCBPostServingLearner,
    run_simulation,
)

# ---- Fixed configuration (paper) -------------------------------------------
D_X = 10            # pre-serving dimension
K = 10              # number of arms
T = 2000            # horizon
C = 25              # corruption budget
LAMBDA = 10000.0    # adversarial-delay budget
MU_TAU = 100.0      # mean stochastic delay
STD_TAU = 100.0     # std stochastic delay
DELAY_MAG = 10000   # strategic-delay budget
KAPPA = 0.25
LAMBDA_REG = 1.0

# figure -> (mappings, post-serving dim e, baseline kind)
FIGURES = {
    "fig0": (["polynomial"], 0, "xonly"),
    "fig1": (["abs"], 10, "xonly"),
    "fig2": (["sinusoidal", "polynomial", "abs"], 10, "ps"),
}

# Seed convention (matches the paper's per-mapping offsets).
PHI_ORDER = ["polynomial", "abs", "cosine", "sinusoidal"]

COLORS = {"Ours": "#1f77b4", "RCDB": "#2ca02c", "ColSTIM": "#ff7f0e",
          "MaxInP": "#9467bd", "MaxPair": "#d62728"}


def make_delay_model(delay_type, rng=None):
    if delay_type == "strategic":
        return StrategicDelay(budget=DELAY_MAG, magnitude=int(np.sqrt(DELAY_MAG)))
    if delay_type == "stochastic":
        return StochasticGaussianDelay(mean_delay=MU_TAU, std_delay=STD_TAU, rng=rng)
    raise ValueError(delay_type)


def build_learners(d, e, kind, torch_seed=None):
    """RCDP-UCB plus the four baselines (X-only or +PS), paper calibration."""
    def make(factory):
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        return factory()

    ours = make(lambda: DuelingGLMLearner(d=d, e=e, lambda_reg=LAMBDA_REG, alpha=0.1,
                                          C=C, Lambda=LAMBDA, mu_tau=MU_TAU, kappa=KAPPA))
    if kind == "xonly":
        beta = np.sqrt(d)                                 # baselines act on X (dim d)
        rcdb_alpha = np.sqrt(d) / (np.sqrt(KAPPA) * C)
        rcdb_beta = beta + rcdb_alpha * C
        return {
            "RCDP-UCB (Ours)": ours,
            "RCDB": make(lambda: RCDBLearner(d, LAMBDA_REG, 0.5, rcdb_alpha, rcdb_beta, KAPPA)),
            "ColSTIM": make(lambda: ColSTIMLearner(d, LAMBDA_REG, 0.5)),
            "MaxInP": make(lambda: MaxInPLearner(d, LAMBDA_REG, 0.5)),
            "MaxPairUCB": make(lambda: MaxPairUCBLearner(d, LAMBDA_REG, beta)),
        }
    # kind == "ps": baselines act on z_hat = (x, phi_hat(x)) (dim d + e).
    dim = d + e
    beta = np.sqrt(dim)
    rcdb_alpha = np.sqrt(dim) / (np.sqrt(KAPPA) * C)
    rcdb_beta = beta + rcdb_alpha * C
    return {
        "RCDP-UCB (Ours)": ours,
        "RCDB+PS": make(lambda: RCDBPostServingLearner(d, e, LAMBDA_REG, 0.5, rcdb_alpha, rcdb_beta, KAPPA)),
        "ColSTIM+PS": make(lambda: ColSTIMPostServingLearner(d, e, LAMBDA_REG, 0.5, C, LAMBDA, MU_TAU, KAPPA)),
        "MaxInP+PS": make(lambda: MaxInPPostServingLearner(d, e, LAMBDA_REG, 0.1, rcdb_alpha, rcdb_beta, KAPPA)),
        "MaxPairUCB+PS": make(lambda: MaxPairUCBPostServingLearner(d, e, LAMBDA_REG, 0.1, rcdb_alpha, rcdb_beta, KAPPA)),
    }


def color_for(name):
    for key, c in COLORS.items():
        if key in name:
            return c
    return "black"


def run_setting(phi, e, kind, delay_type, n_runs, verbose):
    base_seed = 42 + PHI_ORDER.index(phi) * 1000 + 999
    methods = list(build_learners(D_X, e, kind).keys())
    regrets = {m: np.zeros((n_runs, T)) for m in methods}

    for run in range(n_runs):
        run_seed = base_seed + run * 100
        np.random.seed(run_seed)
        run_theta = np.random.randn(D_X) * 0.1
        run_zeta = np.random.randn(e) * 5.0

        def fresh_env():
            context_rng = np.random.RandomState(run_seed)
            feedback_rng = np.random.RandomState(run_seed + 1)
            delay_rng = np.random.RandomState(run_seed + 2)
            param_rng = np.random.RandomState(run_seed + 3)
            env = ContextualDuelingBanditEnv(
                d=D_X, e=e, k=K, phi_type=phi,
                delay_model=make_delay_model(delay_type, rng=delay_rng),
                corruption_model=StrategicOutcomeCorruption(budget=C),
                context_rng=context_rng,
                feedback_rng=feedback_rng,
                param_rng=param_rng)
            env.theta_star, env.zeta_star = run_theta.copy(), run_zeta.copy()
            return env

        for name, learner in build_learners(D_X, e, kind, torch_seed=run_seed).items():
            regrets[name][run] = run_simulation(fresh_env(), learner, T)
        if verbose:
            finals = ", ".join(f"{m}={regrets[m][run][-1]:.0f}" for m in methods)
            print(f"  [{phi}/{delay_type}] run {run + 1}/{n_runs}: {finals}")

    return methods, regrets


def plot(figure, phi, delay_type, methods, regrets, n_runs, output_dir):
    plt.rcParams.update({
        'font.size': 32, 'axes.labelsize': 36, 'xtick.labelsize': 32,
        'ytick.labelsize': 32, 'legend.fontsize': 32, 'lines.linewidth': 7.0,
        'figure.figsize': (14, 11),
    })
    plt.figure()
    t_range = np.arange(T)
    err_idx = np.linspace(0, T - 1, 11).astype(int)
    tag = f"{figure}/{phi}/{delay_type}"
    print(f"\n=== {tag} ({n_runs} runs) ===")
    for name in methods:
        mean = regrets[name].mean(axis=0)
        std = regrets[name].std(axis=0)
        print(f"  {name:18s} final regret = {mean[-1]:.2f}")
        c = color_for(name)
        plt.plot(t_range, mean, label=name, color=c)
        plt.errorbar(t_range[err_idx], mean[err_idx], yerr=std[err_idx],
                     fmt='none', ecolor=c, capsize=5, elinewidth=2.0)
    plt.xlabel('Rounds (t)')
    plt.ylabel('Cumulative Regret')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    # fig0/fig1 use a single mapping (omit phi in the name); fig2 includes it.
    name = f"{figure}_{delay_type}" if figure != "fig2" else f"{figure}_{phi}_{delay_type}"
    path = os.path.join(output_dir, f"{name}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figure', choices=['fig0', 'fig1', 'fig2'], required=True)
    ap.add_argument('--delay', choices=['stochastic', 'strategic', 'both'], default='both')
    ap.add_argument('--n_runs', type=int, default=10)
    ap.add_argument('--output_dir', default='figures')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    mappings, e, kind = FIGURES[args.figure]
    delays = ['stochastic', 'strategic'] if args.delay == 'both' else [args.delay]
    for phi in mappings:
        for delay_type in delays:
            methods, regrets = run_setting(phi, e, kind, delay_type, args.n_runs,
                                           verbose=not args.quiet)
            plot(args.figure, phi, delay_type, methods, regrets, args.n_runs,
                 args.output_dir)


if __name__ == "__main__":
    main()
