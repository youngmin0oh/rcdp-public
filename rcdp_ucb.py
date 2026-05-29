"""
RCDP-UCB: Robust Linear Dueling Bandits with Post-serving Context
under Unknown Delays and Adversarial Corruptions (ICML 2026).

Reference implementation of the proposed algorithm (RCDP-UCB), the baselines,
the synthetic contextual dueling bandit environment, the delay / corruption
models, and the simulation driver used to reproduce Figures fig0 and fig1.

Algorithm map (paper -> class):
    RCDP-UCB (Ours) : DuelingGLMLearner
    RCDB            : RCDBLearner
    ColSTIM         : ColSTIMLearner
    MaxInP          : MaxInPLearner
    MaxPairUCB      : MaxPairUCBLearner
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Environment
# =============================================================================
class ContextualDuelingBanditEnv:
    """Synthetic contextual dueling bandit with a pre -> post-serving mapping.

    Pre-serving contexts x are drawn uniformly from [-pi, pi]^d. Post-serving
    contexts y = phi(x) are revealed only for the played pair. Utility is linear
    in the joint feature z = (x, y): u = <theta_*, x> + <zeta_*, y>. Preference
    outcomes follow the Bradley-Terry-Luce (logistic) link.
    """

    def __init__(self, d=1, e=2, k=10, delay_model=None, corruption_model=None,
                 phi_type='abs', rng=None, context_rng=None, feedback_rng=None,
                 param_rng=None):
        self.d = d
        self.e = e
        self.k = k
        self.delay_model = delay_model if delay_model is not None else NoDelay()
        self.corruption_model = corruption_model if corruption_model is not None else NoCorruption()
        self.phi_type = phi_type
        self.rng = rng if rng is not None else np.random
        self.context_rng = context_rng if context_rng is not None else self.rng
        self.feedback_rng = feedback_rng if feedback_rng is not None else self.rng
        self.param_rng = param_rng if param_rng is not None else self.rng

        # True parameters: small dependence on x, large dependence on y.
        self.theta_star = self.param_rng.randn(d) * 0.1
        self.zeta_star = self.param_rng.randn(e) * 5.0

    def get_context(self):
        return self.context_rng.uniform(-np.pi, np.pi, size=(self.k, self.d))

    def get_true_post_context(self, X):
        """Generate the post-serving contexts y = phi_*(x), shape (K, e)."""
        if self.phi_type == 'polynomial':          # [x^2, sqrt(|x|)]
            raw_y = np.concatenate([X ** 2, np.sqrt(np.abs(X))], axis=1)
        elif self.phi_type == 'sinusoidal':        # [cos(x), sin(x)]
            raw_y = np.concatenate([np.cos(X), np.sin(X)], axis=1)
        elif self.phi_type == 'abs':               # |x|
            raw_y = np.abs(X)
        else:
            raise ValueError(f"Unknown phi_type: {self.phi_type}")

        # Project / tile to exactly e columns.
        if self.e == 0:
            return raw_y[:, :0]
        if raw_y.shape[1] >= self.e:
            return raw_y[:, :self.e]
        repeats = int(np.ceil(self.e / raw_y.shape[1]))
        return np.tile(raw_y, (1, repeats))[:, :self.e]

    def get_utility(self, X, Y):
        return X @ self.theta_star + Y @ self.zeta_star

    def get_feedback(self, u_a, u_b):
        """Return (outcome, delay) under the prioritized interference protocol.

        The adversary favors strategic outcome corruption over delay: subject to
        the corruption budget C, a corrupted sample yields IMMEDIATE falsified
        feedback (delay = 0), whereas delays affect ONLY uncorrupted outcomes.
        The two attacks therefore never co-occur on the same sample.
        """
        prob = sigmoid(u_a - u_b)
        true_outcome = 1 if self.feedback_rng.rand() < prob else 0

        final_outcome = self.corruption_model.corrupt(true_outcome, u_a, u_b)
        is_corrupted = (final_outcome != true_outcome)

        delay = 0 if is_corrupted else self.delay_model.get_delay(u_a, u_b)
        return final_outcome, delay


# =============================================================================
# Delay models
# =============================================================================
class NoDelay:
    def get_delay(self, u_a=None, u_b=None):
        return 0


class StochasticGaussianDelay:
    """Stochastic delay tau ~ N(mean, std^2), clipped to a non-negative integer."""

    def __init__(self, mean_delay, std_delay, rng=None):
        self.mean_delay = mean_delay
        self.std_delay = std_delay
        self.rng = rng if rng is not None else np.random

    def get_delay(self, u_a=None, u_b=None):
        delay = self.rng.normal(self.mean_delay, self.std_delay)
        return int(max(0, np.round(delay)))


class StrategicDelay:
    """Adversarial starvation attack: delay feedback for 'good' pairs (high
    utility) to starve the learner of positive signal, subject to a total
    delay budget."""

    def __init__(self, budget, magnitude=100, threshold_val=0.0):
        self.budget = budget
        self.magnitude = magnitude
        self.threshold = threshold_val
        self.total_delayed = 0

    def get_delay(self, u_a, u_b):
        if self.total_delayed >= self.budget:
            return 0
        if max(u_a, u_b) > self.threshold:
            actual = min(self.magnitude, self.budget - self.total_delayed)
            if actual > 0:
                self.total_delayed += actual
                return actual
        return 0


# =============================================================================
# Corruption models
# =============================================================================
class NoCorruption:
    def corrupt(self, outcome, u_a=None, u_b=None):
        return outcome


class StrategicOutcomeCorruption:
    """Best-arm attack: flip the outcome when it correctly favors the better
    arm, hiding the superiority of the truly preferred arm (within budget)."""

    def __init__(self, budget):
        self.budget = budget
        self.count = 0

    def corrupt(self, outcome, u_a, u_b):
        if self.count >= self.budget:
            return outcome
        if (u_a > u_b and outcome == 1) or (u_b > u_a and outcome == 0):
            self.count += 1
            return 1 - outcome
        return outcome


# =============================================================================
# Post-serving mapping approximator
# =============================================================================
class NeuralApproximator(nn.Module):
    """MLP estimator of phi_*: R^d -> R^e (two hidden layers, 64 units, ReLU)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# RCDP-UCB (Ours)
# =============================================================================
class DuelingGLMLearner:
    """RCDP-UCB.

    Arm selection uses predicted features z_hat = (x, phi_hat(x)) and the full
    information matrix V (paper Eqs. for a_t, b_t). After the selected pair is
    served, the TRUE post-serving contexts of that pair are observed; the robust
    weight omega_t and V are then updated from the observed complete-feature
    difference (Algorithm 1, Step 3). The preference parameter Theta is the
    weighted regularized MLE over arrived outcomes (estimating equation), and
    only that estimation is gated by feedback delay.
    """

    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=1.0, Lambda=1.0,
                 mu_tau=0.0, kappa=0.1):
        self.d = d
        self.e = e
        self.dim = d + e
        self.lambda_reg = lambda_reg
        self.ucb_alpha = alpha          # exploration coefficient c_t (tuned constant)

        # Robustness parameters.
        self.C = C
        self.Lambda = Lambda
        self.mu_tau = mu_tau
        self.kappa = kappa

        # Robust weighting parameter alpha = sqrt(d) / (C + D),  d = d_x + d_y,
        # D = max(sqrt(Lambda), mu_tau).
        D = max(np.sqrt(Lambda), mu_tau) if mu_tau > 0 else np.sqrt(Lambda)
        self.weight_alpha = np.sqrt(self.dim) / (C + D + 1e-6)

        # V: full-history geometry (selection + weighting).
        self.V = lambda_reg * np.eye(self.dim)
        self.V_inv = (1.0 / lambda_reg) * np.eye(self.dim)
        # W: observed (arrived) history.
        self.W = lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / lambda_reg) * np.eye(self.dim)

        self.theta_hat = np.zeros(self.dim)

        # Post-serving mapping approximator phi_hat.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.f_t = NeuralApproximator(d, e).to(self.device) if e > 0 else None
        self.optimizer_f = optim.Adam(self.f_t.parameters(), lr=1e-3) if e > 0 else None
        self.loss_fn_f = nn.MSELoss() if e > 0 else None
        self.data_buffer_x = []
        self.data_buffer_y = []

        # Arrived-outcome history (observed complete-feature delta, outcome, weight)
        # for the weighted-MLE estimator.
        self.hist_dz = []
        self.hist_o = []
        self.hist_w = []

        # RCDP-UCB updates omega_t and V from observed post-serving features.
        self.use_observed_feature_weighting = True

    # --- weighted-MLE estimator --------------------------------------------
    def _solve_mle(self, max_iter=50, tol=1e-7):
        """Theta = argmin (lambda/2)||Theta||^2
                    - sum_{s in H_t} omega_s log g((-1)^{1-o_s} <Theta, dz_s>),
        solved by warm-started Newton (IRLS) over the full arrived history."""
        if not self.hist_dz:
            return
        Z = np.asarray(self.hist_dz)
        o = np.asarray(self.hist_o)
        w = np.asarray(self.hist_w)
        lam = self.lambda_reg
        theta = self.theta_hat.copy()
        eye = np.eye(self.dim)
        for _ in range(max_iter):
            p = sigmoid(Z @ theta)
            grad = lam * theta + Z.T @ (w * (p - o))
            if np.linalg.norm(grad) < tol:
                break
            s = w * p * (1.0 - p)
            H = lam * eye + (Z * s[:, None]).T @ Z
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta = theta - step
        self.theta_hat = theta

    # --- post-serving mapping ----------------------------------------------
    def train_approximator(self, epochs=2):
        if self.e == 0 or not self.data_buffer_x:
            return
        X_tensor = torch.FloatTensor(np.array(self.data_buffer_x)).to(self.device)
        Y_tensor = torch.FloatTensor(np.array(self.data_buffer_y)).to(self.device)
        self.f_t.train()
        for _ in range(epochs):
            self.optimizer_f.zero_grad()
            loss = self.loss_fn_f(self.f_t(X_tensor), Y_tensor)
            loss.backward()
            self.optimizer_f.step()

    def predict_y(self, X):
        if self.e == 0:
            return np.zeros((len(X), 0))
        self.f_t.eval()
        with torch.no_grad():
            return self.f_t(torch.FloatTensor(X).to(self.device)).cpu().numpy()

    def get_features(self, X):
        return np.concatenate([X, self.predict_y(X)], axis=1)

    def _update_observed_covariance(self, delta_z, weight=1.0):
        self.W += weight * np.outer(delta_z, delta_z)
        Aw = self.W_inv @ delta_z
        self.W_inv -= (np.outer(Aw, Aw) * weight) / (1 + np.dot(delta_z, Aw) * weight)

    def _record_mle_feedback(self, delta_z, outcome, weight=1.0, covariance_weight=None):
        cov_weight = weight if covariance_weight is None else covariance_weight
        self._update_observed_covariance(delta_z, cov_weight)
        self.hist_dz.append(delta_z)
        self.hist_o.append(outcome)
        self.hist_w.append(weight)
        self._solve_mle()

    def _max_pair_ucb(self, Z, beta):
        best_i, best_j = 0, 1
        best_score = -np.inf
        for i in range(len(Z)):
            for j in range(len(Z)):
                if i == j:
                    continue
                delta_z = Z[i] - Z[j]
                pair_mean = (Z[i] + Z[j]) @ self.theta_hat
                pair_width = np.sqrt(delta_z @ self.W_inv @ delta_z)
                score = pair_mean + beta * pair_width
                if score > best_score:
                    best_i, best_j, best_score = i, j, score
        return best_i, best_j

    # --- selection / weighting / estimation --------------------------------
    def get_observed_delta(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """Complete-feature difference using the TRUE observed post-serving y."""
        if self.e == 0:
            return X[a_idx] - X[b_idx]
        z_a = np.concatenate([X[a_idx], y_a_obs])
        z_b = np.concatenate([X[b_idx], y_b_obs])
        return z_a - z_b

    def record_observed_features(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """Algorithm 1, Step 3: after observing the selected post-serving
        contexts, compute omega_t = min(1, alpha / ||dz_t||_{V_{t-1}^-1}) and
        update V_t = V_{t-1} + kappa * omega_t * dz_t dz_t^T."""
        delta_z = self.get_observed_delta(X, a_idx, b_idx, y_a_obs, y_b_obs)
        norm_V = np.sqrt(delta_z @ self.V_inv @ delta_z)
        omega_t = 1.0 if norm_V < 1e-9 else min(1.0, self.weight_alpha / norm_V)

        self.V += self.kappa * omega_t * np.outer(delta_z, delta_z)
        v_vec = np.sqrt(self.kappa * omega_t) * delta_z
        Av = self.V_inv @ v_vec
        self.V_inv -= np.outer(Av, Av) / (1 + np.dot(v_vec, Av))

        self.last_delta_z = delta_z
        self.last_omega_t = omega_t
        return delta_z, omega_t

    def select_arms(self, X, env=None):
        """a_t = argmax <Theta, z_hat>;  b_t = argmax <Theta, z_hat> + c_t * width_V.
        omega_t / V are updated later from observed contexts (returns omega=None)."""
        Z = self.get_features(X)
        utilities = Z @ self.theta_hat
        a_t = np.argmax(utilities)

        Delta_Z_a = Z - Z[a_t]
        mean_diff = Delta_Z_a @ self.theta_hat
        width_V = np.sqrt(np.sum((Delta_Z_a @ self.V_inv) * Delta_Z_a, axis=1))
        scores = mean_diff + self.ucb_alpha * width_V
        scores[a_t] = -np.inf
        b_t = np.argmax(scores)
        return a_t, b_t, None

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """Post-serving contexts are revealed immediately (only the preference
        outcome is delayed/corrupted): train phi_hat right after serving."""
        if self.e == 0:
            return
        self.data_buffer_x.append(X[a_idx]); self.data_buffer_y.append(y_a_obs)
        self.data_buffer_x.append(X[b_idx]); self.data_buffer_y.append(y_b_obs)
        self.train_approximator(epochs=2)

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t):
        """Process an arrived outcome: update the observed matrix W and re-solve
        the weighted MLE over all arrived outcomes."""
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            z = self.get_features(X)
            delta_z = z[a_idx] - z[b_idx]

        self._record_mle_feedback(delta_z, outcome, weight=omega_t,
                                  covariance_weight=self.kappa * omega_t)


# =============================================================================
# Baselines (pre-serving context only)
# =============================================================================
class BaselineDuelingGLMLearner:
    """X-only base learner shared by the dueling-bandit baselines."""

    def __init__(self, d, lambda_reg=1.0, alpha=0.1):
        self.d = d
        self.dim = d
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.theta_hat = np.zeros(self.dim)
        self.W = lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / lambda_reg) * np.eye(self.dim)
        self.hist_dz = []
        self.hist_o = []
        self.hist_w = []

    def _solve_mle(self, max_iter=50, tol=1e-7):
        if not self.hist_dz:
            return
        Z = np.asarray(self.hist_dz)
        o = np.asarray(self.hist_o)
        w = np.asarray(self.hist_w)
        lam = self.lambda_reg
        theta = self.theta_hat.copy()
        eye = np.eye(self.dim)
        for _ in range(max_iter):
            p = sigmoid(Z @ theta)
            grad = lam * theta + Z.T @ (w * (p - o))
            if np.linalg.norm(grad) < tol:
                break
            s = w * p * (1.0 - p)
            H = lam * eye + (Z * s[:, None]).T @ Z
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta = theta - step
        self.theta_hat = theta

    def _update_covariance(self, delta_z, weight=1.0):
        self.W += weight * np.outer(delta_z, delta_z)
        Aw = self.W_inv @ delta_z
        self.W_inv -= (np.outer(Aw, Aw) * weight) / (1 + np.dot(delta_z, Aw) * weight)

    def _record_feedback(self, delta_z, outcome, weight=1.0, covariance_weight=None):
        cov_weight = weight if covariance_weight is None else covariance_weight
        self._update_covariance(delta_z, cov_weight)
        self.hist_dz.append(delta_z)
        self.hist_o.append(outcome)
        self.hist_w.append(weight)
        self._solve_mle()

    def _max_pair_ucb(self, Z, beta):
        best_i, best_j = 0, 1
        best_score = -np.inf
        for i in range(len(Z)):
            for j in range(len(Z)):
                if i == j:
                    continue
                delta_z = Z[i] - Z[j]
                pair_mean = (Z[i] + Z[j]) @ self.theta_hat
                pair_width = np.sqrt(delta_z @ self.W_inv @ delta_z)
                score = pair_mean + beta * pair_width
                if score > best_score:
                    best_i, best_j, best_score = i, j, score
        return best_i, best_j

    def select_arms(self, X, env=None):
        utilities = X @ self.theta_hat
        a_t = np.argmax(utilities)
        Delta_Z = X - X[a_t]
        mean_diff = Delta_Z @ self.theta_hat
        width = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
        scores = mean_diff + self.alpha * width
        scores[a_t] = -np.inf
        b_t = np.argmax(scores)
        self.last_delta_z = X[a_t] - X[b_t]
        return a_t, b_t, 1.0

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        return  # X-only: no post-serving mapping to learn.

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = X[a_idx] - X[b_idx]
        self._record_feedback(delta_z, outcome, weight=omega_t)


class RCDBLearner(BaselineDuelingGLMLearner):
    """RCDB (Di et al., 2024): uncertainty-weighted MLE with beta-scaled UCB.
    Weight w_t = min(1, alpha / ||dz||_{W^-1})."""

    def __init__(self, d, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, lambda_reg, alpha)
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta
        self.kappa = kappa

    def select_arms(self, X, env=None):
        i_t, j_t = self._max_pair_ucb(X, self.rcdb_beta)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = X[a_idx] - X[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)
        w_t = 1.0 if norm_val < 1e-9 else min(1.0, self.rcdb_alpha / norm_val)
        self._record_feedback(delta_z, outcome, weight=w_t,
                              covariance_weight=w_t * self.kappa)


class MaxInPLearner(BaselineDuelingGLMLearner):
    """MaxInP (Saha & Krishnamurthy, 2021): the champion is the greedy arm and
    the challenger maximizes relative uncertainty among optimistic candidates."""

    def select_arms(self, X, env=None):
        utilities = X @ self.theta_hat
        i_t = np.argmax(utilities)
        self_widths = np.sqrt(np.sum((X @ self.W_inv) * X, axis=1))
        ucb_scores = utilities + self.alpha * self_widths
        candidates = np.where(ucb_scores >= utilities[i_t])[0]
        candidates = candidates[candidates != i_t]
        if len(candidates) == 0:
            ucb_scores[i_t] = -np.inf
            j_t = np.argmax(ucb_scores)
        else:
            Delta_Z = X[candidates] - X[i_t]
            unc = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
            j_t = candidates[np.argmax(unc)]
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


class MaxPairUCBLearner(BaselineDuelingGLMLearner):
    """MaxPairUCB: maximize <x_i - x_j, theta> + alpha * ||x_i - x_j||_{W^-1}."""

    def select_arms(self, X, env=None):
        i_t, j_t = self._max_pair_ucb(X, self.alpha)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


class ColSTIMLearner(BaselineDuelingGLMLearner):
    """ColSTIM (Bengs et al.): perturbed-greedy (Thompson-style) selection."""

    def select_arms(self, X, env=None):
        utilities = X @ self.theta_hat
        norms = np.sqrt(np.sum((X @ self.W_inv) * X, axis=1))
        ts_scores = utilities + self.alpha * norms * np.random.randn(len(X))
        i_t = np.argmax(ts_scores)
        Delta_Z = X - X[i_t]
        mean_diff = Delta_Z @ self.theta_hat
        width = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
        scores = mean_diff + self.alpha * width
        scores[i_t] = -np.inf
        j_t = np.argmax(scores)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


# =============================================================================
# Baselines augmented with post-serving context (+PS)
#   Same baselines, but acting on predicted features z_hat = (x, phi_hat(x)).
#   Used for the multi-mapping post-serving figure (fig2), where all algorithms
#   learn the pre -> post mapping. They keep their own selection-time weighting
#   (use_observed_feature_weighting = False).
# =============================================================================
class ColSTIMPostServingLearner(DuelingGLMLearner):
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=C, Lambda=Lambda, mu_tau=mu_tau, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.use_true_post_features_for_update = True

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        utilities = Z @ self.theta_hat
        norms = np.sqrt(np.sum((Z @ self.W_inv) * Z, axis=1))
        ts_scores = utilities + self.ucb_alpha * norms * np.random.randn(len(Z))
        i_t = np.argmax(ts_scores)
        Delta_Z = Z - Z[i_t]
        mean_diff = Delta_Z @ self.theta_hat
        width = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
        scores = mean_diff + self.ucb_alpha * width
        scores[i_t] = -np.inf
        j_t = np.argmax(scores)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X); delta_z = Z[a_idx] - Z[b_idx]
        self._record_mle_feedback(delta_z, outcome, weight=omega_t)


class MaxPairUCBPostServingLearner(DuelingGLMLearner):
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.use_true_post_features_for_update = True
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = self._max_pair_ucb(Z, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X); delta_z = Z[a_idx] - Z[b_idx]
        self._record_mle_feedback(delta_z, outcome)


class MaxInPPostServingLearner(DuelingGLMLearner):
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.use_true_post_features_for_update = True
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        utilities = Z @ self.theta_hat
        i_t = np.argmax(utilities)
        self_widths = np.sqrt(np.sum((Z @ self.W_inv) * Z, axis=1))
        ucb_scores = utilities + self.rcdb_beta * self_widths
        candidates = np.where(ucb_scores >= utilities[i_t])[0]
        candidates = candidates[candidates != i_t]
        if len(candidates) == 0:
            ucb_scores[i_t] = -np.inf
            j_t = np.argmax(ucb_scores)
        else:
            Delta_Z = Z[candidates] - Z[i_t]
            unc = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
            j_t = candidates[np.argmax(unc)]
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X); delta_z = Z[a_idx] - Z[b_idx]
        self._record_mle_feedback(delta_z, outcome)


class RCDBPostServingLearner(DuelingGLMLearner):
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.use_true_post_features_for_update = True
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = self._max_pair_ucb(Z, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X); delta_z = Z[a_idx] - Z[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)
        # RCDB weighting is capped at 1: w_t = min(1, alpha / ||dz||_{W^-1}).
        w_t = 1.0 if norm_val < 1e-9 else min(1.0, self.rcdb_alpha / norm_val)
        self._record_mle_feedback(delta_z, outcome, weight=w_t,
                                  covariance_weight=w_t * self.kappa)


# =============================================================================
# Simulation driver
# =============================================================================
def run_simulation(env, learner, T, name=None):
    """Run one learner for T rounds; return the cumulative-regret trajectory."""
    cumulative_regret = 0.0
    regrets = []
    feedback_queue = []

    for t in range(T):
        X = env.get_context()
        a_idx, b_idx, omega_t = learner.select_arms(X, env=env)

        Y = env.get_true_post_context(X)
        y_a, y_b = Y[a_idx], Y[b_idx]

        # Post-serving contexts observed immediately (train phi_hat now).
        learner.observe_context(X, a_idx, b_idx, y_a, y_b)

        if getattr(learner, 'use_observed_feature_weighting', False):
            # RCDP-UCB: weight + V from the observed complete features.
            delta_z, omega_t = learner.record_observed_features(X, a_idx, b_idx, y_a, y_b)
        elif getattr(learner, 'use_true_post_features_for_update', False):
            # +PS baselines select with z_hat, but update on the observed complete
            # feature of the served pair once post-serving contexts are revealed.
            delta_z = learner.get_observed_delta(X, a_idx, b_idx, y_a, y_b)
            if omega_t is None:
                omega_t = 1.0
        else:
            delta_z = getattr(learner, 'last_delta_z', None)
            if omega_t is None:
                omega_t = 1.0

        u_all = np.asarray(env.get_utility(X, Y)).flatten()
        u_a, u_b = u_all[a_idx], u_all[b_idx]

        outcome, delay = env.get_feedback(u_a, u_b)
        feedback_queue.append({
            'arrival_time': t + delay, 'X': X, 'a_idx': a_idx, 'b_idx': b_idx,
            'outcome': outcome, 'y_a': y_a, 'y_b': y_b,
            'omega_t': omega_t, 'delta_z': delta_z,
        })

        # Process all feedback whose (possibly delayed) arrival time has passed.
        remaining = []
        for item in feedback_queue:
            if item['arrival_time'] <= t:
                learner.current_delta_z = item['delta_z']
                learner.update(item['X'], item['a_idx'], item['b_idx'],
                               item['outcome'], item['y_a'], item['y_b'], item['omega_t'])
                learner.current_delta_z = None
            else:
                remaining.append(item)
        feedback_queue = remaining

        # Dueling instantaneous regret r_t = 1/2[(u* - u_a) + (u* - u_b)].
        u_star = u_all[np.argmax(u_all)]
        cumulative_regret += 0.5 * ((u_star - u_a) + (u_star - u_b))
        regrets.append(cumulative_regret)

        if name and (t + 1) % 500 == 0:
            print(f"[{name}] round {t + 1}/{T}, regret {cumulative_regret:.2f}")

    return regrets
