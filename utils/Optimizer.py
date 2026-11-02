"""Bilevel optimiser base class and projected gradient descent solver."""

import numpy as np
from tqdm import tqdm

from utils.T_map  import T_map
from utils.dT_map import dT_map
from utils.g_map  import g_map
from utils.Transformer import Transformer
from utils.Projector import Projector


class Optimizer:
    """Bilevel optimiser: forward pass, backward pass, and error metrics.

    Provides the building blocks for gradient-based optimisation of the
    incentive sequence th_traj via the implicit function theorem.
    """

    def __init__(self, params):
        # ── Sub-maps ──────────────────────────────────────────────────────────
        self.T_map  = T_map(params)
        self.dT_map = dT_map(params)
        self.g_map  = g_map.create(params)

        # ── Dimensions ────────────────────────────────────────────────────────
        self.N     = params.N
        self.N_mu  = 2 * params.N   # dual variable dimension (2 constraints × N each)
        self.T     = params.T

        # ── Side indices ──────────────────────────────────────────────────────
        self.DO = params.DO
        self.PU = params.PU

        # ── Trajectory data ───────────────────────────────────────────────────
        self.z0 = params.z0           # (N+1,) extended initial state
        self.d_traj = params.d_traj   # list [d_do (T,N), d_pu (T,N)]

        # ── Optimiser settings ────────────────────────────────────────────────
        self.th0_traj = params.th0_traj   # (T, N)
        self.maxIt    = params.maxIt
        self.err_thr    = params.err_thr
        self.alph       = params.alph
        self.alph_decay = params.alph_decay

        # ── Projector ─────────────────────────────────────────────────────────
        self.projector = Projector.create(params)

    # ── Step size ─────────────────────────────────────────────────────────────

    def get_alph(self, it):
        """Return the step size for iteration it.

        Power decay: α_k = α_0 / (k+1)^p, which guarantees Σ α_k² < ∞ for p > 0.5.

        Args:
            it: int — current iteration index (0-based).

        Returns:
            float — step size.
        """
        return self.alph / (it + 1) ** self.alph_decay

    # ── Cost ──────────────────────────────────────────────────────────────────

    def get_cost(self, z_traj, th_traj):
        """Compute the total objective J = h(z_T) + Σ_t g(z_t, θ_t, d_t).

        Args:
            z_traj:  np.ndarray of shape (T+1, N+1) — state trajectory from forward_pass.
            th_traj: np.ndarray of shape (T, N)     — incentive sequence.

        Returns:
            float — total cost.
        """
        z_T  = Transformer.z_traj_at_t(z_traj, self.T)
        cost = self.g_map.apply_h(z_T)

        for t in range(self.T):
            th_t = Transformer.th_traj_at_t(th_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)
            z_t  = Transformer.z_traj_at_t(z_traj, t)

            cost += self.g_map.apply_g(z_t, th_t, d_t)

        return cost

    def get_cost_only_Q(self, z_traj):
        """Compute the state-deviation term of the total cost, h(z_T) + Σ_t g_only_Q(z_t).

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — state trajectory from forward_pass.

        Returns:
            float — total state-deviation cost.
        """
        z_T  = Transformer.z_traj_at_t(z_traj, self.T)
        cost = self.g_map.apply_h(z_T)

        for t in range(self.T):
            z_t = Transformer.z_traj_at_t(z_traj, t)

            cost += self.g_map.apply_g_only_Q(z_t)

        return cost

    def get_cost_only_R(self, th_traj):
        """Compute the incentive-deviation term of the total cost, Σ_t g_only_R(θ_t).

        Args:
            th_traj: np.ndarray of shape (T, N) — incentive sequence.

        Returns:
            float — total incentive-deviation cost.
        """
        cost = 0.0

        for t in range(self.T):
            th_t = Transformer.th_traj_at_t(th_traj, t)

            cost += self.g_map.apply_g_only_R(th_t)

        return cost

    def get_cost_t(self, z_t, th_t, d_t):
        """Compute the stage cost g(z_t, θ_t, d_t) for a single time step.

        Args:
            z_t:  np.ndarray of shape (N+1,) — extended state at time t.
            th_t: np.ndarray of shape (N,)   — incentive vector at time t.
            d_t:  np.ndarray of shape (2, N) — demand at time t.

        Returns:
            float — stage cost value.
        """
        return self.g_map.apply_g(z_t, th_t, d_t)


class ForwardBackward(Optimizer):
    """Projected gradient descent solver for the incentive sequence.

    Adds the forward/backward passes, the projection step, and the main
    optimisation loop.
    """

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward_pass(self, th_traj):
        """Forward pass: simulate the system and collect dual variables.

        Args:
            th_traj: np.ndarray of shape (T, N) — current incentive sequence.

        Returns:
            z_traj:   np.ndarray of shape (T+1, N+1) — extended state trajectory.
                      z_traj[0] = z0;
                      z_traj[t+1] = T_map(z_traj[t], th_traj[t], d_t).
            mu_traj:  np.ndarray of shape (T, 2*N) — dual variables per time step.
                      mu_traj[t] = z_star at step t.
        """
        z_traj       = np.zeros((self.T + 1, self.N + 1))
        z_traj[0, :] = self.z0
        mu_traj      = np.zeros((self.T, self.N_mu))

        z_t   = self.z0
        t_counter = tqdm(range(self.T), desc="forward_pass", position=1, leave=False)
        for t in t_counter:
            th_t = Transformer.th_traj_at_t(th_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)

            z_next_t, _, mu_star_t, _ = self.T_map.apply(z_t, th_t, d_t)
            z_traj[t + 1, :]          = z_next_t
            mu_traj[t, :]             = mu_star_t
            z_t                       = z_next_t

        return z_traj, mu_traj

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backpropagation(self, th_traj, z_traj, mu_traj):
        """Backward pass: accumulate the adjoint to obtain the gradient of J w.r.t. θ.

        Iterates backwards from t=T-1 to t=0, propagating the co-states dV_x_t
        and dV_th_traj_t via the chain rule through the dynamics and cost.

        Args:
            th_traj: np.ndarray of shape (T, N)     — current incentive sequence.
            z_traj:  np.ndarray of shape (T+1, N+1) — state trajectory from forward_pass.
            mu_traj: np.ndarray of shape (T, 2*N)   — dual variables from forward_pass.

        Returns:
            dV_th_traj_t: np.ndarray of shape (T, N) — gradient of J w.r.t. θ_t at each time step.
        """
        z_T    = Transformer.z_traj_at_t(z_traj, self.T)
        dh_z   = self.g_map.apply_dh_z(z_T)
        dV_z_t = dh_z
        dV_th_traj_t = np.zeros((self.T, self.N))

        t_counter = tqdm(range(self.T - 1, -1, -1), desc="backpropagation", position=2, leave=False)
        for t in t_counter:
            th_t = Transformer.th_traj_at_t(th_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)
            z_t  = Transformer.z_traj_at_t(z_traj, t)
            mu_t = Transformer.mu_traj_at_t(mu_traj, t)

            dg_z_t  = self.g_map.apply_dg_z(z_t, th_t, d_t)
            dg_th_t = self.g_map.apply_dg_th(z_t, th_t, d_t)

            dT_z_t  = self.dT_map.apply_z(z_t, th_t, d_t, mu_t)
            dT_th_t = self.dT_map.apply_th(z_t, th_t, d_t, mu_t)

            dV_z_next_t  = dg_z_t + dV_z_t @ dT_z_t
            dV_th_next_t = dg_th_t + dV_z_t @ dT_th_t

            dV_z_t             = dV_z_next_t
            dV_th_traj_t[t, :] = dV_th_next_t

        return dV_th_traj_t

    # ── Main loop ─────────────────────────────────────────────────────────────

    def apply(self, th0_traj):
        """Run projected gradient descent to minimise J over th_traj.

        Args:
            th0_traj: np.ndarray of shape (T, N) — initial incentive sequence.
                      Defaults to self.th0_traj if not provided.

        Returns:
            th_traj:      np.ndarray of shape (T, N)          — optimised incentive sequence.
            th_traj_hist: np.ndarray of shape (maxIt+1, T, N) — th_traj per iteration, with th0_traj at index 0.
            cost_hist:    np.ndarray of shape (maxIt,)        — cost J(th_traj) at each iteration.
            res_hist:     np.ndarray of shape (maxIt, 2)      — stationarity residual [stat_res, stat_res_rel]
                          at each iteration (see check_conv).
        """
        th_traj         = th0_traj
        th_traj_hist    = np.zeros((self.maxIt + 1, self.T, self.N))
        th_traj_hist[0] = th0_traj
        cost_hist       = np.zeros(self.maxIt)
        res_hist        = np.zeros((self.maxIt, 2))

        it_counter = tqdm(range(self.maxIt), desc="proj_grad_desc", position=0)
        for it in it_counter:
            z_traj, mu_traj = self.forward_pass(th_traj)
            dV0_th_traj = self.backpropagation(th_traj, z_traj, mu_traj)
            cost_hist[it] = self.get_cost(z_traj, th_traj)

            stat_res         = np.linalg.norm(th_traj - self.projector.proj(th_traj - dV0_th_traj))
            res_hist[it, 0]  = stat_res
            res_hist[it, 1]  = stat_res / np.linalg.norm(th_traj)

            th_traj_next         = self.projector.proj(th_traj - self.get_alph(it) * dV0_th_traj)
            err                  = np.linalg.norm(th_traj_next - th_traj)
            th_traj_hist[it + 1] = th_traj_next
            th_traj              = th_traj_next

            if err < self.err_thr:
                break

        return th_traj, th_traj_hist, cost_hist, res_hist

    # ── Convergence check ─────────────────────────────────────────────────────

    def check_conv(self, th_traj):
        """Higher-level convergence check: projected-gradient stationarity residual,
        plus the explicit KKT residuals for the same problem (see Projector.kkt_residuals).

            stat_res = ||th_traj - proj(th_traj - dV0_th_traj)||,
            zero iff th_traj is a fixed point of the projected gradient map.
            stat_res_rel is the same residual normalised by ||th_traj||.

        Args:
            th_traj: np.ndarray of shape (T, N).

        Returns:
            dict with keys "stat_res", "stat_res_rel" (float), and "kkt":
            {"res_stat", "res_stat_rel", "res_prim", "res_dual", "res_cs"} (float each).
        """
        z_traj, mu_traj = self.forward_pass(th_traj)
        dV0_th_traj     = self.backpropagation(th_traj, z_traj, mu_traj)
        stat_res        = np.linalg.norm(th_traj - self.projector.proj(th_traj - dV0_th_traj))
        stat_res_rel    = stat_res / np.linalg.norm(th_traj)

        res_stat, res_stat_rel, res_prim, res_dual, res_cs = self.projector.kkt_residuals(th_traj, dV0_th_traj)

        return {
            "stat_res":     stat_res,
            "stat_res_rel": stat_res_rel,
            "kkt": {
                "res_stat":     res_stat,
                "res_stat_rel": res_stat_rel,
                "res_prim":     res_prim,
                "res_dual":     res_dual,
                "res_cs":       res_cs,
            },
        }

    # ── Error metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def get_err_hist(th_traj_hist):
        """Compute relative and absolute error trajectories from th_traj_hist.

        rel_err[it] = ||th_traj_hist[it+1] - th_traj_hist[it]||
        abs_err[it] = ||th_traj_hist[it+1] - th_traj_hist[-1]||
        th_traj_hist[0] must be th0_traj.

        Args:
            th_traj_hist: np.ndarray of shape (n_it+1, T, N).

        Returns:
            rel_err: np.ndarray of shape (n_it,).
            abs_err: np.ndarray of shape (n_it,).
        """
        rel_err_hist = np.linalg.norm(np.diff(th_traj_hist, axis=0), axis=(1, 2))
        abs_err_hist = np.linalg.norm(th_traj_hist[1:] - th_traj_hist[-1], axis=(1, 2))
        return rel_err_hist, abs_err_hist


class OneStep(Optimizer):
    """One-step solver for the incentive sequence.

    Inherits step size and projection from Optimizer and adds the main
    optimisation loop.
    """

    # ── One-step pass ─────────────────────────────────────────────────────────

    def one_step(self, th0_t, z_t, d_t):
        """Placeholder for the one-step forward/backward replacement.

        Args:
            th0_t: np.ndarray of shape (N,)   — initial incentive parameters at step t.
            z_t:   np.ndarray of shape (N+1,) — current extended state at step t.
            d_t:   np.ndarray of shape (2, N) — demand at step t.

        Returns:
            th_star_t:    np.ndarray of shape (N,)            — optimised incentive parameters at step t.
            th_traj_hist: np.ndarray of shape (maxIt+1, N)    — th_t per iteration, with th0_t at index 0.
            cost_hist:    np.ndarray of shape (maxIt,)        — cost g(z_t, θ_t, d_t) at each iteration.
        """
        th_t            = th0_t
        th_traj_hist    = np.zeros((self.maxIt + 1, self.N))
        th_traj_hist[0] = th0_t
        cost_hist       = np.zeros(self.maxIt)

        it_counter = tqdm(range(self.maxIt), desc="proj_grad_desc", position=1, leave=False)
        for it in it_counter:
            z_star_t, _, mu_star_t, _ = self.T_map.apply(z_t, th_t, d_t)

            dg_th_t = self.g_map.apply_dg_th(z_t, th_t, d_t)
            dT_th_t = self.dT_map.apply_th(z_t, th_t, d_t, mu_star_t)
            dg_z_t  = self.g_map.apply_dg_z(z_star_t, th_t, d_t)

            dV_th_t = dg_th_t + dg_z_t @ dT_th_t
            cost_hist[it] = self.get_cost_t(z_t, th_t, d_t)

            th_next_t         = self.projector.proj_t(th_t - self.get_alph(it) * dV_th_t)
            err               = np.linalg.norm(th_next_t - th_t)
            th_traj_hist[it + 1] = th_next_t
            th_t              = th_next_t

            if err < self.err_thr:
                break

        return th_t, th_traj_hist, cost_hist

    # ── Main loop ─────────────────────────────────────────────────────────────

    def apply(self, th0_traj):
        """Run the one-step solver to obtain th_traj.

        Args:
            th0_traj: np.ndarray of shape (T, N) — initial incentive sequence.

        Returns:
            th_traj:           np.ndarray of shape (T, N) — optimised incentive sequence.
            th_traj_hist_list: list of length T of np.ndarray of shape (maxIt+1, N)
                               — th_t per inner iteration, one per time step.
            cost_hist_list:    list of length T of np.ndarray of shape (maxIt,)
                               — cost per inner iteration, one per time step.
        """
        th_traj           = np.zeros((self.T, self.N))
        th_traj_hist_list  = []
        cost_hist_list     = []

        z_t = self.z0
        t_counter = tqdm(range(self.T), desc="one_step", position=0)
        for t in t_counter:
            th0_t = Transformer.th_traj_at_t(th0_traj, t)
            d_t   = Transformer.d_traj_at_t(self.d_traj, t)

            th_star_t, th_traj_hist, cost_hist = self.one_step(th0_t, z_t, d_t)

            z_next_t, _, _, _ = self.T_map.apply(z_t, th_star_t, d_t)

            th_traj[t, :] = th_star_t
            th_traj_hist_list.append(th_traj_hist)
            cost_hist_list.append(cost_hist)
            z_t           = z_next_t

        return th_traj, th_traj_hist_list, cost_hist_list

    # ── Convergence check ─────────────────────────────────────────────────────

    def check_conv(self, th_traj):
        """Convergence assessment for both optimisation levels, per time step.

        Higher level: one-step projected-gradient stationarity residual
            ||th_t - proj_t(th_t - dV_th_t)|| at each time step t, where
            dV_th_t = dg_th_t + dg_z_t @ dT_th_t is the same one-step gradient
            computed in one_step (using z_star_t from T_map.apply in place of
            the true costate).

        Lower level: KKT residuals from T_map.check_conv at each time step,
            collected into per-residual vectors of length T.

        Args:
            th_traj: np.ndarray of shape (T, N).

        Returns:
            dict with keys:
                "higher_level": {"stat_res": (T,)}
                "lower_level":  {"res_stat": (T,), "res_prim": (T,),
                                 "res_ineq": (T,), "res_cs":   (T,)}
        """
        stat_res = np.zeros(self.T)

        res_stat = np.zeros(self.T)
        res_prim = np.zeros(self.T)
        res_ineq = np.zeros(self.T)
        res_cs   = np.zeros(self.T)

        z_t = self.z0
        for t in range(self.T):
            th_t = Transformer.th_traj_at_t(th_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)

            z_star_t, p_star_t, mu_star_t, _ = self.T_map.apply(z_t, th_t, d_t)

            # ── Higher level ──────────────────────────────────────────────────
            dg_th_t = self.g_map.apply_dg_th(z_star_t, th_t, d_t)
            dT_th_t = self.dT_map.apply_th(z_t, th_t, d_t, mu_star_t)
            dg_z_t  = self.g_map.apply_dg_z(z_star_t, th_t, d_t)

            dV_th_t     = dg_th_t + dg_z_t @ dT_th_t
            stat_res[t] = np.linalg.norm(th_t - self.projector.proj_t(th_t - dV_th_t))

            # ── Lower level ───────────────────────────────────────────────────
            G_iq_t = self.T_map.get_G_iq(d_t)
            b_iq_t = self.T_map.get_b_iq(z_t)
            r_t    = self.T_map.get_r(th_t)

            res_stat[t], res_prim[t], res_ineq[t], res_cs[t] = \
                self.T_map.check_conv(p_star_t, mu_star_t, G_iq_t, b_iq_t, r_t)

            z_t = z_star_t

        return {
            "higher_level": {
                "stat_res": stat_res,
            },
            "lower_level": {
                "res_stat": res_stat,
                "res_prim": res_prim,
                "res_ineq": res_ineq,
                "res_cs":   res_cs,
            },
        }

    # ── Error metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def get_err_hist(th_traj_hist_list, cost_hist_list):
        """Compute relative and absolute error histories per time step.

        Applies the same rel_err/abs_err formula as ForwardBackward.get_err_hist
        to each time step's th_traj_hist independently.

        Args:
            th_traj_hist_list: list of length T of np.ndarray of shape (maxIt+1, N).
            cost_hist_list:    list of length T of np.ndarray of shape (maxIt,).

        Returns:
            rel_err_list: list of length T of np.ndarray of shape (n_it,).
            abs_err_list: list of length T of np.ndarray of shape (n_it,).
        """
        rel_err_hist_list = []
        abs_err_hist_list = []

        for th_traj_hist in th_traj_hist_list:
            rel_err_hist = np.linalg.norm(np.diff(th_traj_hist, axis=0), axis=1)
            abs_err_hist = np.linalg.norm(th_traj_hist[1:] - th_traj_hist[-1], axis=1)
            rel_err_hist_list.append(rel_err_hist)
            abs_err_hist_list.append(abs_err_hist)

        return rel_err_hist_list, abs_err_hist_list
