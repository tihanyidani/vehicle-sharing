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


class MPCC(Optimizer):
    """Single-level Mathematical Program with Complementarity Constraints (MPCC)
    reformulation of the bilevel incentive-design problem, solved via a
    Scholtes-relaxation homotopy in CasADi/IPOPT.

    The lower-level problem (14) is eliminated in favour of its dual: substituting
    p* = exp(r(θ) - 1 - G(d)ᵀμ) (T_map's own KKT/duality convention, cf.
    T_map.check_conv) removes the primal variable p and the log(p) domain
    restriction, leaving one smooth equality (the simplex constraint) and one
    nonlinear complementarity pair per region per time step:

        G_eq p*(θ_t, μ_t) - b_eq = 0
        0 ≤ μ_iq,t ⊥ s_t ≥ 0,  s_t := b_iq(x_t) - G_iq(d_t) p*(θ_t, μ_t)

    tied together by the (linear, once p* is substituted) dynamics z_{t+1} =
    Φz_t + Γ^pu E(d_t) p*(θ_t,μ_t) + Γ^do d_t^do. Stacking this over t=0,...,T-1
    turns the bilevel problem into one direct-transcription MPCC. It is solved
    by relaxing the complementarity pair to 0 ≤ μ_iq,t · s_t ≤ τ and driving
    τ → 0 across an outer homotopy loop (τ^{k+1} = τ_decay · τ^k), each step
    warm-started from the previous — the same "Scholtes global relaxation"
    baseline described in Pozharskiy et al., "CCOpt: an Open-Source Solver for
    Large-Scale Mathematical Programs with Complementarity Constraints" (2026).
    """

    # ── Cost helpers ──────────────────────────────────────────────────────────
    # g_map.apply_g/apply_h rely on numpy's 1-D vector @ matrix @ vector
    # broadcasting (v @ M @ v with v of shape (n,)); casadi's strictly-2D MX
    # does not support this, so these helpers reuse the already-built weight
    # matrices (g_map.Q, g_map.R, and x_des/Xi_x for the tracking variant)
    # with an explicit quadratic form instead.

    @staticmethod
    def _quad(ca, v, M):
        return ca.mtimes([v.T, M, v])

    def _stage_cost(self, ca, z, th):
        if hasattr(self.g_map, "x_des"):
            e = self.g_map.x_des - self.g_map.Xi_x @ z
            return 0.5 * (self._quad(ca, e, self.g_map.Q) + self._quad(ca, th, self.g_map.R))
        return 0.5 * (self._quad(ca, z, self.g_map.Q) + self._quad(ca, th, self.g_map.R))

    def _terminal_cost(self, ca, z):
        if hasattr(self.g_map, "x_des"):
            e = self.g_map.x_des - self.g_map.Xi_x @ z
            return 0.5 * self._quad(ca, e, self.g_map.Q)
        return 0.5 * self._quad(ca, z, self.g_map.Q)

    # ── Warm start ────────────────────────────────────────────────────────────

    def _forward_pass_warm_start(self, th0_traj):
        """Run the exact lower-level LP (T_map.apply) once under th0_traj to
        obtain a primal-dual feasible warm start for the MPCC variables.

        Args:
            th0_traj: np.ndarray of shape (T, N) — initial incentive sequence.

        Returns:
            z_traj:  np.ndarray of shape (T+1, N+1) — extended state trajectory.
            mu_traj: np.ndarray of shape (T, 2*N)   — [μ_eq; μ_iq] per time step.
            p_traj:  list of length T of np.ndarray of shape (N*(N+1),).
        """
        z_traj       = np.zeros((self.T + 1, self.N + 1))
        z_traj[0, :] = self.z0
        mu_traj      = np.zeros((self.T, self.N_mu))
        p_traj       = []

        z_t = self.z0
        t_counter = tqdm(range(self.T), desc="MPCC warm start", position=1, leave=False)
        for t in t_counter:
            th_t = Transformer.th_traj_at_t(th0_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)

            z_next_t, p_star_t, mu_star_t, _ = self.T_map.apply(z_t, th_t, d_t)
            z_traj[t + 1, :] = z_next_t
            mu_traj[t, :]    = mu_star_t
            p_traj.append(p_star_t)
            z_t = z_next_t

        return z_traj, mu_traj, p_traj

    # ── NLP construction ──────────────────────────────────────────────────────

    def _build_nlp(self, ca):
        """Build the symbolic MPCC transcription described in the class docstring.

        Args:
            ca: the casadi module (passed in rather than imported at module
                level, so importing this file does not require casadi unless
                MPCC.apply is actually called).

        Returns:
            x:          casadi MX — flat decision vector [θ; μ_eq; μ_iq; s; z].
            g:          casadi MX — stacked constraints [equalities; complementarity
                        (relaxed); budget (if a budget projector is configured)].
            cost:       casadi MX — scalar objective, Σ g(z_t,θ_t) + h(z_T).
            tau:        casadi MX — scalar symbolic relaxation parameter (nlpsol's `p`).
            lbx, ubx:   np.ndarray — bounds on x (θ boxed, μ_iq and s ≥ 0, rest free).
            n_eq, n_cc: int — sizes of the equality / complementarity constraint blocks.
            has_budget: bool — whether a budget row was appended to g.
        """
        N, T = self.N, self.T

        th_vars    = [ca.MX.sym(f"th_{t}", N)    for t in range(T)]
        mu_eq_vars = [ca.MX.sym(f"mueq_{t}", N)  for t in range(T)]
        mu_iq_vars = [ca.MX.sym(f"muiq_{t}", N)  for t in range(T)]
        s_vars     = [ca.MX.sym(f"s_{t}", N)     for t in range(T)]
        z_vars     = [ca.MX.sym(f"z_{t}", N + 1) for t in range(T + 1)]
        tau        = ca.MX.sym("tau")

        eq_list = [z_vars[0] - self.z0]
        cc_list = []
        cost    = 0

        for t in range(T):
            d_t = Transformer.d_traj_at_t(self.d_traj, t)
            d_do_t, _ = Transformer.split_d(d_t, self.T_map)

            G_iq_t = self.T_map.get_G_iq(d_t)
            E_t    = self.T_map.get_E(d_t)

            r_t = self.T_map.get_r(th_vars[t])
            p_t = ca.exp(r_t - 1
                         - self.T_map.G_eq.T @ mu_eq_vars[t]
                         - G_iq_t.T @ mu_iq_vars[t])

            b_iq_t = self.T_map.get_b_iq(z_vars[t])
            u_t    = E_t @ p_t

            eq_list.append(self.T_map.G_eq @ p_t - self.T_map.b_eq)
            eq_list.append(s_vars[t] - b_iq_t + G_iq_t @ p_t)
            eq_list.append(z_vars[t + 1] - (self.T_map.Phi @ z_vars[t]
                                             + self.T_map.Gamma_pu @ u_t
                                             + self.T_map.Gamma_do @ d_do_t))
            cc_list.append(mu_iq_vars[t] * s_vars[t] - tau)

            cost = cost + self._stage_cost(ca, z_vars[t], th_vars[t])

        cost = cost + self._terminal_cost(ca, z_vars[T])

        x    = ca.vertcat(*th_vars, *mu_eq_vars, *mu_iq_vars, *s_vars, *z_vars)
        g_eq = ca.vertcat(*eq_list)
        g_cc = ca.vertcat(*cc_list)
        g    = ca.vertcat(g_eq, g_cc)

        th_lo, th_hi = 0.0, self.projector.th_box
        lbx = np.concatenate([
            np.full(T * N, th_lo), np.full(T * N, -np.inf),
            np.zeros(T * N),       np.zeros(T * N),
            np.full((T + 1) * (N + 1), -np.inf),
        ])
        ubx = np.concatenate([
            np.full(T * N, th_hi), np.full(T * N, np.inf),
            np.full(T * N, np.inf), np.full(T * N, np.inf),
            np.full((T + 1) * (N + 1), np.inf),
        ])

        has_budget = hasattr(self.projector, "th_budget")
        if has_budget:
            g = ca.vertcat(g, ca.sum1(ca.vertcat(*th_vars)))

        return x, g, cost, tau, lbx, ubx, g_eq.shape[0], g_cc.shape[0], has_budget

    # ── Main loop ─────────────────────────────────────────────────────────────

    def apply(self, th0_traj, tau0=1.0, tau_decay=0.1, tau_min=1e-8,
              ipopt_opts=None, verbose=False):
        """Solve the incentive design problem as a single-level MPCC.

        Builds the direct-transcription MPCC described in the class docstring
        and solves it with a Scholtes-relaxation homotopy: a sequence of NLPs
        with 0 ≤ μ_iq,t · s_t ≤ τ^k is solved to convergence in IPOPT, warm-
        starting the next from the previous solution, while τ^k → 0
        geometrically (τ^{k+1} = τ_decay · τ^k, starting at τ^0 = tau0).

        Args:
            th0_traj:   np.ndarray of shape (T, N) — initial incentive guess.
                        Also used to seed a lower-level forward pass
                        (T_map.apply) that provides a primal-dual feasible
                        warm start for μ, s, and z.
            tau0:       float — initial relaxation parameter.
            tau_decay:  float — geometric decay factor for τ.
            tau_min:    float — homotopy stops once τ^k < tau_min.
            ipopt_opts: dict or None — extra options forwarded to IPOPT.
            verbose:    bool — if True, IPOPT prints its own iteration log.

        Returns:
            th_traj:   np.ndarray of shape (T, N) — optimised incentive sequence.
            tau_hist:  np.ndarray of shape (n_it,) — τ^k used at each homotopy step.
            cost_hist: np.ndarray of shape (n_it,) — V0(z0, Θ) at each homotopy step.
        """
        import casadi as ca

        N, T = self.N, self.T

        z_traj_ws, mu_traj_ws, p_traj_ws = self._forward_pass_warm_start(th0_traj)

        x, g, cost, tau, lbx, ubx, n_eq, n_cc, has_budget = self._build_nlp(ca)

        opts = {
            "print_time":        verbose,
            "ipopt.print_level": 5 if verbose else 0,
            "ipopt.sb":          "yes",
        }
        if ipopt_opts:
            opts.update(ipopt_opts)

        solver = ca.nlpsol("solver", "ipopt", {"x": x, "p": tau, "f": cost, "g": g}, opts)

        s0_ws = np.concatenate([
            np.maximum(
                self.T_map.get_b_iq(z_traj_ws[t])
                - self.T_map.get_G_iq(Transformer.d_traj_at_t(self.d_traj, t)) @ p_traj_ws[t],
                1e-6,
            )
            for t in range(T)
        ])
        x0 = np.concatenate([
            th0_traj.reshape(-1),
            mu_traj_ws[:, :N].reshape(-1),
            np.maximum(mu_traj_ws[:, N:], 1e-6).reshape(-1),
            s0_ws,
            z_traj_ws.reshape(-1),
        ])

        n_it = int(np.ceil(np.log(tau_min / tau0) / np.log(tau_decay))) + 1
        tau_hist  = np.zeros(n_it)
        cost_hist = np.zeros(n_it)

        tau_k  = tau0
        sol_x0 = x0
        it_counter = tqdm(range(n_it), desc="MPCC homotopy", position=0)
        for it in it_counter:
            lbg = np.concatenate([np.zeros(n_eq), np.full(n_cc, -np.inf)])
            ubg = np.zeros(n_eq + n_cc)
            if has_budget:
                lbg = np.concatenate([lbg, [-np.inf]])
                ubg = np.concatenate([ubg, [self.projector.th_budget]])

            sol    = solver(x0=sol_x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=tau_k)
            sol_x0 = sol["x"]

            tau_hist[it]  = tau_k
            cost_hist[it] = float(sol["f"])
            it_counter.set_postfix(tau=f"{tau_k:.1e}", cost=f"{cost_hist[it]:.4g}")

            if not solver.stats()["success"]:
                tqdm.write(f"[MPCC] IPOPT did not converge at tau={tau_k:.1e}; "
                           f"continuing homotopy from its last iterate.")

            tau_k *= tau_decay

        th_traj = np.array(sol_x0[:T * N]).flatten().reshape(T, N)

        return th_traj, tau_hist, cost_hist
