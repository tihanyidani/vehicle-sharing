"""Inner-level allocation map T and its entropy-regularised LP solver."""

import warnings

import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag

from utils.Transformer import Transformer

class T_map_base:
    """Shared matrix constructions used by both T_map and dT_map.

    Builds and caches the static matrices (Lam_prime, w_prime, G_eq, b_eq,
    Phi, Gamma_pu, Gamma_do) at construction time, and provides helpers for
    the time-varying quantities (G_iq, b_iq, E, r) that depend on the
    current state, incentives, or demand.
    """

    def __init__(self, params):
        self.N  = params.N
        self.dt = params.dt

        self.DO   = params.DO
        self.PU   = params.PU
        self.Xi_x = params.Xi_x

        # Build extended static matrices once at construction time
        self.Lam_prime = self.init_Lam_prime(params)   # extended price-sensitivity matrix
        self.w_prime   = self.init_w_prime(params)     # extended base-utility vector

        self.G_eq      = self.init_G_eq(params)        # simplex constraint matrix
        self.b_eq      = self.init_b_eq(params)        # simplex constraint bound

        self.Phi       = params.Phi                    # extended dynamics matrix
        self.Gamma_pu  = params.Gamma_pu               # extended PU input matrix
        self.Gamma_do  = params.Gamma_do               # extended DO input matrix

    # ── Static matrix constructors ────────────────────────────────────────────

    def init_Lam_prime(self, params):
        """Build extended price-sensitivity matrix Λ of shape (N*(N+1), N).

        One N-row block per destination j, with one zero row appended per
        destination j for the baseline demand variable.
        """
        blocks = []
        for j in range(params.N):
            block = np.vstack([
                np.diag(params.Lam[:, j]),
                np.zeros((1, params.N))
            ])
            blocks.append(block)
        Lam_prime = np.vstack(blocks)

        return Lam_prime

    def init_w_prime(self, params):
        """Build extended base-utility vector w of shape (N*(N+1),).

        Concatenates the pairwise willingness-to-travel W and the
        node-level baseline offsets w0.
        """
        offsets = []
        for j in range(params.N):
            offsets.append(np.concatenate([
                params.W[:, j],
                [params.w_le[j]]
            ]))
        w_prime = np.concatenate(offsets)

        return w_prime

    # ── Constraint matrices ───────────────────────────────────────────────────

    def init_G_eq(self, params):
        """Build simplex constraint matrix G_eq of shape (N, N*(N+1)).

        Each row sums the flow variables for one destination node to 1,
        with an identity block appended for the baseline demand variable.
        """
        G_eq = block_diag(*[
            np.ones((1, self.N + 1)) for j in range(self.N)
        ])
        return G_eq

    def init_b_eq(self, params):
        """Build simplex constraint bound b_eq of shape (N,).

        Each destination node's flow variables must sum to 1.
        """
        return np.ones(self.N)

    def get_G_iq(self, d):
        """Build inequality constraint matrix G_iq of shape (N, N*(N+1)) from signal d.

        Args:
            d: np.ndarray of shape (2, N) — demand at the current time step.

        Returns:
            np.ndarray of shape (N, N*(N+1)).
        """
        _, d_pu = Transformer.split_d(d, self)
        G_iq = np.hstack([d_pu[j] * self.Xi_x for j in range(self.N)])
        return G_iq

    def get_b_iq(self, z):
        """Build inequality constraint bound b_iq of shape (N,) from extended state z.

        Args:
            z: np.ndarray of shape (N+1,) — current extended state.

        Returns:
            np.ndarray of shape (N,).
        """
        x, _ = Transformer.split_z(z, self)
        return x / self.dt

    # ── Matrix helpers ────────────────────────────────────────────────────────

    def get_E(self, d):
        """Build extended inequality constraint matrix E of shape (N+1, N*(N+1)) from signal d.

        Args:
            d: np.ndarray of shape (2, N) — demand at the current time step.

        Returns:
            np.ndarray of shape (N+1, N*(N+1)).
        """
        _, d_pu = Transformer.split_d(d, self)
        E = np.hstack([d_pu[j] * np.eye(self.N + 1) for j in range(self.N)])
        return E

    def get_r(self, th):
        """Compute incentive utility vector r = Λ @ th - w.

        Args:
            th: np.ndarray of shape (N,) — current incentive parameters.

        Returns:
            np.ndarray of shape (N*(N+1),).
        """
        return self.Lam_prime @ th - self.w_prime


class T_map(T_map_base):
    """Forward map T: (z, θ, d) → (z_next, p*, μ*).

    Solves the entropy-regularised allocation LP via cvxpy, then
    applies the system dynamics to advance the state by one step.
    """

    def __init__(self, params):
        super().__init__(params)
        self.lp_tol = params.lp_tol
        self.solver = params.solver
        if self.solver not in cp.installed_solvers():
            raise ValueError(
                f"Solver '{self.solver}' is not installed. "
                f"Installed solvers: {cp.installed_solvers()}"
            )
        print(f"[T_map] LP solver: {self.solver}  tol={self.lp_tol:.1e}")

    # ── Public interface ──────────────────────────────────────────────────────

    def apply(self, z, th, d):
        """Advance the state one step under incentive th and demand d.

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — current incentive parameters.
            d:  np.ndarray of shape (2, N) — current demand signal.

        Returns:
            z_next:  np.ndarray of shape (N+1,) — next extended state.
            p_star:  np.ndarray                 — primal optimal allocation.
            mu_star: np.ndarray                 — dual variables [μ_eq; μ_iq].
            u:       np.ndarray of shape (N+1,) — demand-scaled optimal flow.
        """
        u, p_star, mu_star = self.S_map(z, th, d)
        d_do, _ = Transformer.split_d(d, self)
        z_next  = self.Phi @ z + self.Gamma_pu @ u + self.Gamma_do @ d_do
        return z_next, p_star, mu_star, u

    def S_map(self, z, th, d):
        """Compute the optimal control input u = E @ p*.

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — current incentive parameters.
            d:  np.ndarray of shape (2, N) — current demand signal.

        Returns:
            u:       np.ndarray of shape (N+1,) — demand-scaled optimal flow.
            p_star:  np.ndarray — primal optimal allocation.
            mu_star: np.ndarray — dual variables from P_map.
        """
        p_star, mu_star = self.P_map(z, th, d)
        E = self.get_E(d)
        u = E @ p_star
        return u, p_star, mu_star

    def P_map(self, z, th, d):
        """Solve the entropy-regularised LP for the optimal allocation p*.

        Maximises  p^T r + H(p)  s.t.  G_eq p = b_eq,  G_iq p ≤ b_iq,
        where H(p) = Σ −p_i log p_i is the entropy regulariser that
        ensures a strictly interior (softmax-like) solution.

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — current incentive parameters.
            d:  np.ndarray of shape (2, N) — current demand signal.

        Returns:
            p_star:  np.ndarray — primal optimal allocation.
            mu_star: np.ndarray — concatenated dual variables [μ_eq; μ_iq].
        """
        p = cp.Variable(self.N * (self.N + 1))
        G_iq = self.get_G_iq(d)
        b_iq = self.get_b_iq(z)
        r    = self.get_r(th)

        objective = cp.Maximize(p @ r + cp.sum(cp.entr(p)))

        constraints = [
            self.G_eq @ p == self.b_eq,   # simplex / normalisation
            G_iq @ p      <= b_iq,        # capacity constraints
        ]

        problem = cp.Problem(objective, constraints)
        if self.solver == cp.CLARABEL:
            problem.solve(solver=cp.CLARABEL,
                          tol_gap_abs=self.lp_tol, tol_gap_rel=self.lp_tol,
                          tol_feas=self.lp_tol)
        elif self.solver == cp.SCS:
            eps_scs = max(self.lp_tol, 1e-9)   # SCS minimum practical tolerance
            with warnings.catch_warnings():
                # SCS routinely reports "optimal_inaccurate" within tolerance;
                # status is already checked/reported below, so silence the warning.
                warnings.filterwarnings("ignore", message="Solution may be inaccurate")
                problem.solve(solver=cp.SCS, eps=eps_scs, max_iters=1000)
        else:
            problem.solve(solver=self.solver)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or p.value is None:
            raise RuntimeError(
                f"[T_map] inner LP did not solve: "
                f"solver={problem.solver_stats.solver_name} status={problem.status}. "
                f"r range=[{r.min():.3g}, {r.max():.3g}] finite={np.isfinite(r).all()}; "
                f"check th for non-finite / out-of-range values."
            )

        p_star  = p.value
        mu_star = np.hstack([constr.dual_value for constr in constraints])

        return p_star, mu_star

    def check_conv(self, p_star, mu_star, G_iq, b_iq, r):
        """KKT residuals for the entropy-regularised LP solved in P_map.

        Assesses solution quality independently of the solver's own convergence
        criterion. Returns three residuals — all should be ≈ 0 at optimality.

        Args:
            p_star:  np.ndarray of shape (n_p,)  — primal solution from P_map.
            mu_star: np.ndarray of shape (2*N,)  — dual variables [μ_eq; μ_iq] from P_map.
            G_iq:    np.ndarray of shape (N, n_p) — capacity constraint matrix.
            b_iq:    np.ndarray of shape (N,)     — capacity bound vector.
            r:       np.ndarray of shape (n_p,)   — incentive utility vector.

        Returns:
            res_stat:  float — relative stationarity residual  ‖∇_p L‖ / ‖r‖  (should be ≈ 0).
            res_prim:  float — primal feasibility      ‖G_eq p - b_eq‖ (should be ≈ 0).
            res_ineq:  float — inequality violation    max(G_iq p - b_iq, 0) (should be ≈ 0).
            res_cs:    float — complementary slackness ‖μ_iq ⊙ (G_iq p - b_iq)‖ (should be ≈ 0).
        """
        mu_eq = mu_star[:self.N]    # equality duals
        mu_iq = mu_star[self.N:]    # inequality duals (≥ 0)

        # Stationarity: ∇_p L = -(r - log(p) - 1) + G_eq^T μ_eq + G_iq^T μ_iq = 0
        # Clip before log: SCS can return near-zero or slightly negative values.
        # Normalised by ‖r‖ to remove the 1/tau scale amplification.
        grad_L    = (r - np.log(np.maximum(p_star, 1e-300)) - 1) - self.G_eq.T @ mu_eq - G_iq.T @ mu_iq
        res_stat  = np.linalg.norm(grad_L) / np.linalg.norm(r)

        # Primal feasibility (equality)
        res_prim  = np.linalg.norm(self.G_eq @ p_star - self.b_eq)

        # Primal feasibility (inequality)
        res_ineq  = np.max(np.maximum(G_iq @ p_star - b_iq, 0))

        # Complementary slackness: μ_iq_i (G_iq p - b_iq)_i = 0 for all i
        res_cs    = np.linalg.norm(mu_iq * (G_iq @ p_star - b_iq))

        return res_stat, res_prim, res_ineq, res_cs