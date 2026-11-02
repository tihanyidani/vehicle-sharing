import numpy as np


class Projector:
    """Base class for projections of the incentive sequence onto a feasible set.

    Subclasses implement the concrete projection rule (e.g. box, budget) by
    overriding proj and proj_t. Use Projector.create(params) to instantiate the
    variant selected by params.projector_type.
    """

    @staticmethod
    def create(params):
        """Instantiate the Projector variant selected by params.projector_type."""
        if params.projector_type == "box":
            return BoxProjection(params)
        elif params.projector_type == "budget":
            return BudgetProjection(params)
        else:
            raise ValueError(f"Unknown projector type: {params.projector_type!r}")

    # ── Projection ────────────────────────────────────────────────────────────

    def proj(self, th_traj):
        """Project th_traj onto the feasible set.

        Args:
            th_traj: np.ndarray of shape (T, N).

        Returns:
            np.ndarray of shape (T, N) — the projected incentive sequence.
        """
        raise NotImplementedError

    def proj_t(self, th_t):
        """Project th_t onto the feasible set.

        Args:
            th_t: np.ndarray of shape (N,).

        Returns:
            np.ndarray of shape (N,) — the projected incentive vector.
        """
        raise NotImplementedError

    # ── KKT check ─────────────────────────────────────────────────────────────

    def kkt_residuals(self, theta, grad, tol=1e-8):
        """KKT residuals of min_theta J(theta) s.t. 0 <= theta <= th_box, sum(theta) <= th_budget,
        at the point theta with gradient grad = dJ/dtheta.

        With lam >= 0 (theta >= 0), nu >= 0 (theta <= th_box), mu >= 0 (sum(theta) <= th_budget),
        the KKT system is:
            stationarity:   grad - lam + nu + mu = 0
            primal feas.:   0 <= theta <= th_box,  sum(theta) <= th_budget
            dual feas.:     lam, nu, mu >= 0
            compl. slack.:  lam*theta = 0,  nu*(theta - th_box) = 0,  mu*(sum(theta) - th_budget) = 0

        theta is assumed (box, budget)-active where it sits within tol of a bound; mu is then
        estimated from the interior points' stationarity (grad = -mu there when the budget is
        active), and lam, nu read off from stationarity at the active box bounds. Works for
        BoxProjection too, where th_budget is absent and is treated as +inf (mu == 0 always).

        Args:
            theta: np.ndarray of shape (T, N) or (N,).
            grad:  np.ndarray, same shape as theta — dJ/dtheta at theta.
            tol:   float — tolerance for deciding whether a bound is active.

        Returns:
            res_stat, res_prim, res_dual, res_cs: float residuals, each zero iff the
            corresponding KKT condition holds exactly.
            res_stat_rel: float — res_stat normalised by ‖grad‖, i.e. ‖grad - lam + nu + mu‖ / ‖grad‖.
            Same normalisation as T_map.check_conv's res_stat: dividing by the norm of the raw
            gradient term appearing in the residual cancels problem-scale effects (e.g. from the
            Q/R cost weights or problem size) that would otherwise make res_stat hard to compare
            across runs.

            Returned as (res_stat, res_stat_rel, res_prim, res_dual, res_cs).
        """
        lo, hi = 0.0, self.th_box
        budget = getattr(self, "th_budget", np.inf)

        at_lo    = theta <= lo + tol
        at_hi    = theta >= hi - tol
        interior = ~at_lo & ~at_hi
        budget_active = theta.sum() >= budget - tol

        mu = -grad[interior].mean() if (budget_active and interior.any()) else 0.0

        lam = np.where(at_lo, grad + mu, 0.0)
        nu  = np.where(at_hi, -(grad + mu), 0.0)

        res_stat     = np.linalg.norm(grad - lam + nu + mu)
        res_stat_rel = res_stat / np.linalg.norm(grad)
        res_prim = max(0.0, lo - theta.min(), theta.max() - hi, theta.sum() - budget)
        res_dual = max(0.0, float(-lam.min()), float(-nu.min()), -mu)
        res_cs   = max(
            float(np.max(np.abs(lam * (theta - lo)))),
            float(np.max(np.abs(nu  * (theta - hi)))),
            abs(mu * (theta.sum() - budget)) if np.isfinite(budget) else 0.0,
        )

        return res_stat, res_stat_rel, res_prim, res_dual, res_cs


class BoxProjection(Projector):
    """Projection onto the feasible box [0, th_box]."""

    def __init__(self, params):
        # ── Incentive bounds ──────────────────────────────────────────────────
        self.th_box = params.th_box

    # ── Projection ────────────────────────────────────────────────────────────

    def proj(self, th_traj):
        """Project th_traj onto the feasible box [0, th_box]^(T × N).

        Args:
            th_traj: np.ndarray of shape (T, N).

        Returns:
            np.ndarray of shape (T, N), clipped to [0, th_box].
        """
        return np.clip(th_traj, 0, self.th_box)

    def proj_t(self, th_t):
        """Project th_t onto the feasible box [0, th_box]^N.

        Args:
            th_t: np.ndarray of shape (N,).

        Returns:
            np.ndarray of shape (N,), clipped to [0, th_box].
        """
        return np.clip(th_t, 0, self.th_box)


class BudgetProjection(Projector):
    """Projection onto the box intersected with the budget halfspace,
    {th : 0 <= th <= th_box, sum(th) <= th_budget}.

    This is the BoxProjection feasible set with the extra affine constraint
    that the summed incentive spend not exceed th_budget.
    """

    _BISECT_ITERS = 100   # theta bisection steps (reaches machine precision well before)

    def __init__(self, params):
        # ── Incentive bounds and budget ──────────────────────────────────────
        self.th_box      = params.th_box
        self.th_budget   = params.th_budget
        self.th_budget_t = params.th_budget / params.T   # per-time-step budget share

    # ── Projection ────────────────────────────────────────────────────────────

    @classmethod
    def _proj_box_budget(cls, v, hi, s):
        """Euclidean projection of v onto {0 <= x <= hi, sum(x) <= s}.

        If clipping to the box already satisfies the budget the result is just
        clip(v, 0, hi). Otherwise the budget constraint is active and the
        projection has the form x = clip(v - theta, 0, hi) for a single scalar
        theta >= 0 fixed by sum(x) = s. Since sum(clip(v - theta, 0, hi)) is
        continuous and non-increasing in theta, theta is found by bisection on
        [0, max(v)].

        Args:
            v: np.ndarray of any shape — point to project.
            hi: float or np.ndarray broadcastable to v — box upper bound.
            s: float — budget (assumed >= 0; the set is assumed non-empty).

        Returns:
            np.ndarray of the same shape as v.
        """
        w = np.clip(v, 0.0, hi)
        if w.sum() <= s:
            return w

        hi_arr   = np.broadcast_to(hi, v.shape)
        theta_lo = 0.0                        # sum(clip(v, 0, hi)) - s > 0 here
        theta_hi = float(np.max(v))           # clip(v - theta_hi, 0, hi) == 0 -> sum 0 <= s
        for _ in range(cls._BISECT_ITERS):
            theta = 0.5 * (theta_lo + theta_hi)
            if np.clip(v - theta, 0.0, hi_arr).sum() > s:
                theta_lo = theta
            else:
                theta_hi = theta
        return np.clip(v - theta_hi, 0.0, hi_arr)   # theta_hi side is budget-feasible

    def proj(self, th_traj):
        """Project th_traj onto {Θ : 0 <= Θ <= th_box, Σ_{t,i} Θ_{t,i} <= th_budget}.

        Args:
            th_traj: np.ndarray of shape (T, N).

        Returns:
            np.ndarray of shape (T, N), in [0, th_box] with total sum <= th_budget.
        """
        return self._proj_box_budget(th_traj, self.th_box, self.th_budget)

    def proj_t(self, th_t):
        """Project th_t onto {θ : 0 <= θ <= th_box, Σ_i θ_i <= th_budget_t}, th_budget_t = th_budget / T.

        Per-time-step analogue of proj: the total budget is split evenly across
        the T steps and the same box + budget projection is applied to the
        single step's incentive vector.

        Args:
            th_t: np.ndarray of shape (N,).

        Returns:
            np.ndarray of shape (N,), in [0, th_box] with sum <= th_budget_t.
        """
        return self._proj_box_budget(th_t, self.th_box, self.th_budget_t)
