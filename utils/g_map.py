"""Stage and terminal cost functions and their gradients."""

import numpy as np


class g_map:
    """Base class for the outer optimisation objective's cost functions.

    Subclasses define the concrete stage/terminal cost g/h (and their
    gradients) by implementing init_Q, init_R, apply_g, apply_h, apply_dg_z,
    apply_dg_th, and apply_dh_z. Use g_map.create(params) to instantiate the
    variant selected by params.g_map_type.
    """

    def __init__(self, params):
        self.gamma_R = params.gamma_R  # incentive-deviation weight
        self.gamma_Q = params.gamma_Q  # stage state-deviation weight

    @staticmethod
    def create(params):
        """Instantiate the g_map variant selected by params.g_map_type."""
        if params.g_map_type == "leaving":
            return g_map_leaving(params)
        elif params.g_map_type == "tracking":
            return g_map_tracking(params)
        else:
            raise ValueError(f"Unknown g_map type: {params.g_map_type!r}")


class g_map_leaving(g_map):
    """Cost model that penalises only the outside-state deviation.

    Stage cost:    g(z, θ) = 1/2 (zᵀ Q z + θᵀ R θ)
    Terminal cost: h(z)    = 1/2 zᵀ Q z

    Q — (N+1, N+1); penalises the outside state (last entry of z); only
        Q[-1,-1] = γ_Q. Reused for both the stage and terminal cost.
    R — γ_R I_N, N-by-N, regularises the incentive vector θ uniformly.
    """

    def __init__(self, params):
        super().__init__(params)
        self.Q = self.init_Q(params)  # (N+1, N+1)
        self.R = self.init_R(params)  # (N, N)

    # ── Weight matrix constructors ────────────────────────────────────────────

    def init_Q(self, params):
        """Build stage outside-state cost matrix Q of shape (N+1, N+1).

        Only the outside state s (last entry of z) is penalised.
        """
        Q = np.zeros((params.N + 1, params.N + 1))
        Q[-1, -1] = params.gamma_Q
        return Q

    def init_R(self, params):
        """Build incentive regularisation matrix R of shape (N, N).

        Penalises the incentive vector θ uniformly: R = γ_R I_N.
        """
        return params.gamma_R * np.eye(params.N)

    # ── Cost values ───────────────────────────────────────────────────────────

    def apply_g(self, z, th, d):
        """Evaluate the stage cost g(z, θ) = 1/2 (zᵀ Q z + θᵀ R θ).

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — current incentive vector.
            d:  np.ndarray of shape (2, N) — demand (unused).

        Returns:
            float — stage cost value.
        """
        return 1/2 * (z @ self.Q @ z + th @ self.R @ th)

    def apply_g_only_Q(self, z):
        """Evaluate the state-deviation term of the stage cost, 1/2 zᵀ Q z.

        Args:
            z: np.ndarray of shape (N+1,) — current extended state.

        Returns:
            float — state-deviation cost term.
        """
        return 1/2 * (z @ self.Q @ z)

    def apply_g_only_R(self, th):
        """Evaluate the incentive-deviation term of the stage cost, 1/2 θᵀ R θ.

        Args:
            th: np.ndarray of shape (N,) — current incentive vector.

        Returns:
            float — incentive-deviation cost term.
        """
        return 1/2 * (th @ self.R @ th)

    def apply_h(self, z):
        """Evaluate the terminal cost h(z) = 1/2 zᵀ Q z.

        Args:
            z: np.ndarray of shape (N+1,) — terminal extended state.

        Returns:
            float — terminal cost value.
        """
        return 1/2 * (z @ self.Q @ z)

    # ── Gradients ─────────────────────────────────────────────────────────────

    def apply_dg_z(self, z, th, d):
        """Compute ∂g/∂z = Q z.

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — unused.
            d:  np.ndarray of shape (2, N) — unused.

        Returns:
            np.ndarray of shape (N+1,).
        """
        return self.Q @ z

    def apply_dg_th(self, z, th, d):
        """Compute ∂g/∂θ = R θ.

        Args:
            z:  np.ndarray of shape (N+1,) — unused.
            th: np.ndarray of shape (N,)   — current incentive vector.
            d:  np.ndarray of shape (2, N) — unused.

        Returns:
            np.ndarray of shape (N,).
        """
        return self.R @ th

    def apply_dh_z(self, z):
        """Compute ∂h/∂z = Q z.

        Args:
            z: np.ndarray of shape (N+1,) — terminal extended state.

        Returns:
            np.ndarray of shape (N+1,).
        """
        return self.Q @ z


class g_map_tracking(g_map):
    """Cost model that tracks a desired region-state x_des.

    Stage cost:    g(z, θ) = 1/2 (x_des - Xi_x z)ᵀ Q (x_des - Xi_x z) + 1/2 θᵀ R θ
    Terminal cost: h(z)    = 1/2 (x_des - Xi_x z)ᵀ Q (x_des - Xi_x z)

    Q — γ_Q I_N, N-by-N, penalises deviation of the region states from x_des.
    R — γ_R I_N, N-by-N, regularises the incentive vector θ uniformly.
    """

    def __init__(self, params):
        super().__init__(params)
        self.Q = self.init_Q(params)  # (N, N)
        self.R = self.init_R(params)  # (N, N)
        self.x_des = params.x_des  # (N,) desired region-state vector
        self.Xi_x  = params.Xi_x   # (N, N+1) extracts the region states from z

    # ── Weight matrix constructors ────────────────────────────────────────────

    def init_Q(self, params):
        """Build state-tracking cost matrix Q of shape (N, N).

        Penalises deviation of the region states from x_des: Q = γ_Q I_N.
        """
        return params.gamma_Q * np.eye(params.N)

    def init_R(self, params):
        """Build incentive regularisation matrix R of shape (N, N).

        Penalises the incentive vector θ uniformly: R = γ_R I_N.
        """
        return params.gamma_R * np.eye(params.N)

    # ── Cost values ───────────────────────────────────────────────────────────

    def apply_g(self, z, th, d):
        """Evaluate the stage cost g(z, θ) = 1/2 (x_des - Xi_x z)ᵀ Q (x_des - Xi_x z) + 1/2 θᵀ R θ.

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — current incentive vector.
            d:  np.ndarray of shape (2, N) — demand (unused).

        Returns:
            float — stage cost value.
        """
        e = self.x_des - self.Xi_x @ z
        return 1/2 * (e @ self.Q @ e + th @ self.R @ th)

    def apply_h(self, z):
        """Evaluate the terminal cost h(z) = 1/2 (x_des - Xi_x z)ᵀ Q (x_des - Xi_x z).

        Args:
            z: np.ndarray of shape (N+1,) — terminal extended state.

        Returns:
            float — terminal cost value.
        """
        e = self.x_des - self.Xi_x @ z
        return 1/2 * (e @ self.Q @ e)

    # ── Gradients ─────────────────────────────────────────────────────────────

    def apply_dg_z(self, z, th, d):
        """Compute ∂g/∂z = -Xi_xᵀ Q (x_des - Xi_x z).

        Args:
            z:  np.ndarray of shape (N+1,) — current extended state.
            th: np.ndarray of shape (N,)   — unused.
            d:  np.ndarray of shape (2, N) — unused.

        Returns:
            np.ndarray of shape (N+1,).
        """
        e = self.x_des - self.Xi_x @ z
        return -self.Xi_x.T @ self.Q @ e

    def apply_dg_th(self, z, th, d):
        """Compute ∂g/∂θ = R θ.

        Args:
            z:  np.ndarray of shape (N+1,) — unused.
            th: np.ndarray of shape (N,)   — current incentive vector.
            d:  np.ndarray of shape (2, N) — unused.

        Returns:
            np.ndarray of shape (N,).
        """
        return self.R @ th

    def apply_dh_z(self, z):
        """Compute ∂h/∂z = -Xi_xᵀ Q (x_des - Xi_x z).

        Args:
            z: np.ndarray of shape (N+1,) — terminal extended state.

        Returns:
            np.ndarray of shape (N+1,).
        """
        e = self.x_des - self.Xi_x @ z
        return -self.Xi_x.T @ self.Q @ e
