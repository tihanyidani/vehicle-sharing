"""Jacobians of T_map w.r.t. state z and incentives θ via the implicit function theorem."""

import numpy as np

from utils.T_map import T_map_base


class dT_map(T_map_base):
    """Sensitivity of the forward map T computed via the implicit function theorem.

    Given the primal-dual solution (p*, μ*) from T_map, this class
    differentiates through the KKT stationarity conditions of the inner LP
    to obtain ∂T/∂z and ∂T/∂θ analytically.

    The dual gradient-descent fixed-point map π is used to identify active
    inequality constraints; its Jacobian J_π enters the IFT linear system.
    """

    def __init__(self, params):
        super().__init__(params)
        self.phi   = params.phi    # dual gradient step size

    # ── Extended constraint matrix ────────────────────────────────────────────

    def get_G(self, G_iq):
        """Stack constraint matrices into G = [[G_eq], [G_iq]].

        Args:
            G_iq: np.ndarray of shape (N, N*(N+1)) — inequality constraint matrix.

        Returns:
            np.ndarray of shape (2*N, N*(N+1)).
        """
        return np.vstack((self.G_eq, G_iq))

    def get_b(self, b_iq):
        """Build dual RHS vector b = [b_eq; b_iq].

        Args:
            b_iq: np.ndarray of shape (N,) — inequality constraint bound vector.

        Returns:
            np.ndarray of shape (2*N,).
        """
        return np.concatenate((self.b_eq, b_iq))

    # ── Dual-space quantities ─────────────────────────────────────────────────

    def get_expv(self, mu, r, G):
        """Recover the primal y* from dual mu via KKT stationarity: exp(r - 1 - G^T mu).

        Args:
            mu: np.ndarray of shape (2*N,)         — dual variables.
            r:  np.ndarray of shape (N*(N+1),)      — incentive utility vector.
            G:  np.ndarray of shape (2*N, N*(N+1)) — constraint matrix [[G_eq], [G_iq]].

        Returns:
            np.ndarray of shape (N*(N+1),).
        """
        return np.exp(r - np.ones(self.N * (self.N + 1)) - G.T @ mu)

    def get_db_x(self, z):
        """Compute the mixed partial ∂b/∂z of the dual RHS bound b = [b_eq; b_iq].

        b_eq is constant, so its block is zero. b_iq = Xi_x z / dt is affine
        in z, so this Jacobian is constant; z is accepted only for interface
        consistency with the other dT_map methods and is otherwise unused.

        Args:
            z: np.ndarray of shape (N+1,) — unused.

        Returns:
            np.ndarray of shape (2*N, N+1).
        """
        return np.vstack((
            np.zeros((self.N, self.N + 1)),
            self.Xi_x / self.dt,
        ))

    def get_dd_mu(self, expv, b, G):
        """Compute dual gradient ∇_μ L = -G exp(v) + b.

        Args:
            expv: np.ndarray — primal recovery vector.
            b:    np.ndarray — dual RHS vector.
            G:    np.ndarray — constraint matrix [[G_eq], [G_iq]].

        Returns:
            np.ndarray of shape (2*N,).
        """
        return -G @ expv + b

    def get_ddd_mumu(self, expv, G):
        """Compute dual Hessian ∇²_μ L = G diag(exp(v)) G^T.

        Args:
            expv: np.ndarray — primal recovery vector.
            G:    np.ndarray — constraint matrix [[G_eq], [G_iq]].

        Returns:
            np.ndarray of shape (2*N, 2*N).
        """
        return G @ np.diag(expv) @ G.T

    def get_ddd_muz(self, expv, G, db_x):
        """Compute mixed partial ∇²_{μ,z} L = ∂b/∂z (constant in μ and z).

        Reflects how the dual RHS bound b = [b_eq; b_iq] varies with z:
        b_eq is constant, and b_iq = Xi_x z / dt is affine in z. expv and G
        are accepted only for interface consistency with the other dT_map
        methods and are otherwise unused.

        Args:
            expv: np.ndarray — unused.
            G:    np.ndarray — unused.
            db_x: np.ndarray of shape (2*N, N+1) — mixed partial from get_db_x.

        Returns:
            np.ndarray of shape (2*N, N+1).
        """
        return db_x

    def get_ddd_muth(self, expv, G):
        """Compute mixed partial ∇²_{μ,θ} L = -G diag(exp(v)) Λ.

        Args:
            expv: np.ndarray — primal recovery vector.
            G:    np.ndarray — constraint matrix [[G_eq], [G_iq]].

        Returns:
            np.ndarray of shape (2*N, N).
        """
        return -G @ np.diag(expv) @ self.Lam_prime

    # ── Dual projection Jacobian ──────────────────────────────────────────────

    def get_J_Pi(self, mu, dd_mu):
        """Compute the Jacobian J_π of the dual projection map π.

        π projects the dual iterate onto the non-negative orthant for the
        inequality multipliers μ_iq. J_π is diagonal: 1 for equality duals
        (always active) and active inequality constraints, 0 for inactive ones.

        Args:
            mu:    np.ndarray of shape (2*N,) — current dual variables [μ_eq; μ_iq].
            dd_mu: np.ndarray of shape (2*N,) — dual gradient at mu.

        Returns:
            np.ndarray of shape (2*N, 2*N) — diagonal binary matrix.
        """
        mu_next    = mu - self.phi * dd_mu
        mu_iq_next = mu_next[self.N:]   # inequality-dual part after gradient step
        active     = (mu_iq_next > 0).astype(int)
        return np.diag(np.concatenate((np.ones(self.N), active)))

    # ── IFT linear system ─────────────────────────────────────────────────────

    def get_U(self, J_Pi, ddd_mumu):
        """Compute IFT system matrix U = J_Π (I - γ H) - I.

        Args:
            J_Pi:     np.ndarray of shape (2*N, 2*N) — dual projection Jacobian.
            ddd_mumu: np.ndarray of shape (2*N, 2*N) — dual Hessian.

        Returns:
            np.ndarray of shape (2*N, 2*N).
        """
        return J_Pi @ (np.eye(2 * self.N) - self.phi * ddd_mumu) - np.eye(2 * self.N)

    def get_V_z(self, J_Pi, ddd_muz):
        """Compute IFT RHS for ∂μ*/∂z: V_z = -γ J_Π ∇²_{μ,z} L.

        Args:
            J_Pi:    np.ndarray of shape (2*N, 2*N).
            ddd_muz: np.ndarray of shape (2*N, N+1).

        Returns:
            np.ndarray of shape (2*N, N+1).
        """
        return -self.phi * J_Pi @ ddd_muz

    def get_V_th(self, J_Pi, ddd_muth):
        """Compute IFT RHS for ∂μ*/∂θ: V_th = -γ J_Π ∇²_{μ,θ} L.

        Args:
            J_Pi:     np.ndarray of shape (2*N, 2*N).
            ddd_muth: np.ndarray of shape (2*N, N).

        Returns:
            np.ndarray of shape (2*N, N).
        """
        return -self.phi * J_Pi @ ddd_muth

    # ── IFT solves ────────────────────────────────────────────────────────────

    def get_Mu_z(self, U, V_z):
        """Solve U Mu_z = -V_z for ∂μ*/∂z via pseudoinverse.

        Args:
            U:   np.ndarray of shape (2*N, 2*N).
            V_z: np.ndarray of shape (2*N, N+1).

        Returns:
            np.ndarray of shape (2*N, N+1).
        """
        return -np.linalg.pinv(U) @ V_z

    def get_Mu_th(self, U, V_th):
        """Solve U Mu_th = -V_th for ∂μ*/∂θ via pseudoinverse.

        Args:
            U:    np.ndarray of shape (2*N, 2*N).
            V_th: np.ndarray of shape (2*N, N).

        Returns:
            np.ndarray of shape (2*N, N).
        """
        return -np.linalg.pinv(U) @ V_th

    # ── Primal Jacobians ──────────────────────────────────────────────────────

    def get_P_z(self, expv, G, Mu_z):
        """Compute ∂p*/∂z = diag(exp(v)) (-G^T Mu_z).

        Args:
            expv: np.ndarray of shape (N*(N+1),).
            G:    np.ndarray of shape (2*N, N*(N+1)).
            Mu_z: np.ndarray of shape (2*N, N+1).

        Returns:
            np.ndarray of shape (N*(N+1), N+1).
        """
        return np.diag(expv) @ (-G.T @ Mu_z)

    def get_P_th(self, expv, G, Mu_th):
        """Compute ∂p*/∂θ = diag(exp(v)) (Λ - G^T Mu_th).

        Args:
            expv:  np.ndarray of shape (N*(N+1),).
            G:     np.ndarray of shape (2*N, N*(N+1)).
            Mu_th: np.ndarray of shape (2*N, N).

        Returns:
            np.ndarray of shape (N*(N+1), N).
        """
        return np.diag(expv) @ (self.Lam_prime - G.T @ Mu_th)

    # ── Public interface ──────────────────────────────────────────────────────

    def apply_z(self, z, th, d, mu_star):
        """Compute ∂T/∂z = Phi + Gamma_pu E ∂p*/∂z.

        Args:
            z:       np.ndarray of shape (N+1,) — current extended state.
            th:      np.ndarray of shape (N,) — current incentive parameters.
            d:       np.ndarray of shape (2, N) — current demand signal.
            mu_star: np.ndarray — dual solution returned by T_map.

        Returns:
            np.ndarray of shape (N+1, N+1) — Jacobian of T w.r.t. z.
        """
        G_iq  = self.get_G_iq(d)
        b_iq  = self.get_b_iq(z)
        G     = self.get_G(G_iq)
        b     = self.get_b(b_iq)
        r     = self.get_r(th)
        E     = self.get_E(d)
        expv  = self.get_expv(mu_star, r, G)

        db_x     = self.get_db_x(z)
        dd_mu    = self.get_dd_mu(expv, b, G)
        ddd_mumu = self.get_ddd_mumu(expv, G)
        ddd_muz  = self.get_ddd_muz(expv, G, db_x)
        J_Pi     = self.get_J_Pi(mu_star, dd_mu)

        U    = self.get_U(J_Pi, ddd_mumu)
        V_z  = self.get_V_z(J_Pi, ddd_muz)
        Mu_z = self.get_Mu_z(U, V_z)
        P_z  = self.get_P_z(expv, G, Mu_z)

        return self.Phi + self.Gamma_pu @ E @ P_z

    def apply_th(self, z, th, d, mu_star):
        """Compute ∂T/∂θ = Gamma_pu E ∂p*/∂θ.

        Args:
            z:       np.ndarray of shape (N+1,) — current extended state.
            th:      np.ndarray of shape (N,)   — current incentive parameters.
            d:       np.ndarray of shape (2, N) — current demand signal.
            mu_star: np.ndarray — dual solution returned by T_map.

        Returns:
            np.ndarray of shape (N+1, N) — Jacobian of T w.r.t. θ.
        """
        G_iq  = self.get_G_iq(d)
        b_iq  = self.get_b_iq(z)
        G     = self.get_G(G_iq)
        b     = self.get_b(b_iq)
        r     = self.get_r(th)
        E     = self.get_E(d)
        expv  = self.get_expv(mu_star, r, G)

        dd_mu    = self.get_dd_mu(expv, b, G)
        ddd_mumu = self.get_ddd_mumu(expv, G)
        ddd_muth = self.get_ddd_muth(expv, G)
        J_Pi     = self.get_J_Pi(mu_star, dd_mu)

        U     = self.get_U(J_Pi, ddd_mumu)
        V_th  = self.get_V_th(J_Pi, ddd_muth)
        Mu_th = self.get_Mu_th(U, V_th)
        P_th  = self.get_P_th(expv, G, Mu_th)

        return self.Gamma_pu @ E @ P_th