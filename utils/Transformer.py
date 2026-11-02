"""Transformation utilities for T_map outputs."""

import numpy as np


class Transformer:
    """Collection of static transformation methods for post-processing T_map outputs."""

    # ── Grid utilities ────────────────────────────────────────────────────────

    @staticmethod
    def point_look_up_table(i, params):
        """Return the grid coordinates (r, c) for node index i.

        Args:
            i:      int — node index.
            params: Params — provides points, the (N, 2) coordinate array.

        Returns:
            (r, c): tuple of ints — row and column of node i on the grid.
        """
        return tuple(params.points[i])

    # ── State splitting ───────────────────────────────────────────────────────

    @staticmethod
    def split_z(z, params):
        """Split the extended state vector into the state and the outside/leaving scalar.

        Args:
            z:      np.ndarray of shape (N+1,) — extended state.
            params: Params — provides N.

        Returns:
            x: np.ndarray of shape (N,) — per-node state.
            y: float — outside/leaving state (last element of z).
        """
        x = z[:params.N]
        y = z[params.N]
        return x, y

    @staticmethod
    def split_p_to_matrix(p, params):
        """Split p allocation vector into the P matrix and the leaving row.

        Args:
            p:      np.ndarray of shape ((N+1)*N,) — primal optimal allocation from T_map.P_map.
            params: Params — provides N.

        Returns:
            P:    np.ndarray of shape (N, N) — allocations, sources × destinations.
            p_le: np.ndarray of shape (N,) — leaving users row (outside/extra state per destination).
        """
        N = params.N

        # Each column is a block: reshape with Fortran (column-major) order
        P_full = p.reshape(N + 1, N, order='F')  # (N+1, N)
        P      = P_full[:N, :]                   # (N,   N)
        p_le   = P_full[N, :]                    # (N,)

        return P, p_le

    @staticmethod
    def split_p_traj_to_matrix(p_traj, params):
        """Split a full p trajectory into the P matrix and leaving row trajectories.

        Args:
            p_traj: np.ndarray of shape (T, (N+1)*N) — allocation trajectory from Simulator.run.
            params: Params — provides N.

        Returns:
            P_traj:    np.ndarray of shape (T, N, N) — allocations per time step.
            p_traj_le: np.ndarray of shape (T, N) — leaving users row per time step.
        """
        T = params.T
        N = params.N

        P_traj_full = p_traj.reshape(T, N, N + 1).transpose(0, 2, 1)  # (T, N+1, N)
        P_traj      = P_traj_full[:, :N, :]                           # (T, N,   N)
        p_traj_le   = P_traj_full[:, N, :]                            # (T, N)

        return P_traj, p_traj_le

    # ── Vector splitting ──────────────────────────────────────────────────────

    @staticmethod
    def split_d(d, params):
        """Split the demand matrix into DO and PU components.

        Args:
            d:      np.ndarray of shape (2, N) — demand at the current time step.
            params: Params — provides DO, PU.

        Returns:
            d_do: np.ndarray of shape (N,) — drop-off demand.
            d_pu: np.ndarray of shape (N,) — pick-up demand.
        """
        d_do = d[params.DO, :]
        d_pu = d[params.PU, :]
        return d_do, d_pu

    @staticmethod
    def split_u(u, params):
        """Split the control input vector into the per-node and outside-state components.

        Args:
            u:      np.ndarray of shape (N+1,) — output of T_map.S_map.
            params: Params — provides Xi_x, Xi_le.

        Returns:
            u_x:  np.ndarray of shape (N,) — per-node control input.
            u_le: float — outside-state flow.
        """
        u_x  = params.Xi_x  @ u
        u_le = (params.Xi_le @ u)[0]
        return u_x, u_le

    # ── Trajectory slicing ────────────────────────────────────────────────────

    @staticmethod
    def d_traj_at_t(d_traj, t):
        """Extract the demand matrix at time step t from a trajectory.

        Args:
            d_traj: list [d_do, d_pu] where each is np.ndarray of shape (T, N).
            t:      int — time index.

        Returns:
            np.ndarray of shape (2, N).
        """
        return np.stack([d_traj[0][t, :], d_traj[1][t, :]])

    @staticmethod
    def th_traj_at_t(th_traj, t):
        """Extract the incentive vector at time step t from a trajectory.

        Args:
            th_traj: np.ndarray of shape (T, N) — incentive trajectory.
            t:       int — time index.

        Returns:
            np.ndarray of shape (N,).
        """
        return th_traj[t, :]

    @staticmethod
    def z_traj_at_t(z_traj, t):
        """Extract the extended state vector at time step t from a trajectory.

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — state trajectory.
            t:      int — time index.

        Returns:
            np.ndarray of shape (N+1,).
        """
        return z_traj[t, :]

    @staticmethod
    def p_traj_at_t(p_traj, t):
        """Extract the allocation vector at time step t from a trajectory.

        Args:
            p_traj: np.ndarray of shape (T, (N+1)*N) — allocation trajectory.
            t:      int — time index.

        Returns:
            np.ndarray of shape ((N+1)*N,).
        """
        return p_traj[t, :]

    @staticmethod
    def mu_traj_at_t(mu_traj, t):
        """Extract the dual variable vector at time step t from a trajectory.

        Args:
            mu_traj: np.ndarray of shape (T, 2*N) — dual variable trajectory.
            t:       int — time index.

        Returns:
            np.ndarray of shape (2*N,).
        """
        return mu_traj[t, :]

    @staticmethod
    def u_traj_at_t(u_traj, t):
        """Extract the control input vector at time step t from a trajectory.

        Args:
            u_traj: np.ndarray of shape (T, N+1) — control input trajectory.
            t:      int — time index.

        Returns:
            np.ndarray of shape (N+1,).
        """
        return u_traj[t, :]

    # ── Derived trajectories ──────────────────────────────────────────────────

    @staticmethod
    def get_s_traj(z_traj, params):
        """Compute the saturation trajectory: normalised PU demand vs. per-node state.

        s = (d - x) / (d + x), bounded in [-1, 1]:
          - s -> -1: d -> 0, maximum leverage (demand negligible next to state).
          - s == 0:  d == x, borderline saturation.
          - s -> 1:  d >> x, maximum overshoot (demand swamps state).

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — extended state trajectory.
            params: Params — provides N, T, d_traj, DO, PU.

        Returns:
            s_traj: np.ndarray of shape (T, N) — (d_pu - x) / (d_pu + x)
                    elementwise, 0 wherever both d_pu and x are 0.
        """
        T = params.T
        N = params.N
        s_traj = np.zeros((T, N))
        for t in range(T):
            x_t, _    = Transformer.split_z(Transformer.z_traj_at_t(z_traj, t), params)
            d_t       = Transformer.d_traj_at_t(params.d_traj, t)
            _, d_pu_t = Transformer.split_d(d_t, params)
            denom = d_pu_t + x_t
            s_traj[t, :] = np.where(denom == 0, 0.0, (d_pu_t - x_t) / np.where(denom == 0, 1, denom))

        return s_traj

    @staticmethod
    def get_o_traj(z_traj, params):
        """Compute the overshoot trajectory: PU demand exceeding per-node state.

        o = max(0, d - x).

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — extended state trajectory.
            params: Params — provides N, T, d_traj, DO, PU.

        Returns:
            o_traj: np.ndarray of shape (T, N) — max(0, d_pu - x) elementwise.
        """
        T = params.T
        N = params.N
        o_traj = np.zeros((T, N))
        for t in range(T):
            x_t, _    = Transformer.split_z(Transformer.z_traj_at_t(z_traj, t), params)
            d_t       = Transformer.d_traj_at_t(params.d_traj, t)
            _, d_pu_t = Transformer.split_d(d_t, params)
            o_traj[t, :] = np.maximum(0, d_pu_t - x_t)

        return o_traj

    @staticmethod
    def get_l_traj(z_traj, params):
        """Compute the leverage trajectory: per-node state exceeding PU demand.

        l = max(0, x - d).

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — extended state trajectory.
            params: Params — provides N, T, d_traj, DO, PU.

        Returns:
            l_traj: np.ndarray of shape (T, N) — max(0, x - d_pu) elementwise.
        """
        T = params.T
        N = params.N
        l_traj = np.zeros((T, N))
        for t in range(T):
            x_t, _    = Transformer.split_z(Transformer.z_traj_at_t(z_traj, t), params)
            d_t       = Transformer.d_traj_at_t(params.d_traj, t)
            _, d_pu_t = Transformer.split_d(d_t, params)
            l_traj[t, :] = np.maximum(0, x_t - d_pu_t)

        return l_traj

    # ── Trajectory aggregation ────────────────────────────────────────────────

    @staticmethod
    def agg_z_traj(z_traj, params):
        """Compute the total fleet size and outside state at each time step.

        Args:
            z_traj: np.ndarray of shape (T+1, N+1) — extended state trajectory.
            params: Params — provides N.

        Returns:
            x_agg_traj: np.ndarray of shape (T+1,) — sum of per-node state at each t.
            x_le_traj:  np.ndarray of shape (T+1,) — outside state at each t.
        """
        x_agg_traj = z_traj[:, :params.N].sum(axis=1)
        x_le_traj  = z_traj[:, params.N]
        return x_agg_traj, x_le_traj

    @staticmethod
    def agg_d_traj(d_traj):
        """Compute the total demand at each time step for each side.

        Args:
            d_traj: list [d_do (T, N), d_pu (T, N)] — demand trajectory.

        Returns:
            [d_do_agg_traj, d_pu_agg_traj]: list of two np.ndarray of shape (T,).
        """
        return [d_traj[0].sum(axis=1), d_traj[1].sum(axis=1)]

    @staticmethod
    def agg_th_traj(th_traj):
        """Compute the total incentive at each time step.

        Args:
            th_traj: np.ndarray of shape (T, N) — incentive trajectory.

        Returns:
            th_agg_traj: np.ndarray of shape (T,).
        """
        return th_traj.sum(axis=1)

    @staticmethod
    def agg_b_traj(b_traj):
        """Compute the total capacity bound at each time step.

        Args:
            b_traj: np.ndarray of shape (T, N) — capacity bound trajectory.

        Returns:
            b_agg_traj: np.ndarray of shape (T,).
        """
        return b_traj.sum(axis=1)

    @staticmethod
    def agg_s_traj(s_traj):
        """Compute the total saturation at each time step.

        Args:
            s_traj: np.ndarray of shape (T, N) — saturation trajectory.

        Returns:
            s_agg_traj: np.ndarray of shape (T,).
        """
        return s_traj.sum(axis=1)

    @staticmethod
    def agg_o_traj(o_traj):
        """Compute the total overshoot at each time step.

        Args:
            o_traj: np.ndarray of shape (T, N) — overshoot trajectory.

        Returns:
            o_agg_traj: np.ndarray of shape (T,).
        """
        return o_traj.sum(axis=1)

    @staticmethod
    def agg_l_traj(l_traj):
        """Compute the total leverage at each time step.

        Args:
            l_traj: np.ndarray of shape (T, N) — leverage trajectory.

        Returns:
            l_agg_traj: np.ndarray of shape (T,).
        """
        return l_traj.sum(axis=1)

    # ── Horizon-level diagnostics ─────────────────────────────────────────────

    @staticmethod
    def get_total_u_le(u_traj, params):
        """Sum the outside-state flow u_le over the entire time horizon.

        Args:
            u_traj: np.ndarray of shape (T, N+1) — control input trajectory.
            params: Params — provides Xi_x, Xi_le, dt.

        Returns:
            float — total outside-state flow over all T steps, in users.
        """
        total_u_le = 0.0
        for t in range(len(u_traj)):
            u_t         = Transformer.u_traj_at_t(u_traj, t)
            _, u_le_t   = Transformer.split_u(u_t, params)
            total_u_le += u_le_t

        return total_u_le * params.dt
