"""Forward simulator for the system state trajectory."""

import numpy as np
from tqdm import tqdm

from utils.T_map import T_map
from utils.Transformer import Transformer


class Simulator:
    """Simulates the system forward in time under a fixed incentive sequence.

    Applies T_map recursively from z0, returning the full extended
    state trajectory z_traj of shape (T+1, N+1) and the allocation
    trajectory p_traj of shape (T, N*(N+1)).
    """

    def __init__(self, params):
        # ── Sub-maps ──────────────────────────────────────────────────────────
        self.T_map  = T_map(params)
        self.params = params

        # ── Dimensions ────────────────────────────────────────────────────────
        self.N   = params.N
        self.N_p = params.N * (params.N + 1)  # allocation vector dimension
        self.T   = params.T

        # ── Trajectory data ───────────────────────────────────────────────────
        self.z0 = params.z0   # (N+1,) extended initial state
        self.d_traj = params.d_traj   # list [d_do (T,N), d_pu (T,N)]

    # ── Simulation ────────────────────────────────────────────────────────────

    def run_open_loop(self, th_traj):
        """Run the forward simulation under incentive sequence th_traj.

        Args:
            th_traj: np.ndarray of shape (T, N) — incentive parameters per time step.

        Returns:
            z_traj:     np.ndarray of shape (T+1, N+1)   — extended state trajectory.
                        z_traj[0] = z0;
                        z_traj[t+1] = T_map(z_traj[t], th_traj[t], d_t).
            p_traj:     np.ndarray of shape (T, N*(N+1))   — allocation trajectory.
                        p_traj[t] = p_star at step t.
            b_traj:     np.ndarray of shape (T, N)         — capacity bound trajectory.
                        b_traj[t] = get_b(z_t) at step t.
            u_traj:     np.ndarray of shape (T, N+1)       — control input trajectory.
                        u_traj[t] = u at step t.
        """
        z_traj       = np.zeros((self.T + 1, self.N + 1))
        z_traj[0, :] = self.z0
        p_traj       = np.zeros((self.T, self.N_p))
        b_iq_traj    = np.zeros((self.T, self.N))
        u_traj       = np.zeros((self.T, self.N + 1))

        z_t       = self.z0
        t_counter = tqdm(range(self.T), desc="run")
        for t in t_counter:
            th_t = Transformer.th_traj_at_t(th_traj, t)
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)

            b_iq_traj[t, :]             = self.T_map.get_b_iq(z_t)
            z_next_t, p_next_t, _, u_t  = self.T_map.apply(z_t, th_t, d_t)
            z_traj[t + 1, :]            = z_next_t
            p_traj[t, :]                = p_next_t
            u_traj[t, :]                = u_t
            z_t                         = z_next_t

        return z_traj, p_traj, b_iq_traj, u_traj