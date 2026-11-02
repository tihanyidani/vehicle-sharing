"""Naive baseline approaches: base class plus Trigger and Proportional variants."""

import numpy as np
from tqdm import tqdm

from utils.T_map import T_map
from utils.Transformer import Transformer


class Naive:
    """Base class for naive incentive policies driven off the current state and demand."""

    def __init__(self, params):
        self.params = params
        self.T_map  = T_map(params)

        # ── Dimensions ────────────────────────────────────────────────────────
        self.N = params.N
        self.T = params.T

        # ── Trajectory data ───────────────────────────────────────────────────
        self.z0     = params.z0
        self.d_traj = params.d_traj

        # ── Incentive bounds ──────────────────────────────────────────────────
        self.th_box = params.th_box

    def apply_step(self, z, d):
        """Compute the incentive vector for the current state and demand.

        Args:
            z: np.ndarray of shape (N+1,) — extended state.
            d: np.ndarray of shape (2, N) — demand at the current time step.

        Returns:
            th: np.ndarray of shape (N,) — incentive vector.
        """
        raise NotImplementedError

    def apply(self):
        """Roll out the system from z0, applying the policy at each time step.

        At each step t, th_t is generated from the current state z_t and
        demand d_t via apply_step, and the next state is generated via T_map.apply.

        Returns:
            th_traj: np.ndarray of shape (T, N) — incentive trajectory.
        """
        th_traj = np.zeros((self.T, self.N))

        z_t       = self.z0
        t_counter = tqdm(range(self.T), desc=f"{type(self).__name__}.apply")
        for t in t_counter:
            d_t  = Transformer.d_traj_at_t(self.d_traj, t)
            th_t = self.apply_step(z_t, d_t)

            th_traj[t, :]   = th_t
            z_next_t, _, _, _ = self.T_map.apply(z_t, th_t, d_t)
            z_t             = z_next_t

        return th_traj


class Trigger(Naive):
    """Trigger."""

    def __init__(self, params):
        super().__init__(params)

        # ── Trigger settings ─────────────────────────────────────────────────
        self.trigger_thr = params.trigger_thr
        self.trigger_val = params.trigger_val

    def apply_step(self, z, d):
        """Set the incentive to trigger_val at nodes whose state is above the trigger threshold.

        Args:
            z: np.ndarray of shape (N+1,) — extended state.
            d: np.ndarray of shape (2, N) — demand at the current time step (unused).

        Returns:
            th: np.ndarray of shape (N,) — trigger_val where x > trigger_thr, 0 otherwise,
                clipped to [0, th_box].
        """
        x, _ = Transformer.split_z(z, self.params)
        th = np.where(x > self.trigger_thr, self.trigger_val, 0.0)
        return np.clip(th, 0, self.th_box)


class Proportional(Naive):
    """Proportional."""

    def __init__(self, params):
        super().__init__(params)

        # ── Proportional settings ────────────────────────────────────────────
        self.proportional_P = params.proportional_P

    def apply_step(self, z, d):
        """Compute incentives proportional to unmet pick-up demand.

        Args:
            z: np.ndarray of shape (N+1,) — extended state.
            d: np.ndarray of shape (2, N) — demand at the current time step.

        Returns:
            th: np.ndarray of shape (N,) — proportional_P * error where
                error = x - d_pu is positive, 0 otherwise, clipped to [0, th_box].
        """
        x, _    = Transformer.split_z(z, self.params)
        _, d_pu = Transformer.split_d(d, self.params)
        error   = x - d_pu
        th = np.where(error > 0, self.proportional_P * error, 0.0)
        return np.clip(th, 0, self.th_box)
