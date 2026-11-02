"""System parameters for the incentive-based micromobility rebalancing problem."""

import numpy as np


class Params:
    """Centralised parameter store.

    Defines the grid topology, demand signal, cost weights, and
    optimiser hyperparameters shared across the codebase.

    The PU side uses an extended (N+1)-dimensional state to model agents
    arriving from outside the system, governed by a baseline
    willingness-to-travel w0 that is not incentive-sensitive (Lam row = 0).
    """

    def __init__(self, cfg, map, demand):
        # cfg is the merged scenario dict from Interface.load_yaml: dataset-level
        # keys such as "dt"/"T" may originate from a shared metaparams.yaml
        # (via the scenario's "metadata" key) rather than the scenario file itself.
        # ── Side indices ──────────────────────────────────────────────────────
        self.DO = 0
        self.PU = 1

        # ── Grid topology (from the Map instance) ───────────────────────────────
        self.map    = map
        self.N      = map.N

        self.points = map.points  # (N, 2)
        MDist       = map.MDist   # (N, N)

        # ── User decision model ───────────────────────────────────────────────
        lam_joint  = cfg["lam_joint"]
        w_le_joint = cfg["w_le_joint"]
        tau        = cfg["tau"]

        Lam_pu = lam_joint * np.ones((self.N, self.N))
        W_pu   = MDist
        w_le   = w_le_joint * np.ones(self.N)

        self.Lam  = Lam_pu / tau
        self.W    = W_pu / tau
        self.w_le = w_le / tau

        # ── Dynamics ──────────────────────────────────────────────────────────
        self.dt    = cfg["dt"]
        self.T     = cfg["T"]

        self.Xi_x  = np.hstack((np.eye(self.N), np.zeros((self.N, 1))))
        self.Xi_le = np.hstack((np.zeros((1, self.N)), np.ones((1, 1))))

        self.A = np.eye(self.N)

        self.B_pu = -self.dt * self.Xi_x
        self.B_do =  self.dt * np.eye(self.N)

        self.C = np.zeros((1, self.N))

        self.D_pu = self.dt * self.Xi_le
        self.D_do = np.zeros((1, self.N))

        self.Phi = np.block([
            [self.A, np.zeros((self.N, 1))],
            [self.C, np.zeros((1, 1))],
        ])

        self.Gamma_pu = np.vstack([self.B_pu, self.D_pu])
        self.Gamma_do = np.vstack([self.B_do, self.D_do])

        # ── Optimiser hyperparameters ─────────────────────────────────────────
        self.phi     = cfg["phi"]
        self.maxIt   = cfg["maxIt"]
        self.err_thr = cfg["err_thr"]
        self.alph       = cfg["alph"]
        self.alph_decay = cfg["alph_decay"]
        self.lp_tol     = cfg.get("lp_tol", 1e-8)         # inner LP solver tolerance
        self.solver     = cfg.get("solver", "CLARABEL")   # inner LP solver: "CLARABEL" or "SCS"

        # ── Projector parameters ──────────────────────────────────────────────
        self.th_box         = cfg["th_box"]
        self.projector_type = cfg.get("projector_type", "box")   # "box" or "budget"
        self.th_budget      = cfg.get("th_budget")   # only required when projector_type == "budget"

        # ── Naive approach parameters (only used by utils/Naive.py) ────────────
        self.trigger_thr = cfg.get("trigger_thr")
        self.trigger_val = cfg.get("trigger_val")
        self.proportional_P = cfg.get("proportional_P")

        # ── Demand trajectory (from the Demand instance) ───────────────────────
        assert demand.N == self.N, "Demand region count doesn't match Map region count"
        assert demand.T == self.T, "Demand time steps don't match Params T"
        self.demand = demand
        self.d_traj = demand.d_traj

        # ── Initial conditions ────────────────────────────────────────────────
        self.x0 = self.resolve_state(cfg["x0"], self.N)
        self.z0 = np.append(self.x0, 0)

        # ── Cost weights ──────────────────────────────────────────────────────
        self.gamma_R = cfg["gamma_R"]
        self.gamma_Q = cfg["gamma_Q"]

        # ── Cost model selection ──────────────────────────────────────────────
        self.g_map_type = cfg.get("g_map_type", "leaving")  # "leaving" or "tracking"
        self.init_x_des(cfg)

        # ── Initial incentive sequence ────────────────────────────────────────
        self.th0_traj = np.zeros((self.T, self.N))

    def init_x_des(self, cfg):
        """Set self.x_des, the desired-state vector used by the tracking cost.

        Only required when g_map_type is "tracking". cfg["x_des"] is either
        a scalar (uniform across all N regions) or a list of N values.
        """
        if self.g_map_type != "tracking":
            return

        if "x_des" not in cfg:
            raise ValueError('g_map_type is "tracking" but cfg["x_des"] is missing')

        self.x_des = self.resolve_state(cfg["x_des"], self.N)

    @staticmethod
    def resolve_state(state_cfg, N):
        """Resolve an x0/xT_des config entry into an (N,) state vector.

        state_cfg is one of:
            scalar              -> uniform value, broadcast to all N regions.
            list of N values    -> explicit per-region values.

        Args:
            state_cfg: scalar or list.
            N:         int — number of regions.

        Returns:
            np.ndarray of shape (N,).
        """
        if np.isscalar(state_cfg):
            return state_cfg * np.ones(N)
        else:
            return np.array(state_cfg)