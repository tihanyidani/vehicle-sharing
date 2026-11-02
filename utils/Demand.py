"""Demand utilities: synthetic and file-based DO/PU demand trajectories."""

import numpy as np
import pandas as pd


class Demand:
    """Drop-off (DO) / pick-up (PU) demand trajectories.

    Built from a cfg dict, similar to Map. cfg["type"] selects the source:

        {"type": "synthetic", "N": <int>, "T": <int>, "d_traj": <scalar or list>}
            Synthetic demand, built the same way Params previously did it
            in-line. d_traj is either a scalar (uniform across both sides,
            all nodes, all time steps) or a list of sparse entries:
                {side: DO|PU, t_start: int, t_end: int, node: int|null, value: float}
            node: null means all nodes; t_end is exclusive.

        {"type": "file", "dropoff_csv": <str>, "pickup_csv": <str>}
            Real demand from the demand CSVs produced by 0_pre_proc.py
            (rows = 15-min time clusters, columns = region ids). dropoff_csv
            -> DO, pickup_csv -> PU. N = number of region columns, T = number
            of time-cluster rows. Both files must share the same region
            columns, in the same order.

            A scenario file spanning several days ("dir"/"start_date"/
            "n_scenarios" instead of "dropoff_csv"/"pickup_csv") is expanded
            into one such per-day cfg per day by MainIterate_demand — see
            utils/MainIterate.py.
    """

    DO = 0
    PU = 1

    def __init__(self, cfg):
        demand_type = cfg["type"]
        if demand_type == "synthetic":
            index, N, T, d_traj = self._build_synthetic(cfg["N"], cfg["T"], cfg.get("d_traj", []))
        elif demand_type == "file":
            index, N, T, d_traj = self._build_from_file(cfg["dropoff_csv"], cfg["pickup_csv"])
        else:
            raise ValueError(f"Unknown demand type: {demand_type!r}")

        self.index  = index    # length-N list of region indices/ids
        self.N      = N
        self.T      = T
        self.d_traj = d_traj   # [d_do_traj, d_pu_traj], each (T, N)

    @classmethod
    def _build_synthetic(cls, N, T, d_traj_cfg):
        d_do_traj = np.zeros((T, N))
        d_pu_traj = np.zeros((T, N))
        if np.isscalar(d_traj_cfg):
            d_do_traj[:] = d_traj_cfg
            d_pu_traj[:] = d_traj_cfg
        else:
            side_arrays = [d_do_traj, d_pu_traj]
            side_map    = {"DO": cls.DO, "PU": cls.PU}
            for entry in d_traj_cfg:
                arr  = side_arrays[side_map[entry["side"]]]
                node = entry.get("node")       # None → all nodes
                idx  = slice(None) if node is None else node
                arr[entry["t_start"]:entry["t_end"], idx] = entry["value"]

        index = list(range(N))
        return index, N, T, [d_do_traj, d_pu_traj]

    @staticmethod
    def _build_from_file(dropoff_csv, pickup_csv):
        dropoff = pd.read_csv(dropoff_csv, index_col=0)
        pickup  = pd.read_csv(pickup_csv, index_col=0)

        if list(dropoff.columns) != list(pickup.columns):
            raise ValueError("dropoff_csv and pickup_csv must have matching region columns")

        index = list(dropoff.columns.astype(int))
        N     = len(index)
        T     = len(dropoff)

        d_do_traj = dropoff.to_numpy(dtype=float)
        d_pu_traj = pickup.to_numpy(dtype=float)
        return index, N, T, [d_do_traj, d_pu_traj]
