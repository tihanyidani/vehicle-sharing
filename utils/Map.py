"""Region map utilities: grid and file-based topologies."""

import numpy as np
import pandas as pd


class Map:
    """Region topology: count, indices, coordinates, and pairwise distances.

    Built from a cfg dict, similar to Params. cfg["type"] selects the
    topology:

        {"type": "grid", "n": <int>, "m": <int>}
            Synthetic n x m grid.

        {"type": "file", "csv_path": <str>}
            Real region layout from a metadata CSV (same column structure
            as dataset/procdata/metadata.csv, built by pre_proc.py: id, x, y,
            x2, y2, n_stations). Only id/x2/y2 are read. x2/y2 are the
            projected (metric) coordinates (EPSG:3879, ETRS-GK25FIN) and are
            used for the distance matrix; x/y (WGS84 lon/lat) are not.
            cfg["type"]/cfg["csv_path"] normally come from a scenario's
            "map" key, itself usually resolved from a shared
            dataset/procdata/metaparams.yaml via Interface.load_yaml's
            "metadata" indirection rather than being set inline per-scenario.
    """

    def __init__(self, cfg):
        map_type = cfg["type"]
        if map_type == "grid":
            index, points = self._build_grid(cfg["n"], cfg["m"])
            crs = None
        elif map_type == "file":
            index, points = self._build_from_file(cfg["csv_path"])
            crs = "EPSG:3879"
        else:
            raise ValueError(f"Unknown map type: {map_type!r}")

        self.index  = index                                   # length-N list of region indices/ids
        self.points = np.array(points)                        # (N, 2) coordinates
        self.N      = len(self.index)
        self.MDist  = self.points_to_distances(self.points)  # (N, N)
        self.crs    = crs                                      # CRS of points, or None if not geospatial

    @staticmethod
    def _build_grid(n, m):
        index  = list(range(n * m))
        points = [(i, j) for i in range(n) for j in range(m)]
        return index, points

    @staticmethod
    def _build_from_file(csv_path):
        meta   = pd.read_csv(csv_path)
        index  = list(meta["id"])
        points = list(zip(meta["x2"], meta["y2"]))
        return index, points

    @staticmethod
    def points_to_distances(points):
        """Compute the pairwise Euclidean distance matrix for a set of 2-D points.

        Args:
            points: np.ndarray of shape (N, 2).

        Returns:
            np.ndarray of shape (N, N).
        """
        points = np.asarray(points)
        return np.linalg.norm(points[:, None] - points[None, :], axis=2)
