"""File I/O utilities for saving and loading results and animations."""

import os
import pickle
import yaml
from matplotlib.animation import PillowWriter


class Interface:
    """Static methods for persisting data to disk.

    Supports:
    - Generic pickle save/load for any Python object.
    - Convenience bundling for the standard optimisation outputs.
    - GIF export for one or more FuncAnimation objects.
    """

    # ── YAML I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def load_yaml(path):
        """Load a YAML scenario file and return its contents as a dict.

        If the scenario has a top-level "metadata" key, it's treated as a
        path to a shared metadata YAML (e.g. dataset/procdata/metaparams.yaml)
        holding dataset-derived keys such as "map", "dt", "T". That file is
        loaded and merged in as the base, with the scenario's own keys taking
        precedence on conflicts.

        Args:
            path: str — path to the .yaml file.

        Returns:
            dict — parsed YAML contents.
        """
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"[Interface] Scenario loaded from '{path}'")

        if "metadata" in cfg:
            meta_path = cfg.pop("metadata")
            with open(meta_path, "r") as f:
                meta_cfg = yaml.safe_load(f)
            print(f"[Interface] Metadata loaded from '{meta_path}'")
            cfg = {**meta_cfg, **cfg}

        return cfg

    # ── Pickle I/O ────────────────────────────────────────────────────────────

    @staticmethod
    def save(data, path):
        """Save any Python object to a pickle file.

        A .pkl extension is appended automatically if absent.
        The destination directory is created if it does not exist.

        Args:
            data: any serialisable Python object.
            path: str — destination file path.
        """
        if not path.endswith(".pkl"):
            path += ".pkl"

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"[Interface] Saved to '{path}'")

    @staticmethod
    def load(path):
        """Load a Python object from a pickle file.

        Args:
            path: str — source file path.

        Returns:
            The deserialised Python object.

        Raises:
            FileNotFoundError: if the file does not exist.
        """
        if not path.endswith(".pkl"):
            path += ".pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(f"[Interface] No file found at '{path}'")

        with open(path, "rb") as f:
            data = pickle.load(f)

        print(f"[Interface] Loaded from '{path}'")
        return data

    # ── Optimisation run bundle ───────────────────────────────────────────────

    @staticmethod
    def save_run(th_traj, z_traj, err_traj, path):
        """Bundle and save the standard optimisation outputs together.

        Args:
            th_traj:  np.ndarray of shape (T, N)      — optimised incentive sequence.
            z_traj:   np.ndarray of shape (T+1, N+1)  — simulated state trajectory.
            err_traj: np.ndarray of shape (maxIt,)    — gradient norm per iteration.
            path:     str — destination file path.
        """
        bundle = {
            "th_traj": th_traj,
            "z_traj":  z_traj,
            "err_traj": err_traj,
        }
        Interface.save(bundle, path)

    @staticmethod
    def load_run(path):
        """Load a bundle previously saved by save_run().

        Args:
            path: str — source file path.

        Returns:
            (th_traj, z_traj, err_traj): tuple of np.ndarray.
        """
        bundle = Interface.load(path)
        return bundle["th_traj"], bundle["z_traj"], bundle["err_traj"]

    # ── GIF export ────────────────────────────────────────────────────────────

    @staticmethod
    def save_animation(anim, path, fps=5):
        """Save a single FuncAnimation to a GIF file.

        Args:
            anim: FuncAnimation — animation to export.
            path: str — destination file path. A .gif extension is appended if absent.
            fps:  int — frames per second (default 5, matching 200 ms interval).
        """
        if not path.endswith(".gif"):
            path += ".gif"

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        anim.save(path, writer=PillowWriter(fps=fps))
        print(f"[Interface] Animation saved to '{path}'")

    @staticmethod
    def save_figure(fig, path):
        """Save a matplotlib Figure to a PDF (vector graphics) file.

        Args:
            fig:  matplotlib.figure.Figure — figure to export.
            path: str — destination file path. A .pdf extension is appended if absent.
        """
        if not path.endswith(".pdf"):
            path += ".pdf"

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fig.savefig(path)
        print(f"[Interface] Figure saved to '{path}'")

    @staticmethod
    def save_csv(df, path):
        """Save a pandas DataFrame to a CSV file.

        Args:
            df:   pandas.DataFrame — table to export.
            path: str — destination file path. A .csv extension is appended if absent.
        """
        if not path.endswith(".csv"):
            path += ".csv"

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        df.to_csv(path)
        print(f"[Interface] Table saved to '{path}'")

    @staticmethod
    def save_table(fig, path):
        """Save an annotated table figure (e.g. from Plotter.gen_tab_u_le_comp) to a PDF file.

        Args:
            fig:  matplotlib.figure.Figure — table figure to export.
            path: str — destination file path. A .pdf extension is appended if absent.
        """
        Interface.save_figure(fig, path)

    @staticmethod
    def save_figures(fig_dict, dir_path):
        """Save multiple Figures to PDF files in one call.

        Args:
            fig_dict: dict[str, Figure] — maps filename stem to figure.
            dir_path: str — directory to write all PDFs into.
        """
        for name, fig in fig_dict.items():
            Interface.save_figure(fig, os.path.join(dir_path, name))

    @staticmethod
    def save_animations(anim_dict, dir_path, fps=5):
        """Save multiple FuncAnimations to GIF files in one call.

        Args:
            anim_dict: dict[str, FuncAnimation] — maps filename stem to animation,
                       e.g. {"state": anim_x, "demand_pu": anim_d_pu}.
            dir_path:  str — directory to write all GIFs into.
            fps:       int — frames per second (default 5).
        """
        for name, anim in anim_dict.items():
            Interface.save_animation(anim, os.path.join(dir_path, name), fps=fps)