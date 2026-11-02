"""Scenario sweeps over a single parameter (th0, demand, or budget)."""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils.Map       import Map
from utils.Demand    import Demand
from utils.Params    import Params
from utils.Optimizer import ForwardBackward
from utils.Interface import Interface
from utils.Plotter   import Plotter


class MainIterate:
    """Base class for sweeping a scenario over multiple values of one parameter.

    Subclasses define the concrete sweep behaviour for each parameter kind.
    Use MainIterate.create(cfg) to instantiate the variant selected by
    cfg["iterate_type"].
    """

    def __init__(self, cfg):
        self.cfg           = cfg
        self.iterates       = cfg["iterates"]
        self.iterate_type   = cfg.get("iterate_type")  # unused when iterates is False

    @staticmethod
    def create(cfg):
        """Instantiate the MainIterate variant selected by cfg["iterates"]/cfg["iterate_type"].

        When cfg["iterates"] is False, x0/th0/demand are each held to their
        single configured value and MainIterate_single runs the one scenario,
        regardless of what cfg["iterate_type"] is set to.
        """
        if not cfg["iterates"]:
            return MainIterate_single(cfg)

        iterate_type = cfg["iterate_type"]
        if iterate_type == "th0":
            return MainIterate_th0(cfg)
        elif iterate_type == "demand":
            return MainIterate_demand(cfg)
        elif iterate_type == "budget":
            return MainIterate_budget(cfg)
        else:
            raise ValueError(f"Unknown iterate_type: {iterate_type!r}")

    @staticmethod
    def expand_dates(dir, start_date, n_scenarios):
        """Build the per-day file cfgs for a contiguous run of daily scenarios.

        Assumes the naming convention produced by dataset/proc_dataset.py:
        "{dir}/{date}_demand_dropoff.csv" / "{dir}/{date}_demand_pickup.csv",
        one pair per calendar day, dates running contiguously.

        Args:
            dir:         str — directory holding the daily demand CSVs.
            start_date:  str — first day to run, "YYYY-MM-DD".
            n_scenarios: int — number of consecutive days to run.

        Returns:
            list[dict] — length n_scenarios, each {"date", "dropoff_csv", "pickup_csv"}
            in calendar order, one entry per day to run as its own scenario.
        """
        start = date.fromisoformat(str(start_date))
        series = []
        for i in range(n_scenarios):
            d = (start + timedelta(days=i)).isoformat()
            series.append({
                "date":        d,
                "dropoff_csv": f"{dir}/{d}_demand_dropoff.csv",
                "pickup_csv":  f"{dir}/{d}_demand_pickup.csv",
            })
        return series

    def build_fixed_demand(self, map):
        """Build the single Demand instance shared across every run.

        Used by iterators whose swept parameter isn't "demand": cfg["demand"]
        is expected to resolve to exactly one day (n_scenarios: 1, or an
        explicit dropoff_csv/pickup_csv pair).
        """
        cfg = self.cfg
        demand_cfg = cfg["demand"]
        if "dropoff_csv" in demand_cfg:
            day_cfg = {**demand_cfg}
        else:
            day_cfg = {"type": "file", **self.expand_dates(demand_cfg["dir"], demand_cfg["start_date"], demand_cfg["n_scenarios"])[0]}
        day_cfg.pop("date", None)
        return Demand({**day_cfg, "N": map.N, "T": cfg["T"]})

    def iterate(self, experiment, cont=False):
        """Run every scenario in the sweep for this iterator's parameter.

        Args:
            experiment: str — experiment name; results go to results/{experiment}/.
            cont:       bool — if True, skip scenarios whose run.pkl already exists.
        """
        raise NotImplementedError

    @staticmethod
    def skip_done(experiment, tag, cont):
        """Return True (and say so) if cont is set and this scenario's results are already saved.

        run.pkl is the last file written for a scenario, so its presence means
        the scenario finished.
        """
        if cont and os.path.exists(f"results/{experiment}/{tag}/run.pkl"):
            print(f"[1_main] ── Skipping '{experiment}' / {tag} (already saved) ──")
            return True
        return False

    @staticmethod
    def save_conv_table(conv, results_dir):
        """Save the check_conv output (stationarity + KKT residuals) as a one-row CSV table.

        Args:
            conv:        dict — as returned by Optimizer.check_conv.
            results_dir: str — "results/{experiment}/{tag}" directory for this scenario.
        """
        kkt = conv["kkt"]
        conv_df = pd.DataFrame([{
            "stat_res":         conv["stat_res"],
            "stat_res_rel":     conv["stat_res_rel"],
            "kkt_res_stat":     kkt["res_stat"],
            "kkt_res_stat_rel": kkt["res_stat_rel"],
            "kkt_res_prim":     kkt["res_prim"],
            "kkt_res_dual":     kkt["res_dual"],
            "kkt_res_cs":       kkt["res_cs"],
        }])
        Interface.save_csv(conv_df, f"{results_dir}/tables/conv")


class MainIterate_single(MainIterate):
    """Runs a single scenario, with x0/th0/demand each held to one value."""

    def iterate(self, experiment, cont=False):
        cfg = self.cfg
        tag = "run"
        if self.skip_done(experiment, tag, cont):
            return
        print(f"\n[1_main] ── Running scenario '{experiment}' / {tag} ──")

        RESULTS_PATH = f"results/{experiment}/{tag}/run"
        FIG_DIR      = f"results/{experiment}/{tag}/figs"

        # ── Map topology, demand, and params all fixed for the one run ────────
        map    = Map(cfg["map"])
        demand = self.build_fixed_demand(map)
        params = Params(cfg, map, demand)

        th0_traj = np.tile(Params.resolve_state(cfg["th0"], params.N), (params.T, 1))

        optimizer = ForwardBackward(params)
        th_traj, th_traj_hist, cost_hist, res_hist = optimizer.apply(th0_traj)
        conv         = optimizer.check_conv(th_traj)
        stat_res     = conv["stat_res"]
        print(f"Stationarity residual: {stat_res:.6e}")
        kkt = conv["kkt"]
        print(f"KKT residuals: stat={kkt['res_stat']:.3e}  prim={kkt['res_prim']:.3e}  "
              f"dual={kkt['res_dual']:.3e}  cs={kkt['res_cs']:.3e}")
        self.save_conv_table(conv, f"results/{experiment}/{tag}")
        rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
        fig_convergence = Plotter.plot_convergence(rel_err_hist, abs_err_hist, cost_hist, show=True)
        Interface.save_figures({"convergence": fig_convergence}, dir_path=FIG_DIR)
        Interface.save({"th_traj": th_traj, "th_traj_hist": th_traj_hist, "cost_hist": cost_hist, "res_hist": res_hist, "conv": conv}, RESULTS_PATH)


class MainIterate_th0(MainIterate):
    def iterate(self, experiment, cont=False):
        """Run every scenario in the sweep for this iterator's parameter."""
        cfg = self.cfg

        th0_values = cfg["th0"]
        n_runs     = len(th0_values)

        # ── Map topology, demand, and params are fixed; only th0 varies ───────
        map       = Map(cfg["map"])
        demand    = self.build_fixed_demand(map)
        params    = Params(cfg, map, demand)
        optimizer = ForwardBackward(params)

        for i, th0 in enumerate(th0_values):
            tag = f"th0_{i}"
            if self.skip_done(experiment, tag, cont):
                continue
            print(f"\n[1_main] ── Running scenario '{experiment}' / {tag} ──")

            RESULTS_PATH = f"results/{experiment}/{tag}/run"
            FIG_DIR      = f"results/{experiment}/{tag}/figs"

            th0_traj = np.tile(Params.resolve_state(th0, params.N), (params.T, 1))

            th_traj, th_traj_hist, cost_hist, res_hist = optimizer.apply(th0_traj)
            conv         = optimizer.check_conv(th_traj)
            stat_res     = conv["stat_res"]
            print(f"Stationarity residual: {stat_res:.6e}")
            kkt = conv["kkt"]
            print(f"KKT residuals: stat={kkt['res_stat']:.3e}  prim={kkt['res_prim']:.3e}  "
                  f"dual={kkt['res_dual']:.3e}  cs={kkt['res_cs']:.3e}")
            self.save_conv_table(conv, f"results/{experiment}/{tag}")
            rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
            fig_convergence = Plotter.plot_convergence(rel_err_hist, abs_err_hist, cost_hist, show=(n_runs == 1))
            Interface.save_figures({"convergence": fig_convergence}, dir_path=FIG_DIR)
            Interface.save({"th_traj": th_traj, "th_traj_hist": th_traj_hist, "cost_hist": cost_hist, "res_hist": res_hist, "conv": conv}, RESULTS_PATH)


class MainIterate_budget(MainIterate):
    def iterate(self, experiment, cont=False):
        """Run every scenario in the sweep for this iterator's parameter.

        th_budget varies feeds the Projector at ForwardBackward-construction
        time (unlike th0), so Params and the optimizer are rebuilt fresh for
        each budget value, with x0/th0/demand held fixed.
        """
        cfg = self.cfg

        budget_values = cfg["th_budget"]
        n_runs        = len(budget_values)

        # ── Map topology and demand are fixed; only th_budget varies ──────────
        map    = Map(cfg["map"])
        demand = self.build_fixed_demand(map)

        for i, th_budget in enumerate(budget_values):
            tag = f"budget_{th_budget}"
            if self.skip_done(experiment, tag, cont):
                continue
            print(f"\n[1_main] ── Running scenario '{experiment}' / {tag} ──")

            RESULTS_PATH = f"results/{experiment}/{tag}/run"
            FIG_DIR      = f"results/{experiment}/{tag}/figs"

            params    = Params({**cfg, "th_budget": th_budget}, map, demand)
            optimizer = ForwardBackward(params)

            th0_traj = np.tile(Params.resolve_state(cfg["th0"], params.N), (params.T, 1))

            th_traj, th_traj_hist, cost_hist, res_hist = optimizer.apply(th0_traj)
            conv         = optimizer.check_conv(th_traj)
            stat_res     = conv["stat_res"]
            print(f"Stationarity residual: {stat_res:.6e}")
            kkt = conv["kkt"]
            print(f"KKT residuals: stat={kkt['res_stat']:.3e}  prim={kkt['res_prim']:.3e}  "
                  f"dual={kkt['res_dual']:.3e}  cs={kkt['res_cs']:.3e}")
            self.save_conv_table(conv, f"results/{experiment}/{tag}")
            rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
            fig_convergence = Plotter.plot_convergence(rel_err_hist, abs_err_hist, cost_hist, show=(n_runs == 1))
            Interface.save_figures({"convergence": fig_convergence}, dir_path=FIG_DIR)
            Interface.save({"th_traj": th_traj, "th_traj_hist": th_traj_hist, "cost_hist": cost_hist, "res_hist": res_hist, "conv": conv}, RESULTS_PATH)


class MainIterate_demand(MainIterate):
    def iterate(self, experiment, cont=False):
        """Run every scenario in the sweep for this iterator's parameter."""
        cfg = self.cfg

        # ── Expand the demand block into one file-cfg per day to run ──────────
        demand_cfg = cfg["demand"]
        if "dropoff_csv" in demand_cfg:
            day_cfgs = [{**demand_cfg, "date": None}]
        else:
            day_cfgs = [
                {"type": "file", **day}
                for day in self.expand_dates(demand_cfg["dir"], demand_cfg["start_date"], demand_cfg["n_scenarios"])
            ]
        n_days = len(day_cfgs)

        # ── Run each day as its own, independent scenario ─────────────────────
        map = Map(cfg["map"])   # shared topology, same across all days
        for day_cfg in day_cfgs:
            date = day_cfg.pop("date")
            tag  = date or "run"
            if self.skip_done(experiment, tag, cont):
                continue
            print(f"\n[1_main] ── Running scenario '{experiment}' / {tag} ──")

            RESULTS_PATH = f"results/{experiment}/{tag}/run"
            FIG_DIR      = f"results/{experiment}/{tag}/figs"

            demand = Demand({**day_cfg, "N": map.N, "T": cfg["T"]})
            params = Params(cfg, map, demand)

            optimizer = ForwardBackward(params)
            th_traj, th_traj_hist, cost_hist, res_hist = optimizer.apply(params.th0_traj)
            conv         = optimizer.check_conv(th_traj)
            stat_res     = conv["stat_res"]
            print(f"Stationarity residual: {stat_res:.6e}")
            kkt = conv["kkt"]
            print(f"KKT residuals: stat={kkt['res_stat']:.3e}  prim={kkt['res_prim']:.3e}  "
                  f"dual={kkt['res_dual']:.3e}  cs={kkt['res_cs']:.3e}")
            self.save_conv_table(conv, f"results/{experiment}/{tag}")
            rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
            fig_convergence = Plotter.plot_convergence(rel_err_hist, abs_err_hist, cost_hist, show=(n_days == 1))
            Interface.save_figures({"convergence": fig_convergence}, dir_path=FIG_DIR)
            Interface.save({"th_traj": th_traj, "th_traj_hist": th_traj_hist, "cost_hist": cost_hist, "res_hist": res_hist, "conv": conv}, RESULTS_PATH)
