"""Post-processing over a scenario sweep (th0, demand, or budget)."""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.Demand      import Demand
from utils.Params      import Params
from utils.Optimizer   import ForwardBackward, OneStep
from utils.Interface   import Interface
from utils.Plotter     import Plotter
from utils.Simulator   import Simulator
from utils.Transformer import Transformer
from utils.Naive       import Trigger, Proportional


class ProcIterate:
    """Base class for post-processing a scenario swept over multiple values of one parameter.

    Subclasses define the concrete post-processing behaviour for each parameter kind.
    Use ProcIterate.create(cfg) to instantiate the variant selected by
    cfg["iterate_type"].
    """

    def __init__(self, cfg):
        self.cfg           = cfg
        self.iterates       = cfg["iterates"]
        self.iterate_type   = cfg.get("iterate_type")  # unused when iterates is False

    @staticmethod
    def create(cfg):
        """Instantiate the ProcIterate variant selected by cfg["iterates"]/cfg["iterate_type"].

        When cfg["iterates"] is False, ProcIterate_single post-processes the
        one scenario run by MainIterate_single, regardless of what
        cfg["iterate_type"] is set to.
        """
        if not cfg["iterates"]:
            return ProcIterate_single(cfg)

        iterate_type = cfg["iterate_type"]
        if iterate_type == "th0":
            return ProcIterate_th0(cfg)
        elif iterate_type == "demand":
            return ProcIterate_demand(cfg)
        elif iterate_type == "budget":
            return ProcIterate_budget(cfg)
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

    def iterate(self, experiment, map, demand, params, naive=True, gifs=True, calc=True):
        """Post-process every scenario in the sweep for this iterator's parameter.

        naive: bool — forwarded to ProcIterate_single.iterate (see there). When
            False, each scenario's post-processing is restricted to the baseline
            (zero-incentive) trajectory. Ignored by variants that don't delegate
            to ProcIterate_single.
        gifs: bool — forwarded to ProcIterate_single.iterate (see there). When
            False, no .gif animations are rendered or saved. Ignored by variants
            that don't delegate to ProcIterate_single.
        calc: bool — when True, run the variant's calculate (the slow
            simulations) and pickle its output; when False, skip calculate and
            load that pickle instead, then plot from it either way.
        """
        raise NotImplementedError

    @staticmethod
    def _calculate_or_load(path, calc, calculate):
        """Run calculate() and pickle its output to path when calc is True; otherwise load the output a previous calc=True run saved there."""
        if calc:
            comp = calculate()
            Interface.save(comp, path)
        else:
            comp = Interface.load(path)
        return comp

    @staticmethod
    def _save_u_le_comp_across_tags(experiment, tags):
        """Lay each tagged run's u_le_comp.csv out side by side, grouped by tag.

        Reuses the per-tag u_le_comp.csv already saved by
        ProcIterate_single.iterate (zero-incentive/Algorithm 1/trigger/
        proportional/one-step rows in the full path, zero-incentive/Algorithm 1
        rows only when naive=False) and stacks the per-tag columns side by side. Shared by
        every variant that delegates per-scenario post-processing to
        ProcIterate_single (e.g. ProcIterate_demand, ProcIterate_budget).
        """
        u_le_comp_cols = {}
        for tag in tags:
            tag_u_le_df = pd.read_csv(f"results/{experiment}/{tag}/tables/u_le_comp.csv").set_index("Algorithms")
            u_le_comp_cols[(tag, "Leaving users")]           = tag_u_le_df["Leaving users"]
            u_le_comp_cols[(tag, "Reduction")]               = tag_u_le_df["Reduction"]
            u_le_comp_cols[(tag, "Incentive usage")]         = tag_u_le_df["Incentive usage"]
            u_le_comp_cols[(tag, "Reduction per incentive")] = tag_u_le_df["Reduction per incentive"]

        u_le_comp_df = pd.DataFrame(u_le_comp_cols)
        Interface.save_csv(u_le_comp_df, f"results/{experiment}/comp_results/tables/u_le_comp")


class ProcIterate_single(ProcIterate):
    def iterate(self, experiment, map, demand, params, tag="run", naive=True, gifs=True, calc=True):
        """Post-process the single scenario run by MainIterate_single.

        Runs the naive baselines (zero incentives, trigger, proportional), the
        one-step myopic solver, and loads the optimized (bi-level
        ForwardBackward) incentives saved by MainIterate_single, one by one,
        simulating and plotting each. The work is split into calculate (the
        slow simulations/solvers) followed by plot (comparison figure and
        u_le_comp table); calculate's output is pickled in between to
        results/{experiment}/{tag}/comp.pkl (comp_simple.pkl when naive=False).

        Args:
            tag: str — sweep tag identifying which run to post-process, matching
                 the tag MainIterate used to save it (e.g. "run" for a plain
                 single scenario, or a date when reused per-day by
                 ProcIterate_demand). Defaults to "run".
            naive: bool — when True (default), post-process every strategy
                 (zero-incentive, Algorithm 1, trigger, proportional, one-step).
                 When False, post-process only the zero-incentive and
                 Algorithm 1 trajectories, still emitting the corresponding
                 plot_agg_comp figure and u_le_comp table (those two rows only).
            gifs: bool — when True (default), render and save the per-strategy
                 combined incentive/leaving-users .gif animation. Set False to
                 skip the (slow) animations entirely and only produce the
                 comparison figures/tables.
            calc: bool — when True (default), run calculate and save its output.
                 When False, skip calculate and load the output saved by a
                 previous calc=True run with the same naive setting (no gifs
                 are rendered in that case).
        """
        comp_name = "comp" if naive else "comp_simple"
        comp = self._calculate_or_load(
            f"results/{experiment}/{tag}/{comp_name}", calc,
            lambda: self.calculate(experiment, params, tag=tag, naive=naive, gifs=gifs))
        self.plot(experiment, params, comp, tag=tag, naive=naive)

    def calculate(self, experiment, params, tag="run", naive=True, gifs=True):
        """Simulate every compared strategy (the slow part of iterate).

        Args: see iterate.

        Returns:
            dict — per-strategy lists, aligned with "tags":
                "z", "th", "o", "l", "u": trajectories of each strategy;
                "tags": strategy names, zero-incentive first.
        """
        GIF_DIR = f"results/{experiment}/{tag}/gifs"

        simulator = Simulator(params)

        def simulate_and_plot(th, label):
            """Run the simulator under incentive th; save the combined incentive/leaving-users animation unless gifs is False."""
            z_traj, p_traj, _, u_traj = simulator.run_open_loop(th)

            if gifs:
                _, p_le_traj = Transformer.split_p_traj_to_matrix(p_traj, params)

                anim_combined_incentive_leaving = Plotter.anim_combined_incentive_leaving(
                    th, p_traj, p_le_traj, params, params.th_box)
                Interface.save_animations({
                    "combined_incentive_leaving": anim_combined_incentive_leaving,
                }, dir_path=f"{GIF_DIR}/{label}")

            return z_traj, u_traj

        # Baseline: zero incentives — always computed.
        th_zero = np.zeros((params.T, params.N))
        z_traj_baseline, u_traj_baseline = simulate_and_plot(th_zero, "zero-incentive")
        o_traj_baseline = Transformer.get_o_traj(z_traj_baseline, params)
        l_traj_baseline = Transformer.get_l_traj(z_traj_baseline, params)

        # Comparison entries: baseline and optimized always; the trigger /
        # proportional / one-step strategies only in the full ("naive") path.
        z_traj_comp_list  = [z_traj_baseline]
        th_traj_comp_list = [th_zero]
        o_traj_comp_list  = [o_traj_baseline]
        l_traj_comp_list  = [l_traj_baseline]
        u_traj_comp_list  = [u_traj_baseline]
        tags_comp         = ["zero-incentive"]

        # Optimized incentives (bi-level ForwardBackward result saved by MainIterate_single)
        data              = Interface.load(f"results/{experiment}/{tag}/run")
        th_traj_optimized = data["th_traj"]
        z_traj_optimized, u_traj_optimized = simulate_and_plot(th_traj_optimized, "Algorithm 1")
        o_traj_optimized = Transformer.get_o_traj(z_traj_optimized, params)
        l_traj_optimized = Transformer.get_l_traj(z_traj_optimized, params)

        z_traj_comp_list  += [z_traj_optimized]
        th_traj_comp_list += [th_traj_optimized]
        o_traj_comp_list  += [o_traj_optimized]
        l_traj_comp_list  += [l_traj_optimized]
        u_traj_comp_list  += [u_traj_optimized]
        tags_comp         += ["Algorithm 1"]

        if naive:
            # Naive trigger
            th_traj_trigger = Trigger(params).apply()
            z_traj_trigger, u_traj_trigger = simulate_and_plot(th_traj_trigger, "trigger")
            o_traj_trigger = Transformer.get_o_traj(z_traj_trigger, params)
            l_traj_trigger = Transformer.get_l_traj(z_traj_trigger, params)

            # Naive proportional
            th_traj_proportional = Proportional(params).apply()
            z_traj_proportional, u_traj_proportional = simulate_and_plot(th_traj_proportional, "proportional")
            o_traj_proportional = Transformer.get_o_traj(z_traj_proportional, params)
            l_traj_proportional = Transformer.get_l_traj(z_traj_proportional, params)

            # One-step myopic solver
            th_traj_onestep, _, _ = OneStep(params).apply(params.th0_traj)
            z_traj_onestep, u_traj_onestep = simulate_and_plot(th_traj_onestep, "one-step")
            o_traj_onestep = Transformer.get_o_traj(z_traj_onestep, params)
            l_traj_onestep = Transformer.get_l_traj(z_traj_onestep, params)

            z_traj_comp_list  += [z_traj_trigger, z_traj_proportional, z_traj_onestep]
            th_traj_comp_list += [th_traj_trigger, th_traj_proportional, th_traj_onestep]
            o_traj_comp_list  += [o_traj_trigger, o_traj_proportional, o_traj_onestep]
            l_traj_comp_list  += [l_traj_trigger, l_traj_proportional, l_traj_onestep]
            u_traj_comp_list  += [u_traj_trigger, u_traj_proportional, u_traj_onestep]
            tags_comp         += ["trigger", "proportional", "one-step"]

        return {
            "z":    z_traj_comp_list,
            "th":   th_traj_comp_list,
            "o":    o_traj_comp_list,
            "l":    l_traj_comp_list,
            "u":    u_traj_comp_list,
            "tags": tags_comp,
        }

    def plot(self, experiment, params, comp, tag="run", naive=True):
        """Save the plot_agg_comp figure and u_le_comp table from calculate's output.

        Args:
            comp: dict — as returned by calculate.
            tag, naive: see iterate.
        """
        z_traj_comp_list  = comp["z"]
        th_traj_comp_list = comp["th"]
        o_traj_comp_list  = comp["o"]
        l_traj_comp_list  = comp["l"]
        u_traj_comp_list  = comp["u"]
        tags_comp         = comp["tags"]

        d_traj_comp_list = [params.d_traj] * len(tags_comp)

        # plot_agg_comp's panel ranges are fixed rather than autoscaled, so derive
        # them from the actual data being compared (the function adds its own loose
        # margin on top of these).
        d_max_agg    = max(Transformer.agg_d_traj(params.d_traj)[i].max() for i in (0, 1))
        th_max_agg   = max(Transformer.agg_th_traj(th).max() for th in th_traj_comp_list)
        u_le_max_agg = max(Transformer.agg_z_traj(z, params)[1].max() for z in z_traj_comp_list)
        o_max_agg    = max(Transformer.agg_o_traj(o).max() for o in o_traj_comp_list)
        fig_agg_comp = Plotter.plot_agg_comp(
            z_traj_comp_list, th_traj_comp_list, d_traj_comp_list,
            o_traj_comp_list, l_traj_comp_list, tags_comp, params,
            d_max_agg, o_max_agg, th_max_agg, u_le_max_agg,
            title=f"Aggregate trajectories — {'/'.join(tags_comp)}",
            show=False)

        if naive:
            Interface.save_figures({"agg_comp": fig_agg_comp},
                                    dir_path=f"results/{experiment}/{tag}/figs")
        else:
            Interface.save_figures({"agg_comp_simple": fig_agg_comp},
                                    dir_path=f"results/{experiment}/{tag}/figs")

        # Total outside-state flow (u_le) comparison against the baseline
        # (zero-incentive, always first)
        total_u_le_baseline = Transformer.get_total_u_le(u_traj_comp_list[0], params)

        totals    = [Transformer.get_total_u_le(u, params) for u in u_traj_comp_list]
        pct_diffs = [(total - total_u_le_baseline) / total_u_le_baseline * 100 for total in totals]
        th_usages = [np.sum(th) for th in th_traj_comp_list]
        pct_diffs_per_usage = [pct / usage if usage != 0 else np.nan
                               for pct, usage in zip(pct_diffs, th_usages)]

        # The zero-incentive row (always first) is the reference, so its
        # comparison columns are shown as "-".
        pct_diffs[0] = th_usages[0] = pct_diffs_per_usage[0] = "-"

        u_le_df = pd.DataFrame({
            "Algorithms":              tags_comp,
            "Leaving users":           totals,
            "Reduction":               pct_diffs,
            "Incentive usage":         th_usages,
            "Reduction per incentive": pct_diffs_per_usage,
        })
        Interface.save_csv(u_le_df, f"results/{experiment}/{tag}/tables/u_le_comp")


class ProcIterate_th0(ProcIterate):
    def iterate(self, experiment, map, demand, params, naive=True, gifs=True, calc=True):
        """Post-process every scenario in the sweep for this iterator's parameter.

        Split into calculate (loading each run and simulating it) followed by
        plot (cross-th0 comparison figures/tables); calculate's output is
        pickled in between to results/{experiment}/comp_results/comp.pkl.

        naive and gifs are accepted for signature uniformity but have no effect
        here: this variant doesn't delegate to ProcIterate_single.
        calc: bool — when True (default), run calculate and save its output.
            When False, skip calculate and load the output saved by a previous
            calc=True run.
        """
        comp = self._calculate_or_load(
            f"results/{experiment}/comp_results/comp", calc,
            lambda: self.calculate(experiment, params))
        self.plot(experiment, params, comp)

    def calculate(self, experiment, params):
        """Load every th0 run and simulate it, plus the zero-incentive baseline (the slow part of iterate).

        Returns:
            dict — per-run lists aligned with "tags" ("th_traj_list",
            "cost_hist_list", "z_traj_list", "u_traj_list", "o_traj_list",
            "l_traj_list", "total_cost_list", "incentive_cost_list"), the
            per-iteration "th_traj_max_err_hist", and the zero-incentive
            baseline "z_traj_baseline", "o_traj_baseline", "l_traj_baseline".
        """
        th0_values = self.cfg["th0"]
        n_runs     = len(th0_values)

        th_traj_list      = []
        th_traj_hist_list = []
        cost_hist_list    = []
        tags              = []

        for i, th0 in enumerate(th0_values):
            tag = f"th0_{i}"
            RESULTS_PATH = f"results/{experiment}/{tag}/run"

            data         = Interface.load(RESULTS_PATH)
            th_traj      = data["th_traj"]
            th_traj_hist = data["th_traj_hist"]
            cost_hist    = data["cost_hist"]

            th_traj_list.append(th_traj)
            th_traj_hist_list.append(th_traj_hist)
            cost_hist_list.append(cost_hist)
            tags.append(tag)

        # ── Max pairwise distance between runs' th_traj, at each opt. iteration ─
        # th_traj_hist_i[j] is run i's th_traj after iteration j; th_traj_max_err_hist[j]
        # is the biggest euclidean distance between any two runs' th_traj at that iteration.
        th_traj_hist_stack = np.stack(th_traj_hist_list, axis=0)  # (n_runs, n_it+1, T, N)
        n_it_p1 = th_traj_hist_stack.shape[1]

        th_traj_max_err_hist = np.zeros(n_it_p1)
        for j in range(n_it_p1):
            th_traj_j  = th_traj_hist_stack[:, j].reshape(n_runs, -1)
            pair_dists = np.linalg.norm(th_traj_j[:, None, :] - th_traj_j[None, :, :], axis=-1)
            th_traj_max_err_hist[j] = pair_dists.max()

        # ── Simulate each th0 iterate ───────────────────────────────────────────
        simulator = Simulator(params)
        optimizer = ForwardBackward(params)

        z_traj_list         = []
        u_traj_list         = []
        o_traj_list         = []
        l_traj_list         = []
        total_cost_list     = []
        incentive_cost_list = []
        for th_traj in th_traj_list:
            z_traj, p_traj, b_iq_traj, u_traj = simulator.run_open_loop(th_traj)
            z_traj_list.append(z_traj)
            u_traj_list.append(u_traj)
            o_traj_list.append(Transformer.get_o_traj(z_traj, params))
            l_traj_list.append(Transformer.get_l_traj(z_traj, params))
            total_cost_list.append(optimizer.get_cost(z_traj, th_traj))
            incentive_cost_list.append(optimizer.get_cost_only_R(th_traj))

        # ── Baseline: zero incentives, added as a reference to the aggregate comparison ─
        th_zero = np.zeros((params.T, params.N))
        z_traj_baseline, p_traj_baseline, b_iq_traj_baseline, u_traj_baseline = \
            simulator.run_open_loop(th_zero)
        o_traj_baseline = Transformer.get_o_traj(z_traj_baseline, params)
        l_traj_baseline = Transformer.get_l_traj(z_traj_baseline, params)

        return {
            "tags":                 tags,
            "th_traj_list":         th_traj_list,
            "cost_hist_list":       cost_hist_list,
            "th_traj_max_err_hist": th_traj_max_err_hist,
            "z_traj_list":          z_traj_list,
            "u_traj_list":          u_traj_list,
            "o_traj_list":          o_traj_list,
            "l_traj_list":          l_traj_list,
            "total_cost_list":      total_cost_list,
            "incentive_cost_list":  incentive_cost_list,
            "z_traj_baseline":      z_traj_baseline,
            "o_traj_baseline":      o_traj_baseline,
            "l_traj_baseline":      l_traj_baseline,
        }

    def plot(self, experiment, params, comp):
        """Save the cross-th0 comparison figures/tables from calculate's output.

        Args:
            comp: dict — as returned by calculate.
        """
        tags                 = comp["tags"]
        th_traj_list         = comp["th_traj_list"]
        cost_hist_list       = comp["cost_hist_list"]
        th_traj_max_err_hist = comp["th_traj_max_err_hist"]
        z_traj_list          = comp["z_traj_list"]
        u_traj_list          = comp["u_traj_list"]
        o_traj_list          = comp["o_traj_list"]
        l_traj_list          = comp["l_traj_list"]
        total_cost_list      = comp["total_cost_list"]
        incentive_cost_list  = comp["incentive_cost_list"]
        z_traj_baseline      = comp["z_traj_baseline"]
        o_traj_baseline      = comp["o_traj_baseline"]
        l_traj_baseline      = comp["l_traj_baseline"]
        n_runs               = len(tags)

        fig_convergence_comp = Plotter.plot_convergence_comp(
            cost_hist_list, show=False)
        Interface.save_figures({"convergence_comp": fig_convergence_comp},
                                dir_path=f"results/{experiment}/comp_results/figs")

        # ── Max pairwise distance between runs' th_traj, at each opt. iteration ─
        n_it_p1 = len(th_traj_max_err_hist)

        fig_max_err, ax_max_err = plt.subplots(figsize=(8, 4))
        ax_max_err.plot(np.arange(n_it_p1), th_traj_max_err_hist, color="black", linewidth=1.5)
        ax_max_err.set_xlabel("Iteration")
        ax_max_err.set_ylabel("max ‖θ_i − θ_i'‖")
        ax_max_err.set_title("Max pairwise th_traj distance across th0 sweep — per iteration")
        ax_max_err.set_ylim(bottom=0)
        ax_max_err.grid(True, linestyle="--", alpha=0.5)
        Interface.save_figures({"th_traj_max_err": fig_max_err},
                                dir_path=f"results/{experiment}/comp_results/figs")

        # ── Aggregate trajectory comparison across th0 iterates ────────────────
        d_traj_list = [params.d_traj] * n_runs

        # ── Pairwise distance between the converged th_traj of each run ───────
        dist_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(n_runs):
                dist_matrix[i, j] = np.linalg.norm(th_traj_list[i] - th_traj_list[j])

        dist_df = pd.DataFrame(dist_matrix, index=tags, columns=tags)
        dist_df["incentive cost"] = incentive_cost_list
        dist_df["total cost"]     = total_cost_list
        Interface.save_csv(dist_df, f"results/{experiment}/comp_results/tables/th_traj_dist")

        # ── Same distances, normalised by ||th_traj_list[i]|| (row-relative) ──
        th_traj_norms = [np.linalg.norm(th_traj) for th_traj in th_traj_list]

        dist_rel_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            dist_rel_matrix[i, :] = dist_matrix[i, :] / th_traj_norms[i]

        dist_rel_df = pd.DataFrame(dist_rel_matrix, index=tags, columns=tags)
        dist_rel_df["incentive cost"] = incentive_cost_list
        dist_rel_df["total cost"]     = total_cost_list
        Interface.save_csv(dist_rel_df, f"results/{experiment}/comp_results/tables/th_traj_dist_rel")

        # ── Final (converged) cost per run ─────────────────────────────────────
        final_costs = [cost_hist[-1] for cost_hist in cost_hist_list]

        max_dev_norm = (max(final_costs) - min(final_costs)) / np.mean(final_costs)
        print(f"Biggest final-cost deviation, normalised by mean cost: {max_dev_norm:.6f}")

        # ── Baseline: zero incentives, added as a reference to the aggregate comparison ─
        th_zero = np.zeros((params.T, params.N))

        # plot_agg_comp's panel ranges are fixed rather than autoscaled, so derive
        # them from the actual data being compared (the function adds its own loose
        # margin on top of these).
        z_traj_agg  = z_traj_list  + [z_traj_baseline]
        th_traj_agg = th_traj_list + [th_zero]
        d_traj_agg  = d_traj_list  + [params.d_traj]
        o_traj_agg  = o_traj_list  + [o_traj_baseline]
        l_traj_agg  = l_traj_list  + [l_traj_baseline]
        tags_agg    = tags + ["zero-incentive"]

        d_max_agg    = max(Transformer.agg_d_traj(params.d_traj)[i].max() for i in (0, 1))
        th_max_agg   = max(Transformer.agg_th_traj(th).max() for th in th_traj_agg)
        u_le_max_agg = max(Transformer.agg_z_traj(z, params)[1].max() for z in z_traj_agg)
        s_max_agg    = max(
            max(Transformer.agg_o_traj(o).max() for o in o_traj_agg),
            max(Transformer.agg_l_traj(l).max() for l in l_traj_agg),
        )
        fig_agg_comp = Plotter.plot_agg_comp(
            z_traj_agg, th_traj_agg, d_traj_agg, o_traj_agg, l_traj_agg,
            tags_agg, params, d_max_agg, s_max_agg, th_max_agg, u_le_max_agg,
            title="Aggregate trajectories — th0 sweep", show=False)
        Interface.save_figures({"agg_comp": fig_agg_comp},
                                dir_path=f"results/{experiment}/comp_results/figs")

        # ── Final cost & total leaving users per run, side by side ─────────────
        total_u_le_list = [Transformer.get_total_u_le(u_traj, params) for u_traj in u_traj_list]
        fig_cost_u_le   = Plotter.plot_final_comp(final_costs, total_u_le_list, show=False)
        Interface.save_figures({"cost_u_le_comp": fig_cost_u_le},
                                dir_path=f"results/{experiment}/comp_results/figs")


class ProcIterate_budget(ProcIterate):
    def iterate(self, experiment, map, demand, params, naive=True, gifs=True, calc=True):
        """Post-process every scenario in the sweep for this iterator's parameter.

        Rebuilds each budget value's Params (mirroring MainIterate_budget.iterate,
        since th_budget feeds the Projector at ForwardBackward-construction time
        and so must be a scalar there — demand/map stay fixed) and delegates the
        actual post-processing of each run to ProcIterate_single (forwarding
        gifs, so per-run animations are only produced when gifs=True), tagged by
        "budget_{value}" so it loads/saves under
        results/{experiment}/budget_{value}/... — matching where
        MainIterate_budget saved that run's optimized incentives. Only the
        cross-budget comparison plots/tables below are generated here.

        Split into calculate (every run's ProcIterate_single.calculate plus the
        cross-budget simulations) followed by plot (every run's
        ProcIterate_single.plot plus the cross-budget figures/tables);
        calculate's output is pickled in between to
        results/{experiment}/comp_results/comp.pkl (comp_simple.pkl when
        naive=False).

        naive: bool — forwarded to each run's ProcIterate_single.iterate. When
            False, each run post-processes only the zero-incentive and
            Algorithm 1 trajectories (no trigger/proportional/one-step). The cross-budget
            comparison plots/tables below always run either way, since they only
            ever compare the optimized result against the baseline.
        calc: bool — when True (default), run calculate and save its output.
            When False, skip calculate and load the output saved by a previous
            calc=True run with the same naive setting (no gifs are rendered in
            that case).
        """
        comp_name = "comp" if naive else "comp_simple"
        comp = self._calculate_or_load(
            f"results/{experiment}/comp_results/{comp_name}", calc,
            lambda: self.calculate(experiment, map, demand, params, naive=naive, gifs=gifs))
        self.plot(experiment, map, demand, params, comp, naive=naive)

    def _budget_runs(self, map, demand):
        """List (tag, Params) for every budget value in the sweep, tagged "budget_{value}"."""
        return [(f"budget_{th_budget}", Params({**self.cfg, "th_budget": th_budget}, map, demand))
                for th_budget in self.cfg["th_budget"]]

    def calculate(self, experiment, map, demand, params, naive=True, gifs=True):
        """Run every budget's ProcIterate_single.calculate and simulate the cross-budget comparison (the slow part of iterate).

        Returns:
            dict — "single" maps each tag to its ProcIterate_single.calculate
            output; per-run lists aligned with "tags" ("rel_err_list",
            "abs_err_list", "cost_hist_list", "total_u_le_opt_list",
            "total_u_le_base_list", "th_traj_list", "z_traj_list",
            "o_traj_list", "l_traj_list"); and the zero-incentive baseline
            "z_traj_baseline", "o_traj_baseline", "l_traj_baseline".
        """
        single       = ProcIterate_single(self.cfg)
        single_comps = {}

        rel_err_list   = []
        abs_err_list   = []
        cost_hist_list = []
        tags           = []

        total_u_le_opt_list  = []
        total_u_le_base_list = []

        th_traj_list = []
        z_traj_list  = []
        o_traj_list  = []
        l_traj_list  = []

        for tag, budget_params in self._budget_runs(map, demand):
            print(f"\n[2_post_proc] ── Calculating scenario '{experiment}' / {tag} ──")

            single_comps[tag] = single.calculate(experiment, budget_params, tag=tag, naive=naive, gifs=gifs)

            data         = Interface.load(f"results/{experiment}/{tag}/run")
            th_traj      = data["th_traj"]
            th_traj_hist = data["th_traj_hist"]
            cost_hist    = data["cost_hist"]

            rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
            rel_err_list.append(rel_err_hist)
            abs_err_list.append(abs_err_hist)
            cost_hist_list.append(cost_hist)
            tags.append(tag)

            # ── Optimized vs baseline (zero incentives) total leaving users ─
            # Reuses single.calculate's simulations: entry 0 is always
            # zero-incentive and entry 1 always Algorithm 1 (the same th_traj).
            single_comp = single_comps[tag]
            u_traj_base = single_comp["u"][0]
            u_traj_opt  = single_comp["u"][1]

            total_u_le_opt_list.append(Transformer.get_total_u_le(u_traj_opt, budget_params))
            total_u_le_base_list.append(Transformer.get_total_u_le(u_traj_base, budget_params))

            # ── Kept for the cross-budget aggregate-trajectory comparison ─
            th_traj_list.append(th_traj)
            z_traj_list.append(single_comp["z"][1])
            o_traj_list.append(single_comp["o"][1])
            l_traj_list.append(single_comp["l"][1])

        # th_budget doesn't affect the dynamics/demand (only the Projector at
        # optimization time, already baked into each saved th_traj), so every
        # budget's zero-incentive simulation is the same — reuse the first one
        # as the cross-budget baseline.
        baseline_comp   = single_comps[tags[0]]
        z_traj_baseline = baseline_comp["z"][0]
        o_traj_baseline = baseline_comp["o"][0]
        l_traj_baseline = baseline_comp["l"][0]

        return {
            "single":               single_comps,
            "tags":                 tags,
            "rel_err_list":         rel_err_list,
            "abs_err_list":         abs_err_list,
            "cost_hist_list":       cost_hist_list,
            "total_u_le_opt_list":  total_u_le_opt_list,
            "total_u_le_base_list": total_u_le_base_list,
            "th_traj_list":         th_traj_list,
            "z_traj_list":          z_traj_list,
            "o_traj_list":          o_traj_list,
            "l_traj_list":          l_traj_list,
            "z_traj_baseline":      z_traj_baseline,
            "o_traj_baseline":      o_traj_baseline,
            "l_traj_baseline":      l_traj_baseline,
        }

    def plot(self, experiment, map, demand, params, comp, naive=True):
        """Run every budget's ProcIterate_single.plot and save the cross-budget figures/tables from calculate's output.

        Args:
            comp: dict — as returned by calculate.
            naive: see iterate.
        """
        tags                 = comp["tags"]
        rel_err_list         = comp["rel_err_list"]
        abs_err_list         = comp["abs_err_list"]
        cost_hist_list       = comp["cost_hist_list"]
        total_u_le_opt_list  = comp["total_u_le_opt_list"]
        total_u_le_base_list = comp["total_u_le_base_list"]
        th_traj_list         = comp["th_traj_list"]
        z_traj_list          = comp["z_traj_list"]
        o_traj_list          = comp["o_traj_list"]
        l_traj_list          = comp["l_traj_list"]
        z_traj_baseline      = comp["z_traj_baseline"]
        o_traj_baseline      = comp["o_traj_baseline"]
        l_traj_baseline      = comp["l_traj_baseline"]

        single = ProcIterate_single(self.cfg)
        for tag, budget_params in self._budget_runs(map, demand):
            print(f"\n[2_post_proc] ── Plotting scenario '{experiment}' / {tag} ──")
            single.plot(experiment, budget_params, comp["single"][tag], tag=tag, naive=naive)

        # ── Output dirs for the cross-budget comparative plots/tables ──────────
        os.makedirs(f"results/{experiment}/comp_results/figs", exist_ok=True)
        os.makedirs(f"results/{experiment}/comp_results/tables", exist_ok=True)

        # ── Convergence comparison across budgets ───────────────────────────────
        fig_convergence_comp = Plotter.plot_convergence(
            rel_err_list, abs_err_list, cost_hist_list,
            legend_name_list=tags, title="Convergence — budget sweep", show=False)
        Interface.save_figures({"convergence_comp": fig_convergence_comp},
                                dir_path=f"results/{experiment}/comp_results/figs")

        # ── Total leaving users per budget — baseline vs optimized clustered bars ─
        fig_leaving_days = Plotter.plot_leaving_days(
            total_u_le_base_list, total_u_le_opt_list,
            legend_name_list=tags, title="Total leaving users - across different budgets", show=False)

        Interface.save_figures({"leaving_users_comp_days": fig_leaving_days},
                                dir_path=f"results/{experiment}/comp_results/figs")

        self._save_u_le_comp_across_tags(experiment, tags)

        # ── Aggregate trajectory comparison across budget iterates ─────────────
        th_zero = np.zeros((params.T, params.N))

        d_traj_list = [params.d_traj] * len(tags)

        z_traj_agg  = z_traj_list  + [z_traj_baseline]
        th_traj_agg = th_traj_list + [th_zero]
        d_traj_agg  = d_traj_list  + [params.d_traj]
        o_traj_agg  = o_traj_list  + [o_traj_baseline]
        l_traj_agg  = l_traj_list  + [l_traj_baseline]
        tags_agg    = tags + ["zero-incentive"]

        # plot_agg_comp's panel ranges are fixed rather than autoscaled, so derive
        # them from the actual data being compared (the function adds its own loose
        # margin on top of these).
        d_max_agg    = max(Transformer.agg_d_traj(params.d_traj)[i].max() for i in (0, 1))
        th_max_agg   = max(Transformer.agg_th_traj(th).max() for th in th_traj_agg)
        u_le_max_agg = max(Transformer.agg_z_traj(z, params)[1].max() for z in z_traj_agg)
        o_max_agg    = max(
            max(Transformer.agg_o_traj(o).max() for o in o_traj_agg),
            max(Transformer.agg_l_traj(l).max() for l in l_traj_agg),
        )
        fig_agg_comp = Plotter.plot_agg_comp(
            z_traj_agg, th_traj_agg, d_traj_agg, o_traj_agg, l_traj_agg,
            tags_agg, params, d_max_agg, o_max_agg, th_max_agg, u_le_max_agg,
            title="Aggregate trajectories — budget sweep", show=False)
        Interface.save_figures({"agg_comp": fig_agg_comp},
                                dir_path=f"results/{experiment}/comp_results/figs")


class ProcIterate_demand(ProcIterate):
    def iterate(self, experiment, map, demand, params, naive=True, gifs=True, calc=True):
        """Post-process every scenario in the sweep for this iterator's parameter.

        Rebuilds each day's Demand/Params (mirroring MainIterate_demand.iterate's
        day expansion) and delegates the actual post-processing of each day's run
        to ProcIterate_single (forwarding gifs, so per-day animations are only
        produced when gifs=True), tagged by date so it loads/saves under
        results/{experiment}/{date}/... — matching where MainIterate_demand
        saved that day's optimized incentives. Only the cross-day comparison
        plots/tables below are generated here.

        Split into calculate (every day's ProcIterate_single.calculate plus the
        cross-day simulations) followed by plot (every day's
        ProcIterate_single.plot plus the cross-day figures/tables);
        calculate's output is pickled in between to
        results/{experiment}/comp_results/comp.pkl (comp_simple.pkl when
        naive=False).

        naive: bool — forwarded to each day's ProcIterate_single.iterate. When
            False, each day post-processes only the zero-incentive and
            Algorithm 1 trajectories (no trigger/proportional/one-step). The cross-day
            comparison plots/tables below always run either way, since they only
            ever compare the optimized result against the baseline.
        calc: bool — when True (default), run calculate and save its output.
            When False, skip calculate and load the output saved by a previous
            calc=True run with the same naive setting (no gifs are rendered in
            that case).
        """
        comp_name = "comp" if naive else "comp_simple"
        comp = self._calculate_or_load(
            f"results/{experiment}/comp_results/{comp_name}", calc,
            lambda: self.calculate(experiment, map, naive=naive, gifs=gifs))
        self.plot(experiment, map, comp, naive=naive)

    def _day_runs(self, map):
        """List (tag, Params) for every day in the sweep, tagged by date ("run" for an explicit single-file demand)."""
        cfg = self.cfg

        # ── Expand the demand block into one file-cfg per day, as MainIterate_demand did ─
        demand_cfg = cfg["demand"]
        if "dropoff_csv" in demand_cfg:
            day_cfgs = [{**demand_cfg, "date": None}]
        else:
            day_cfgs = [
                {"type": "file", **day}
                for day in self.expand_dates(demand_cfg["dir"], demand_cfg["start_date"], demand_cfg["n_scenarios"])
            ]

        runs = []
        for day_cfg in day_cfgs:
            date       = day_cfg.pop("date")
            day_demand = Demand({**day_cfg, "N": map.N, "T": cfg["T"]})
            runs.append((date or "run", Params(cfg, map, day_demand)))
        return runs

    def calculate(self, experiment, map, naive=True, gifs=True):
        """Run every day's ProcIterate_single.calculate and simulate the cross-day comparison (the slow part of iterate).

        Returns:
            dict — "single" maps each tag to its ProcIterate_single.calculate
            output; per-day lists aligned with "tags" ("rel_err_list",
            "abs_err_list", "cost_hist_list", "total_u_le_opt_list",
            "total_u_le_base_list").
        """
        single       = ProcIterate_single(self.cfg)
        single_comps = {}

        rel_err_list   = []
        abs_err_list   = []
        cost_hist_list = []
        tags           = []

        total_u_le_opt_list  = []
        total_u_le_base_list = []

        for tag, day_params in self._day_runs(map):
            print(f"\n[2_post_proc] ── Calculating scenario '{experiment}' / {tag} ──")

            single_comps[tag] = single.calculate(experiment, day_params, tag=tag, naive=naive, gifs=gifs)

            data         = Interface.load(f"results/{experiment}/{tag}/run")
            th_traj_hist = data["th_traj_hist"]
            cost_hist    = data["cost_hist"]

            rel_err_hist, abs_err_hist = ForwardBackward.get_err_hist(th_traj_hist)
            rel_err_list.append(rel_err_hist)
            abs_err_list.append(abs_err_hist)
            cost_hist_list.append(cost_hist)
            tags.append(tag)

            # ── Optimized vs baseline (zero incentives) total leaving users ─
            # Reuses single.calculate's simulations: entry 0 is always
            # zero-incentive and entry 1 always Algorithm 1 (the same th_traj).
            u_traj_base = single_comps[tag]["u"][0]
            u_traj_opt  = single_comps[tag]["u"][1]

            total_u_le_opt_list.append(Transformer.get_total_u_le(u_traj_opt, day_params))
            total_u_le_base_list.append(Transformer.get_total_u_le(u_traj_base, day_params))

        return {
            "single":               single_comps,
            "tags":                 tags,
            "rel_err_list":         rel_err_list,
            "abs_err_list":         abs_err_list,
            "cost_hist_list":       cost_hist_list,
            "total_u_le_opt_list":  total_u_le_opt_list,
            "total_u_le_base_list": total_u_le_base_list,
        }

    def plot(self, experiment, map, comp, naive=True):
        """Run every day's ProcIterate_single.plot and save the cross-day figures/tables from calculate's output.

        Args:
            comp: dict — as returned by calculate.
            naive: see iterate.
        """
        tags                 = comp["tags"]
        rel_err_list         = comp["rel_err_list"]
        abs_err_list         = comp["abs_err_list"]
        cost_hist_list       = comp["cost_hist_list"]
        total_u_le_opt_list  = comp["total_u_le_opt_list"]
        total_u_le_base_list = comp["total_u_le_base_list"]

        single = ProcIterate_single(self.cfg)
        for tag, day_params in self._day_runs(map):
            print(f"\n[2_post_proc] ── Plotting scenario '{experiment}' / {tag} ──")
            single.plot(experiment, day_params, comp["single"][tag], tag=tag, naive=naive)

        # ── Output dirs for the cross-day comparative plots/tables ─────────────
        os.makedirs(f"results/{experiment}/comp_results/figs", exist_ok=True)
        os.makedirs(f"results/{experiment}/comp_results/tables", exist_ok=True)

        # ── Convergence comparison across days ──────────────────────────────────
        fig_convergence_comp = Plotter.plot_convergence(
            rel_err_list, abs_err_list, cost_hist_list,
            legend_name_list=tags, title="Convergence — demand sweep", show=False)
        Interface.save_figures({"convergence_comp": fig_convergence_comp},
                                dir_path=f"results/{experiment}/comp_results/figs")

        # ── Total leaving users per day — baseline vs optimized clustered bars ─
        fig_leaving_days = Plotter.plot_leaving_days(
            total_u_le_base_list, total_u_le_opt_list,
            legend_name_list=tags, show=False)

        Interface.save_figures({"leaving_users_comp_days": fig_leaving_days},
                                dir_path=f"results/{experiment}/comp_results/figs")

        self._save_u_le_comp_across_tags(experiment, tags)
