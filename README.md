# Incentive-based Fleet Rebalancing in Vehicle-Sharing Systems

Code and data for the paper

> D. Tihanyi and G. Belgioioso, "Incentive-based Fleet Rebalancing in Vehicle-Sharing Systems via Equilibrium-Constrained Optimal Control," 2026.

The operator of a free-floating vehicle-sharing system offers fare discounts (incentives) that steer users toward alternative pick-up regions. Users' responses are modelled as the variational generalized Nash equilibrium (v-GNE) of logit models coupled through shared vehicle-availability constraints. The incentives over a day of operation solve an optimal control problem with nested equilibrium constraints, which this code solves by projected gradient descent, backpropagating through the equilibria via nonsmooth implicit differentiation. The case study uses Helsinki city-bike data from May 2025.

## Installation

Tested with Python 3.13 on Linux.

```bash
git clone https://github.com/tihanyidani/vehicle-sharing.git
cd vehicle-sharing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the package versions used to produce the paper's results. The equilibria are computed with CVXPY and the SCS solver.

## Pipeline

All scripts are run from the repository root. Each script has a "Quick access" block at the top for selecting the experiment and options.

| Step | Script | Input | Output |
|---|---|---|---|
| 1. Pre-processing | `0_pre_proc.py` | `dataset/rawdata/2025-05.csv`, `dataset/metadata/*.csv` | `dataset/procdata/` (regions, per-day demand), `dataset/figures/` (region maps) |
| 2. Optimisation | `1_main.py` | `experiments/<EXPERIMENT>.yaml`, `dataset/procdata/` | `results/<EXPERIMENT>/<tag>/run.pkl`, convergence figure and table |
| 3. Post-processing | `2_post_proc.py` | `results/<EXPERIMENT>/` | comparison figures and tables under `results/<EXPERIMENT>/` |

The processed data and all results are included in the repository, so the figures can be regenerated without running steps 1 and 2.

**Pre-processing.** `0_pre_proc.py` keeps the trips whose departure and return stations both appear in the station metadata, restricts the network to central Helsinki, and groups stations into N = 46 regions by agglomerative clustering (300 m threshold). It then writes one pick-up and one drop-off demand file per day in 15-minute intervals (T = 96), and draws the region maps. The map background is downloaded from Esri on the first run (see [Data](#data)).

**Optimisation.** `1_main.py` runs Algorithm 1 for every scenario of the selected experiment. With `CONTINUE = True` it skips scenarios whose results are already saved, so set `CONTINUE = False` to recompute them.

**Post-processing.** `2_post_proc.py` simulates the optimised incentives and the baselines, and produces the comparison figures and tables. The options are:

- `NAIVE`: compare against all baselines (zero-incentive, trigger, proportional, one-step) instead of only the zero-incentive baseline.
- `GIFS`: also render per-strategy animations (slow).
- `CALC`: recompute the simulations; set to `False` to re-plot from the comparison saved by an earlier run.

## Reproducing the paper's results

Set `EXPERIMENT` (and `NAIVE` where listed) at the top of `1_main.py` and `2_post_proc.py`, then run them in order.

| Paper | `EXPERIMENT` | `NAIVE` | Output |
|---|---|---|---|
| Fig. 2 (regions) | n/a: produced by `0_pre_proc.py` | n/a | `dataset/figures/metadata_region_bounds.pdf` |
| Fig. 3 (incentives and leaving users over the day) | `Helsinki_case_study` | `False` | `results/Helsinki_case_study/run/figs/agg_comp_simple.pdf` |
| Table I (comparison with baselines) | `Helsinki_case_study` | `True` | `results/Helsinki_case_study/run/tables/u_le_comp.csv` |
| Fig. 4 (leaving users over 10 days) | `Helsinki_comp_days` | `False` | `results/Helsinki_comp_days/comp_results/figs/leaving_users_comp_days.pdf` |
| Fig. 5 (100 random initialisations) | `Helsinki_rand_init_100` | ignored | `results/Helsinki_rand_init_100/comp_results/figs/convergence_comp.pdf` |

Running `Helsinki_case_study` with `NAIVE = False` overwrites `u_le_comp.csv` with the zero-incentive and Algorithm 1 rows only. Run it with `NAIVE = True` afterwards to restore the full Table I.

The experiments share all model and solver parameters (see the YAML files): incentive sensitivity λ = 500 m, base cost of leaving w_∅ = 100 m, temperature τ = 50 m, cost weights γ_R = 1 and γ_Q = 10, maximum incentive θ̄ = 1, initial distribution of 10 vehicles per region, and 20 iterations of Algorithm 1 with step sizes α_k = 1/(k+1)³.

## Code structure

```
0_pre_proc.py          data pre-processing
1_main.py              optimisation entry point
2_post_proc.py         post-processing entry point
experiments/           one YAML file per experiment
dataset/
  rawdata/             HSL trip data (May 2025)
  metadata/            HSL station metadata
  procdata/            regions, per-day demand, shared parameters (metaparams.yaml)
  figures/             region maps
results/               saved runs, figures and tables per experiment
utils/                 model, algorithm and helpers (below)
```

How the modules in `utils/` relate to the paper:

| Module | Paper |
|---|---|
| `Params.py` | Model parameters: rewards (5), dynamics (2), cost weights (13) |
| `Map.py`, `Demand.py` | Regions and inter-region distances; demand flows d^pu, d^do |
| `T_map.py` | v-GNE (7) solved with CVXPY, primal–dual relation (21), state update map (14) |
| `dT_map.py` | Algorithm 4: implicit differentiation through the dual fixed point (22) |
| `g_map.py` | Stage cost (13), (15) and terminal cost |
| `Optimizer.py` | `ForwardBackward`: Algorithms 1–3 (projected gradient descent, backpropagation, forward pass); `OneStep`: one-step baseline |
| `Projector.py` | Projection onto the feasible set Ω |
| `Naive.py` | Trigger and proportional baselines |
| `Simulator.py` | Simulation of the closed system under a given incentive sequence |
| `MainIterate.py`, `ProcIterate.py` | Experiment sweeps over days or initial incentives, for `1_main.py` and `2_post_proc.py` |
| `Plotter.py`, `Interface.py`, `Transformer.py` | Plotting, file I/O and array helpers |

Notation used in the code: `th` is the incentive θ, `p` the users' choice distribution, `mu` the dual variables μ, and `d` the demand. The state `z` has N + 1 entries: the vehicle distribution x and one entry for users leaving the service.

## Data

- **Trips:** `dataset/rawdata/2025-05.csv` comes from [City bike stations' Origin-Destination (OD) data](https://hri.fi/data/en/dataset/11b6916b-2529-474e-b068-b8f05c59309a) by Helsinki Region Transport (HSL) and City Bike Finland, published on Helsinki Region Infoshare under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Stations:** `dataset/metadata/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin_*.csv` comes from [Helsinki Region Transport's (HSL) city bicycle stations](https://hri.fi/data/en_GB/dataset/hsl-n-kaupunkipyoraasemat) by HSL, published on Helsinki Region Infoshare under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Processed data:** the files in `dataset/procdata/` are derived from the two datasets above by `0_pre_proc.py`.
- **Basemap:** the map figures use the Esri World Light Gray Canvas basemap (Tiles © Esri — Esri, DeLorme, NAVTEQ). The tiles are not redistributed. `0_pre_proc.py` downloads them and caches them locally in `dataset/procdata/basemap.npz`, which is ignored by git.

## Citation

If you use this code, please cite:

```bibtex
@misc{tihanyi2026incentive,
  title  = {Incentive-based Fleet Rebalancing in Vehicle-Sharing Systems via Equilibrium-Constrained Optimal Control},
  author = {Tihanyi, Daniel and Belgioioso, Giuseppe},
  year   = {2026},
  note   = {Code available at https://github.com/tihanyidani/vehicle-sharing}
}
```

## License

The code is released under the [MIT License](LICENSE). The HSL data in `dataset/` remains under its original CC BY 4.0 license.

## Contact

Daniel Tihanyi (tihanyi@kth.se) and Giuseppe Belgioioso (giubel@kth.se), Division of Decision and Control Systems, KTH Royal Institute of Technology, Stockholm, Sweden.

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.
