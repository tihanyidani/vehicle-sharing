"""Post-processing — loads optimisation results and visualises them."""

import os

# Resolve all relative paths (experiments/, dataset/, results/) against this
# script's folder, so it runs from any working directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.Map         import Map
from utils.Params      import Params
from utils.Interface   import Interface
from utils.ProcIterate import ProcIterate

# ── Quick access ──────────────────────────────────────────────────────────────
# Paper experiments: "Helsinki_case_study" (Fig. 3, Table I; NAIVE = True),
# "Helsinki_comp_days" (Fig. 4; NAIVE = False), "Helsinki_rand_init_100" (Fig. 5; NAIVE ignored).
EXPERIMENT  = "Helsinki_case_study"  # experiment name — must match a file in experiments/
NAIVE       = True                  # True: post-process every strategy; False: zero-incentive and Algorithm 1 only
GIFS        = False                  # True: render and save .gif animations (slow); False: skip them
CALC        = True                   # True: recompute the comparison results and save them; False: load them from the previous CALC=True run

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_PATH = f"experiments/{EXPERIMENT}.yaml"
GIF_DIR         = f"results/{EXPERIMENT}/gifs"
FIG_DIR         = f"results/{EXPERIMENT}/figs"

# cfg["map"]/cfg["dt"]/cfg["T"] may come straight from the scenario file, or be
# merged in from a shared dataset/procdata/metaparams.yaml via the scenario's
# "metadata" key — see Interface.load_yaml. Must match what 1_main.py used to
# produce results for this EXPERIMENT. Per-tag results (one per sweep value)
# live under results/{EXPERIMENT}/{tag}/run — loaded inside ProcIterate.iterate.
cfg      = Interface.load_yaml(EXPERIMENT_PATH)
iterator = ProcIterate.create(cfg)

map    = Map(cfg["map"])
demand = iterator.build_fixed_demand(map)
params = Params(cfg, map, demand)

iterator.iterate(EXPERIMENT, map, demand, params, naive=NAIVE, gifs=GIFS, calc=CALC)