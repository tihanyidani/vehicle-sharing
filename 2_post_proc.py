"""Post-processing — loads optimisation results and visualises them."""

from utils.Map         import Map
from utils.Params      import Params
from utils.Interface   import Interface
from utils.ProcIterate import ProcIterate

# ── Quick access ──────────────────────────────────────────────────────────────
EXPERIMENT  = "Helsinki_rand_init_100"      # experiment name — must match a file in experiments/
NAIVE       = False                      # True: post-process every strategy; False: baseline trajectory only
GIFS        = False                      # True: render and save .gif animations (slow); False: skip them
CALC        = True                      # True: recompute the comparison results and save them; False: load them from the previous CALC=True run

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_PATH = f"experiments/{EXPERIMENT}.yaml"
GIF_DIR         = f"results/{EXPERIMENT}/gifs"
FIG_DIR         = f"results/{EXPERIMENT}/figs"

# cfg["map"]/cfg["dt"]/cfg["T"] may come straight from the scenario file, or be
# merged in from a shared dataset/procdata/metaparams.yaml via the scenario's
# "metadata" key — see Interface.load_yaml. Must match what main.py used to
# produce results for this EXPERIMENT. Per-tag results (one per sweep value)
# live under results/{EXPERIMENT}/{tag}/run — loaded inside ProcIterate.iterate.
cfg      = Interface.load_yaml(EXPERIMENT_PATH)
iterator = ProcIterate.create(cfg)

map    = Map(cfg["map"])
demand = iterator.build_fixed_demand(map)
params = Params(cfg, map, demand)

iterator.iterate(EXPERIMENT, map, demand, params, naive=NAIVE, gifs=GIFS, calc=CALC)