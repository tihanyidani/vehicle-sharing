"""Entry point — runs the optimisation pipeline and saves results.

Runs one scenario per day when the YAML's "demand" block gives a date
range ("dir"/"start_date"/"n_scenarios"), or a single scenario when it
gives an explicit "dropoff_csv"/"pickup_csv" pair.
"""

import os

# Resolve all relative paths (experiments/, dataset/, results/) against this
# script's folder, so it runs from any working directory (e.g. a VS Code
# workspace opened one level up).
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.Interface import Interface
from utils.MainIterate import MainIterate

# ── Quick access ──────────────────────────────────────────────────────────────
# Paper experiments: "Helsinki_case_study" (Fig. 3, Table I), "Helsinki_comp_days"
# (Fig. 4), "Helsinki_rand_init_100" (Fig. 5).
EXPERIMENT  = "Helsinki_case_study"  # experiment name — must match a file in experiments/
CONTINUE    = True  # True: skip scenarios already saved in results/{EXPERIMENT}/; False: rerun all from the start

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_PATH = f"experiments/{EXPERIMENT}.yaml"

# cfg["map"]/cfg["dt"]/cfg["T"] may come straight from the scenario file, or be
# merged in from a shared dataset/procdata/metaparams.yaml via the scenario's
# "metadata" key — see Interface.load_yaml.
cfg = Interface.load_yaml(EXPERIMENT_PATH)

iterator = MainIterate.create(cfg)
iterator.iterate(EXPERIMENT, cont=CONTINUE)