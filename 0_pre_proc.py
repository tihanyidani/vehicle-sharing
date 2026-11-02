"""
Build a multi-day, cross-checked trip + station statistical dataset.

Expected folder structure (relative to the repo's dataset/ folder, or set
BASE_DIR below):

    rawdata/
        2025-05.csv
    metadata/
        <station metadata csv>   (e.g. Helsingin_ja_Espoon_kaupunkipyoraasemat...csv)
    procdata/                (created automatically if missing)
    figures/                 (created automatically if missing)

Output:
    procdata/metadata.csv                   -> shared region metadata (all days)
    procdata/metadata_stations.csv           -> shared station metadata (all days), with region_id
    procdata/2025-05-01_demand_pickup.csv    -> pick-up demand matrix for that day
    procdata/2025-05-01_demand_dropoff.csv   -> drop-off demand matrix for that day
    ... one demand_pickup / demand_dropoff pair per day in TARGET_DATE_LIST
    figures/metadata_map.pdf                 -> region map
    figures/metadata_region_bounds.pdf       -> region bounds (Voronoi) map
    procdata/basemap.npz                     -> map figures' basemap image (if DOWNLOAD_BASEMAP)

Cross-checking logic:
    - A trip is kept only if BOTH its departure and return station IDs exist
      in the metadata (numeric match; non-numeric / malformed IDs such as "*40"
      are treated as unmatched).
    - The set of stations actually used (as departure or return) is computed
      ONCE across the union of all days in TARGET_DATE_LIST, and the resulting
      station set / region assignment is shared by every day's output. This
      guarantees all per-day datasets use the exact same stations, station
      ids, and regions — only the demand matrices vary from day to day.
"""

import calendar
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import Delaunay

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # directory this script lives in
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.Plotter import Plotter  # noqa: E402 (needs REPO_ROOT on sys.path first)
BASE_DIR = os.path.join(REPO_ROOT, "dataset")  # rawdata/metadata/procdata live here
RAW_TRIPS_FILE = "2025-05.csv"      # filename inside rawdata/

# Every day of the month covered by RAW_TRIPS_FILE (e.g. all of May 2025).
_year, _month = (int(part) for part in RAW_TRIPS_FILE.removesuffix(".csv").split("-"))
_n_days_in_month = calendar.monthrange(_year, _month)[1]
TARGET_DATE_LIST = [f"{_year:04d}-{_month:02d}-{day:02d}" for day in range(1, _n_days_in_month + 1)]

# Bounding box (lon_min, lon_max, lat_min, lat_max) using the metadata's
# x/y columns (WGS84). Default: central Helsinki (Kamppi/Töölö/Kruununhaka/
# Kaivopuisto core, ~67 stations). Adjust to taste — tighten to shrink
# the network further, widen to include more of it.
LON_MIN, LON_MAX = 24.90, 24.96
LAT_MIN, LAT_MAX = 60.155, 60.185

# Stations are grouped into regions via average-linkage agglomerative
# clustering on the projected metric coordinates (x2/y2, meters): stations
# end up in the same region only if they're within REGION_DISTANCE_THRESHOLD
# of each other (transitively, through the cluster chain). Raise/lower to
# get fewer/more regions.
REGION_DISTANCE_THRESHOLD = 300  # meters

# The map figures' basemap tiles take minutes to download. True downloads
# them and saves the image to BASEMAP_FILE; False reuses that saved image.
# Download again whenever the map's area changes (e.g. LON_*/LAT_* above).
DOWNLOAD_BASEMAP = True

RAWDATA_DIR = os.path.join(BASE_DIR, "rawdata")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
PROCDATA_DIR = os.path.join(BASE_DIR, "procdata")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
BASEMAP_FILE = os.path.join(PROCDATA_DIR, "basemap.npz")

CLUSTER_MINUTES = 15
RATE_SCALE = 60 / CLUSTER_MINUTES  # trips-per-cluster -> trips-per-hour
n_clusters = (24 * 60) // CLUSTER_MINUTES
cluster_labels = [
    f"{(i * CLUSTER_MINUTES) // 60:02d}:{(i * CLUSTER_MINUTES) % 60:02d}"
    for i in range(n_clusters)
]

# ----------------------------------------------------------------------
# 1. Locate input files
# ----------------------------------------------------------------------
trips_path = os.path.join(RAWDATA_DIR, RAW_TRIPS_FILE)
if not os.path.exists(trips_path):
    sys.exit(f"Trip file not found: {trips_path}")

meta_candidates = glob.glob(os.path.join(METADATA_DIR, "*.csv"))
if not meta_candidates:
    sys.exit(f"No metadata CSV found in {METADATA_DIR}")
meta_path = meta_candidates[0]
if len(meta_candidates) > 1:
    print(f"Note: multiple CSVs found in {METADATA_DIR}, using: {meta_path}")

os.makedirs(PROCDATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
print(f"Loading trips:    {trips_path}")
trips = pd.read_csv(
    trips_path,
    dtype={"Departure station id": str, "Return station id": str},
)

print(f"Loading metadata:  {meta_path}")
meta = pd.read_csv(meta_path)

trips["Departure"] = pd.to_datetime(trips["Departure"])
trips["Return"] = pd.to_datetime(trips["Return"])

month_dates = {pd.to_datetime(d).date() for d in TARGET_DATE_LIST}
month_trips = trips[trips["Departure"].dt.date.isin(month_dates)].copy()
print(f"Trips in {_year:04d}-{_month:02d}: {len(month_trips)} (of {len(trips)} in the raw file)")

if month_trips.empty:
    sys.exit(f"No trips found for {TARGET_DATE_LIST[0]}..{TARGET_DATE_LIST[-1]} — "
              f"check TARGET_DATE_LIST / the month covered by {RAW_TRIPS_FILE}.")

# ----------------------------------------------------------------------
# 3. Normalize station IDs and cross-check against metadata
#    (metadata 'ID' is int; trip IDs are zero-padded strings and can
#    contain malformed values like "*40" — those are treated as unmatched)
# ----------------------------------------------------------------------
meta["ID"] = meta["ID"].astype(int)

n_meta_before_bbox = len(meta)
meta = meta[
    meta["x"].between(LON_MIN, LON_MAX) & meta["y"].between(LAT_MIN, LAT_MAX)
].copy()
print(f"Stations inside bounding box: {len(meta)} (of {n_meta_before_bbox})")

meta_ids = set(meta["ID"])


def to_int_safe(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


month_trips["dep_id_int"] = month_trips["Departure station id"].apply(to_int_safe)
month_trips["ret_id_int"] = month_trips["Return station id"].apply(to_int_safe)

dep_ok = month_trips["dep_id_int"].isin(meta_ids)
ret_ok = month_trips["ret_id_int"].isin(meta_ids)
both_ok = dep_ok & ret_ok

n_total = len(month_trips)
n_removed = n_total - both_ok.sum()
print(f"Trips removed (unmatched departure/return station): {n_removed} "
      f"({n_removed / n_total:.2%})")

clean_trips_all = month_trips[both_ok].copy()

# ----------------------------------------------------------------------
# 4. Filter metadata down to stations used at least once anywhere in the
#    month. This station set (and the region assignment derived from it)
#    is shared by every day, so all per-day datasets stay consistent.
# ----------------------------------------------------------------------
used_ids = set(clean_trips_all["dep_id_int"]) | set(clean_trips_all["ret_id_int"])
META_COLUMNS = ["ID", "x", "y", "x2", "y2"]
META_COLUMN_RENAME = {"ID": "id"}
clean_meta = meta.loc[meta["ID"].isin(used_ids), META_COLUMNS].rename(columns=META_COLUMN_RENAME).copy()
clean_meta = clean_meta.sort_values("id").reset_index(drop=True)

print(f"Stations used across the month: {len(used_ids)}")
print(f"Stations kept in metadata:       {len(clean_meta)}")

# ----------------------------------------------------------------------
# 5. Cluster stations into regions (shared across all days)
#    See REGION_DISTANCE_THRESHOLD in CONFIG. Clustering runs on x2/y2
#    (projected, meters) rather than x/y (lon/lat) so the distance
#    threshold is physically meaningful.
# ----------------------------------------------------------------------
if len(clean_meta) > 1:
    Z = linkage(clean_meta[["x2", "y2"]].to_numpy(), method="average", metric="euclidean")
    clean_meta["region_id"] = fcluster(Z, t=REGION_DISTANCE_THRESHOLD, criterion="distance")
else:
    clean_meta["region_id"] = 1

region_agg = (
    clean_meta.groupby("region_id")
    .agg(x=("x", "mean"), y=("y", "mean"),
         x2=("x2", "mean"), y2=("y2", "mean"), n_stations=("id", "count"))
    .sort_values("x2")
)

# Renumber regions 0..n_regions-1 (west -> east, by mean x2) so ids are
# stable/readable, and propagate the same ids back onto clean_meta so the
# station-level and region-level outputs stay consistent.
old_to_new_region_id = {old_id: new_id for new_id, old_id in enumerate(region_agg.index)}
clean_meta["region_id"] = clean_meta["region_id"].map(old_to_new_region_id)

region_meta = region_agg.reset_index(drop=True)
region_meta.insert(0, "id", region_meta.index)

print(f"Stations grouped into {len(region_meta)} regions "
      f"(distance threshold={REGION_DISTANCE_THRESHOLD} m); "
      f"stations/region min={region_meta['n_stations'].min()} "
      f"max={region_meta['n_stations'].max()} "
      f"mean={region_meta['n_stations'].mean():.1f}")

# ----------------------------------------------------------------------
# 6. Save the shared metadata (region-level and station-level)
# ----------------------------------------------------------------------
meta_out = os.path.join(PROCDATA_DIR, "metadata.csv")
station_meta_out = os.path.join(PROCDATA_DIR, "metadata_stations.csv")

region_meta.to_csv(meta_out, index=False)
clean_meta.sort_values("id").to_csv(station_meta_out, index=False)

print(f"\nSaved: {meta_out}  ({len(region_meta)} regions)")
print(f"Saved: {station_meta_out}  ({len(clean_meta)} stations, with region_id)")

if DOWNLOAD_BASEMAP:
    basemap_img = None  # plot_region_map downloads the tiles
else:
    if not os.path.exists(BASEMAP_FILE):
        sys.exit(f"Basemap not found: {BASEMAP_FILE} (set DOWNLOAD_BASEMAP = True)")
    with np.load(BASEMAP_FILE) as saved:
        basemap_img = (saved["img"], tuple(saved["extent"]))
    print(f"Loaded basemap: {BASEMAP_FILE}")

map_fig = Plotter.plot_region_map(
    region_meta, title=f"Regions ({len(region_meta)}, threshold={REGION_DISTANCE_THRESHOLD} m)",
    show=False, basemap_img=basemap_img,
)
if DOWNLOAD_BASEMAP:
    # Keep the downloaded basemap for the bounds map below and for later runs.
    image = map_fig.axes[0].images[0]
    basemap_img = (np.asarray(image.get_array()), tuple(image.get_extent()))
    np.savez_compressed(BASEMAP_FILE, img=basemap_img[0], extent=basemap_img[1])
    print(f"Saved: {BASEMAP_FILE}")
map_out = os.path.join(FIGURES_DIR, "metadata_map.pdf")
map_fig.savefig(map_out, dpi=600)
print(f"Saved: {map_out}")

# EXPERIMENT: region bounds via Voronoi tessellation (nearest-center partition).
# Reuse map_fig's axes extent so this plot focuses on the same area as the
# dots-only map above, rather than the wider, padded Voronoi-clipping envelope.
map_ax = map_fig.axes[0]
bounds_fig = Plotter.plot_region_bounds_map(
    region_meta, title=None, show=False,
    xlim=map_ax.get_xlim(), ylim=map_ax.get_ylim(), basemap_img=basemap_img,
)
bounds_out = os.path.join(FIGURES_DIR, "metadata_region_bounds.pdf")
# Tight bbox with no padding crops the white margin around the map.
bounds_fig.savefig(bounds_out, dpi=600, bbox_inches="tight", pad_inches=0)
print(f"Saved: {bounds_out}")
plt.close(map_fig)
plt.close(bounds_fig)

# ----------------------------------------------------------------------
# 7. Restrict trips to stations kept in the shared metadata, and format
#    the pick-up / drop-off columns used to build the demand matrices.
# ----------------------------------------------------------------------
clean_trips_all = clean_trips_all[
    clean_trips_all["dep_id_int"].isin(used_ids) & clean_trips_all["ret_id_int"].isin(used_ids)
].copy()

TRIP_COLUMNS = ["Departure", "Return", "Departure station id", "Return station id"]
TRIP_COLUMN_RENAME = {
    "Departure": "pick-up time",
    "Return": "drop-off time",
    "Departure station id": "pick-up station id",
    "Return station id": "drop-off station id",
}
clean_trips_all = clean_trips_all[["Departure", "Return", "dep_id_int", "ret_id_int"]].rename(
    columns={"Departure": "pick-up time", "Return": "drop-off time",
             "dep_id_int": "pick-up station id", "ret_id_int": "drop-off station id"}
)
clean_trips_all["date"] = clean_trips_all["pick-up time"].dt.date

# ----------------------------------------------------------------------
# 8. Build pick-up / drop-off demand matrices per day
#    rows = 15-min time clusters (fixed, covering the full day)
#    columns = region ids (from the shared region metadata)
#    cell = demand rate in trips/hour, picking up/dropping off within that
#           region during that cluster (raw 15-min trip count scaled by
#           60 / CLUSTER_MINUTES so cluster length doesn't change the unit)
# ----------------------------------------------------------------------
station_to_region = dict(zip(clean_meta["id"], clean_meta["region_id"]))
region_ids = list(region_meta["id"])


def build_demand(day_trips, time_col, station_col):
    minutes = day_trips[time_col].dt.hour * 60 + day_trips[time_col].dt.minute
    cluster_idx = minutes // CLUSTER_MINUTES
    region = day_trips[station_col].map(station_to_region)
    demand = pd.crosstab(cluster_idx, region)
    demand = demand.reindex(index=range(n_clusters), columns=region_ids, fill_value=0)
    demand.index = pd.Index(cluster_labels, name="time")
    return demand * RATE_SCALE


print()
for target_date in TARGET_DATE_LIST:
    date_obj = pd.to_datetime(target_date).date()
    day_trips = clean_trips_all[clean_trips_all["date"] == date_obj]
    if day_trips.empty:
        print(f"{target_date}: no trips — skipping")
        continue

    pickup_demand = build_demand(day_trips, "pick-up time", "pick-up station id")
    dropoff_demand = build_demand(day_trips, "drop-off time", "drop-off station id")

    # The demand matrix columns must be identical, and in the same order, to
    # the region IDs in the shared region metadata file.
    assert list(pickup_demand.columns) == list(region_meta["id"])
    assert list(dropoff_demand.columns) == list(region_meta["id"])

    pickup_out = os.path.join(PROCDATA_DIR, f"{target_date}_demand_pickup.csv")
    dropoff_out = os.path.join(PROCDATA_DIR, f"{target_date}_demand_dropoff.csv")
    pickup_demand.to_csv(pickup_out)
    dropoff_demand.to_csv(dropoff_out)

    print(f"{target_date}: {len(day_trips)} trips -> "
          f"{os.path.basename(pickup_out)}, {os.path.basename(dropoff_out)} "
          f"(d_max pickup={pickup_demand.values.max():.1f}, dropoff={dropoff_demand.values.max():.1f})")

print(f"\nDone. N (regions)={len(region_ids)}, T (time steps)={n_clusters}, "
      f"days processed={len(TARGET_DATE_LIST)}")

# ----------------------------------------------------------------------
# 9. Distances between neighboring region centers in the final map.
#    "Neighboring" = Voronoi-adjacent (cells share an edge), i.e. joined
#    by a Delaunay-triangulation edge — the dual of the region-bounds
#    Voronoi tessellation. x2/y2 are projected meters, so distances are
#    in meters.
# ----------------------------------------------------------------------
region_pts = region_meta[["x2", "y2"]].to_numpy(dtype=float)
if len(region_pts) >= 4:
    simplices = Delaunay(region_pts).simplices
    edges = set()
    for tri in simplices:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
            edges.add((a, b) if a < b else (b, a))
    edges = np.array(sorted(edges))
    nbr_dists = np.linalg.norm(region_pts[edges[:, 0]] - region_pts[edges[:, 1]], axis=1)
    print(f"\nNeighboring-region distances (m, Voronoi-adjacent, {len(edges)} pairs): "
          f"min={nbr_dists.min():.1f}  max={nbr_dists.max():.1f}  mean={nbr_dists.mean():.1f}")
elif len(region_pts) > 1:
    dmat = np.linalg.norm(region_pts[:, None] - region_pts[None, :], axis=2)
    pair_dists = dmat[np.triu_indices(len(region_pts), k=1)]
    print(f"\nNeighboring-region distances (m, all {len(pair_dists)} pairs — too few for Delaunay): "
          f"min={pair_dists.min():.1f}  max={pair_dists.max():.1f}  mean={pair_dists.mean():.1f}")
else:
    print("\nNeighboring-region distances: only one region, nothing to compare.")