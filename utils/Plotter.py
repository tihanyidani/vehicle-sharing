"""Static plotting and animation utilities."""

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from tqdm import tqdm

from utils.Transformer import Transformer

# ── Font family ───────────────────────────────────────────────────────────────
# One Times-style serif for every plot, matching the paper's body text (the
# cssconf/IEEEtran class typesets in Times). The list is a fallback chain:
# STIXGeneral ships with matplotlib, so some Times face is always found.
# STIX math is Times-styled, so math matches the text. Type 42 embeds
# TrueType rather than Type 3 fonts in PDF/PS output, as IEEE requires.
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman",
                         "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


class Plotter:
    """Collection of static plotting methods for diagnostics and visualisation."""

    # ── Colour code ───────────────────────────────────────────────────────────
    # Fixed across every plot in this class, so a single edit here restyles
    # every snapshot consistently.
    COLOR_X    = "blue"     # state (x)
    COLOR_D_PU = "gold"     # PU demand (d_pu)
    COLOR_D_DO = "green"    # DO demand (d_do)
    COLOR_B_IQ = COLOR_X    # capacity bound (b_iq)
    COLOR_P_LE = "crimson"  # leaving users (p_le)
    COLOR_TH   = "orange"   # incentive (th)
    COLOR_P    = "black"    # allocation (p)
    COLOR_O    = "red"      # overshoot (o)
    COLOR_L    = "green"    # leverage (l)

    # ── Font sizes ────────────────────────────────────────────────────────────
    # Shared by every plot in this class, so a single edit here resizes the
    # text of every figure consistently.
    TITLE_FS  = 23    # panel/figure titles
    LABEL_FS  = 24    # axis labels
    TICK_FS   = 22    # tick labels
    LEGEND_FS = 22    # legends
    OFFSET_FS = TICK_FS - 2    # axis scale/offset labels (e.g. "1e6")

    # ── Pre-processing maps ─────────────────────────────────────────────────

    @staticmethod
    def _draw_basemap(ax, basemap_img=None):
        """Draw the Esri grey-canvas basemap under ax's current view (EPSG:3857).

        Downloading the tiles is slow, so a basemap drawn once can be passed
        back in (or saved and reloaded) to draw it again without downloading.

        Args:
            ax:          matplotlib Axes in EPSG:3857, with its final view set.
            basemap_img: (img, extent) or None — a basemap returned by an earlier
                         call; None downloads the tiles for the current view.

        Returns:
            (img, extent) — the drawn basemap image and its
            (xmin, xmax, ymin, ymax) extent in EPSG:3857.
        """
        import contextily as cx
        source = cx.providers.Esri.WorldGrayCanvas

        if basemap_img is None:
            import contextily.tile as cx_tile
            import mercantile

            zoom_adjust   = 2     # ~430 dpi basemap, the provider's max zoom (16) here; each step doubles the detail and ~quadruples the tiles
            n_connections = 8     # tiles downloaded in parallel
            timeout       = 30    # seconds per tile request, so a stalled one fails

            # contextily has no progress hook, so the bar's total repeats its own
            # tile selection (as in contextily.tile.bounds2img), and each tile is
            # counted by briefly wrapping contextily's per-tile fetch.
            xmin, xmax, ymin, ymax = ax.axis()
            w, s = cx_tile._sm2ll(xmin, ymin)
            e, n = cx_tile._sm2ll(xmax, ymax)
            zoom = cx_tile._validate_zoom(cx_tile._calculate_zoom(w, s, e, n) + zoom_adjust,
                                          cx_tile._process_source(source), auto=True)
            n_tiles = len(list(mercantile.tiles(w, s, e, n, [zoom])))

            # The tile server's IPv6 addresses are tried first but can be
            # unreachable, and each then costs a full timeout before IPv4 is
            # tried (~40 s per tile) — so resolve IPv4 only while downloading.
            import socket
            import urllib3.util.connection as urllib3_connection
            allowed_gai_family = urllib3_connection.allowed_gai_family

            fetch_tile = cx_tile._fetch_tile
            with tqdm(total=n_tiles, desc="Downloading basemap", unit="tile") as pbar:
                def _fetch_tile_counted(*args, **kwargs):
                    tile = fetch_tile(*args, **kwargs)
                    pbar.update(1)
                    return tile
                cx_tile._fetch_tile = _fetch_tile_counted
                urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
                try:
                    # use_cache=False makes contextily download in threads, which
                    # share the wrapped fetch above (its cached mode uses processes).
                    basemap_img = cx.bounds2img(xmin, ymin, xmax, ymax, source=source,
                                                zoom_adjust=zoom_adjust,
                                                n_connections=n_connections,
                                                use_cache=False, timeout=timeout)
                finally:
                    cx_tile._fetch_tile = fetch_tile
                    urllib3_connection.allowed_gai_family = allowed_gai_family

        img, extent = basemap_img
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if not (extent[0] <= min(xlim) and max(xlim) <= extent[1]
                and extent[2] <= min(ylim) and max(ylim) <= extent[3]):
            raise ValueError("The saved basemap does not cover this map's view; "
                             "download it again.")
        # Same drawing as contextily's add_basemap; imshow resets the view, so restore it.
        ax.imshow(img, extent=extent, interpolation="bilinear")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # The tile provider's source credit, as add_basemap writes it.
        cx.add_attribution(ax, source.attribution)
        return img, tuple(extent)

    BASEMAP_FILE = "dataset/procdata/basemap.npz"  # saved by 0_pre_proc.py
    _basemap_cache = None

    @staticmethod
    def _saved_basemap():
        """Load the basemap 0_pre_proc.py saved to BASEMAP_FILE, once per process.

        Returns:
            (img, extent), or None if the file is missing (_draw_basemap then
            downloads the tiles).
        """
        if Plotter._basemap_cache is None:
            import os
            if not os.path.exists(Plotter.BASEMAP_FILE):
                print(f"Saved basemap not found ({Plotter.BASEMAP_FILE}); downloading the tiles.")
                return None
            with np.load(Plotter.BASEMAP_FILE) as saved:
                Plotter._basemap_cache = (saved["img"], tuple(saved["extent"]))
        return Plotter._basemap_cache

    @staticmethod
    def _draw_saved_basemap(ax):
        """Draw the saved basemap under ax, downloading it at most once per process.

        If BASEMAP_FILE is missing, the tiles downloaded for the first snapshot
        are kept in memory and reused by every later one (e.g. animation frames).
        """
        Plotter._basemap_cache = Plotter._draw_basemap(ax, Plotter._saved_basemap())

    @staticmethod
    def plot_region_map(region_meta, title="Regions", show=True, ax=None, basemap=True,
                        lon_col="x", lat_col="y", region_col="id", crs="EPSG:4326",
                        basemap_img=None):
        """Plot region locations from a region metadata DataFrame.

        Plots the raw region metadata produced by 0_pre_proc.py
        (columns: id, x, y, x2, y2, n_stations). All markers are drawn the
        same size, annotated with region_col.

        Args:
            region_meta:   pd.DataFrame — region metadata with lon/lat and
                           an id column (as saved by 0_pre_proc.py's
                           metadata.csv).
            title:         str — plot title.
            show:          bool — call plt.show() at the end.
            ax:            matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:       bool — attempt an Esri grey-canvas basemap via contextily.
            lon_col:       str — longitude column name.
            lat_col:       str — latitude column name.
            region_col:    str — column name used for point annotations.
            crs:           str — CRS of lon_col/lat_col (default WGS84).
            basemap_img:   (img, extent) or None — a previously downloaded basemap
                           (see _draw_basemap); None downloads the tiles.

        Returns:
            matplotlib Figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        points = region_meta[[lon_col, lat_col]].to_numpy(dtype=float)
        used_basemap = False

        if basemap:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in points], crs=crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        ax.scatter(points[:, 0], points[:, 1], s=40,
                  c="#e63946", edgecolors="white", linewidths=0.6,
                  alpha=0.9, zorder=2)

        for (x, y), region_id in zip(points, region_meta[region_col]):
            ax.annotate(str(region_id), (x, y), fontsize=Plotter.TICK_FS,
                        xytext=(3, 3), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None,
                        # Semi-transparent circle behind the number, as in
                        # plot_leaving_days, so lines under it don't hide it. Drawn under the
                        # region dots (zorder 2), above the bounds (zorder 1).
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.85),
                        zorder=1.5)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_basemap(ax, basemap_img)
            ax.set_axis_off()
        else:
            ax.set_xlabel(lon_col, fontsize=Plotter.LABEL_FS)
            ax.set_ylabel(lat_col, fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    @staticmethod
    def plot_region_bounds_map(region_meta, title="Region bounds", show=True, ax=None, basemap=True,
                               x2_col="x2", y2_col="y2", id_col="id", crs="EPSG:3879", margin=2000.0,
                               xlim=None, ylim=None, basemap_img=None):
        """Plot region centers together with their Voronoi tessellation bounds.

        EXPERIMENTAL. Partitions the plane into one cell per region such that
        every point in a cell is closer to that region's center than to any
        other (a Voronoi tessellation), computed on the projected x2/y2
        (metric) coordinates so cell shapes reflect true nearest-center
        distance — uses x2/y2-native, optional-basemap coordinates rather
        than plot_region_map's lon/lat-native ones, since the Voronoi cells
        and the scatter must stay in the same coordinate space as whatever
        the basemap reprojection produces.

        Args:
            region_meta: pd.DataFrame — region metadata with x2/y2 (projected
                         meters) and id columns (as saved by 0_pre_proc.py's
                         metadata.csv).
            title:       str or None — plot title; None/"" omits it.
            show:        bool — call plt.show() at the end.
            ax:          matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:     bool — attempt an Esri grey-canvas basemap via contextily.
            x2_col:      str — projected x column name.
            y2_col:      str — projected y column name.
            id_col:      str — region id column name (used for point annotations).
            crs:         str — CRS of x2_col/y2_col (default EPSG:3879, matching
                         Map's file-based CRS).
            margin:      float — padding (in CRS units, metres) added around
                         the point bounding box before clipping the Voronoi
                         diagram, so edge regions get closed, finite cells.
            xlim:        tuple (min, max) or None — force the axes' x extent
                         (in EPSG:3857 if basemap is on, else in x2_col's
                         units) to match another plot's framing; None
                         autoscales to the region points, ignoring the
                         padded envelope.
            ylim:        tuple (min, max) or None — same as xlim, for y.
            basemap_img: (img, extent) or None — a previously downloaded basemap
                         (see _draw_basemap); None downloads the tiles.

        Returns:
            matplotlib Figure.
        """
        from shapely.geometry import MultiPoint, box
        from shapely.ops import voronoi_diagram

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        points_proj = region_meta[[x2_col, y2_col]].to_numpy(dtype=float)
        minx, miny = points_proj.min(axis=0) - margin
        maxx, maxy = points_proj.max(axis=0) + margin
        envelope = box(minx, miny, maxx, maxy)

        vor = voronoi_diagram(MultiPoint(points_proj), envelope=envelope)

        # voronoi_diagram doesn't preserve input order/identity, so match each
        # cell back to its owning region by nearest center.
        cell_polys = [None] * len(points_proj)
        for poly in vor.geoms:
            rep = np.array(poly.representative_point().coords[0])
            idx = int(np.argmin(np.linalg.norm(points_proj - rep, axis=1)))
            cell_polys[idx] = poly

        points = points_proj
        polys = cell_polys
        used_basemap = False

        if basemap:
            try:
                import contextily as cx
                import geopandas as gpd

                gdf_pts = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(points_proj[:, 0], points_proj[:, 1]), crs=crs,
                ).to_crs(epsg=3857)
                points = np.column_stack((gdf_pts.geometry.x, gdf_pts.geometry.y))

                gdf_poly = gpd.GeoDataFrame(geometry=polys, crs=crs).to_crs(epsg=3857)
                polys = list(gdf_poly.geometry)
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        # Scatter (and only the points) first, to establish a default view
        # extent matching the dots-only framing — the Voronoi cells are
        # clipped to a padded envelope (so edge regions get closed cells)
        # that's larger than the point extent and would otherwise stretch the
        # visible focus out to include that padding. If the caller passed an
        # explicit xlim/ylim (e.g. taken from an existing figure), that takes
        # precedence and is applied at the end instead.
        ax.scatter(points[:, 0], points[:, 1], s=100, c="#e63946",
                  edgecolors="white", linewidths=0.8, alpha=0.9, zorder=2)
        if xlim is None or ylim is None:
            ax.relim()
            ax.autoscale_view()
            auto_xlim, auto_ylim = ax.get_xlim(), ax.get_ylim()
            xlim = xlim if xlim is not None else auto_xlim
            ylim = ylim if ylim is not None else auto_ylim

        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=1.0, alpha=0.6, zorder=1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        for (x, y), region_id in zip(points, region_meta[id_col]):
            ax.annotate(str(region_id), (x, y), fontsize=Plotter.TICK_FS,
                        xytext=(5, 5), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None,
                        # Semi-transparent circle behind the number, as in
                        # plot_leaving_days, so region bounds under it don't hide it. Drawn under the
                        # region dots (zorder 2), above the bounds (zorder 1).
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.85),
                        zorder=1.5)

        if used_basemap:
            Plotter._draw_basemap(ax, basemap_img)
            ax.set_axis_off()
        else:
            ax.set_xlabel(x2_col, fontsize=Plotter.LABEL_FS)
            ax.set_ylabel(y2_col, fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.grid(alpha=0.3)
        # Equal aspect in both branches, so the map is never stretched.
        ax.set_aspect("equal")

        if title:
            ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── Snapshot diagnostics ──────────────────────────────────────────────────

    @staticmethod
    def snapshot_state(x, d_pu, d_do, params, x_max, d_max, title="State", show=True, ax=None, basemap=True):
        """Plot the state x on real region coordinates as blue dots, with PU/DO
        demand overlaid as up/down arrows.

        Real-coordinate counterpart of the old plot_z_map, but instead of
        colour-coding markers via a colormap, every marker is blue and its
        size/transparency scales with x relative to x_max, on
        params.map.points. d_pu is drawn as an upward-pointing yellow arrow
        and d_do as a downward-pointing green arrow at each region, with
        strength/width/transparency scaling with value relative to d_max.
        Tries an OSM basemap underneath when params.map.crs is set, falling
        back to a plain scatter otherwise.

        Args:
            x:       np.ndarray of shape (N,) — per-region state.
            d_pu:    np.ndarray of shape (N,) — per-region PU demand.
            d_do:    np.ndarray of shape (N,) — per-region DO demand.
            params:  Params — provides region coordinates/ids/CRS via params.map.
            x_max:   float — state value at which a marker is fully sized/opaque.
            d_max:   float — demand value (shared by d_pu/d_do) at which an
                     arrow is fully sized/opaque.
            title:   str — plot title.
            show:    bool — call plt.show() at the end.
            ax:      matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap: bool — attempt the OSM basemap when params.map.crs is set.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        map    = params.map
        points = map.points
        used_basemap = False

        if basemap and getattr(map, "crs", None) is not None:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in map.points],
                    crs=map.crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        prop   = np.clip(x / x_max, 0.0, 1.0)
        sizes  = 20 + 200 * prop
        colors = np.tile(to_rgba(Plotter.COLOR_X), (len(x), 1))
        colors[:, 3] = 0.2 + 0.8 * prop  # alpha

        prop_pu    = np.clip(d_pu / d_max, 0.0, 1.0)
        prop_do    = np.clip(d_do / d_max, 0.0, 1.0)
        arrow_len  = 0.05 * (points[:, 1].max() - points[:, 1].min())
        for (px, py), pu_i, do_i in zip(points, prop_pu, prop_do):
            if pu_i > 0:
                ax.annotate("", xy=(px, py + arrow_len * pu_i), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_D_PU,
                                            alpha=0.3 + 0.7 * pu_i, lw=1 + 3 * pu_i),
                            zorder=1)
            if do_i > 0:
                ax.annotate("", xy=(px, py - arrow_len * do_i), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_D_DO,
                                            alpha=0.3 + 0.7 * do_i, lw=1 + 3 * do_i),
                            zorder=1)

        ax.scatter(points[:, 0], points[:, 1], s=sizes, color=colors,
                  edgecolors="white", linewidths=0.6, zorder=2)

        for (px, py), region_id in zip(points, map.index):
            ax.annotate(str(region_id), (px, py), fontsize=6,
                        xytext=(3, 3), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_saved_basemap(ax)
            ax.set_axis_off()
        else:
            ax.set_xlabel("x", fontsize=Plotter.LABEL_FS)
            ax.set_ylabel("y", fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")

        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_X,
                   markeredgecolor="white", markersize=8, label="state (x)"),
            Line2D([0], [0], marker="^", color=Plotter.COLOR_D_PU, markerfacecolor=Plotter.COLOR_D_PU,
                   markersize=8, label="PU demand (d_pu)"),
            Line2D([0], [0], marker="v", color=Plotter.COLOR_D_DO, markerfacecolor=Plotter.COLOR_D_DO,
                   markersize=8, label="DO demand (d_do)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=Plotter.LEGEND_FS, framealpha=0.9)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    @staticmethod
    def snapshot_leaving_users(b_iq, d_pu, p_le, params, b_iq_max, d_max, p_le_max=1,
                               title="Leaving users", show=True, ax=None, basemap=True):
        """Plot PU demand, its capacity bound, and leaving users on real region coordinates.

        Companion to snapshot_state, same per-region layout on params.map.points:
          - d_pu is a gold dot, sized/faded by d_pu relative to d_max.
          - b_iq is a black ring centred on the same point, sized by b_iq
            relative to b_iq_max via the same size mapping as the d_pu dot —
            so when b_iq == d_pu (and b_iq_max == d_max) the ring sits right
            on the dot's edge.
          - p_le is a straight, leftward-pointing crimson arrow, with
            strength/width/size/transparency scaling with value relative to
            p_le_max (fixed at 1), mirroring the d_pu/d_do arrows in
            snapshot_state.

        Tries an OSM basemap underneath when params.map.crs is set, falling
        back to a plain scatter otherwise.

        Args:
            b_iq:     np.ndarray of shape (N,) — per-region capacity bound (PU).
            d_pu:     np.ndarray of shape (N,) — per-region PU demand.
            p_le:     np.ndarray of shape (N,) — per-region leaving-users fraction.
            params:   Params — provides region coordinates/ids/CRS via params.map.
            b_iq_max: float — bound value at which the ring is fully sized/opaque.
            d_max:    float — demand value at which the dot is fully sized/opaque.
            p_le_max: float — leaving-users value at which the arrow is fully
                      sized/opaque (fixed at 1, since p_le is a fraction).
            title:    str — plot title.
            show:     bool — call plt.show() at the end.
            ax:       matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:  bool — attempt the OSM basemap when params.map.crs is set.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        map    = params.map
        points = map.points
        used_basemap = False

        if basemap and getattr(map, "crs", None) is not None:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in map.points],
                    crs=map.crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        def _size(prop):
            return 20 + 200 * prop

        # ── p_le straight leftward arrow ─────────────────────────────────────
        prop_le   = np.clip(p_le / p_le_max, 0.0, 1.0)
        arrow_len = 0.05 * (points[:, 1].max() - points[:, 1].min())
        for (px, py), le_i in zip(points, prop_le):
            if le_i > 0:
                d = arrow_len * le_i
                ax.annotate("", xy=(px - d, py), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_P_LE,
                                            alpha=0.3 + 0.7 * le_i, lw=1 + 3 * le_i),
                            zorder=1)

        # ── d_pu dot ────────────────────────────────────────────────────────
        prop_pu = np.clip(d_pu / d_max, 0.0, 1.0)
        colors_pu = np.tile(to_rgba(Plotter.COLOR_D_PU), (len(d_pu), 1))
        colors_pu[:, 3] = 0.2 + 0.8 * prop_pu

        ax.scatter(points[:, 0], points[:, 1], s=_size(prop_pu), color=colors_pu,
                  edgecolors="white", linewidths=0.6, zorder=2)

        # ── b_iq ring, drawn on top so it stays visible even when smaller
        #    than the d_pu dot it's centred on ─────────────────────────────
        prop_biq = np.clip(b_iq / b_iq_max, 0.0, 1.0)
        edge_colors_biq = np.tile(to_rgba(Plotter.COLOR_B_IQ), (len(b_iq), 1))
        edge_colors_biq[:, 3] = 0.2 + 0.8 * prop_biq

        ax.scatter(points[:, 0], points[:, 1], s=_size(prop_biq), facecolor="none",
                  edgecolor=edge_colors_biq, linewidths=1.5, zorder=3)

        for (px, py), region_id in zip(points, map.index):
            ax.annotate(str(region_id), (px, py), fontsize=6,
                        xytext=(3, 3), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_saved_basemap(ax)
            ax.set_axis_off()
        else:
            ax.set_xlabel("x", fontsize=Plotter.LABEL_FS)
            ax.set_ylabel("y", fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")

        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_D_PU,
                   markeredgecolor="white", markersize=8, label="PU demand (d_pu)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=Plotter.COLOR_B_IQ, markersize=8, label="capacity bound (b_iq)"),
            Line2D([0], [0], marker=r"$\leftarrow$", color=Plotter.COLOR_P_LE,
                   linestyle="None", markersize=10, label="leaving users (p_le)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=Plotter.LEGEND_FS, framealpha=0.9)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    @staticmethod
    def snapshot_incentive_response(th, p, params, th_max=1, p_max=1, arrow_thr=0.05,
                                    title="Incentive response", show=True, ax=None, basemap=True):
        """Plot incentives and the resulting allocation on real region coordinates.

        Real-coordinate counterpart of the old plot_th_p_map, restyled to be
        consistent with the other snapshot_* plots: th is an orange dot per
        region, sized/faded by th relative to th_max, rather than a
        colour-mapped marker with a colorbar. p (the allocation matrix,
        obtained from the flat p vector via Transformer.split_p_to_matrix) is
        drawn as black arrows between region pairs — arrowhead at the
        destination i, tail at the source j — with strength/width/
        transparency scaling with P[i, j] relative to p_max; pairs at or
        below arrow_thr are skipped entirely, matching plot_map_heat_arrow's
        default threshold. Tries an OSM basemap underneath when
        params.map.crs is set, falling back to a plain scatter otherwise.

        Args:
            th:        np.ndarray of shape (N,) — per-region incentive.
            p:         np.ndarray of shape (N*(N+1),) — optimal allocation from T_map.P_map.
            params:    Params — provides region coordinates/ids/CRS via params.map, and N.
            th_max:    float — incentive value at which a dot is fully sized/opaque (fixed at 1).
            p_max:     float — allocation value at which an arrow is fully sized/opaque (fixed at 1).
            arrow_thr: float — minimum P[i, j] value to display an arrow.
            title:     str — plot title.
            show:      bool — call plt.show() at the end.
            ax:        matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:   bool — attempt the OSM basemap when params.map.crs is set.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        map    = params.map
        points = map.points
        used_basemap = False

        if basemap and getattr(map, "crs", None) is not None:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in map.points],
                    crs=map.crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        def _size(prop):
            return 20 + 200 * prop

        # ── p arrows between region pairs ──────────────────────────────────
        P, _ = Transformer.split_p_to_matrix(p, params)
        N    = params.N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                val = float(P[i, j]) / p_max
                if val <= arrow_thr:
                    continue
                val = min(val, 1.0)
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                # Arrowhead at i, tail at j.
                ax.annotate("", xy=(x_i, y_i), xytext=(x_j, y_j),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_P,
                                            alpha=0.3 + 0.7 * val, lw=1 + 3 * val,
                                            connectionstyle="arc3,rad=0.15"),
                            zorder=1)

        # ── th dot ──────────────────────────────────────────────────────────
        prop_th   = np.clip(th / th_max, 0.0, 1.0)
        colors_th = np.tile(to_rgba(Plotter.COLOR_TH), (len(th), 1))
        colors_th[:, 3] = 0.2 + 0.8 * prop_th

        ax.scatter(points[:, 0], points[:, 1], s=_size(prop_th), color=colors_th,
                  edgecolors="white", linewidths=0.6, zorder=2)

        for (px, py), region_id in zip(points, map.index):
            ax.annotate(str(region_id), (px, py), fontsize=6,
                        xytext=(10, 10), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_saved_basemap(ax)
            ax.set_axis_off()
        else:
            ax.set_xlabel("x", fontsize=Plotter.LABEL_FS)
            ax.set_ylabel("y", fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")

        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_TH,
                   markeredgecolor="white", markersize=8, label="incentive (th)"),
            Line2D([0], [0], marker=r"$\rightarrow$", color=Plotter.COLOR_P,
                   linestyle="None", markersize=10, label="allocation (p)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=Plotter.LEGEND_FS, framealpha=0.9)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    @staticmethod
    def snapshot_combined_incentive_leaving(th, p, p_le, params, th_max=1, p_max=1, p_le_max=1,
                                            arrow_thr=0.05, title="Incentive response & leaving users",
                                            show=True, ax=None, basemap=True):
        """Plot incentives, the resulting allocation, and leaving users on real region coordinates.

        Identical to snapshot_incentive_response — th as an orange dot per
        region (sized/faded by th relative to th_max) and the allocation p as
        black arrows between region pairs (arrowhead at destination i, tail at
        source j, scaled by P[i, j] relative to p_max, pairs at or below
        arrow_thr skipped) — with the leaving-users field p_le overlaid
        exactly as snapshot_leaving_users draws it: a straight
        leftward-pointing crimson arrow per region, with strength/width/
        transparency scaling with value relative to p_le_max (fixed at 1).
        Tries an OSM basemap underneath when params.map.crs is set, falling
        back to a plain scatter otherwise.

        Args:
            th:        np.ndarray of shape (N,) — per-region incentive.
            p:         np.ndarray of shape (N*(N+1),) — optimal allocation from T_map.P_map.
            p_le:      np.ndarray of shape (N,) — per-region leaving-users fraction.
            params:    Params — provides region coordinates/ids/CRS via params.map, and N.
            th_max:    float — incentive value at which a dot is fully sized/opaque (fixed at 1).
            p_max:     float — allocation value at which an arrow is fully sized/opaque (fixed at 1).
            p_le_max:  float — leaving-users value at which the arrow is fully
                       sized/opaque (fixed at 1, since p_le is a fraction).
            arrow_thr: float — minimum P[i, j] value to display an allocation arrow.
            title:     str — plot title.
            show:      bool — call plt.show() at the end.
            ax:        matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:   bool — attempt the OSM basemap when params.map.crs is set.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        map    = params.map
        points = map.points
        used_basemap = False

        if basemap and getattr(map, "crs", None) is not None:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in map.points],
                    crs=map.crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        def _size(prop):
            return 20 + 200 * prop

        # ── p arrows between region pairs ──────────────────────────────────
        P, _ = Transformer.split_p_to_matrix(p, params)
        N    = params.N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                val = float(P[i, j]) / p_max
                if val <= arrow_thr:
                    continue
                val = min(val, 1.0)
                x_i, y_i = points[i]
                x_j, y_j = points[j]
                # Arrowhead at i, tail at j.
                ax.annotate("", xy=(x_i, y_i), xytext=(x_j, y_j),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_P,
                                            alpha=0.3 + 0.7 * val, lw=1 + 3 * val,
                                            connectionstyle="arc3,rad=0.15"),
                            zorder=1)

        # ── p_le straight leftward arrow (mirrors snapshot_leaving_users) ──
        prop_le   = np.clip(p_le / p_le_max, 0.0, 1.0)
        arrow_len = 0.05 * (points[:, 1].max() - points[:, 1].min())
        for (px, py), le_i in zip(points, prop_le):
            if le_i > 0:
                d = arrow_len * le_i
                ax.annotate("", xy=(px - d, py), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_P_LE,
                                            alpha=0.3 + 0.7 * le_i, lw=1 + 3 * le_i),
                            zorder=1)

        # ── th dot ──────────────────────────────────────────────────────────
        prop_th   = np.clip(th / th_max, 0.0, 1.0)
        colors_th = np.tile(to_rgba(Plotter.COLOR_TH), (len(th), 1))
        colors_th[:, 3] = 0.2 + 0.8 * prop_th

        ax.scatter(points[:, 0], points[:, 1], s=_size(prop_th), color=colors_th,
                  edgecolors="white", linewidths=0.6, zorder=2)

        for (px, py), region_id in zip(points, map.index):
            ax.annotate(str(region_id), (px, py), fontsize=6,
                        xytext=(10, 10), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_saved_basemap(ax)
            ax.set_axis_off()
        else:
            ax.set_xlabel("x", fontsize=Plotter.LABEL_FS)
            ax.set_ylabel("y", fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")

        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_TH,
                   markeredgecolor="white", markersize=8, label="incentive (th)"),
            Line2D([0], [0], marker=r"$\rightarrow$", color=Plotter.COLOR_P,
                   linestyle="None", markersize=10, label="allocation (p)"),
            Line2D([0], [0], marker=r"$\leftarrow$", color=Plotter.COLOR_P_LE,
                   linestyle="None", markersize=10, label="leaving users (p_le)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=Plotter.LEGEND_FS, framealpha=0.9)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    @staticmethod
    def snapshot_saturation(o, l, p_le, params, s_max, p_le_max=1,
                            title="Saturation", show=True, ax=None, basemap=True):
        """Plot overshoot/leverage and leaving users on real region coordinates.

        Companion to the other snapshot_* plots, same per-region layout on
        params.map.points. o and l are mutually exclusive at any given region
        (Transformer.get_o_traj/get_l_traj: o = max(0, d - x), l = max(0, x
        - d), so at most one is nonzero) — so only the larger of the two is
        drawn per region, as a dot sized/faded by its value relative to the
        shared s_max: red for overshoot (o), green for leverage (l). p_le is
        drawn the same way as in snapshot_leaving_users: a straight
        leftward-pointing crimson arrow, with strength/width/size/
        transparency scaling with value relative to p_le_max (fixed at 1).
        Tries an OSM basemap underneath when params.map.crs is set, falling
        back to a plain scatter otherwise.

        Args:
            o:        np.ndarray of shape (N,) — per-region overshoot.
            l:        np.ndarray of shape (N,) — per-region leverage.
            p_le:     np.ndarray of shape (N,) — per-region leaving-users fraction.
            params:   Params — provides region coordinates/ids/CRS via params.map.
            s_max:    float — overshoot/leverage value at which a dot is fully sized/opaque.
            p_le_max: float — leaving-users value at which the arrow is fully
                      sized/opaque (fixed at 1, since p_le is a fraction).
            title:    str — plot title.
            show:     bool — call plt.show() at the end.
            ax:       matplotlib Axes or None — draw on this axes; creates a new figure if None.
            basemap:  bool — attempt the OSM basemap when params.map.crs is set.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
        else:
            fig = ax.get_figure()

        map    = params.map
        points = map.points
        used_basemap = False

        if basemap and getattr(map, "crs", None) is not None:
            try:
                import contextily as cx
                import geopandas as gpd
                from shapely.geometry import Point

                gdf = gpd.GeoDataFrame(
                    geometry=[Point(xy) for xy in map.points],
                    crs=map.crs,
                ).to_crs(epsg=3857)  # Web Mercator, required by tile providers
                points = np.column_stack((gdf.geometry.x, gdf.geometry.y))
                used_basemap = True
            except Exception as e:
                print(f"Basemap unavailable ({type(e).__name__}: {e}); "
                      f"falling back to a plain scatter plot.")

        def _size(prop):
            return 20 + 200 * prop

        # ── p_le straight leftward arrow ─────────────────────────────────────
        prop_le   = np.clip(p_le / p_le_max, 0.0, 1.0)
        arrow_len = 0.05 * (points[:, 1].max() - points[:, 1].min())
        for (px, py), le_i in zip(points, prop_le):
            if le_i > 0:
                d = arrow_len * le_i
                ax.annotate("", xy=(px - d, py), xytext=(px, py),
                            arrowprops=dict(arrowstyle="-|>", color=Plotter.COLOR_P_LE,
                                            alpha=0.3 + 0.7 * le_i, lw=1 + 3 * le_i),
                            zorder=1)

        # ── o/l dot: whichever is larger per region, red for o, green for l ──
        mask_o = o >= l
        value  = np.where(mask_o, o, l)
        prop   = np.clip(value / s_max, 0.0, 1.0)

        colors = np.where(mask_o[:, None], to_rgba(Plotter.COLOR_O), to_rgba(Plotter.COLOR_L))
        colors[:, 3] = 0.2 + 0.8 * prop

        ax.scatter(points[:, 0], points[:, 1], s=_size(prop), color=colors,
                  edgecolors="white", linewidths=0.6, zorder=2)

        for (px, py), region_id in zip(points, map.index):
            ax.annotate(str(region_id), (px, py), fontsize=6,
                        xytext=(10, 10), textcoords="offset points",
                        alpha=0.8 if used_basemap else 0.7,
                        color="black" if used_basemap else None)

        if used_basemap:
            # The basemap covers the axes' current data limits, so it must
            # be drawn after the scatter.
            Plotter._draw_saved_basemap(ax)
            ax.set_axis_off()
        else:
            ax.set_xlabel("x", fontsize=Plotter.LABEL_FS)
            ax.set_ylabel("y", fontsize=Plotter.LABEL_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.set_aspect("equal")

        legend_elements = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_O,
                   markeredgecolor="white", markersize=8, label="overshoot (o)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=Plotter.COLOR_L,
                   markeredgecolor="white", markersize=8, label="leverage (l)"),
            Line2D([0], [0], marker=r"$\leftarrow$", color=Plotter.COLOR_P_LE,
                   linestyle="None", markersize=10, label="leaving users (p_le)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=Plotter.LEGEND_FS, framealpha=0.9)

        ax.set_title(title, fontsize=Plotter.TITLE_FS)

        if show:
            plt.tight_layout()
            plt.show()
        return fig

    # ── Trajectory animations ─────────────────────────────────────────────────

    @staticmethod
    def anim_state(x_traj, d_pu_traj, d_do_traj, params, x_max, d_max, title="State", interval=200):
        """Animate snapshot_state over a trajectory as a GIF.

        Args:
            x_traj:    np.ndarray of shape (T, N) — state trajectory.
            d_pu_traj: np.ndarray of shape (T, N) — PU demand trajectory.
            d_do_traj: np.ndarray of shape (T, N) — DO demand trajectory.
            params:    Params — passed through to snapshot_state.
            x_max:     float — passed through to snapshot_state.
            d_max:     float — passed through to snapshot_state.
            title:     str — base title; time step is appended per frame.
            interval:  int — milliseconds between frames.

        Returns:
            matplotlib FuncAnimation.
        """
        T   = len(x_traj)
        fig = plt.figure(figsize=(9, 9))

        def update(t):
            fig.clear()
            ax = fig.add_subplot(111)
            Plotter.snapshot_state(x_traj[t], d_pu_traj[t], d_do_traj[t], params, x_max, d_max,
                                   title=f"{title}  t={t}", show=False, ax=ax)

        anim = FuncAnimation(fig, update, frames=tqdm(range(T), desc="anim_state"), interval=interval)
        plt.close(fig)
        return anim

    @staticmethod
    def anim_leaving_users(b_iq_traj, d_pu_traj, p_le_traj, params, b_iq_max, d_max, p_le_max=1,
                           title="Leaving users", interval=200):
        """Animate snapshot_leaving_users over a trajectory as a GIF.

        Args:
            b_iq_traj: np.ndarray of shape (T, N) — capacity bound trajectory (PU).
            d_pu_traj: np.ndarray of shape (T, N) — PU demand trajectory.
            p_le_traj: np.ndarray of shape (T, N) — leaving-users trajectory.
            params:    Params — passed through to snapshot_leaving_users.
            b_iq_max:  float — passed through to snapshot_leaving_users.
            d_max:     float — passed through to snapshot_leaving_users.
            p_le_max:  float — passed through to snapshot_leaving_users (fixed at 1).
            title:     str — base title; time step is appended per frame.
            interval:  int — milliseconds between frames.

        Returns:
            matplotlib FuncAnimation.
        """
        T   = len(b_iq_traj)
        fig = plt.figure(figsize=(9, 9))

        def update(t):
            fig.clear()
            ax = fig.add_subplot(111)
            Plotter.snapshot_leaving_users(b_iq_traj[t], d_pu_traj[t], p_le_traj[t], params,
                                           b_iq_max, d_max, p_le_max,
                                           title=f"{title}  t={t}", show=False, ax=ax)

        anim = FuncAnimation(fig, update, frames=tqdm(range(T), desc="anim_leaving_users"), interval=interval)
        plt.close(fig)
        return anim

    @staticmethod
    def anim_incentive_response(th_traj, p_traj, params, th_max=1, p_max=1, arrow_thr=0.05,
                                title="Incentive response", interval=200):
        """Animate snapshot_incentive_response over a trajectory as a GIF.

        Args:
            th_traj:   np.ndarray of shape (T, N)        — incentive trajectory.
            p_traj:    np.ndarray of shape (T, N*(N+1))  — allocation trajectory from Simulator.
            params:    Params — passed through to snapshot_incentive_response.
            th_max:    float — passed through to snapshot_incentive_response (fixed at 1).
            p_max:     float — passed through to snapshot_incentive_response (fixed at 1).
            arrow_thr: float — passed through to snapshot_incentive_response.
            title:     str — base title; time step is appended per frame.
            interval:  int — milliseconds between frames.

        Returns:
            matplotlib FuncAnimation.
        """
        T   = len(th_traj)
        fig = plt.figure(figsize=(9, 9))

        def update(t):
            fig.clear()
            ax   = fig.add_subplot(111)
            th_t = Transformer.th_traj_at_t(th_traj, t)
            p_t  = Transformer.p_traj_at_t(p_traj, t)
            Plotter.snapshot_incentive_response(th_t, p_t, params, th_max, p_max, arrow_thr,
                                                title=f"{title}  t={t}", show=False, ax=ax)

        anim = FuncAnimation(fig, update, frames=tqdm(range(T), desc="anim_incentive_response"), interval=interval)
        plt.close(fig)
        return anim

    @staticmethod
    def anim_combined_incentive_leaving(th_traj, p_traj, p_le_traj, params, th_max=1, p_max=1,
                                        p_le_max=1, arrow_thr=0.05,
                                        title="Incentive response & leaving users", interval=200):
        """Animate snapshot_combined_incentive_leaving over a trajectory as a GIF.

        Args:
            th_traj:   np.ndarray of shape (T, N)        — incentive trajectory.
            p_traj:    np.ndarray of shape (T, N*(N+1))  — allocation trajectory from Simulator.
            p_le_traj: np.ndarray of shape (T, N)        — leaving-users trajectory.
            params:    Params — passed through to snapshot_combined_incentive_leaving.
            th_max:    float — passed through to snapshot_combined_incentive_leaving (fixed at 1).
            p_max:     float — passed through to snapshot_combined_incentive_leaving (fixed at 1).
            p_le_max:  float — passed through to snapshot_combined_incentive_leaving (fixed at 1).
            arrow_thr: float — passed through to snapshot_combined_incentive_leaving.
            title:     str — base title; time step is appended per frame.
            interval:  int — milliseconds between frames.

        Returns:
            matplotlib FuncAnimation.
        """
        T   = len(th_traj)
        fig = plt.figure(figsize=(9, 9))

        def update(t):
            fig.clear()
            ax   = fig.add_subplot(111)
            th_t = Transformer.th_traj_at_t(th_traj, t)
            p_t  = Transformer.p_traj_at_t(p_traj, t)
            Plotter.snapshot_combined_incentive_leaving(th_t, p_t, p_le_traj[t], params,
                                                        th_max, p_max, p_le_max, arrow_thr,
                                                        title=f"{title}  t={t}", show=False, ax=ax)

        anim = FuncAnimation(fig, update,
                             frames=tqdm(range(T), desc="anim_combined_incentive_leaving"),
                             interval=interval)
        plt.close(fig)
        return anim

    @staticmethod
    def anim_saturation(o_traj, l_traj, p_le_traj, params, s_max, p_le_max=1,
                        title="Saturation", interval=200):
        """Animate snapshot_saturation over a trajectory as a GIF.

        Args:
            o_traj:    np.ndarray of shape (T, N) — overshoot trajectory.
            l_traj:    np.ndarray of shape (T, N) — leverage trajectory.
            p_le_traj: np.ndarray of shape (T, N) — leaving-users trajectory.
            params:    Params — passed through to snapshot_saturation.
            s_max:     float — passed through to snapshot_saturation.
            p_le_max:  float — passed through to snapshot_saturation (fixed at 1).
            title:     str — base title; time step is appended per frame.
            interval:  int — milliseconds between frames.

        Returns:
            matplotlib FuncAnimation.
        """
        T   = len(o_traj)
        fig = plt.figure(figsize=(9, 9))

        def update(t):
            fig.clear()
            ax = fig.add_subplot(111)
            Plotter.snapshot_saturation(o_traj[t], l_traj[t], p_le_traj[t], params, s_max, p_le_max,
                                        title=f"{title}  t={t}", show=False, ax=ax)

        anim = FuncAnimation(fig, update, frames=tqdm(range(T), desc="anim_saturation"), interval=interval)
        plt.close(fig)
        return anim

    # ── Shared legend helper ──────────────────────────────────────────────────

    @staticmethod
    def _legend_above(fig, ax, handles, labels, fontsize, clear_pt, ceiling, set_top, max_cols=5,
                      handlelength=3):
        """Place a shared legend above ax, its right edge flush with ax's right edge.

        The legend sits clear_pt points above the top of ax (enough to clear
        ax's title), laid out in rows of at most max_cols entries. The figure
        is then drawn once to measure the legend, and set_top shifts the
        subplot area's top edge so that the legend's top lands exactly at
        `ceiling` (figure fraction) — however many rows it needed.

        Args:
            fig:      matplotlib Figure.
            ax:       Axes to place the legend above.
            handles:  list of artists — legend handles.
            labels:   list of str — legend labels.
            fontsize: float — legend font size.
            clear_pt: float — gap (points) between ax's top edge and the legend's bottom.
            ceiling:  float — figure-fraction height the legend's top should reach.
            set_top:  callable(delta) — move the subplot area's top edge down by delta
                      (figure fraction; negative moves it up).
            max_cols: int — maximum entries per legend row.
            handlelength: float — length of the legend's colour samples, in font-size units.

        Returns:
            matplotlib Legend.
        """
        from matplotlib.transforms import offset_copy

        ncol   = min(len(labels), max_cols)
        anchor = offset_copy(ax.transAxes, fig=fig, y=clear_pt, units="points")
        legend = ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                           bbox_transform=anchor, borderaxespad=0.0, ncol=ncol,
                           fontsize=fontsize, handlelength=handlelength, borderpad=0.8, columnspacing=1.5)

        fig.canvas.draw()
        legend_top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
        set_top(legend_top - ceiling)
        return legend

    # ── Optimiser diagnostics ─────────────────────────────────────────────────

    @staticmethod
    def plot_convergence(rel_err_traj, abs_err_traj, cost_hist, legend_name_list=None,
                         title="Convergence", show=True):
        """Plot relative error, absolute error, and cost convergence curves from proj_grad_desc.

        Accepts either a single run or multiple runs to overlay on the same
        three panels, one colour per run (matplotlib's default colour cycle,
        in run order — as in plot_agg_comp). When legend_name_list is given,
        the runs share one legend placed above the top panel, right-aligned
        with it (see _legend_above).

        Args:
            rel_err_traj: np.ndarray of shape (n_it,) — step-to-step change norm per iteration —
                          or a list of such, one per run.
            abs_err_traj: np.ndarray of shape (n_it,) — distance to converged value per iteration —
                          or a list of such, one per run.
            cost_hist:    np.ndarray of shape (n_it,) — objective cost per iteration —
                          or a list of such, one per run.
            legend_name_list: list of str — legend label per run; only meaningful
                          (and required) when passing lists of runs above.
            title:        str — figure title.
            show:         bool — call plt.show() at the end.

        Returns:
            matplotlib Figure.
        """
        multi = isinstance(rel_err_traj, list)
        rel_err_traj_list  = rel_err_traj  if multi else [rel_err_traj]
        abs_err_traj_list  = abs_err_traj  if multi else [abs_err_traj]
        cost_hist_list     = cost_hist     if multi else [cost_hist]

        colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, axes = plt.subplots(3, 1, figsize=(8, 9))
        suptitle = fig.suptitle(title, y=0.99, va="top", fontsize=Plotter.TITLE_FS)
        # Fixed margins (no tight_layout) so the shared legend above the top
        # panel stays inside the saved figure; top is refined by _legend_above.
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.07, hspace=0.45)

        for i, (rel_err, abs_err, cost) in enumerate(
                zip(rel_err_traj_list, abs_err_traj_list, cost_hist_list)):
            color = colors[i % len(colors)]
            name  = legend_name_list[i] if legend_name_list else None
            iters = np.arange(1, len(rel_err) + 1)

            axes[0].semilogy(iters, rel_err, color=color, linewidth=1.5, label=name)
            axes[1].semilogy(iters, abs_err, color=color, linewidth=1.5, label=name)
            # cost[k] is J(θ^k), with θ^0 the initial guess → 0-based
            axes[2].plot(np.arange(len(cost)), cost, color=color, linewidth=1.5, label=name)

        axes[0].set_ylabel("||θ_next − θ||", fontsize=Plotter.LABEL_FS)
        axes[0].set_title("relative error", fontsize=Plotter.TITLE_FS)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].set_ylabel("||θ − θ*||", fontsize=Plotter.LABEL_FS)
        axes[1].set_title("absolute error", fontsize=Plotter.TITLE_FS)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        axes[2].set_xlabel("Iteration", fontsize=Plotter.LABEL_FS)
        axes[2].set_ylabel("J(θ)", fontsize=Plotter.LABEL_FS)
        axes[2].set_title("cost", fontsize=Plotter.TITLE_FS)
        axes[2].set_ylim(bottom=0)
        axes[2].set_xlim(left=0)      # cost starts at θ^0
        axes[2].grid(True, linestyle="--", alpha=0.5)

        for ax in axes:
            ax.tick_params(labelsize=Plotter.TICK_FS)

        # One shared legend above the top panel, right-aligned with it — every
        # panel carries the same runs. Its top stops just under the suptitle.
        if legend_name_list:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.canvas.draw()
            suptitle_bottom = suptitle.get_window_extent().transformed(fig.transFigure.inverted()).y0
            title_h_pt      = axes[0].title.get_fontsize() * 1.2
            Plotter._legend_above(
                fig, axes[0], handles, labels, Plotter.LEGEND_FS,
                clear_pt=plt.rcParams["axes.titlepad"] + title_h_pt + 4,
                ceiling=suptitle_bottom - 0.01,
                set_top=lambda d: fig.subplots_adjust(top=fig.subplotpars.top - d),
                max_cols=4)

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_convergence_comp(cost_hist_list, show=True):
        """Plot the cost convergence of several runs as a fan chart: the median
        cost per iteration as a line, over a 25–75 % band and a lighter
        5–95 % band (percentiles across runs). Runs that stopped early are
        extended with their final cost, since a converged run stays there.
        Right of the last iterate, two short horizontal lines mark the lowest
        and the highest final cost, each pointed at by an arrow from outside
        (above the highest, below the lowest) marking the spread between runs;
        the spread as a % of the median final cost is written right of them
        (as in plot_final_comp, which uses the mean).
        The legend sits above the panel, right-aligned with it (see _legend_above).

        Args:
            cost_hist_list: list of np.ndarray of shape (n_it_i,) — objective cost per
                            iteration, one per run; lengths may differ.
            show:           bool — call plt.show() at the end.

        Returns:
            matplotlib Figure.
        """
        color     = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        # Pad every run to the longest one with its final cost → (n_runs, n_it)
        n_it  = max(len(cost) for cost in cost_hist_list)
        costs = np.vstack([np.pad(np.asarray(cost, dtype=float), (0, n_it - len(cost)), mode="edge")
                           for cost in cost_hist_list])
        iters = np.arange(n_it)    # cost[k] is J(θ^k), with θ^0 the initial guess
        p5, p25, p50, p75, p95 = np.percentile(costs, [5, 25, 50, 75, 95], axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))   # height leaves room for the legend above

        ax.fill_between(iters, p5,  p95, color=color, alpha=0.15, linewidth=0, label="5–95 %")
        ax.fill_between(iters, p25, p75, color=color, alpha=0.35, linewidth=0, label="25–75 %")
        ax.plot(iters, p50, color=color, linewidth=2, label="median")

        # Spread of the final cost across runs, right of the last iterate:
        # short lines at the lowest and highest final cost, each with an
        # arrow pointing at it from outside (fixed length in points, so it
        # stays visible however close the lines are), with the spread as a %
        # of the median beside them.
        final = costs[:, -1]
        lo, hi = final.min(), final.max()
        k_last = n_it - 1   # last iterate, in iterations
        dx = 0.02 * n_it    # horizontal step, in iterations
        arrow_pt = 18       # arrow length, in points
        ax.hlines([lo, hi], k_last + dx, k_last + 3 * dx, color="black", linewidth=1.5)
        for y, sign in ((hi, 1), (lo, -1)):
            ax.annotate("", xy=(k_last + 2 * dx, y), xytext=(0, sign * arrow_pt),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2,
                                        shrinkA=0, shrinkB=0))
        spread_pct = (hi - lo) / np.median(final) * 100
        spread_label = ax.text(k_last + 4 * dx, (hi + lo) / 2, f"{spread_pct:.1f} %",
                               ha="left", va="center", fontsize=Plotter.TICK_FS)
        ax.set_xlim(0, k_last + 12 * dx)    # start at θ^0; provisional right end

        ax.set_xlabel(r"iteration $k$", fontsize=Plotter.LABEL_FS)
        ax.set_ylabel(r"$V_0(x_0,\boldsymbol{\theta}^k)$", fontsize=Plotter.LABEL_FS)
        ax.tick_params(labelsize=Plotter.TICK_FS)
        # Start at 0; top derived from the data (5 % headroom above the
        # 95th percentile) so it survives cost-scale changes (e.g. a different dt).
        y_hi = p95.max()
        ax.set_ylim(0, 1.05 * y_hi if y_hi > 0 else 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)

        ax.yaxis.get_offset_text().set_fontsize(Plotter.OFFSET_FS)

        # Fit the margins to the labels, so larger fonts are never clipped.
        plt.tight_layout()

        # Pull the right end in to just past the spread label. The label's
        # width is fixed in pixels, so in data units it depends on the x-range:
        # solve x_left + pad + w_px · R / W_px = R for the right end R.
        fig.canvas.draw()
        label_px = spread_label.get_window_extent().width
        ax_px    = ax.get_window_extent().width
        ax.set_xlim(0, (k_last + 4 * dx + dx) / (1 - label_px / ax_px))

        # Legend above the panel, right-aligned with it, as in plot_agg_comp;
        # median first, then the bands from narrow to wide.
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 1, 0]
        Plotter._legend_above(
            fig, ax, [handles[k] for k in order], [labels[k] for k in order],
            Plotter.LEGEND_FS,
            clear_pt=Plotter.OFFSET_FS + 6,    # clear the y-axis scale label ("1e6")
            ceiling=0.99,
            set_top=lambda d: fig.subplots_adjust(top=fig.subplotpars.top - d),
            handlelength=0.8)

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_final_comp(final_cost_list, total_u_le_list, show=True):
        """Plot the final cost and the total leaving users of every run, side
        by side, one dot per run, all on the same vertical line (x = 1, marked
        by a single unlabelled tick); the colour matches plot_convergence_comp.
        Two short horizontal lines right of the dots mark the lowest and the
        highest value, joined by a double-headed arrow showing the spread
        between runs; the spread as a % of the mean is written right of it.

        Args:
            final_cost_list: list of float — final (converged) cost J(θ), one per run.
            total_u_le_list: list of float — total leaving users, one per run.
            show:            bool — call plt.show() at the end.

        Returns:
            matplotlib Figure.
        """
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        for ax, values, subtitle in (
                (axes[0], final_cost_list, "J(θ) - Total cost"),
                (axes[1], total_u_le_list, "Total leaving users")):
            ax.scatter(np.ones(len(values)), values, color=color, alpha=0.7, zorder=3)
            # Lowest and highest value, as short lines just right of the dots
            ax.hlines([min(values), max(values)], 1.06, 1.10, color="black", linewidth=1.5)
            # Double-headed arrow spanning the two lines, marking the spread
            ax.annotate("", xy=(1.08, max(values)), xytext=(1.08, min(values)),
                        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2,
                                        shrinkA=0, shrinkB=0))
            # Spread (highest − lowest) relative to the mean, right of the arrow
            spread_pct = (max(values) - min(values)) / np.mean(values) * 100
            ax.text(1.12, (max(values) + min(values)) / 2, f"{spread_pct:.1f} %",
                    ha="left", va="center")
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([1])
            ax.set_xticklabels([])
            ax.set_title(subtitle, fontsize=Plotter.TITLE_FS)
            ax.tick_params(labelsize=Plotter.TICK_FS)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.set_ylim(0, 1.08 * max(values))    # headroom above the top line

        fig.tight_layout()

        if show:
            plt.show()
        return fig

    # ── Comparative plots ─────────────────────────────────────────────────────

    @staticmethod
    def plot_agg_comp(z_traj_list, th_traj_list, d_traj_list, o_traj_list, l_traj_list,
                      legend_name_list, params, d_max, o_max, th_max, u_le_max,
                      title="Aggregate trajectories — comparison", show=True,
                      bundle_label=None):
        """Plot aggregated demand, incentive, and leaving-users trajectories,
        overlaying one series per entry per metric.

        Panel order (top to bottom): pick-up demand, incentives, leaving users.
        Panels carry no titles; each is identified by its y-axis label.

        Demand is identical across series, so it is drawn once (from the
        first entry) with no legend. Every other fixed-bound panel's y-range
        is set to its metric's known bound rather than autoscaled, so panels
        are directly comparable across figures/series: demand in [0, d_max],
        incentives in [0, th_max], leaving users in [0, u_le_max].
        Every y-range is padded by 5% of its span on both ends. The series share one
        legend, placed above the top (demand) panel and right-aligned with it
        (see _legend_above); every
        visible x-axis is labelled "time" and ticked every 3 hours
        (12 timesteps of 15 min) as HH:00 from 00:00.

        Args:
            z_traj_list:      list of np.ndarray, each of shape (T+1, N+1) — extended state trajectories.
            th_traj_list:     list of np.ndarray, each of shape (T, N)     — incentive trajectories.
            d_traj_list:      list of [d_do (T, N), d_pu (T, N)] — demand trajectories.
            o_traj_list:      list of np.ndarray, each of shape (T, N)     — overshoot
                              trajectories; accepted for call-site compatibility, not plotted.
            l_traj_list:      list of np.ndarray, each of shape (T, N)     — leverage
                              trajectories; accepted for call-site compatibility, not plotted.
            legend_name_list: list of str — legend label per trajectory tuple.
            params:           Params — provides N.
            d_max:            float — demand panel range is [0, d_max], on the
                              *aggregate* (summed over N) scale, not per-node.
            o_max:            float — accepted for call-site compatibility, unused.
            th_max:           float — incentives panel range is [0, th_max], on the aggregate scale.
            u_le_max:         float — leaving-users panel range is [0, u_le_max]; x_le is
                              z_traj's single outside-state column, already scalar (not summed).
            title:            str — accepted for call-site compatibility; the figure has
                              no titles.
            show:             bool — call plt.show() at the end.
            bundle_label:     str or None — if given, every series except the last is
                              drawn as one thin, translucent bundle in a single colour
                              under this one legend label, and the last series (e.g. the
                              zero-incentive reference) is drawn bold under its own name.
                              Use for sweeps with many runs (e.g. 100 random th0), where
                              per-run colours repeat and a per-run legend can't fit.

        Returns:
            matplotlib Figure.
        """
        colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_groups  = 3
        margin    = 0.05  # y-range padding on both ends, as a fraction of the span
        label_fs  = Plotter.LABEL_FS
        tick_fs   = Plotter.TICK_FS
        legend_fs = Plotter.LEGEND_FS
        legend_ceiling = 0.99  # figure-fraction height the shared legend's top reaches
        legend_gap = 6    # gap (points) between the top panel and the shared legend

        def _ylim(ax, lo, hi):
            """Set the y-range to [lo, hi], padded by margin * (hi - lo) on both ends."""
            if hi <= lo:  # degenerate range (e.g. all-zero series) — fall back to unit span
                hi = lo + 1.0
            pad = margin * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        # Each timestep is a 15-minute window from 00:00, so 4 steps make an hour.
        STEPS_PER_HOUR = 4
        HOURS_PER_TICK = 3
        hour_ticks     = np.arange(0, params.T + 1, STEPS_PER_HOUR * HOURS_PER_TICK)
        hour_labels    = [f"{t // STEPS_PER_HOUR:02d}:00" for t in hour_ticks]

        def _time_axis(ax):
            """Label the x-axis in clock time — one tick every HOURS_PER_TICK hours, starting at 00:00.
            The range is exactly [00:00, 24:00] (timesteps [0, T]), with no margin."""
            ax.set_xlim(0, params.T)
            ax.set_xticks(hour_ticks)
            ax.set_xticklabels(hour_labels)
            ax.set_xlabel("time", fontsize=label_fs)

        fig    = plt.figure(figsize=(12, 3 * n_groups))
        # top is refined by _legend_above so the legend above the top panel fits.
        outer  = fig.add_gridspec(n_groups, 1, left=0.08, right=0.97, top=0.92, bottom=0.08,
                                  hspace=0.5)

        ax_pu = fig.add_subplot(outer[0])
        ax_th = fig.add_subplot(outer[1])
        ax_le = fig.add_subplot(outer[2])

        for i, (z_traj, th_traj, d_traj, name) in enumerate(
                zip(z_traj_list, th_traj_list, d_traj_list, legend_name_list)):
            color = colors[i % len(colors)]
            _, x_le  = Transformer.agg_z_traj(z_traj, params)
            th_agg   = Transformer.agg_th_traj(th_traj)

            t_x  = np.arange(len(x_le))
            t_th = np.arange(len(th_agg))

            # Demand is identical across series — draw it once, no legend.
            if i == 0:
                _, d_pu_agg = Transformer.agg_d_traj(d_traj)
                ax_pu.step(np.arange(len(d_pu_agg)), d_pu_agg, where="post", color="black")

            style = {"color": color, "label": name}
            if bundle_label is not None:
                if i < len(z_traj_list) - 1:
                    # One legend entry for the whole bundle (labels starting "_" are hidden).
                    style = {"color": colors[0], "alpha": 0.3, "linewidth": 0.8,
                             "label": bundle_label if i == 0 else "_bundle"}
                else:
                    style = {"color": colors[3], "linewidth": 2.0, "label": name}

            ax_th.step(t_th, th_agg, where="post", **style)
            ax_le.step(t_x,  x_le,   where="post", **style)

        ax_pu.set_ylabel("demand (pick-up)", fontsize=label_fs)
        _time_axis(ax_pu)
        _ylim(ax_pu, 0, d_max)
        ax_pu.grid(True, linestyle="--", alpha=0.5)

        ax_th.set_ylabel("incentives", fontsize=label_fs)
        _time_axis(ax_th)
        _ylim(ax_th, 0, th_max)
        ax_th.grid(True, linestyle="--", alpha=0.5)

        ax_le.set_ylabel("leaving users", fontsize=label_fs)
        _time_axis(ax_le)
        _ylim(ax_le, 0, u_le_max)
        ax_le.grid(True, linestyle="--", alpha=0.5)

        for ax in (ax_pu, ax_th, ax_le):
            ax.tick_params(labelsize=tick_fs)

        # Line the y-labels up in one column despite differing tick-label widths.
        fig.align_ylabels()

        # One shared legend above the top panel, right-aligned with it — every
        # legended panel carries the same series.
        handles, labels = ax_th.get_legend_handles_labels()
        legend = Plotter._legend_above(
            fig, ax_pu, handles, labels, legend_fs,
            clear_pt=legend_gap,
            ceiling=legend_ceiling,
            set_top=lambda d: outer.update(top=outer.top - d))
        if bundle_label is not None:
            # The bundle's lines are faint by design; keep its legend sample readable.
            for line in legend.get_lines():
                line.set_alpha(1.0)
                line.set_linewidth(2.0)

        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_leaving_days(total_u_le_base_list, total_u_le_opt_list,
                          legend_name_list,
                          title="Total leaving users - across different days",
                          show=True):
        """Per-day comparison of baseline vs optimized total leaving users
        across a demand sweep.

        One panel, clustered bar chart: for each day a "zero-incentive" bar and an
        "Algorithm 1" bar side by side. A vertical arrow spans the two bar tops
        of each day, labelled with the optimized reduction relative to the
        baseline in percent ("-x%", or "+x%" if optimized is worse).

        Colours and legend follow plot_agg_comp: "zero-incentive" and
        "Algorithm 1" take the first two colours of matplotlib's default cycle
        (blue, orange). The legend sits above the panel, right-aligned with it
        (see _legend_above).

        Args:
            total_u_le_base_list: list of float — baseline total leaving users per day.
            total_u_le_opt_list:  list of float — optimized total leaving users per day.
            legend_name_list:     list of str — x-axis label per day. "YYYY-MM-DD"
                                  dates are shown as "MM-DD"; other tags as given.
            title:                str — unused; the figure is drawn without a title.
            show:                bool — call plt.show() at the end.

        Returns:
            matplotlib Figure.
        """
        n_days = len(legend_name_list)
        spacing = 1.35  # distance between neighbouring days, in x units; just clears the % labels
        x       = np.arange(n_days) * spacing
        width   = 0.45  # per bar; leaves a spacing - 2*width gap between days

        base = np.asarray(total_u_le_base_list, dtype=float)
        opt  = np.asarray(total_u_le_opt_list,  dtype=float)

        # Optimized reduction relative to baseline, in percent (positive = fewer
        # leaving users). Guard against a zero baseline.
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_reduction = np.where(base != 0, (base - opt) / base * 100.0, np.nan)

        # Same colours as plot_agg_comp, where "zero-incentive" and "Algorithm 1"
        # are the first two series of the default colour cycle.
        colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
        label_fs  = Plotter.LABEL_FS
        tick_fs   = Plotter.TICK_FS
        legend_fs = Plotter.LEGEND_FS
        margin    = 0.05  # y-range padding above the tallest bar, as a fraction of it
        legend_ceiling = 0.99  # figure-fraction height the legend's top reaches
        legend_gap = 6    # gap (points) between the panel and the legend

        # "YYYY-MM-DD" → "MM-DD"; any other tag (e.g. "run") is kept as is.
        day_labels = [name[5:] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name) else name
                      for name in legend_name_list]

        # ~0.9 in per unit of day spacing for horizontal date labels.
        fig, ax = plt.subplots(figsize=(max(8, n_days * spacing * 0.9 + 1), 5))

        ax.bar(x - width / 2, base, width,
               color=colors[0], label="zero-incentive", zorder=2)
        ax.bar(x + width / 2, opt, width,
               color=colors[1], label="Algorithm 1", zorder=2)

        # Vertical reduction arrow per day, drawn at the cluster centre from the
        # baseline bar top down to the optimized bar top, labelled "-x%".
        for xi, base_i, opt_i, pct_i in zip(x, base, opt, pct_reduction):
            if np.isnan(pct_i):
                continue
            ax.annotate("", xy=(xi, opt_i), xytext=(xi, base_i),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, mutation_scale=22,
                                        shrinkA=0, shrinkB=0),
                        zorder=4)
            ax.text(xi, (base_i + opt_i) / 2, f"{-pct_i:+.1f}%",
                    ha="center", va="center", fontsize=tick_fs, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

        ax.set_xticks(x)
        ax.set_xticklabels(day_labels)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_xlabel("days", fontsize=label_fs)
        ax.set_ylabel("number of leaving users", fontsize=label_fs)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        bar_max = np.nanmax(np.concatenate([base, opt])) if n_days else 0.0
        ax.set_ylim(0, bar_max * (1 + margin) if bar_max > 0 else 1.0)

        plt.tight_layout()

        # Legend above the panel, right-aligned with it, as in plot_agg_comp;
        # top is refined by _legend_above so the legend fits in the figure.
        handles, labels = ax.get_legend_handles_labels()
        Plotter._legend_above(
            fig, ax, handles, labels, legend_fs,
            clear_pt=legend_gap,
            ceiling=legend_ceiling,
            set_top=lambda d: fig.subplots_adjust(top=fig.subplotpars.top - d))

        if show:
            plt.show()
        return fig

    # ── Comparison tables ─────────────────────────────────────────────────────

    @staticmethod
    def gen_tab_u_le_comp(u_traj_baseline, u_traj_list, legend_name_list, params,
                          title="Outside-state flow comparison", show=True):
        """Build an annotated table comparing total outside-state flow (u_le)
        for one or more trajectories against a baseline.

        Args:
            u_traj_baseline:  np.ndarray of shape (T, N+1) — baseline control input trajectory.
            u_traj_list:      list of np.ndarray, each of shape (T, N+1) — control trajectories to compare.
            legend_name_list: list of str — label per entry in u_traj_list.
            params:           Params — provides Xi_x, Xi_le, dt (passed through to
                              Transformer.get_total_u_le).
            title:            str — table title.
            show:             bool — call plt.show() at the end.

        Returns:
            matplotlib Figure containing the annotated comparison table.
        """
        total_u_le_baseline = Transformer.get_total_u_le(u_traj_baseline, params)

        names  = ["zero-incentive"] + list(legend_name_list)
        totals = [total_u_le_baseline] + [Transformer.get_total_u_le(u, params) for u in u_traj_list]
        pct_diffs = [(total - total_u_le_baseline) / total_u_le_baseline * 100 for total in totals]

        cell_text = [[name, f"{total:.2f}", f"{pct:+.2f}%"]
                     for name, total, pct in zip(names, totals, pct_diffs)]

        fig, ax = plt.subplots(figsize=(6, 0.6 * len(names) + 1))
        ax.axis("off")
        fig.suptitle(title, fontsize=Plotter.TITLE_FS)

        table = ax.table(cellText=cell_text,
                          colLabels=["Algorithms", "Leaving users", "Reduction"],
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        if show:
            plt.tight_layout()
            plt.show()
        return fig
