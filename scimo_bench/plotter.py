"""Plots from the result files of one run folder (no physics here).

A plot is a function `fn(p) -> figure` decorated with `@plot` (`scimo_bench.catalogue`),
in the dataset's `plots.py`; `p` is a `PlotContext` with the run's files and helpers:

    @plot
    def jet_pt(p):
        fig, ax, rax = p.figure(ratio=True)
        p.hist(ax, "jet_pt", np.linspace(0, 200, 41), ratio_ax=rax)
        p.legend(ax)
        return fig

The figure is saved as `plots/<function name>.<fmt>`; a figure without data is skipped.
Helpers label every curve with its level: truth is "truth", a model's output
"<model> (reco)"; an axis says "(truth)" or "(reco)" when only one level is drawn.
Generic plots (any dataset): `ranking` here, and the Pareto plot (`plots.pareto`).
The look of all plots comes from `Style` (`plots.style`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scimo_bench.catalogue import Catalogue, plot, split  # noqa: E402
from scimo_bench.scorer import Metric, better_key, overall  # noqa: E402

log = logging.getLogger(__name__)

PARETO_COLUMNS = [
    "plot",
    "model",
    "condition",
    "x_metric",
    "y_metric",
    "x",
    "x_lower",
    "x_upper",
    "y",
    "y_lower",
    "y_upper",
    "on_front",
]


@dataclass
class Style:
    """The look of all plots (config `plots.style`)."""

    # categorical order validated for colour-vision deficiency; fixed, never cycled
    palette: list[str] = field(
        default_factory=lambda: [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ]
    )
    linestyles: list[str] = field(default_factory=lambda: ["-", "--", "-.", ":"])
    truth_color: str = "#b4b3ad"
    ink: str = "#0b0b0b"
    muted: str = "#52514e"
    grid_color: str = "grey"
    grid_linestyle: str = ":"
    grid_alpha: float = 0.3
    font_size: float = 10
    line_width: float = 2.5
    hist_line_width: float = 1.5  # model lines in histograms
    ratio_marker_size: float = 3.5  # points in ratio panels
    figsize: Sequence[float] = (6.0, 4.0)
    ratio_figsize: Sequence[float] = (6.0, 5.0)
    dpi: int = 150
    legend: str = "above"  # above the axes, or a matplotlib `loc`
    rc: dict = field(default_factory=dict)  # extra matplotlib rcParams, applied last

    @classmethod
    def from_cfg(cls, cfg: Mapping | None) -> Style:
        """From the `plots.style` config (unknown keys are an error)."""
        cfg = dict(cfg or {})
        unknown = set(cfg) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown plots.style keys {sorted(unknown)}")
        style = cls(**cfg)
        for key in ("truth_color", "ink", "muted", "grid_color"):
            check_colour(getattr(style, key), f"plots.style.{key}")
        for c in style.palette:
            check_colour(c, "plots.style.palette")
        return style

    def rc_params(self) -> dict:
        """matplotlib rcParams for this style."""
        return {
            "axes.edgecolor": self.ink,
            "axes.labelcolor": self.ink,
            "axes.grid": True,
            "grid.color": self.grid_color,
            "grid.linestyle": self.grid_linestyle,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": True,  # a full frame around every plot
            "axes.spines.right": True,
            "xtick.color": self.ink,
            "ytick.color": self.ink,
            "legend.frameon": False,
            "lines.linewidth": self.line_width,
            "font.size": self.font_size,
            "svg.hashsalt": "scimo_bench",
            **dict(self.rc),
        }


def check_colour(colour: str, where: str) -> None:
    """A colour must be a matplotlib colour (a name like `red`, `tab:blue`, or hex)."""
    if not mcolors.is_color_like(colour):
        raise ValueError(f"{where}: unknown colour '{colour}' (use a matplotlib name, e.g. red)")


def _slot(name: str, n: int) -> int:
    """Palette slot of a model name (stable hash)."""
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % n


def model_styles(
    names: list[str], style: Style | None = None, overrides: Mapping | None = None
) -> dict[str, dict]:
    """Colour and line style per model, the same in every plot and run.

    Colour: a palette slot from a hash of the name (collisions resolved in sorted name
    order). Line style, the second cue besides colour: position in the sorted names. Past
    the palette, colours repeat with a line style not yet used with that colour; past all
    colour × style pairs, the pairs repeat. `overrides` {model: {color, linestyle}} win.
    """
    style = style or Style()
    palette, linestyles = list(style.palette), list(style.linestyles)
    n_col, n_ls = len(palette), len(linestyles)
    taken: set[tuple[int, int]] = set()
    out = {}
    for i, name in enumerate(sorted(set(names))):
        if len(taken) == n_col * n_ls:
            taken.clear()
        slot, ls = _slot(name, n_col), i % n_ls
        used = {c for c, _ in taken}
        if len(used) < n_col:
            while slot in used:
                slot = (slot + 1) % n_col
        else:
            free = [(c, s) for c in range(n_col) for s in range(n_ls) if (c, s) not in taken]
            slot, ls = min(free, key=lambda p: ((p[0] - slot) % n_col, (p[1] - ls) % n_ls))
        taken.add((slot, ls))
        out[name] = dict(color=palette[slot], linestyle=linestyles[ls])
        for key, value in dict((overrides or {}).get(name) or {}).items():
            if value is not None:
                out[name][key] = value
    return out


@dataclass
class PlotContext:
    """What a plot function gets: the run's result files, the models with their labels and
    styles, the task's metrics, the dataset's catalogue (labels, units), and helpers."""

    folder: Path
    report: pd.DataFrame
    rankings: pd.DataFrame | None
    per_event: ak.Array | None
    models: list[str]
    labels: dict[str, str]
    metrics: dict[str, Metric]
    catalogue: Catalogue = field(default_factory=Catalogue)
    formats: list[str] = field(default_factory=lambda: ["pdf"])
    task: str = ""
    style: Style = field(default_factory=Style)
    model_overrides: dict = field(default_factory=dict)  # {model: {color, linestyle}}
    pareto_rows: list[dict] = field(default_factory=list)

    @property
    def styles(self) -> dict[str, dict]:
        """Colour and line style per model."""
        return model_styles(self.models, self.style, self.model_overrides)

    def label(self, model: str) -> str:
        """Display label of a model."""
        return self.labels.get(model, model)

    def select(self, subset: Sequence[str] | None = None) -> list[str]:
        """The run's models, restricted to `subset` if given (order kept)."""
        return [m for m in self.models if subset is None or m in set(subset)]

    def values(self, side: str, name: str) -> np.ndarray | None:
        """Finite values of an observable or comparison from per_event.parquet, for
        `side` = "truth" or a model name (its reco output); flattened if per object."""
        pe = self.per_event
        if pe is None or name not in pe.fields:
            return None
        v = pe[pe["model"] == side][name]
        v = v[~ak.is_none(v, axis=0)]
        if len(v) == 0:
            return None
        if v.ndim > 1:
            v = ak.flatten(v, axis=1)
        a = ak.to_numpy(v).astype(np.float64)
        return a[np.isfinite(a)]

    # --- building blocks for plot functions ------------------------------------------------

    def figure(self, ratio: bool = False, figsize: Sequence[float] | None = None):
        """A figure: (fig, ax), or (fig, ax, ratio_ax) with a ratio panel below."""
        if ratio:
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=tuple(figsize or self.style.ratio_figsize),
                sharex=True,
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08),
            )
            return fig, ax, rax
        fig, ax = plt.subplots(figsize=tuple(figsize or self.style.figsize))
        return fig, ax

    def hist(
        self,
        ax,
        name: str,
        bins: Sequence[float],
        ratio_ax=None,
        truth: bool = True,
        models: Sequence[str] | None = None,
        density: bool = True,
    ) -> None:
        """Histogram of an observable or comparison, with bin edges `bins`.

        Observable: the truth as a grey filled histogram ("truth"), each model's output as a
        line ("<model> (reco)"), and with `ratio_ax` the ratio pred / truth as points at the
        bin centres in the model's colour (range 0.5–1.5). Comparison: one
        line per model. The x label comes from the catalogue; it says "(truth)" or "(reco)"
        when only one level is drawn. Nothing is drawn if there is no data.
        """
        info = self.catalogue.info(name)
        is_obs = info.kind == "observable"
        edges = np.asarray(bins, dtype=float)
        t = self.values("truth", name) if is_obs and truth else None
        t = t if t is not None and len(t) else None
        data = {m: self.values(m, name) for m in self.select(models)}
        data = {m: v for m, v in data.items() if v is not None and len(v)}
        if t is None and not data:
            log.warning("hist %s: no data", name)
            return
        h_t = None
        if t is not None:
            h_t, _ = np.histogram(t, edges, density=density)
            ax.stairs(h_t, edges, fill=True, color=self.style.truth_color, label="truth", lw=0)
        for m, v in data.items():
            st = self.styles[m]
            h, _ = np.histogram(v, edges, density=density)
            label = self.label(m) + (" (reco)" if is_obs else "")
            ax.stairs(h, edges, label=label, lw=self.style.hist_line_width, **st)
            if ratio_ax is not None and h_t is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(h_t > 0, h / h_t, np.nan)
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    ratio_ax.plot(
                        centres,
                        r,
                        "o",
                        ms=self.style.ratio_marker_size,
                        color=st["color"],
                        ls="none",
                    )
        if info.reference is not None:
            ax.axvline(info.reference, color=self.style.muted, lw=1, ls=":")
        ax.set_ylabel("entries (normalized)" if density else "entries")
        side = None
        if is_obs and not data:
            side = "truth"
        elif is_obs and t is None:
            side = "reco"
        (ratio_ax if ratio_ax is not None else ax).set_xlabel(info.axis_label(side))
        if ratio_ax is not None:
            ratio_ax.axhline(1.0, color=self.style.muted, lw=1)
            ratio_ax.set_ylabel("pred / truth")
            ratio_ax.set_ylim(0.5, 1.5)
            ratio_ax.set_yticks([0.5, 1.0, 1.5])

    def metric_vs_x(
        self, ax, metric: str, log_x: bool = False, models: Sequence[str] | None = None
    ) -> None:
        """A binned metric from report.csv against its x, one line per model with its
        interval band. The x label names the side of x (e.g. "jet $p_T$ (truth)")."""
        m = self.metrics.get(metric)
        rows = self.report[(self.report["metric"] == metric) & self.report["bin_low"].notna()]
        if m is None or m.x is None or rows.empty:
            log.warning("metric_vs_x %s: no binned rows", metric)
            return
        drawn = False
        for model in self.select(models):
            r = rows[rows["model"] == model].sort_values("bin_low")
            ok = r["status"] == "ok"
            if not ok.any():
                continue
            lo_e, hi_e = r["bin_low"].to_numpy(), r["bin_high"].to_numpy()
            x = np.sqrt(lo_e * hi_e) if log_x and lo_e.min() > 0 else 0.5 * (lo_e + hi_e)
            y = np.where(ok, r["value"], np.nan)
            lo, hi = np.where(ok, r["lower"], np.nan), np.where(ok, r["upper"], np.nan)
            st = self.styles[model]
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, lw=0)
            ax.plot(x, y, label=self.label(model), marker="o", ms=4, **st)
            drawn = True
        if not drawn:
            return
        ref = self.catalogue.info(m.variable).reference
        if m.location and ref is not None:
            ax.axhline(ref, color=self.style.muted, lw=1, ls=":")
        side, _ = split(m.x)
        ax.set_xlabel(self.catalogue.info(m.x).axis_label(side))
        ax.set_ylabel(m.label)
        if log_x:
            self.log_x(ax)

    @staticmethod
    def log_x(ax) -> None:
        """Log x axis with plain number labels."""
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    def legend(self, ax, title: str | None = None) -> None:
        """Legend above the axes (never over the data) or at `style.legend`, with the task
        (or `title`) as its title; one series is named in the title instead."""
        title = title if title is not None else self.task
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) < 2:
            title = f"{title} · {labels[0]}" if labels else title
            ax.set_title(title, loc="left", fontsize=self.style.font_size - 1, color=self.style.ink)
            return
        if self.style.legend == "above":
            leg = ax.legend(
                loc="lower left",
                bbox_to_anchor=(0, 1.02),
                borderaxespad=0,
                title=title,
                alignment="left",
            )
        else:
            leg = ax.legend(loc=self.style.legend, title=title, alignment="left")
        leg.get_title().set_color(self.style.ink)
        leg.get_title().set_fontsize(self.style.font_size - 1)

    def save(self, fig, name: str) -> list[Path]:
        """Write the figure in every format to `plots/<name>.<fmt>` (deterministic files)."""
        out = self.folder / "plots"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for fmt in self.formats:
            p = out / f"{name}.{fmt}"
            meta = {"CreationDate": None} if fmt == "pdf" else {"Software": None}
            fig.savefig(p, metadata=meta, bbox_inches="tight", dpi=self.style.dpi)
            paths.append(p)
        plt.close(fig)
        return paths


# --- generic plots -------------------------------------------------------------------------


@plot
def ranking(p: PlotContext):
    """Rank of every model per metric (rows), the value written in each cell."""
    rk = p.rankings
    if rk is None or rk.empty:
        log.warning("plot ranking: no rankings.csv")
        return None
    metrics = list(dict.fromkeys(rk["metric"]))
    models = [m for m in p.select() if m in set(rk["model"])]
    ranks = np.full((len(metrics), len(models)), np.nan)
    vmax = max(len(models), 2) + 1.5
    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(models), 0.9 + 0.45 * len(metrics)))
    for i, met in enumerate(metrics):
        for j, m in enumerate(models):
            r = rk[(rk["metric"] == met) & (rk["model"] == m)].iloc[0]
            ranks[i, j] = r["rank"]
            txt = f"#{int(r['rank'])}" if r["status"] == "ok" else r["status"]
            if r["status"] == "ok":
                txt += f"\n{r['value']:.4g}"
            dark = r["rank"] <= 1 + 0.4 * (vmax - 1)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=p.style.font_size - 2,
                color="white" if dark else p.style.ink,
            )
    ax.imshow(ranks, cmap="Blues_r", vmin=1, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(models)), [p.label(m) for m in models], rotation=20, ha="right")
    ax.set_yticks(range(len(metrics)), metrics)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    title = f"{p.task}: rank per metric (1 = best)"
    ax.set_title(title, loc="left", fontsize=p.style.font_size - 1, color=p.style.ink)
    return fig


GENERIC_PLOTS: dict[str, Callable] = {"ranking": ranking}


def parse_condition(cond: str) -> dict[str, str]:
    """`gen=…;scenario=…;batch_size=…;device=…` → dict."""
    out = {}
    for part in str(cond).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def pareto_front(x, y, y_better="lower", x_err=None, y_err=None, robust=False) -> np.ndarray:
    """Non-dominated points; x lower is better, y by `y_better`.

    `x_err`/`y_err` = (lower, upper) arrays; with `robust`, a point only dominates another
    when its whole interval is at least as good as the other's best edge in both axes.
    """
    x = np.asarray(x, dtype=float)
    ky = better_key(y_better, y)
    if robust and x_err is not None and y_err is not None:
        x_best, x_worst = np.asarray(x_err[0], float), np.asarray(x_err[1], float)
        k_lo, k_hi = better_key(y_better, y_err[0]), better_key(y_better, y_err[1])
        y_best, y_worst = np.minimum(k_lo, k_hi), np.maximum(k_lo, k_hi)
        if isinstance(y_better, Mapping):
            c = float(y_better["closest_to"])
            inside = (np.asarray(y_err[0]) <= c) & (c <= np.asarray(y_err[1]))
            y_best = np.where(inside, 0.0, y_best)
    else:
        x_best = x_worst = x
        y_best = y_worst = ky
    n = len(x)
    front = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weak = x_worst[j] <= x_best[i] and y_worst[j] <= y_best[i]
            strict = x_worst[j] < x_best[i] or y_worst[j] < y_best[i]
            if weak and strict:
                front[i] = False
                break
    return front


class ParetoPlot:
    """Quality metric `y` against cost `x` (a perf metric under one fixed perf condition,
    e.g. {scenario: single, batch_size: 1}); writes its points to `pareto.csv`.
    `connect_sweeps`: join a model's points; `robust_dominance`: use the intervals;
    `xlim`/`ylim`: axis ranges [low, high]. Each model is named next to its point (no
    legend)."""

    def __init__(
        self,
        y: str,
        x: str,
        perf_condition: Mapping | None = None,
        x_log: bool = True,
        y_log: bool = False,
        connect_sweeps: bool = True,
        robust_dominance: bool = False,
        show_front: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: Sequence[float] | None = None,
        ylim: Sequence[float] | None = None,
        title: str | None = None,
        figsize: list[float] | None = None,
        models: list[str] | None = None,
        name: str | None = None,
    ):
        self.y = y
        self.x = x
        self.perf_condition = dict(perf_condition or {})
        self.x_log = x_log
        self.y_log = y_log
        self.connect_sweeps = connect_sweeps
        self.robust_dominance = robust_dominance
        self.show_front = show_front
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.figsize = figsize
        self.models = models
        self.name = name or f"pareto_{y}_vs_{x}"

    def matches(self, cond: dict[str, str]) -> bool:
        """Whether a perf condition matches `perf_condition` (device: substring match)."""
        for k, v in self.perf_condition.items():
            have = cond.get(k)
            if have is None:
                return False
            if k == "device":
                if str(v).lower() not in have.lower():
                    return False
            elif str(v) != have:
                return False
        return True

    def points(self, report: pd.DataFrame) -> pd.DataFrame:
        """(quality, cost) pairs: quality and perf rows of the same model and generation."""
        q = overall(report[(report["kind"] == "quality") & (report["metric"] == self.y)])
        q = q[q["status"] == "ok"].copy()
        q["gen"] = [parse_condition(c).get("gen") for c in q["condition"]]
        p = report[(report["kind"] == "perf") & (report["metric"] == self.x)].copy()
        p = p[p["status"] == "ok"]
        conds = [parse_condition(c) for c in p["condition"]]
        p = p[np.array([self.matches(c) for c in conds], dtype=bool)].copy()
        p["gen"] = [parse_condition(c).get("gen") for c in p["condition"]]
        merged = q.merge(p, on=["model", "gen"], suffixes=("_y", "_x"))
        return pd.DataFrame(
            dict(
                plot=self.name,
                model=merged["model"],
                condition=merged["condition_x"],
                x_metric=self.x,
                y_metric=self.y,
                x=merged["value_x"],
                x_lower=merged["lower_x"],
                x_upper=merged["upper_x"],
                y=merged["value_y"],
                y_lower=merged["lower_y"],
                y_upper=merged["upper_y"],
            )
        )

    def draw(self, pc: PlotContext) -> list[Path]:
        pts = self.points(pc.report)
        pts = pts[pts["model"].isin(pc.select(self.models))]
        if pts.empty:
            log.warning("plot %s: no (quality, perf) pairs for %s vs %s", self.name, self.y, self.x)
            return []
        metric = pc.metrics.get(self.y)
        better = metric.better if metric is not None else "lower"
        pts = pts.assign(
            on_front=pareto_front(
                pts["x"],
                pts["y"],
                better,
                (pts["x_lower"], pts["x_upper"]),
                (pts["y_lower"], pts["y_upper"]),
                self.robust_dominance,
            )
        )
        pc.pareto_rows.extend(pts[PARETO_COLUMNS].to_dict("records"))
        fig, ax = pc.figure(figsize=self.figsize or (6.0, 4.5))
        if self.show_front:
            front = pts[pts["on_front"]].sort_values("x")
            ax.plot(
                front["x"],
                front["y"],
                color=pc.style.ink,
                lw=1,
                ls="--",
                zorder=1,
            )
        for m in pc.select(self.models):
            r = pts[pts["model"] == m].sort_values("x")
            if r.empty:
                continue
            st = pc.styles[m]
            xerr = np.abs(np.vstack([r["x"] - r["x_lower"], r["x_upper"] - r["x"]]))
            yerr = np.abs(np.vstack([r["y"] - r["y_lower"], r["y_upper"] - r["y"]]))
            ax.errorbar(
                r["x"],
                r["y"],
                xerr=np.nan_to_num(xerr),
                yerr=np.nan_to_num(yerr),
                fmt="o",
                ms=8,
                color=st["color"],
                mec="white",
                mew=1.5,
                elinewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                pc.label(m),
                (r["x"].iloc[0], r["y"].iloc[0]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=pc.style.font_size - 1,
                color=st["color"],
                zorder=4,
            )
            if self.connect_sweeps and len(r) > 1:
                ax.plot(r["x"], r["y"], color=st["color"], ls=st["linestyle"], lw=1.2, zorder=2)
        if self.x_log:
            pc.log_x(ax)
        if self.y_log:
            ax.set_yscale("log")
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        ax.set_xlabel(self.xlabel or self.x + _condition_suffix(self.perf_condition))
        ax.set_ylabel(self.ylabel or (metric.label if metric is not None else self.y))
        title = self.title if self.title is not None else f"{pc.task}: quality vs cost"
        ax.set_title(title, loc="left", fontsize=pc.style.font_size - 1, color=pc.style.ink)
        return pc.save(fig, self.name)


def _condition_suffix(cond: dict) -> str:
    """` (k=v, ...)` for axis labels."""
    return f"  ({', '.join(f'{k}={v}' for k, v in cond.items())})" if cond else ""


def draw(pc: PlotContext, plots: list) -> list[Path]:
    """Draw plot functions (saved under their name; skipped without data) and Pareto
    plots; write `pareto.csv` if any Pareto plot produced points. A failing plot is
    logged and skipped."""
    paths: list[Path] = []
    with plt.rc_context(pc.style.rc_params()):
        for p in plots:
            name = p.info.name if hasattr(p, "info") else p.name
            try:
                if not hasattr(p, "info"):
                    paths += p.draw(pc)
                    continue
                fig = p(pc)
                if fig is None or not any(ax.has_data() for ax in fig.axes):
                    log.warning("plot %s: no data, skipped", name)
                    if fig is not None:
                        plt.close(fig)
                    continue
                paths += pc.save(fig, name)
            except Exception as e:  # a broken plot must not stop the others
                log.warning("plot %s failed: %s", name, e)
    if pc.pareto_rows:
        pd.DataFrame(pc.pareto_rows, columns=PARETO_COLUMNS).to_csv(
            pc.folder / "pareto.csv", index=False
        )
    return paths
