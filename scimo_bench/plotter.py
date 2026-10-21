"""Plots from the result files of one run folder (no physics here).

Reads `report.csv`, `rankings.csv` and `per_event.parquet`; draws every model into the same
axes with a fixed style per model. Plot kinds are classes registered with
`@register_plot(name)`; every constructor argument is a config option (`plots.specs.<plot>`).
The look of all plots comes from `Style` (`plots.style`). A plot name resolves to a
catalogue spec, else a registered plot, else a variable (drawn as a distribution).

A new plot kind: a class with a `name` attribute and `draw(pc: PlotContext) -> list[Path]`,
decorated with `@register_plot("kind")`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
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

from scimo_bench.scorer import Metric, better_key, overall  # noqa: E402
from scimo_bench.variables import exists, info  # noqa: E402

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
    grid_color: str = "#e4e3df"
    font_size: float = 10
    line_width: float = 2.0
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
            "axes.edgecolor": self.muted,
            "axes.labelcolor": self.ink,
            "axes.grid": True,
            "grid.color": self.grid_color,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": self.muted,
            "ytick.color": self.muted,
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
    """Everything a plot may read: the run's result files, the models with their labels
    and styles, the task's metrics, and the style."""

    folder: Path
    report: pd.DataFrame
    rankings: pd.DataFrame | None
    per_event: ak.Array | None
    models: list[str]
    labels: dict[str, str]
    metrics: dict[str, Metric]
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
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

    def select(self, subset: Sequence[str] | None) -> list[str]:
        """The run's models, restricted to `subset` if given (order kept)."""
        return [m for m in self.models if subset is None or m in set(subset)]

    def values(self, model: str, variable: str) -> np.ndarray | None:
        """Finite values of a variable for a model (or 'truth') from per_event.parquet."""
        pe = self.per_event
        if pe is None or variable not in pe.fields:
            return None
        v = pe[pe["model"] == model][variable]
        v = v[~ak.is_none(v, axis=0)]
        if len(v) == 0:
            return None
        if v.ndim > 1:
            v = ak.flatten(v, axis=1)
        a = ak.to_numpy(v).astype(np.float64)
        return a[np.isfinite(a)]

    def figure(self, figsize: Sequence[float] | None, ratio: bool = False):
        """A figure (with a ratio panel below if `ratio`): (fig, ax, ratio_ax or None)."""
        if ratio:
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=tuple(figsize or self.style.ratio_figsize),
                sharex=True,
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05),
            )
            return fig, ax, rax
        fig, ax = plt.subplots(figsize=tuple(figsize or self.style.figsize))
        return fig, ax, None

    def legend(self, ax, title: str | None, show: bool = True) -> None:
        """Legend above the axes (never over the data) or at `style.legend`, with the task
        (or `title`) as its title; one series (or `show=False`) gets only the title."""
        title = title if title is not None else self.task
        handles, _ = ax.get_legend_handles_labels()
        if not show or len(handles) < 2:
            ax.set_title(
                title, loc="left", fontsize=self.style.font_size - 1, color=self.style.muted
            )
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
        leg.get_title().set_color(self.style.muted)
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


PLOTS: dict[str, type] = {}


def register_plot(name: str):
    """Class decorator: make a plot kind available by `name`."""

    def wrap(cls):
        PLOTS[name] = cls
        cls.kind = name
        return cls

    return wrap


def _log_x(ax) -> None:
    """Log x axis with plain number labels."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


@register_plot("distribution")
class Distribution:
    """Histogram of a variable: truth as a grey filled histogram, models as step lines,
    optional ratio panel (model / truth).

    `bins`: number (range from `x_range` or the data's 0.5–99.5 % quantiles) or edges.
    `models`: draw only these models. `truth`: draw the truth counterpart. `reference`:
    draw the variable's ideal value as a dotted line.
    """

    def __init__(
        self,
        variable: str,
        bins: int | list[float] = 50,
        x_range: list[float] | None = None,
        density: bool = True,
        ratio: bool = True,
        ratio_range: list[float] = (0.5, 1.5),
        ratio_label: str = "model / truth",
        log_x: bool = False,
        log_y: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        truth: bool = True,
        truth_label: str = "truth",
        reference: bool = True,
        legend: bool = True,
        figsize: list[float] | None = None,
        models: list[str] | None = None,
        name: str | None = None,
    ):
        self.variable = variable
        self.bins = bins
        self.x_range = x_range
        self.density = density
        self.ratio = ratio
        self.ratio_range = ratio_range
        self.ratio_label = ratio_label
        self.log_x = log_x
        self.log_y = log_y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.truth = truth
        self.truth_label = truth_label
        self.reference = reference
        self.legend = legend
        self.figsize = figsize
        self.models = models
        self.name = name or variable

    def draw(self, pc: PlotContext) -> list[Path]:
        meta = info(self.variable)
        data = {m: pc.values(m, self.variable) for m in pc.select(self.models)}
        data = {m: v for m, v in data.items() if v is not None and len(v)}
        truth = pc.values("truth", meta.truth) if meta.truth and self.truth else None
        if meta.truth_only:
            truth, data = pc.values("truth", self.variable), {}
        if not data and truth is None:
            log.warning("plot %s: no data", self.name)
            return []
        edges = self._edges([*data.values(), *([truth] if truth is not None else [])])
        ratio = self.ratio and truth is not None and len(data) > 0
        fig, ax, rax = pc.figure(self.figsize, ratio)
        h_truth = None
        if truth is not None:
            h_truth, _ = np.histogram(truth, edges, density=self.density)
            ax.stairs(
                h_truth, edges, fill=True, color=pc.style.truth_color, label=self.truth_label, lw=0
            )
        for m, v in data.items():
            st = pc.styles[m]
            h, _ = np.histogram(v, edges, density=self.density)
            ax.stairs(h, edges, label=pc.label(m), **st)
            if ratio:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rax.stairs(np.where(h_truth > 0, h / h_truth, np.nan), edges, **st)
        if self.reference and meta.reference is not None:
            ax.axvline(meta.reference, color=pc.style.muted, lw=1, ls=":")
        ax.set_ylabel(self.ylabel or ("density" if self.density else "entries"))
        if self.log_y:
            ax.set_yscale("log")
        pc.legend(ax, self.title, self.legend)
        last = rax if ratio else ax
        if ratio:
            rax.axhline(1.0, color=pc.style.muted, lw=1)
            rax.set_ylabel(self.ratio_label)
            rax.set_ylim(*self.ratio_range)
        last.set_xlabel(self.xlabel or meta.axis_label)
        if self.log_x:
            _log_x(last)
        return pc.save(fig, self.name)

    def _edges(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Bin edges: given, or from `x_range` / the data's quantiles."""
        if not isinstance(self.bins, int):
            return np.asarray(self.bins, dtype=float)
        if self.x_range is not None:
            lo, hi = self.x_range
        else:
            lo, hi = np.quantile(np.concatenate(arrays), [0.005, 0.995])
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        if self.log_x and lo > 0:
            return np.geomspace(lo, hi, self.bins + 1)
        return np.linspace(lo, hi, self.bins + 1)


@register_plot("metric_vs_x")
class MetricVsX:
    """A binned metric from report.csv against its x variable, one line per model with
    its interval band. `reference`: the variable's ideal value as a dotted line."""

    def __init__(
        self,
        metric: str,
        log_x: bool = False,
        log_y: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        y_range: list[float] | None = None,
        title: str | None = None,
        band: bool = True,
        markers: bool = True,
        reference: bool = True,
        legend: bool = True,
        figsize: list[float] | None = None,
        models: list[str] | None = None,
        name: str | None = None,
    ):
        self.metric = metric
        self.log_x = log_x
        self.log_y = log_y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.y_range = y_range
        self.title = title
        self.band = band
        self.markers = markers
        self.reference = reference
        self.legend = legend
        self.figsize = figsize
        self.models = models
        self.name = name or f"{metric}_vs_x"

    def draw(self, pc: PlotContext) -> list[Path]:
        metric = pc.metrics.get(self.metric)
        rows = pc.report[(pc.report["metric"] == self.metric) & pc.report["bin_low"].notna()]
        if metric is None or metric.x is None or rows.empty:
            log.warning("plot %s: no binned rows for metric %s", self.name, self.metric)
            return []
        fig, ax, _ = pc.figure(self.figsize)
        drawn = False
        for m in pc.select(self.models):
            r = rows[(rows["model"] == m)].sort_values("bin_low")
            ok = r["status"] == "ok"
            if not ok.any():
                continue
            lo_e, hi_e = r["bin_low"].to_numpy(), r["bin_high"].to_numpy()
            x = np.sqrt(lo_e * hi_e) if self.log_x and lo_e.min() > 0 else 0.5 * (lo_e + hi_e)
            y = np.where(ok, r["value"], np.nan)
            lo, hi = np.where(ok, r["lower"], np.nan), np.where(ok, r["upper"], np.nan)
            st = pc.styles[m]
            if self.band:
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, lw=0)
            marker = dict(marker="o", ms=4) if self.markers else {}
            ax.plot(x, y, label=pc.label(m), **marker, **st)
            drawn = True
        if not drawn:
            plt.close(fig)
            return []
        ref = info(metric.variable).reference if exists(metric.variable) else None
        if self.reference and metric.location and ref is not None:
            ax.axhline(ref, color=pc.style.muted, lw=1, ls=":")
        ax.set_xlabel(self.xlabel or info(metric.x).axis_label)
        ax.set_ylabel(self.ylabel or metric.label)
        if self.y_range is not None:
            ax.set_ylim(*self.y_range)
        if self.log_x:
            _log_x(ax)
        if self.log_y:
            ax.set_yscale("log")
        pc.legend(ax, self.title, self.legend)
        return pc.save(fig, self.name)


@register_plot("ranking")
class RankingPlot:
    """Rank of every model per metric (rows), the value written in each cell.
    `metrics`: only these rows (default all, incl. `mean_rank`)."""

    def __init__(
        self,
        metrics: list[str] | None = None,
        cmap: str = "Blues_r",
        show_values: bool = True,
        title: str | None = None,
        figsize: list[float] | None = None,
        models: list[str] | None = None,
        name: str = "ranking",
    ):
        self.metrics = metrics
        self.cmap = cmap
        self.show_values = show_values
        self.title = title
        self.figsize = figsize
        self.models = models
        self.name = name

    def draw(self, pc: PlotContext) -> list[Path]:
        rk = pc.rankings
        if rk is None or rk.empty:
            log.warning("plot ranking: no rankings.csv")
            return []
        metrics = [m for m in dict.fromkeys(rk["metric"]) if not self.metrics or m in self.metrics]
        models = [m for m in pc.select(self.models) if m in set(rk["model"])]
        ranks = np.full((len(metrics), len(models)), np.nan)
        vmax = max(len(models), 2) + 1.5
        size = self.figsize or (1.6 + 1.6 * len(models), 0.9 + 0.45 * len(metrics))
        fig, ax = plt.subplots(figsize=tuple(size))
        for i, met in enumerate(metrics):
            for j, m in enumerate(models):
                r = rk[(rk["metric"] == met) & (rk["model"] == m)].iloc[0]
                ranks[i, j] = r["rank"]
                txt = f"#{int(r['rank'])}" if r["status"] == "ok" else r["status"]
                if self.show_values and r["status"] == "ok":
                    txt += f"\n{r['value']:.4g}"
                dark = r["rank"] <= 1 + 0.4 * (vmax - 1)
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=pc.style.font_size - 2,
                    color="white" if dark else pc.style.ink,
                )
        ax.imshow(ranks, cmap=self.cmap, vmin=1, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(models)), [pc.label(m) for m in models], rotation=20, ha="right")
        ax.set_yticks(range(len(metrics)), metrics)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        title = self.title or f"{pc.task}: rank per metric (1 = best)"
        ax.set_title(title, loc="left", fontsize=pc.style.font_size - 1, color=pc.style.muted)
        return pc.save(fig, self.name)


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


@register_plot("pareto")
class ParetoPlot:
    """Quality metric `y` against cost `x` (a perf metric under one fixed perf condition,
    e.g. {scenario: single, batch_size: 1}); writes its points to `pareto.csv`.
    `connect_sweeps`: join a model's points; `robust_dominance`: use the intervals."""

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
        fig, ax, _ = pc.figure(self.figsize or (6.0, 4.5))
        if self.show_front:
            front = pts[pts["on_front"]].sort_values("x")
            ax.plot(
                front["x"],
                front["y"],
                color=pc.style.ink,
                lw=1,
                ls="--",
                label="Pareto front",
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
                label=pc.label(m),
                zorder=3,
            )
            if self.connect_sweeps and len(r) > 1:
                ax.plot(r["x"], r["y"], color=st["color"], ls=st["linestyle"], lw=1.2, zorder=2)
        if self.x_log:
            _log_x(ax)
        if self.y_log:
            ax.set_yscale("log")
        ax.set_xlabel(self.xlabel or self.x + _condition_suffix(self.perf_condition))
        ax.set_ylabel(self.ylabel or (metric.label if metric is not None else self.y))
        pc.legend(ax, self.title if self.title is not None else f"{pc.task}: quality vs cost")
        return pc.save(fig, self.name)


def _condition_suffix(cond: dict) -> str:
    """` (k=v, ...)` for axis labels."""
    return f"  ({', '.join(f'{k}={v}' for k, v in cond.items())})" if cond else ""


def resolve(name: str, specs: Mapping) -> object | None:
    """A plot name → plot object: catalogue spec, registered plot, or a distribution of
    the variable `name`; None if none of these."""
    if name in specs:
        return specs[name]
    if name in PLOTS:
        return PLOTS[name]()
    if exists(name):
        return Distribution(name)
    return None


def draw(pc: PlotContext, plots: list[object]) -> list[Path]:
    """Draw plots in the context's style; write `pareto.csv` if any Pareto plot produced
    points. A failing plot is logged and skipped."""
    paths: list[Path] = []
    with plt.rc_context(pc.style.rc_params()):
        for p in plots:
            try:
                paths += p.draw(pc)
            except Exception as e:  # a broken plot must not stop the others
                log.warning("plot %s failed: %s", getattr(p, "name", p), e)
    if pc.pareto_rows:
        pd.DataFrame(pc.pareto_rows, columns=PARETO_COLUMNS).to_csv(
            pc.folder / "pareto.csv", index=False
        )
    return paths
