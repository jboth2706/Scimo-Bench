"""The physics of a dataset as plain functions: observables, comparisons and plots.

A dataset package (e.g. `scimo_bench/datasets/cocoa/`) has three modules of decorated
functions; the function name is the name used in configs and result files:

- `observables.py`: `@observable` — a quantity of ONE set of events. It is computed on the
  truth and on every model's output (reco), so it is written once:

      @observable(label="jet $p_T$", unit="GeV")
      def jet_pt(ev):                       # ev: events, e.g. ev.particles.pt
          return jet(ev.particles)["pt"]    # one value per event, or a list per event

- `comparisons.py`: `@comparison` — a quantity of the truth AND one model, row by row:

      @comparison(label="$E_{reco} / E_{truth}$ (jet)", reference=1.0)
      def jet_e_response(truth, reco):
          return jet_e(reco) / jet_e(truth)

  Task parameters are keyword arguments with defaults (`def f(truth, reco, match_dr=0.05)`);
  the task's `params` override them.

- `plots.py`: `@plot` — one figure from the run's result files (see `plotter.Plots`).

The package's `__init__.py` collects them: `CATALOGUE = Catalogue.collect(observables,
comparisons, plots)`.

In metrics and plots, an observable is named with its side, `truth.<name>` or
`reco.<name>`; a comparison by its name alone.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType

import awkward as ak

SIDES = ("truth", "reco")


class Missing(Exception):
    """A quantity could not be computed for a model (its output lacks a field)."""


@dataclass(frozen=True)
class Info:
    """Name, kind and axis metadata of an observable, comparison or plot."""

    name: str
    kind: str  # "observable", "comparison" or "plot"
    label: str = ""
    unit: str = ""
    reference: float | None = None  # the ideal value of a comparison (drawn as a line)

    def axis_label(self, side: str | None = None) -> str:
        """`label (side) [unit]`; the side is left out when None."""
        text = self.label + (f" ({side})" if side else "")
        return f"{text} [{self.unit}]" if self.unit else text


def _mark(kind: str, label: str, unit: str = "", reference: float | None = None):
    def wrap(fn):
        fn.info = Info(fn.__name__, kind, label or fn.__name__, unit, reference)
        return fn

    return wrap


def observable(label: str = "", unit: str = ""):
    """Decorator: `fn(ev)` is an observable (one value per event, or a list per event)."""
    return _mark("observable", label=label, unit=unit)


def comparison(label: str = "", unit: str = "", reference: float | None = None):
    """Decorator: `fn(truth, reco, **params)` compares the truth with one model."""
    return _mark("comparison", label=label, unit=unit, reference=reference)


def plot(fn):
    """Decorator: `fn(p) -> figure` is a plot, saved as `plots/<name>.<fmt>`."""
    fn.info = Info(fn.__name__, "plot")
    return fn


def _collect(module: ModuleType, kind: str) -> dict[str, Callable]:
    """The functions of `kind` defined in `module` (not imported into it), in order."""
    return {
        name: fn
        for name, fn in vars(module).items()
        if inspect.isfunction(fn)
        and isinstance(getattr(fn, "info", None), Info)
        and fn.info.kind == kind
        and fn.__module__ == module.__name__
    }


@dataclass
class Catalogue:
    """All observables, comparisons and plots of one dataset."""

    observables: dict[str, Callable] = field(default_factory=dict)
    comparisons: dict[str, Callable] = field(default_factory=dict)
    plots: dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def collect(cls, observables: ModuleType, comparisons: ModuleType, plots: ModuleType):
        """From the dataset's three modules."""
        cat = cls(
            _collect(observables, "observable"),
            _collect(comparisons, "comparison"),
            _collect(plots, "plot"),
        )
        both = set(cat.observables) & set(cat.comparisons)
        if both:
            raise ValueError(f"names used as observable and comparison: {sorted(both)}")
        return cat

    def info(self, name: str) -> Info:
        """Metadata of `name`: a comparison, an observable, or `truth.` / `reco.` + one."""
        side, base = split(name)
        fn = self.observables.get(base) if side or base in self.observables else None
        fn = fn or self.comparisons.get(base)
        if fn is None:
            raise KeyError(f"unknown observable or comparison '{name}'")
        return fn.info

    def check_name(self, name: str, where: str) -> None:
        """A name metrics may use: a comparison, or `truth.<obs>` / `reco.<obs>`."""
        side, base = split(name)
        if side is None and base in self.comparisons:
            return
        if side is not None and base in self.observables:
            return
        if side is None and base in self.observables:
            raise ValueError(f"{where}: say which side, truth.{base} or reco.{base}")
        raise ValueError(
            f"{where}: unknown '{name}'; comparisons {sorted(self.comparisons)}, "
            f"observables (as truth.<name> / reco.<name>) {sorted(self.observables)}"
        )

    def params(self) -> set[str]:
        """Keyword parameters the comparisons accept (what task `params` may set)."""
        return {p for fn in self.comparisons.values() for p in keywords(fn)}


def split(name: str) -> tuple[str | None, str]:
    """`truth.jet_pt` → ("truth", "jet_pt"); `jet_dr` → (None, "jet_dr")."""
    side, dot, base = name.partition(".")
    return (side, base) if dot and side in SIDES else (None, name)


def keywords(fn: Callable) -> list[str]:
    """Names of a comparison's keyword parameters (after truth, reco)."""
    return list(inspect.signature(fn).parameters)[2:]


def call_comparison(fn: Callable, truth: ak.Array, reco: ak.Array, params: dict):
    """`fn(truth, reco)` with the task parameters it accepts."""
    return fn(truth, reco, **{k: v for k, v in params.items() if k in keywords(fn)})


_FIELD = re.compile(r"no field named '([^']+)'")


def missing_field(error: Exception) -> str | None:
    """The field name if `error` is awkward's "no field named ..." error, else None."""
    if isinstance(error, AttributeError | ak.errors.FieldNotFoundError):
        m = _FIELD.search(str(error))
        return m.group(1) if m else None
    return None


def load(module: str) -> Catalogue:
    """The `CATALOGUE` of a dataset package, e.g. `scimo_bench.datasets.cocoa`."""
    return importlib.import_module(module).CATALOGUE
