"""Solver-neutral interfaces for feasibility gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from planar_instance_generation.io.pglib import PglibCase


FloatVector = NDArray[np.float64]


def _finite_vector(name: str, values: FloatVector) -> FloatVector:
    vector = np.array(values, dtype=np.float64, copy=True)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class DcOpfSolution:
    """A DC-OPF solution, with vectors aligned to the input case rows.

    Power quantities use the MATPOWER case units (MW), so branch flows can be
    consumed directly by cut-flow compensation.
    """

    bus_angles: FloatVector
    branch_from_flows: FloatVector
    branch_to_flows: FloatVector
    generator_dispatch: FloatVector

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus_angles", _finite_vector("bus_angles", self.bus_angles))
        object.__setattr__(
            self,
            "branch_from_flows",
            _finite_vector("branch_from_flows", self.branch_from_flows),
        )
        object.__setattr__(
            self,
            "branch_to_flows",
            _finite_vector("branch_to_flows", self.branch_to_flows),
        )
        object.__setattr__(
            self,
            "generator_dispatch",
            _finite_vector("generator_dispatch", self.generator_dispatch),
        )

    def validate_for(self, case: PglibCase) -> None:
        """Check that every vector has the row order and size of ``case``."""

        expected = {
            "bus_angles": case.bus.shape[0],
            "branch_from_flows": case.branch.shape[0],
            "branch_to_flows": case.branch.shape[0],
            "generator_dispatch": case.gen.shape[0],
        }
        for name, size in expected.items():
            actual = getattr(self, name).shape[0]
            if actual != size:
                raise ValueError(
                    f"{case.name}: {name} has length {actual}; expected {size}"
                )


@dataclass(frozen=True, slots=True)
class DcOpfResult:
    """Outcome and timings from one base-case DC-OPF solve.

    ``runtime_seconds`` is end-to-end gate wall time. ``optimizer_seconds`` is
    the backend-reported solve time when the backend exposes it.
    """

    feasible: bool
    status: str
    objective: float | None
    runtime_seconds: float
    solution: DcOpfSolution | None = None
    optimizer_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise ValueError("status must not be empty")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be finite and nonnegative")
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError("objective must be finite when provided")
        if self.optimizer_seconds is not None and (
            not math.isfinite(self.optimizer_seconds) or self.optimizer_seconds < 0
        ):
            raise ValueError("optimizer_seconds must be finite and nonnegative")
        if self.feasible and self.solution is None:
            raise ValueError("a feasible result must contain a solution")
        if self.feasible and self.objective is None:
            raise ValueError("a feasible result must contain an objective")
        if not self.feasible and self.solution is not None:
            raise ValueError("an infeasible result cannot contain a solution")


@runtime_checkable
class DcOpfGate(Protocol):
    """Interface implemented by every base-case DC-OPF backend."""

    def solve(self, case: PglibCase, /) -> DcOpfResult:
        """Solve one case without mutating it."""
