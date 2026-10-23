"""Syrupy snapshot extension with float-tolerance comparison.

JSON-backed; the matcher parses both sides and walks structures, applying
math.isclose at numeric leaves so snapshots survive cross-machine FP/BLAS
rounding skew.
"""

import json
import math
from typing import Any, Iterator

from syrupy.extensions.json import JSONSnapshotExtension

# 5e-6 atol covers bin-mean residuals near zero where rtol can't help; 1e-5
# rtol covers the ~5-7e-6 relative drift observed on larger-magnitude bin
# means and stds across (a) Linux OpenBLAS vs macOS Accelerate and (b)
# numpy/sklearn version differences between py3.10/3.11/3.12 under tox.
# Bin residuals have smaller magnitudes than the underlying predictions,
# so their relative noise floor is higher than per-element prediction noise.
# Snapshots remain informative for genuine algorithmic regressions (which
# shift outputs by >> 1e-4 relative; PR 5's clustering reorg shifted
# per-hour spread by 3-12%, four orders of magnitude above this floor).
FLOAT_ATOL = 5e-6
FLOAT_RTOL = 1e-5


class TolerantJSONSnapshotExtension(JSONSnapshotExtension):
    """JSON snapshot extension that compares floats with math.isclose tolerance.

    Default behavior on numeric leaves: ``math.isclose(actual, expected,
    abs_tol={atol}, rel_tol={rtol})``, with integers compared exactly. NaN
    compares equal to NaN. Lists, dicts, and other structures compare
    element-wise; structural mismatch is a hard fail.

    On failure, replaces syrupy's per-line unified diff with a compact summary
    of the largest numeric divergences so long arrays produce short output.
    """

    atol = FLOAT_ATOL
    rtol = FLOAT_RTOL
    atol_keys: frozenset[str] | None = None

    @classmethod
    def with_tolerance(
        cls, *, atol: float, rtol: float, atol_keys: frozenset[str] | None = None
    ) -> type["TolerantJSONSnapshotExtension"]:
        """Return a subclass comparing numeric leaves at ``rtol`` and ``atol``; when ``atol_keys``
        is given, ``atol`` applies only under those top-level keys and ``FLOAT_ATOL`` under the
        rest."""
        # syrupy finds an extension by module and name, and type() alone would set the module to abc
        attrs = {"atol": atol, "rtol": rtol, "atol_keys": atol_keys, "__module__": cls.__module__}

        return type(cls.__name__, (cls,), attrs)

    def _atol_for(self, key: str) -> float:
        if self.atol_keys is None or key in self.atol_keys:
            return self.atol

        return FLOAT_ATOL

    def _tolerance_text(self) -> str:
        if self.atol_keys is None:
            return f"atol={self.atol:g}, rtol={self.rtol:g}"

        keys = ", ".join(sorted(self.atol_keys))

        return f"atol={self.atol:g} under {keys}, {FLOAT_ATOL:g} elsewhere; rtol={self.rtol:g}"

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        if serialized_data == snapshot_data:
            return True
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            return False

        if isinstance(actual, dict) and isinstance(expected, dict):
            if actual.keys() != expected.keys():
                return False

            return all(
                _deep_close(actual[k], expected[k], atol=self._atol_for(k), rtol=self.rtol)
                for k in actual
            )

        return _deep_close(actual, expected, atol=self.atol, rtol=self.rtol)

    def diff_lines(self, serialized_data, snapshot_data) -> Iterator[str]:
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs: list[tuple[str, float, float]] = []
        if isinstance(actual, dict) and isinstance(expected, dict):
            for k in actual.keys() | expected.keys():
                _collect_diffs(
                    actual.get(k), expected.get(k), k, diffs, atol=self._atol_for(k), rtol=self.rtol
                )
        else:
            _collect_diffs(actual, expected, "", diffs, atol=self.atol, rtol=self.rtol)
        if not diffs:
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs.sort(key=lambda d: -_score(d[1], d[2]))
        yield f"{len(diffs)} value(s) outside tolerance ({self._tolerance_text()})"
        for path, a, e in diffs[:10]:
            yield f"  {path}: received {a!r}, snapshot {e!r}  (abs={abs(a - e):.3g}, rel={_rel(a, e):.3g})"
        if len(diffs) > 10:
            yield f"  ... {len(diffs) - 10} more"


def _deep_close(a: Any, b: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False

        return all(_deep_close(a[k], b[k], atol=atol, rtol=rtol) for k in a)

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False

        return all(_deep_close(a[i], b[i], atol=atol, rtol=rtol) for i in range(len(a)))

    if isinstance(a, int) and isinstance(b, int):
        return a == b

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a_nan = isinstance(a, float) and math.isnan(a)
        b_nan = isinstance(b, float) and math.isnan(b)
        if a_nan and b_nan:
            return True
        if a_nan or b_nan:
            return False

        return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)

    return a == b


def _collect_diffs(
    a: Any,
    b: Any,
    path: str,
    diffs: list,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Walk structures and record (path, actual, expected) for numeric leaves outside tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            _collect_diffs(a.get(k), b.get(k), f"{path}.{k}" if path else k, diffs, atol=atol, rtol=rtol)
        return

    if isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            ai = a[i] if i < len(a) else None
            bi = b[i] if i < len(b) else None
            _collect_diffs(ai, bi, f"{path}[{i}]", diffs, atol=atol, rtol=rtol)
        return

    if isinstance(a, int) and isinstance(b, int):
        if a != b:
            diffs.append((path, a, b))
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a_f = float(a)
        b_f = float(b)
        if math.isnan(a_f) and math.isnan(b_f):
            return
        if not math.isclose(a_f, b_f, abs_tol=atol, rel_tol=rtol):
            diffs.append((path, a_f, b_f))
        return

    if a != b:
        diffs.append((path, a, b))


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))

    return abs(a - b) / scale if scale > 0 else 0.0


def _score(a, b) -> float:
    try:
        return max(abs(float(a) - float(b)), _rel(float(a), float(b)))
    except (TypeError, ValueError):
        return 0.0
