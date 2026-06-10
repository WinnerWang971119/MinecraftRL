"""logging — Structured metrics logger for training and evaluation runs.

A single, backend-agnostic :class:`MetricsLogger` used by the M1 throughput
benchmark (T11) and the M2 eval (T19). It prefers Weights & Biases, then
TensorBoard, and falls back to a **dependency-free** on-disk JSON-lines + summary
sink so the exact same API works on a machine with neither installed (the common
case — ``torch``/``wandb``/``tensorboard`` are optional heavy deps here). T19
imports this module read-only, so the public API below is the stable seam:

    log = MetricsLogger("bench", backend="auto", log_dir="runs")
    log.log({"transitions_per_s": 4.9, "p99_latency_ms": 41.0}, step=0)
    log.log_scalar("tps", 19.7, step=1)
    log.summary({"max_arenas": 3})
    log.close()

------------------------------------------------------------------------------
Backend resolution
------------------------------------------------------------------------------
``backend="auto"`` (the default) resolves in this priority order, taking the
first one that imports successfully:

    1. ``"wandb"``        — Weights & Biases (``import wandb``); rich hosted runs.
    2. ``"tensorboard"``  — ``torch.utils.tensorboard.SummaryWriter`` (or the
                            standalone ``tensorboardX``); local event files.
    3. ``"jsonl"``        — the always-available fallback: append-only
                            ``metrics.jsonl`` + a ``summary.json`` under
                            ``<log_dir>/<run_name>/``. No third-party deps.

Each backend is imported **lazily inside a try/except** at construction time, so
importing this module never requires W&B or TensorBoard to be installed. Passing
an explicit ``backend=`` skips resolution and **raises** if that backend's
dependency is missing (so a caller that truly wants W&B finds out loudly), except
``backend="jsonl"`` which is always available. The resolved backend is exposed as
:attr:`MetricsLogger.backend` and the on-disk paths (for the fallback) as
:attr:`MetricsLogger.metrics_path` / :attr:`MetricsLogger.summary_path`.

The JSON-lines fallback is also **readable back** via :func:`read_jsonl` and
:func:`read_summary`, which the benchmark's own tests use to prove a logged run
round-trips. W&B / TensorBoard runs are not read back here (their stores are the
source of truth); only the fallback owns an on-disk format this module reads.

Owner: T11 (Eval/infra track) — SOLE writer. T19 imports it read-only.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "MetricsLogger",
    "BackendUnavailableError",
    "AUTO_BACKEND_ORDER",
    "read_jsonl",
    "read_summary",
]


#: The fixed priority order :class:`MetricsLogger` tries when ``backend="auto"``.
#: First importable wins; ``"jsonl"`` is the always-available terminal fallback.
AUTO_BACKEND_ORDER: tuple = ("wandb", "tensorboard", "jsonl")

#: File names written by the dependency-free JSON-lines backend, under
#: ``<log_dir>/<run_name>/``.
_METRICS_FILENAME = "metrics.jsonl"
_SUMMARY_FILENAME = "summary.json"


class BackendUnavailableError(RuntimeError):
    """Raised when an *explicitly requested* logging backend cannot be loaded.

    Only raised for an explicit ``backend="wandb"`` / ``backend="tensorboard"``
    whose dependency is not importable. ``backend="auto"`` never raises this — it
    silently falls through to the next backend and ultimately to the always-
    available JSON-lines sink.
    """


# ---------------------------------------------------------------------------
# JSON sanitization.
#
# W&B/TensorBoard and json.dump all want plain Python scalars. Metric dicts often
# carry numpy scalars (e.g. a float32 percentile) or non-finite floats; coerce
# them to portable JSON values so no backend chokes on a numpy type or a NaN.
# ---------------------------------------------------------------------------


def _coerce_scalar(value: Any) -> Any:
    """Coerce one metric value to a JSON/backend-safe Python scalar.

    - numpy scalars / 0-d arrays -> their Python ``item()``;
    - ``bool`` stays ``bool`` (it is a valid JSON value);
    - finite numbers stay as ``int`` / ``float``;
    - non-finite floats (NaN/Inf) -> ``None`` (JSON has no NaN; backends reject it);
    - everything else is passed through unchanged (strings, etc.).
    """
    # Unwrap numpy scalars / 0-d arrays without importing numpy at module load.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            # Not a 0-d numeric scalar (e.g. a multi-element array); leave as-is.
            pass

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _coerce_mapping(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with string keys and JSON/backend-safe scalar values."""
    return {str(k): _coerce_scalar(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# The logger.
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Backend-agnostic metrics logger (W&B → TensorBoard → JSON-lines fallback).

    The public surface — :meth:`log`, :meth:`log_scalar`, :meth:`summary`,
    :meth:`close` — is identical regardless of the resolved backend, so callers
    (T11 benchmark, T19 eval) never branch on which one is active.

    Args:
        run_name: A human-readable run name. Used as the W&B run name, the
            TensorBoard sub-directory, and the JSON-lines run directory name.
        backend: ``"auto"`` (default) resolves W&B → TensorBoard → JSON-lines,
            taking the first importable one. An explicit ``"wandb"`` /
            ``"tensorboard"`` / ``"jsonl"`` forces that backend and RAISES
            :class:`BackendUnavailableError` if its dependency is missing (only
            ``"jsonl"`` is guaranteed available).
        log_dir: Root directory for on-disk artifacts (TensorBoard event files
            and the JSON-lines fallback). Defaults to ``"runs"`` under the cwd.
            Created lazily on first write.
        config: Optional run config/hyperparameters recorded once at start (W&B
            ``config``; otherwise embedded in the summary file).
        wandb_project: Optional W&B project name (ignored by other backends).

    Attributes:
        run_name: The run name.
        backend: The RESOLVED backend string (``"wandb"`` / ``"tensorboard"`` /
            ``"jsonl"``) — what is actually active after resolution.
        run_dir: The on-disk directory for this run's fallback/TensorBoard
            artifacts (``None`` for a pure W&B run that writes nothing locally).
        metrics_path: Path to ``metrics.jsonl`` for the JSON-lines backend
            (``None`` for other backends).
        summary_path: Path to ``summary.json`` for the JSON-lines backend
            (``None`` for other backends).
    """

    def __init__(
        self,
        run_name: str,
        backend: str = "auto",
        log_dir: str = "runs",
        config: Optional[Mapping[str, Any]] = None,
        wandb_project: Optional[str] = None,
    ) -> None:
        self.run_name = str(run_name)
        self._requested_backend = str(backend)
        self._log_dir = str(log_dir)
        self._config: Dict[str, Any] = dict(config) if config else {}
        self._wandb_project = wandb_project
        self._closed = False

        # Backend handles (exactly one is non-None after resolution).
        self._wandb = None  # the wandb module
        self._wandb_run = None  # the wandb.Run
        self._tb_writer = None  # a SummaryWriter
        self._jsonl_file = None  # an open text file for metrics.jsonl

        # On-disk paths (populated by the tensorboard/jsonl backends).
        self.run_dir: Optional[str] = None
        self.metrics_path: Optional[str] = None
        self.summary_path: Optional[str] = None

        # Accumulated summary values (flushed to disk for the jsonl backend; set
        # as wandb.summary / a TensorBoard scalar group for the others).
        self._summary: Dict[str, Any] = {}

        self.backend = self._resolve_and_init(self._requested_backend)

    # -- backend resolution -----------------------------------------------

    def _resolve_and_init(self, requested: str) -> str:
        """Resolve and initialize a backend, returning its resolved name.

        For ``"auto"``, try each backend in :data:`AUTO_BACKEND_ORDER` and take
        the first that initializes. For an explicit backend, initialize exactly
        that one and raise :class:`BackendUnavailableError` if it cannot load.
        """
        if requested == "auto":
            for name in AUTO_BACKEND_ORDER:
                if self._try_init_backend(name):
                    return name
            # AUTO_BACKEND_ORDER ends in "jsonl", which never fails to init, so
            # this is unreachable in practice — guard anyway rather than return a
            # bogus backend name.
            raise BackendUnavailableError(  # pragma: no cover - jsonl always inits
                "no logging backend could be initialized (jsonl fallback failed)"
            )

        if requested not in AUTO_BACKEND_ORDER:
            raise ValueError(
                f"unknown backend {requested!r}; expected 'auto' or one of "
                f"{list(AUTO_BACKEND_ORDER)}"
            )
        if not self._try_init_backend(requested):
            raise BackendUnavailableError(
                f"requested logging backend {requested!r} is unavailable "
                f"(its dependency is not importable). Install it or use "
                f"backend='auto' to fall back to the dependency-free JSON-lines sink."
            )
        return requested

    def _try_init_backend(self, name: str) -> bool:
        """Attempt to initialize backend ``name``; return True on success.

        Each branch imports its dependency lazily inside try/except so a missing
        package is a clean ``False`` (fall through), never an import error at
        module load.
        """
        if name == "wandb":
            return self._init_wandb()
        if name == "tensorboard":
            return self._init_tensorboard()
        if name == "jsonl":
            return self._init_jsonl()
        return False  # pragma: no cover - guarded by _resolve_and_init

    def _init_wandb(self) -> bool:
        try:
            import wandb  # type: ignore
        except Exception:
            return False
        try:
            run = wandb.init(
                project=self._wandb_project,
                name=self.run_name,
                config=self._config or None,
                reinit=True,
            )
        except Exception:
            # wandb importable but init failed (e.g. no API key / offline misconfig).
            # Treat as unavailable so auto-resolution falls through cleanly.
            return False
        self._wandb = wandb
        self._wandb_run = run
        return True

    def _init_tensorboard(self) -> bool:
        SummaryWriter = None
        try:
            from torch.utils.tensorboard import SummaryWriter as _SW  # type: ignore

            SummaryWriter = _SW
        except Exception:
            try:
                from tensorboardX import SummaryWriter as _SW  # type: ignore

                SummaryWriter = _SW
            except Exception:
                return False
        try:
            run_dir = os.path.join(self._log_dir, self.run_name)
            os.makedirs(run_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=run_dir)
        except Exception:
            return False
        self.run_dir = run_dir
        return True

    def _init_jsonl(self) -> bool:
        """Initialize the dependency-free JSON-lines backend (always succeeds).

        Creates ``<log_dir>/<run_name>/`` and opens ``metrics.jsonl`` for append.
        Returns ``False`` only on an unexpected OS error (so auto-resolution can,
        in principle, report total failure rather than hang).
        """
        try:
            run_dir = os.path.join(self._log_dir, self.run_name)
            os.makedirs(run_dir, exist_ok=True)
            metrics_path = os.path.join(run_dir, _METRICS_FILENAME)
            summary_path = os.path.join(run_dir, _SUMMARY_FILENAME)
            # Append mode so re-running a run name accumulates rather than truncates
            # mid-run; the summary is rewritten wholesale on each flush.
            self._jsonl_file = open(metrics_path, "a", encoding="utf-8")
        except OSError:
            return False
        self.run_dir = run_dir
        self.metrics_path = metrics_path
        self.summary_path = summary_path
        # Seed the summary file with config so a reader sees the run's parameters
        # even before any summary() call.
        if self._config:
            self._summary.update({f"config.{k}": v for k, v in self._config.items()})
            self._flush_summary()
        return True

    # -- logging API (identical across backends) --------------------------

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of named scalar metrics at an optional integer ``step``.

        Values are coerced to JSON/backend-safe scalars (numpy scalars unwrapped,
        non-finite floats dropped to ``None``). A ``None`` ``step`` lets each
        backend use its own running counter.

        Args:
            metrics: Mapping of metric name -> scalar value.
            step: Optional monotonic step index (e.g. a decision/episode counter).

        Raises:
            RuntimeError: if called after :meth:`close`.
        """
        self._ensure_open()
        if not metrics:
            return
        clean = _coerce_mapping(metrics)

        if self._wandb_run is not None:
            # wandb accepts step=None (uses its own counter). Pass through.
            self._wandb_run.log(clean, step=step)
            return

        if self._tb_writer is not None:
            tb_step = 0 if step is None else int(step)
            for name, value in clean.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._tb_writer.add_scalar(name, value, tb_step)
                else:
                    # Non-numeric (or dropped non-finite) values: record as text so
                    # nothing is silently lost.
                    self._tb_writer.add_text(name, str(value), tb_step)
            return

        # JSON-lines fallback: one record per log() call.
        record: Dict[str, Any] = {"step": step, "wall_time": time.time()}
        record.update(clean)
        assert self._jsonl_file is not None  # set by _init_jsonl
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def log_scalar(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a single named scalar — convenience wrapper over :meth:`log`."""
        self.log({name: value}, step=step)

    def summary(self, values: Mapping[str, Any]) -> None:
        """Record/overwrite run-level summary values (final/aggregate metrics).

        Summary values are the run's headline numbers (e.g. the benchmark's
        ``max_arenas`` or p99 latency) as opposed to per-step series. They are set
        on ``wandb.run.summary``, written as a ``summary/<key>`` scalar group for
        TensorBoard, and flushed to ``summary.json`` for the JSON-lines backend.

        Repeated keys overwrite; partial updates merge into the accumulated
        summary.

        Raises:
            RuntimeError: if called after :meth:`close`.
        """
        self._ensure_open()
        if not values:
            return
        clean = _coerce_mapping(values)
        self._summary.update(clean)

        if self._wandb_run is not None:
            for name, value in clean.items():
                self._wandb_run.summary[name] = value
            return

        if self._tb_writer is not None:
            for name, value in clean.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._tb_writer.add_scalar(f"summary/{name}", value, 0)
                else:
                    self._tb_writer.add_text(f"summary/{name}", str(value), 0)
            return

        # JSON-lines fallback: rewrite the whole summary file.
        self._flush_summary()

    def close(self) -> None:
        """Flush and release the active backend. Idempotent.

        Finishes the W&B run, closes the TensorBoard writer, or flushes+closes the
        JSON-lines file. Safe to call multiple times and safe to call inside a
        ``finally`` even if construction half-failed.
        """
        if self._closed:
            return
        self._closed = True

        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None
            return

        if self._tb_writer is not None:
            try:
                self._tb_writer.flush()
            except Exception:
                pass
            try:
                self._tb_writer.close()
            except Exception:
                pass
            self._tb_writer = None
            return

        if self._jsonl_file is not None:
            try:
                self._flush_summary()
            except OSError:
                pass
            try:
                self._jsonl_file.flush()
                self._jsonl_file.close()
            except OSError:
                pass
            self._jsonl_file = None

    # -- context-manager sugar --------------------------------------------

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                f"MetricsLogger({self.run_name!r}) is closed; cannot log after close()"
            )

    def _flush_summary(self) -> None:
        """Atomically rewrite ``summary.json`` (JSON-lines backend only)."""
        if self.summary_path is None:
            return
        tmp_path = self.summary_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.summary_path)


# ---------------------------------------------------------------------------
# Read-back helpers (dependency-free JSON-lines backend only).
#
# Only the fallback owns an on-disk format this module both writes AND reads.
# W&B / TensorBoard stores are read by their own tooling, not here.
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a ``metrics.jsonl`` file written by the JSON-lines backend.

    Args:
        path: Path to the ``metrics.jsonl`` file.

    Returns:
        A list of per-``log()`` record dicts, in write order. Blank lines are
        skipped. Each record carries the logged metrics plus ``"step"`` and
        ``"wall_time"``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a non-blank line is not valid JSON.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: not a valid JSON metrics record: {exc}"
                ) from exc
    return records


def read_summary(path: str) -> Dict[str, Any]:
    """Read a ``summary.json`` file written by the JSON-lines backend.

    Args:
        path: Path to the ``summary.json`` file.

    Returns:
        The decoded summary dict (empty if the file holds an empty object).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a JSON object.
    """
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: summary must be a JSON object, got {type(data).__name__}")
    return data
