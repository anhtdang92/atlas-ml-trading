"""
Experiment Tracking for Stock ML Training Pipeline

Tracks hyperparameters, metrics, and model artifacts for each training run.
Supports both local CSV logging and optional MLflow integration.

Usage:
    tracker = ExperimentTracker()
    with tracker.start_run(symbol="AAPL", params={...}) as run:
        # ... train model ...
        run.log_metrics({"mse": 0.001, "directional_accuracy": 0.62})
    tracker.get_best_run("AAPL", metric="directional_accuracy")
"""

import csv
import json
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class RunRecord:
    """Single experiment run record."""
    run_id: str
    symbol: str
    timestamp: str
    status: str  # "running", "completed", "failed"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    model_path: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten for CSV storage."""
        flat = {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "status": self.status,
            "model_path": self.model_path or "",
            "duration_seconds": self.duration_seconds or 0,
        }
        for k, v in self.params.items():
            flat[f"param_{k}"] = v
        for k, v in self.metrics.items():
            flat[f"metric_{k}"] = v
        for k, v in self.tags.items():
            flat[f"tag_{k}"] = v
        return flat


class RunContext:
    """Context manager for an active experiment run."""

    def __init__(self, record: RunRecord, tracker: "ExperimentTracker"):
        self.record = record
        self._tracker = tracker
        self._start_time = datetime.now()

    def log_params(self, params: Dict[str, Any]) -> None:
        self.record.params.update(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        self.record.metrics.update(metrics)

    def log_metric(self, key: str, value: float) -> None:
        self.record.metrics[key] = value

    def set_tag(self, key: str, value: str) -> None:
        self.record.tags[key] = value

    def set_model_path(self, path: str) -> None:
        self.record.model_path = path


class ExperimentTracker:
    """Track ML experiment runs with metrics, params, and artifacts.

    Stores results as CSV (always) and optionally logs to MLflow.

    Args:
        results_dir: Directory for experiment logs and artifacts.
        experiment_name: Name for grouping runs (used as MLflow experiment).
        use_mlflow: Enable MLflow logging if available.
    """

    def __init__(
        self,
        results_dir: str = "results/experiments",
        experiment_name: str = "stock-lstm-predictions",
        use_mlflow: bool = False,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and HAS_MLFLOW
        self.csv_path = self.results_dir / "experiment_log.csv"
        self._runs: List[RunRecord] = []
        self._load_existing_runs()

        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        logger.info(f"ExperimentTracker initialized (dir={results_dir}, mlflow={self.use_mlflow})")

    def _load_existing_runs(self) -> None:
        """Load previous runs from CSV."""
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    params = {k[6:]: v for k, v in row.items() if k.startswith("param_")}
                    metrics = {}
                    for k, v in row.items():
                        if k.startswith("metric_"):
                            try:
                                metrics[k[7:]] = float(v)
                            except (ValueError, TypeError):
                                pass
                    tags = {k[4:]: v for k, v in row.items() if k.startswith("tag_")}
                    record = RunRecord(
                        run_id=row.get("run_id", ""),
                        symbol=row.get("symbol", ""),
                        timestamp=row.get("timestamp", ""),
                        status=row.get("status", "completed"),
                        params=params,
                        metrics=metrics,
                        tags=tags,
                        model_path=row.get("model_path", None),
                        duration_seconds=float(row.get("duration_seconds", 0) or 0),
                    )
                    self._runs.append(record)
            logger.info(f"Loaded {len(self._runs)} previous experiment runs")
        except Exception as e:
            logger.warning(f"Could not load experiment log: {e}")

    @contextmanager
    def start_run(
        self,
        symbol: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Start a tracked experiment run.

        Args:
            symbol: Stock ticker being trained.
            params: Hyperparameters for this run.
            tags: Optional tags (e.g., model_type, data_version).

        Yields:
            RunContext for logging metrics during training.
        """
        run_id = str(uuid.uuid4())[:8]
        record = RunRecord(
            run_id=run_id,
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            status="running",
            params=params or {},
            tags=tags or {},
        )
        ctx = RunContext(record, self)
        start_time = datetime.now()

        mlflow_run = None
        if self.use_mlflow:
            mlflow_run = mlflow.start_run(run_name=f"{symbol}_{run_id}")
            mlflow.log_params(record.params)
            if record.tags:
                for k, v in record.tags.items():
                    mlflow.set_tag(k, v)
            mlflow.set_tag("symbol", symbol)

        try:
            yield ctx
            record.status = "completed"
        except Exception as e:
            record.status = "failed"
            record.tags["error"] = str(e)
            raise
        finally:
            record.duration_seconds = (datetime.now() - start_time).total_seconds()

            if self.use_mlflow and mlflow_run:
                if record.metrics:
                    mlflow.log_metrics(record.metrics)
                if record.model_path:
                    mlflow.log_artifact(record.model_path)
                mlflow.end_run()

            self._runs.append(record)
            self._save_run(record)
            logger.info(
                f"Run {run_id} for {symbol}: {record.status} "
                f"({record.duration_seconds:.1f}s)"
            )

    def _save_run(self, record: RunRecord) -> None:
        """Append a run to the CSV log."""
        flat = record.to_flat_dict()
        file_exists = self.csv_path.exists()

        if file_exists:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
        else:
            existing_fields = []

        all_fields = list(dict.fromkeys(existing_fields + list(flat.keys())))

        rows = []
        if file_exists:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        rows.append(flat)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        # Also save as JSON for easy programmatic access
        json_path = self.results_dir / f"run_{record.run_id}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(record), f, indent=2, default=str)

    def get_runs(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunRecord]:
        """Get filtered experiment runs."""
        runs = self._runs
        if symbol:
            runs = [r for r in runs if r.symbol == symbol]
        if status:
            runs = [r for r in runs if r.status == status]
        return runs

    def get_best_run(
        self,
        symbol: str,
        metric: str = "directional_accuracy",
        higher_is_better: bool = True,
    ) -> Optional[RunRecord]:
        """Get the best run for a symbol by a specific metric."""
        runs = [
            r for r in self._runs
            if r.symbol == symbol
            and r.status == "completed"
            and metric in r.metrics
        ]
        if not runs:
            return None

        return max(runs, key=lambda r: r.metrics[metric] * (1 if higher_is_better else -1))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all runs."""
        completed = [r for r in self._runs if r.status == "completed"]
        failed = [r for r in self._runs if r.status == "failed"]
        symbols = set(r.symbol for r in completed)

        per_symbol = {}
        for sym in symbols:
            sym_runs = [r for r in completed if r.symbol == sym]
            best = self.get_best_run(sym)
            per_symbol[sym] = {
                "total_runs": len(sym_runs),
                "best_directional_accuracy": best.metrics.get("directional_accuracy") if best else None,
                "best_rmse": self.get_best_run(sym, "rmse", higher_is_better=False).metrics.get("rmse")
                if self.get_best_run(sym, "rmse", higher_is_better=False) else None,
                "avg_duration_seconds": sum(r.duration_seconds or 0 for r in sym_runs) / len(sym_runs),
            }

        return {
            "total_runs": len(self._runs),
            "completed": len(completed),
            "failed": len(failed),
            "symbols_trained": len(symbols),
            "per_symbol": per_symbol,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of all experiments."""
        summary = self.get_summary()
        print(f"\n{'='*70}")
        print(f"EXPERIMENT TRACKING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Runs: {summary['total_runs']} "
              f"(Completed: {summary['completed']}, Failed: {summary['failed']})")
        print(f"Symbols Trained: {summary['symbols_trained']}")

        if summary["per_symbol"]:
            print(f"\n{'Symbol':<8} {'Runs':>5} {'Best DA':>10} {'Best RMSE':>12} {'Avg Time':>10}")
            print("-" * 50)
            for sym, data in sorted(summary["per_symbol"].items()):
                da = f"{data['best_directional_accuracy']:.2%}" if data["best_directional_accuracy"] else "N/A"
                rmse = f"{data['best_rmse']:.6f}" if data["best_rmse"] else "N/A"
                dur = f"{data['avg_duration_seconds']:.1f}s"
                print(f"{sym:<8} {data['total_runs']:>5} {da:>10} {rmse:>12} {dur:>10}")
        print(f"{'='*70}")
