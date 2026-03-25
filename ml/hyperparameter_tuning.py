"""
Hyperparameter Tuning for Stock LSTM Model

Bayesian optimization of LSTM hyperparameters using Optuna (preferred) with
sklearn RandomizedSearchCV fallback. Uses walk-forward cross-validation
(time-series splits) to respect temporal ordering -- never random splits.

Tuned Parameters:
- lstm_units: 32-256 (hidden state dimensionality)
- dropout_rate: 0.1-0.5 (regularization strength)
- l2_reg: 1e-5 to 1e-2 (weight decay, log scale)
- learning_rate: 1e-4 to 1e-2 (optimizer step size, log scale)
- batch_size: 16, 32, or 64 (training mini-batch size)
- lookback: 15-60 (input sequence length in trading days)

Objective: Directional accuracy (the metric that matters for trading).
A model that correctly predicts up/down is more valuable than one that
minimizes MSE but gets the direction wrong.

Usage:
    tuner = HyperparameterTuner()
    best_params = tuner.tune(X, y, n_trials=50, n_cv_splits=3)
    summary_df = tuner.get_study_summary()
    fig = tuner.plot_optimization_history()
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.exceptions import TrialPruned
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward time-series splitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """Walk-forward cross-validation for time-series data.

    Unlike sklearn's TimeSeriesSplit which uses expanding windows, this uses
    a fixed training window that slides forward -- closer to how you would
    actually validate a trading model in production.

    Each fold:
        train: [i * step : i * step + train_size]
        val:   [i * step + train_size : i * step + train_size + val_size]
    """

    def __init__(self, n_splits: int = 3, val_ratio: float = 0.15) -> None:
        self.n_splits = n_splits
        self.val_ratio = val_ratio

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation index arrays for each fold.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        val_size = max(int(n_samples * self.val_ratio), 1)
        train_size = n_samples - val_size - (self.n_splits - 1) * val_size
        if train_size < val_size:
            # Not enough data -- fall back to a single split
            split_point = int(n_samples * 0.8)
            train_idx = np.arange(0, split_point)
            val_idx = np.arange(split_point, n_samples)
            logger.warning(
                "Insufficient data for %d walk-forward splits; using single 80/20 split",
                self.n_splits,
            )
            return [(train_idx, val_idx)]

        step = val_size
        folds: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            val_start = train_end
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            folds.append((train_idx, val_idx))

        if not folds:
            split_point = int(n_samples * 0.8)
            folds = [(np.arange(0, split_point), np.arange(split_point, n_samples))]
            logger.warning("Walk-forward produced no folds; using single 80/20 split")

        logger.info(
            "Walk-forward splitter: %d folds, ~%d train / ~%d val samples each",
            len(folds), len(folds[0][0]), len(folds[0][1]),
        )
        return folds


# ---------------------------------------------------------------------------
# Core training + evaluation helper
# ---------------------------------------------------------------------------

def _build_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: Dict[str, Any],
    epochs: int = 100,
    verbose: int = 0,
) -> float:
    """Build an LSTM with *params*, train on fold, return directional accuracy.

    This is intentionally a module-level function so it can be called from both
    the Optuna objective and the sklearn fallback path.

    Args:
        X: Full input array (samples, timesteps, features).
        y: Full target array (samples,).
        train_idx: Indices for the training portion of this fold.
        val_idx: Indices for the validation portion of this fold.
        params: Hyperparameter dict with keys: lstm_units, dropout_rate,
                l2_reg, learning_rate, batch_size.
        epochs: Maximum training epochs (early stopping may cut short).
        verbose: Keras verbosity level.

    Returns:
        Directional accuracy on the validation set (float in [0, 1]).
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required for hyperparameter tuning")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    lstm_units = int(params['lstm_units'])
    dropout_rate = float(params['dropout_rate'])
    l2_reg = float(params['l2_reg'])
    learning_rate = float(params['learning_rate'])
    batch_size = int(params['batch_size'])

    num_features = X_train.shape[2]
    lookback = X_train.shape[1]
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = keras.Sequential([
        layers.Input(shape=(lookback, num_features)),
        layers.LSTM(lstm_units, return_sequences=True,
                    recurrent_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.LSTM(lstm_units, return_sequences=False,
                    recurrent_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=0.1),
        metrics=['mae'],
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=0,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    predictions = model.predict(X_val, verbose=0).flatten()
    pred_direction = predictions > 0
    true_direction = y_val > 0
    directional_accuracy = float(np.mean(pred_direction == true_direction))

    # Clean up to free GPU memory between trials
    del model
    tf.keras.backend.clear_session()

    return directional_accuracy


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _optuna_objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    epochs: int,
) -> float:
    """Optuna objective function: mean directional accuracy across walk-forward folds."""
    params = {
        'lstm_units': trial.suggest_int('lstm_units', 32, 256, step=32),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
        'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }

    fold_scores: List[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        score = _build_and_evaluate(X, y, train_idx, val_idx, params, epochs=epochs)
        fold_scores.append(score)

        # Report intermediate value for potential pruning
        trial.report(np.mean(fold_scores), fold_idx)
        if trial.should_prune():
            raise TrialPruned()

    mean_score = float(np.mean(fold_scores))
    logger.info(
        "Trial %d: dir_acc=%.4f  params=%s",
        trial.number, mean_score,
        {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()},
    )
    return mean_score


# ---------------------------------------------------------------------------
# sklearn fallback
# ---------------------------------------------------------------------------

def _sklearn_random_search(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    epochs: int = 100,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fallback hyperparameter search when Optuna is not installed.

    Uses simple random sampling over the same parameter space.

    Returns:
        best_params: Dict of best hyperparameters found.
        all_trials: List of dicts with 'params' and 'score' keys.
    """
    rng = np.random.RandomState(42)
    all_trials: List[Dict[str, Any]] = []
    best_score = -1.0
    best_params: Dict[str, Any] = {}

    for trial_idx in range(n_trials):
        params = {
            'lstm_units': int(rng.choice(range(32, 257, 32))),
            'dropout_rate': round(rng.uniform(0.1, 0.5), 2),
            'l2_reg': float(10 ** rng.uniform(-5, -2)),
            'learning_rate': float(10 ** rng.uniform(-4, -2)),
            'batch_size': int(rng.choice([16, 32, 64])),
        }

        fold_scores: List[float] = []
        for train_idx, val_idx in folds:
            score = _build_and_evaluate(X, y, train_idx, val_idx, params, epochs=epochs)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        all_trials.append({'params': params, 'score': mean_score})

        logger.info(
            "Trial %d/%d: dir_acc=%.4f  params=%s",
            trial_idx + 1, n_trials, mean_score,
            {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()},
        )

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()

    return best_params, all_trials


# ---------------------------------------------------------------------------
# Main tuner class
# ---------------------------------------------------------------------------

class HyperparameterTuner:
    """Bayesian hyperparameter optimization for the ATLAS LSTM model.

    Uses Optuna when available, falls back to random search otherwise.
    Always uses walk-forward (time-series) cross-validation -- never random splits.

    Example:
        tuner = HyperparameterTuner()
        best = tuner.tune(X, y, n_trials=50, n_cv_splits=3)
        print(best)
        # {'lstm_units': 128, 'dropout_rate': 0.2, 'l2_reg': 0.0003, ...}
    """

    def __init__(self) -> None:
        self._study: Optional[Any] = None  # optuna.Study or None
        self._all_trials: Optional[List[Dict[str, Any]]] = None  # sklearn fallback
        self._best_params: Optional[Dict[str, Any]] = None
        self._backend: Optional[str] = None

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        n_cv_splits: int = 3,
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            X: Input sequences, shape (samples, timesteps, features).
            y: Target returns, shape (samples,).
            n_trials: Number of hyperparameter configurations to evaluate.
            n_cv_splits: Number of walk-forward cross-validation folds.
            epochs: Max training epochs per fold (early stopping applies).

        Returns:
            Dictionary of best hyperparameters found.
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for hyperparameter tuning")

        logger.info("=" * 60)
        logger.info("Starting hyperparameter tuning")
        logger.info("   Samples: %d", len(X))
        logger.info("   Timesteps: %d, Features: %d", X.shape[1], X.shape[2])
        logger.info("   Trials: %d, CV Splits: %d", n_trials, n_cv_splits)
        logger.info("   Backend: %s", "Optuna" if HAS_OPTUNA else "RandomSearch (sklearn fallback)")
        logger.info("=" * 60)

        splitter = WalkForwardSplitter(n_splits=n_cv_splits)
        folds = splitter.split(len(X))
        start_time = time.time()

        if HAS_OPTUNA:
            self._backend = "optuna"
            self._study = optuna.create_study(
                direction='maximize',
                study_name='atlas_lstm_tuning',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            )
            self._study.optimize(
                lambda trial: _optuna_objective(trial, X, y, folds, epochs),
                n_trials=n_trials,
                show_progress_bar=True,
            )
            self._best_params = self._study.best_params.copy()
        else:
            self._backend = "random_search"
            logger.info("Optuna not installed; using random search fallback")
            self._best_params, self._all_trials = _sklearn_random_search(
                X, y, folds, n_trials=n_trials, epochs=epochs,
            )

        elapsed = time.time() - start_time
        logger.info("Tuning complete in %.1f seconds", elapsed)
        logger.info("Best params: %s", self._best_params)

        return self._best_params

    def get_study_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all trials.

        Columns: trial_number, score (directional accuracy), and one column
        per hyperparameter.

        Returns:
            DataFrame sorted by score descending.
        """
        if self._backend == "optuna" and self._study is not None:
            records = []
            for trial in self._study.trials:
                if trial.state.name != 'COMPLETE':
                    continue
                row = {'trial_number': trial.number, 'score': trial.value}
                row.update(trial.params)
                records.append(row)
            df = pd.DataFrame(records)
        elif self._all_trials is not None:
            records = []
            for i, t in enumerate(self._all_trials):
                row = {'trial_number': i, 'score': t['score']}
                row.update(t['params'])
                records.append(row)
            df = pd.DataFrame(records)
        else:
            logger.warning("No tuning results available. Run tune() first.")
            return pd.DataFrame()

        if not df.empty:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
        return df

    def plot_optimization_history(self) -> plt.Figure:
        """Plot objective value (directional accuracy) across trials.

        Returns:
            matplotlib Figure showing per-trial score and running best.
        """
        summary = self.get_study_summary()
        if summary.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No tuning data available', ha='center', va='center')
            return fig

        # Sort by trial number for chronological view
        summary = summary.sort_values('trial_number')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(summary['trial_number'], summary['score'],
                   alpha=0.6, s=30, label='Trial score', color='#00d4ff')

        running_best = summary['score'].cummax()
        ax.plot(summary['trial_number'], running_best,
                color='#ff6e40', linewidth=2, label='Best so far')

        ax.set_xlabel('Trial')
        ax.set_ylabel('Directional Accuracy')
        ax.set_title('Hyperparameter Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_param_importances(self) -> plt.Figure:
        """Plot hyperparameter importance based on correlation with score.

        For Optuna: uses built-in importance evaluator when available.
        For random search: uses absolute Spearman rank correlation.

        Returns:
            matplotlib Figure showing parameter importance bars.
        """
        summary = self.get_study_summary()
        if summary.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No tuning data available', ha='center', va='center')
            return fig

        # If Optuna is available and was used, try its built-in importance
        if self._backend == "optuna" and HAS_OPTUNA and self._study is not None:
            try:
                importances = optuna.importance.get_param_importances(self._study)
                params = list(importances.keys())
                values = list(importances.values())

                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.barh(params, values, color='#00d4ff', edgecolor='#0097a7')
                ax.set_xlabel('Importance')
                ax.set_title('Hyperparameter Importance (Optuna fANOVA)')
                ax.grid(True, alpha=0.3, axis='x')
                fig.tight_layout()
                return fig
            except Exception as e:
                logger.warning("Optuna importance failed (%s), falling back to correlation", e)

        # Fallback: rank correlation
        param_cols = [c for c in summary.columns if c not in ('trial_number', 'score')]
        importances: Dict[str, float] = {}

        try:
            from scipy.stats import spearmanr
            for col in param_cols:
                if summary[col].nunique() > 1:
                    corr, _ = spearmanr(summary[col], summary['score'])
                    importances[col] = abs(corr) if not np.isnan(corr) else 0.0
                else:
                    importances[col] = 0.0
        except ImportError:
            # No scipy -- use simple Pearson via numpy
            for col in param_cols:
                if summary[col].nunique() > 1:
                    corr = np.corrcoef(summary[col].astype(float), summary['score'])[0, 1]
                    importances[col] = abs(corr) if not np.isnan(corr) else 0.0
                else:
                    importances[col] = 0.0

        # Sort by importance
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        params = [p[0] for p in sorted_params]
        values = [p[1] for p in sorted_params]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(params, values, color='#00d4ff', edgecolor='#0097a7')
        ax.set_xlabel('|Correlation| with Directional Accuracy')
        ax.set_title('Hyperparameter Importance (Rank Correlation)')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    """Demonstrate hyperparameter tuning on synthetic data."""
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Install with: pip install tensorflow")
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    print("=" * 60)
    print("Hyperparameter Tuning Demo (Synthetic Data)")
    print("=" * 60)
    print(f"Backend: {'Optuna (Bayesian)' if HAS_OPTUNA else 'Random Search (fallback)'}")

    # Generate synthetic time-series data that mimics the LSTM pipeline output
    np.random.seed(42)
    n_samples = 300
    lookback = 30
    n_features = 29

    X = np.random.randn(n_samples, lookback, n_features).astype(np.float32)
    # Create a target with weak signal: sum of last-day features + noise
    signal = X[:, -1, :3].sum(axis=1) * 0.01
    noise = np.random.randn(n_samples) * 0.05
    y = (signal + noise).astype(np.float32)

    print(f"\nSynthetic data: X={X.shape}, y={y.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

    # Run tuning with small trial count for demo
    tuner = HyperparameterTuner()
    best_params = tuner.tune(X, y, n_trials=5, n_cv_splits=2, epochs=10)

    print(f"\nBest hyperparameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Show study summary
    summary = tuner.get_study_summary()
    print(f"\nStudy summary ({len(summary)} completed trials):")
    print(summary.to_string(index=False))

    # Save plots
    fig_history = tuner.plot_optimization_history()
    fig_history.savefig('/tmp/tuning_history.png', dpi=100)
    print("\nOptimization history plot saved to /tmp/tuning_history.png")

    fig_importance = tuner.plot_param_importances()
    fig_importance.savefig('/tmp/param_importances.png', dpi=100)
    print("Parameter importance plot saved to /tmp/param_importances.png")

    plt.close('all')
    print("\nHyperparameter tuning demo complete!")


if __name__ == "__main__":
    main()
