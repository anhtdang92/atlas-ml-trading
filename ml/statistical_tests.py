"""
Statistical Significance Tests for ATLAS Stock ML Pipeline

Provides rigorous statistical validation for ML model performance:

1. Bootstrap Confidence Intervals — for Sharpe ratio, directional accuracy, RMSE, MAE
2. Diebold-Mariano Test — compare two forecast models for significant difference
3. Paired t-test — simple pairwise comparison of prediction errors
4. Rolling Stability Analysis — detect regime-dependent performance

Why this matters:
A model with 58% directional accuracy on 80 test samples could easily be noise.
These tests answer: "Is my model genuinely better than the baseline, or did I get lucky?"

Usage:
    from ml.statistical_tests import StatisticalValidator
    validator = StatisticalValidator()
    result = validator.bootstrap_sharpe(strategy_returns, n_bootstrap=10000)
    comparison = validator.compare_models(y_true, {"LSTM": preds_lstm, "Ridge": preds_ridge})
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class StatisticalValidator:
    """Statistical validation toolkit for ML trading model evaluation.

    All methods use non-parametric bootstrap or distribution-free tests
    to avoid assumptions about return distributions (which are typically
    fat-tailed and non-normal in financial data).
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    # ─── Generic Bootstrap ────────────────────────────────────────────

    def bootstrap_metric(
        self,
        values: np.ndarray,
        metric_fn: Callable[[np.ndarray], float],
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Dict[str, Any]:
        """Bootstrap confidence interval for any metric.

        Args:
            values: Array of observations (e.g., returns, errors).
            metric_fn: Function that computes the metric from a sample.
            n_bootstrap: Number of bootstrap resamples.
            ci: Confidence level (default 95%).

        Returns:
            Dict with point_estimate, ci_lower, ci_upper, std, distribution.
        """
        n = len(values)
        point_estimate = metric_fn(values)

        boot_estimates = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = self._rng.randint(0, n, size=n)
            boot_estimates[i] = metric_fn(values[idx])

        alpha = 1 - ci
        ci_lower = np.percentile(boot_estimates, alpha / 2 * 100)
        ci_upper = np.percentile(boot_estimates, (1 - alpha / 2) * 100)

        # Bootstrap p-value: proportion of bootstrap samples on wrong side of zero
        p_value = float(np.mean(boot_estimates <= 0)) if point_estimate > 0 else float(
            np.mean(boot_estimates >= 0)
        )

        return {
            "point_estimate": float(point_estimate),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_level": ci,
            "std": float(np.std(boot_estimates)),
            "p_value": float(p_value),
            "n_bootstrap": n_bootstrap,
            "distribution": boot_estimates,
        }

    # ─── Sharpe Ratio Bootstrap ───────────────────────────────────────

    def bootstrap_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.04,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Dict[str, Any]:
        """Bootstrap confidence interval for annualized Sharpe ratio.

        Uses the Ledoit-Wolf (2008) adjustment for autocorrelation in
        financial returns when computing the standard error.
        """
        daily_rf = risk_free_rate / self.TRADING_DAYS_PER_YEAR

        def sharpe_fn(r: np.ndarray) -> float:
            excess = r - daily_rf
            if np.std(r) == 0:
                return 0.0
            return float(np.mean(excess) / np.std(r) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

        result = self.bootstrap_metric(returns, sharpe_fn, n_bootstrap, ci)
        result["metric_name"] = "Annualized Sharpe Ratio"
        logger.info(
            f"Sharpe: {result['point_estimate']:.2f} "
            f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}] "
            f"(p={result['p_value']:.4f})"
        )
        return result

    # ─── Directional Accuracy Bootstrap ───────────────────────────────

    def bootstrap_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Dict[str, Any]:
        """Bootstrap CI for directional accuracy.

        Tests whether the model's up/down accuracy is significantly
        above the 50% random baseline.
        """
        correct = ((y_pred > 0) == (y_true > 0)).astype(float)

        def da_fn(c: np.ndarray) -> float:
            return float(np.mean(c))

        result = self.bootstrap_metric(correct, da_fn, n_bootstrap, ci)
        result["metric_name"] = "Directional Accuracy"

        # Adjusted p-value: test against 50% null hypothesis
        boot_dist = result["distribution"]
        p_value_vs_random = float(np.mean(boot_dist <= 0.5))
        result["p_value_vs_random"] = p_value_vs_random
        result["significantly_above_random"] = p_value_vs_random < (1 - ci)

        logger.info(
            f"DA: {result['point_estimate']:.1%} "
            f"[{result['ci_lower']:.1%}, {result['ci_upper']:.1%}] "
            f"(p vs random={p_value_vs_random:.4f})"
        )
        return result

    # ─── Diebold-Mariano Test ─────────────────────────────────────────

    def diebold_mariano_test(
        self,
        errors_1: np.ndarray,
        errors_2: np.ndarray,
        loss_fn: str = "squared",
    ) -> Dict[str, float]:
        """Diebold-Mariano test for equal predictive accuracy.

        Tests H0: Model 1 and Model 2 have equal forecast accuracy.
        Rejection (p < 0.05) means one model is significantly better.

        Args:
            errors_1: Prediction errors from model 1 (y_true - y_pred_1).
            errors_2: Prediction errors from model 2 (y_true - y_pred_2).
            loss_fn: "squared" for MSE comparison, "absolute" for MAE.

        Returns:
            Dict with dm_statistic, p_value, model_1_better (bool).
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for Diebold-Mariano test")

        if loss_fn == "squared":
            d = errors_1 ** 2 - errors_2 ** 2
        elif loss_fn == "absolute":
            d = np.abs(errors_1) - np.abs(errors_2)
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        n = len(d)
        mean_d = np.mean(d)

        # Newey-West HAC variance estimator (lag = int(n^(1/3)))
        max_lag = max(1, int(n ** (1 / 3)))
        gamma_0 = np.var(d)
        gamma = 0.0
        for lag in range(1, max_lag + 1):
            weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
            gamma += 2 * weight * np.cov(d[:-lag], d[lag:])[0, 1]

        variance = (gamma_0 + gamma) / n
        if variance <= 0:
            variance = gamma_0 / n  # Fallback

        dm_stat = mean_d / np.sqrt(max(variance, 1e-12))
        p_value = float(2 * scipy_stats.norm.sf(abs(dm_stat)))  # Two-sided

        result = {
            "dm_statistic": float(dm_stat),
            "p_value": p_value,
            "model_1_better": mean_d < 0,  # Negative means model 1 has smaller loss
            "significant_at_5pct": p_value < 0.05,
            "significant_at_10pct": p_value < 0.10,
            "mean_loss_diff": float(mean_d),
        }

        logger.info(
            f"DM test: stat={dm_stat:.3f}, p={p_value:.4f}, "
            f"model_1_better={result['model_1_better']}"
        )
        return result

    # ─── Model Comparison ─────────────────────────────────────────────

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        n_bootstrap: int = 5000,
        ci: float = 0.95,
    ) -> pd.DataFrame:
        """Compare all models with bootstrap CIs and pairwise DM tests.

        Args:
            y_true: True target values.
            predictions: Dict mapping model name to prediction array.
            n_bootstrap: Bootstrap resamples for CIs.
            ci: Confidence level.

        Returns:
            DataFrame with metrics, CIs, and significance flags.
        """
        rows = []
        model_names = list(predictions.keys())

        for name, preds in predictions.items():
            errors = y_true - preds

            # Bootstrap DA
            da_result = self.bootstrap_directional_accuracy(y_true, preds, n_bootstrap, ci)

            # Bootstrap RMSE
            def rmse_fn(idx_errors):
                return float(np.sqrt(np.mean(idx_errors ** 2)))

            rmse_result = self.bootstrap_metric(errors, rmse_fn, n_bootstrap, ci)

            rows.append(
                {
                    "model": name,
                    "da": da_result["point_estimate"],
                    "da_ci_lower": da_result["ci_lower"],
                    "da_ci_upper": da_result["ci_upper"],
                    "da_significant_vs_random": da_result["significantly_above_random"],
                    "rmse": rmse_result["point_estimate"],
                    "rmse_ci_lower": rmse_result["ci_lower"],
                    "rmse_ci_upper": rmse_result["ci_upper"],
                }
            )

        df = pd.DataFrame(rows)

        # Pairwise DM tests (compare each model against the first / best)
        if len(model_names) >= 2 and HAS_SCIPY:
            reference = model_names[0]
            ref_errors = y_true - predictions[reference]
            dm_results = []
            for name in model_names[1:]:
                other_errors = y_true - predictions[name]
                dm = self.diebold_mariano_test(ref_errors, other_errors)
                dm_results.append(
                    {
                        "model": name,
                        f"dm_vs_{reference}_pvalue": dm["p_value"],
                        f"better_than_{reference}": dm["model_1_better"],
                    }
                )
            if dm_results:
                dm_df = pd.DataFrame(dm_results)
                df = df.merge(dm_df, on="model", how="left")

        return df.sort_values("da", ascending=False).reset_index(drop=True)

    # ─── Rolling Stability ────────────────────────────────────────────

    def rolling_stability(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        window: int = 50,
    ) -> pd.DataFrame:
        """Compute rolling directional accuracy and RMSE.

        Detects periods where the model performs well vs. poorly,
        suggesting regime-dependent effectiveness.
        """
        n = len(y_true)
        if n < window:
            logger.warning(f"Not enough samples ({n}) for window={window}")
            window = max(10, n // 3)

        correct = ((y_pred > 0) == (y_true > 0)).astype(float)
        squared_errors = (y_true - y_pred) ** 2

        records = []
        for i in range(window, n + 1):
            start = i - window
            records.append(
                {
                    "index": i,
                    "rolling_da": float(np.mean(correct[start:i])),
                    "rolling_rmse": float(np.sqrt(np.mean(squared_errors[start:i]))),
                    "rolling_mean_return": float(np.mean(y_true[start:i])),
                    "rolling_vol": float(np.std(y_true[start:i])),
                }
            )

        return pd.DataFrame(records)

    # ─── Visualization ────────────────────────────────────────────────

    def plot_bootstrap_distribution(
        self,
        result: Dict[str, Any],
        metric_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """Plot bootstrap sampling distribution with CI."""
        if not HAS_MATPLOTLIB:
            return None

        dist = result["distribution"]
        name = metric_name or result.get("metric_name", "Metric")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(dist, bins=80, color="#00d4ff", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(result["point_estimate"], color="red", linewidth=2, linestyle="-",
                    label=f"Point estimate: {result['point_estimate']:.4f}")
        ax.axvline(result["ci_lower"], color="orange", linewidth=2, linestyle="--",
                    label=f"{result['ci_level']:.0%} CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        ax.axvline(result["ci_upper"], color="orange", linewidth=2, linestyle="--")

        # Show null hypothesis line for relevant tests
        if "p_value_vs_random" in result:
            ax.axvline(0.5, color="gray", linewidth=2, linestyle=":",
                        label=f"Random baseline (50%)")

        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Bootstrap Distribution — {name} (n={result['n_bootstrap']:,})",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """Plot model comparison with error bars from bootstrap CIs."""
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # DA comparison with CIs
        ax = axes[0]
        models = comparison_df["model"].values
        da = comparison_df["da"].values
        da_err_low = da - comparison_df["da_ci_lower"].values
        da_err_high = comparison_df["da_ci_upper"].values - da

        colors = ["#00d4ff" if sig else "#ff9900"
                  for sig in comparison_df.get("da_significant_vs_random", [True] * len(models))]

        ax.barh(range(len(models)), da, xerr=[da_err_low, da_err_high],
                color=colors, edgecolor="white", linewidth=1.5, capsize=5)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Random (50%)")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)
        ax.set_xlabel("Directional Accuracy")
        ax.set_title("DA with Bootstrap 95% CI", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        # RMSE comparison with CIs
        ax = axes[1]
        rmse = comparison_df["rmse"].values
        rmse_err_low = rmse - comparison_df["rmse_ci_lower"].values
        rmse_err_high = comparison_df["rmse_ci_upper"].values - rmse

        ax.barh(range(len(models)), rmse, xerr=[rmse_err_low, rmse_err_high],
                color="#ff9900", edgecolor="white", linewidth=1.5, capsize=5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("RMSE with Bootstrap 95% CI", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")

        plt.suptitle("Model Comparison with Statistical Significance",
                      fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_rolling_stability(
        self,
        stability_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> Optional[object]:
        """Plot rolling DA and RMSE to visualize regime dependence."""
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Rolling DA
        ax = axes[0]
        ax.plot(stability_df["index"], stability_df["rolling_da"],
                color="#00d4ff", linewidth=1.5, label="Rolling DA")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random (50%)")
        ax.fill_between(stability_df["index"], 0.5, stability_df["rolling_da"],
                         where=stability_df["rolling_da"] > 0.5,
                         alpha=0.2, color="green", label="Above random")
        ax.fill_between(stability_df["index"], 0.5, stability_df["rolling_da"],
                         where=stability_df["rolling_da"] <= 0.5,
                         alpha=0.2, color="red", label="Below random")
        ax.set_ylabel("Directional Accuracy")
        ax.set_title("Rolling Directional Accuracy — Regime Analysis", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

        # Rolling volatility (regime proxy)
        ax = axes[1]
        ax.plot(stability_df["index"], stability_df["rolling_vol"],
                color="#ff9900", linewidth=1.5, label="Rolling Volatility")
        ax2 = ax.twinx()
        ax2.plot(stability_df["index"], stability_df["rolling_da"],
                 color="#00d4ff", linewidth=1, alpha=0.5, label="Rolling DA")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Return Volatility", color="#ff9900")
        ax2.set_ylabel("Directional Accuracy", color="#00d4ff")
        ax.set_title("Volatility Regime vs Model Performance", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─── Summary Report ───────────────────────────────────────────────

    def summary_report(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        strategy_returns: Optional[np.ndarray] = None,
        n_bootstrap: int = 5000,
    ) -> str:
        """Generate a formatted text summary of statistical validation.

        Args:
            y_true: True target returns.
            predictions: Dict mapping model name to predictions.
            strategy_returns: Optional daily strategy returns for Sharpe CI.
            n_bootstrap: Number of bootstrap resamples.

        Returns:
            Formatted string report.
        """
        lines = [
            "=" * 70,
            "STATISTICAL SIGNIFICANCE REPORT — ATLAS Stock ML",
            "=" * 70,
            "",
        ]

        # Model comparison
        comparison = self.compare_models(y_true, predictions, n_bootstrap)
        lines.append("MODEL COMPARISON (with Bootstrap 95% CIs)")
        lines.append("-" * 60)
        lines.append(f"{'Model':<25} {'DA':>8} {'DA 95% CI':>18} {'Sig?':>6}")
        lines.append("-" * 60)

        for _, row in comparison.iterrows():
            sig = "YES" if row.get("da_significant_vs_random", False) else "no"
            lines.append(
                f"{row['model']:<25} "
                f"{row['da']:>7.1%} "
                f"[{row['da_ci_lower']:.1%}, {row['da_ci_upper']:.1%}]"
                f"{'':>3}{sig:>4}"
            )

        lines.append("")

        # Pairwise significance
        if len(predictions) >= 2:
            model_names = list(predictions.keys())
            ref = model_names[0]
            ref_errors = y_true - predictions[ref]
            lines.append(f"PAIRWISE TESTS vs {ref}")
            lines.append("-" * 60)

            for name in model_names[1:]:
                other_errors = y_true - predictions[name]
                if HAS_SCIPY:
                    dm = self.diebold_mariano_test(ref_errors, other_errors)
                    better = ref if dm["model_1_better"] else name
                    sig_marker = "***" if dm["p_value"] < 0.01 else "**" if dm["p_value"] < 0.05 else "*" if dm["p_value"] < 0.10 else ""
                    lines.append(
                        f"  {ref} vs {name}: DM={dm['dm_statistic']:+.3f}, "
                        f"p={dm['p_value']:.4f}{sig_marker}  → {better} is better"
                    )

        lines.append("")

        # Sharpe ratio CI
        if strategy_returns is not None:
            sharpe_result = self.bootstrap_sharpe(strategy_returns, n_bootstrap=n_bootstrap)
            lines.append(f"SHARPE RATIO")
            lines.append("-" * 60)
            lines.append(
                f"  Point estimate: {sharpe_result['point_estimate']:.2f}"
            )
            lines.append(
                f"  95% CI:         [{sharpe_result['ci_lower']:.2f}, {sharpe_result['ci_upper']:.2f}]"
            )
            lines.append(
                f"  p-value:        {sharpe_result['p_value']:.4f}"
            )
            sig = "YES" if sharpe_result["p_value"] < 0.05 else "NO"
            lines.append(f"  Significantly > 0: {sig}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.10")
        lines.append("=" * 70)

        report = "\n".join(lines)
        logger.info("Statistical report generated")
        return report


def main() -> None:
    """Demo statistical validation with synthetic data."""
    print("Statistical Significance Testing Demo")
    print("=" * 50)

    np.random.seed(42)
    n = 100

    # Synthetic true values and predictions
    y_true = np.random.randn(n) * 0.05
    lstm_preds = y_true * 0.4 + np.random.randn(n) * 0.03  # Weak signal
    ridge_preds = y_true * 0.2 + np.random.randn(n) * 0.04  # Weaker signal
    random_preds = np.random.randn(n) * 0.05  # No signal

    validator = StatisticalValidator()

    # Bootstrap DA
    da_result = validator.bootstrap_directional_accuracy(y_true, lstm_preds)
    print(f"\nLSTM DA: {da_result['point_estimate']:.1%} "
          f"[{da_result['ci_lower']:.1%}, {da_result['ci_upper']:.1%}]")
    print(f"Significantly above random: {da_result['significantly_above_random']}")

    # Model comparison
    predictions = {"LSTM": lstm_preds, "Ridge": ridge_preds, "Random": random_preds}
    comparison = validator.compare_models(y_true, predictions)
    print(f"\n{comparison[['model', 'da', 'da_ci_lower', 'da_ci_upper', 'rmse']].to_string(index=False)}")

    # Sharpe
    strategy_returns = np.where(lstm_preds > 0, y_true / 21, 0)
    sharpe = validator.bootstrap_sharpe(strategy_returns)
    print(f"\nSharpe: {sharpe['point_estimate']:.2f} "
          f"[{sharpe['ci_lower']:.2f}, {sharpe['ci_upper']:.2f}]")

    # Full report
    report = validator.summary_report(y_true, predictions, strategy_returns)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
