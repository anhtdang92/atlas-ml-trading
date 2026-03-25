"""
Feature Importance Analysis for ATLAS Stock ML Pipeline

Provides multiple methods to assess which of the 25+ technical indicators
carry the most predictive signal for 21-day forward returns:

1. Permutation Importance - Model-agnostic, measures accuracy drop when feature is shuffled
2. SHAP (SHapley Additive exPlanations) - Game-theoretic feature attribution
3. Mutual Information - Non-linear dependency between feature and target
4. Correlation-based - Simple linear relationship with target

Usage:
    from ml.feature_importance import FeatureImportanceAnalyzer
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.full_analysis(X_train, y_train, X_test, y_test, feature_names)
    analyzer.plot_summary(results)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.feature_selection import mutual_info_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple complementary methods.

    Combines model-agnostic (permutation, SHAP) and statistical (mutual info,
    correlation) approaches to provide a robust view of feature relevance.
    """

    def __init__(self, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required: pip install scikit-learn")
        self.random_state = random_state
        self._results: Dict[str, pd.DataFrame] = {}

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D LSTM sequences to 2D for tree/linear models.

        Uses only the last timestep (most recent data) to preserve
        interpretability. Alternative: flatten all timesteps.
        """
        if X.ndim == 3:
            return X[:, -1, :]  # Last timestep only
        return X

    def correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Compute absolute Pearson and Spearman correlation with target.

        Simple but effective baseline for linear relationships.
        """
        X_flat = self._flatten_sequences(X)
        df = pd.DataFrame(X_flat, columns=feature_names)

        pearson = df.corrwith(pd.Series(y, name="target")).abs()
        spearman = df.corrwith(pd.Series(y, name="target"), method="spearman").abs()

        result = pd.DataFrame({
            "feature": feature_names,
            "pearson_abs": pearson.values,
            "spearman_abs": spearman.values,
            "correlation_score": (pearson.values + spearman.values) / 2,
        }).sort_values("correlation_score", ascending=False).reset_index(drop=True)

        self._results["correlation"] = result
        logger.info("Correlation importance computed")
        return result

    def mutual_information_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Compute mutual information between each feature and target.

        Captures non-linear dependencies that correlation misses.
        """
        X_flat = self._flatten_sequences(X)
        mi_scores = mutual_info_regression(
            X_flat, y, random_state=self.random_state, n_neighbors=5
        )

        result = pd.DataFrame({
            "feature": feature_names,
            "mutual_information": mi_scores,
        }).sort_values("mutual_information", ascending=False).reset_index(drop=True)

        self._results["mutual_info"] = result
        logger.info("Mutual information importance computed")
        return result

    def permutation_importance_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """Compute permutation importance using a Ridge regression surrogate.

        Measures how much model performance degrades when each feature is
        randomly shuffled. Model-agnostic and intuitive.
        """
        X_train_flat = self._flatten_sequences(X_train)
        X_test_flat = self._flatten_sequences(X_test)

        # Use Ridge as surrogate (fast, reasonable approximation)
        model = Ridge(alpha=1.0)
        model.fit(X_train_flat, y_train)

        baseline_mse = mean_squared_error(y_test, model.predict(X_test_flat))
        logger.info(f"Permutation importance baseline MSE: {baseline_mse:.6f}")

        perm_result = permutation_importance(
            model, X_test_flat, y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_squared_error",
        )

        result = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self._results["permutation"] = result
        logger.info("Permutation importance computed")
        return result

    def tree_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_estimators: int = 200,
    ) -> pd.DataFrame:
        """Compute feature importance from Random Forest (MDI).

        Measures how much each feature reduces impurity across all trees.
        Complementary to permutation importance.
        """
        X_flat = self._flatten_sequences(X_train)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_flat, y_train)

        result = pd.DataFrame({
            "feature": feature_names,
            "rf_importance": rf.feature_importances_,
        }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

        self._results["tree"] = result
        logger.info("Random Forest feature importance computed")
        return result

    def shap_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        max_samples: int = 200,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Compute SHAP values for feature attribution.

        Uses TreeExplainer with a Random Forest surrogate for speed.
        Returns both summary statistics and raw SHAP values for plotting.
        """
        if not HAS_SHAP:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return pd.DataFrame(), None

        X_train_flat = self._flatten_sequences(X_train)
        X_test_flat = self._flatten_sequences(X_test)

        # Subsample for speed
        if len(X_test_flat) > max_samples:
            idx = np.random.RandomState(self.random_state).choice(
                len(X_test_flat), max_samples, replace=False
            )
            X_test_flat = X_test_flat[idx]

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            random_state=self.random_state, n_jobs=-1,
        )
        rf.fit(X_train_flat, y_train)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test_flat)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        result = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        self._results["shap"] = result
        logger.info("SHAP importance computed")
        return result, shap_values

    def full_analysis(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Run all importance methods and produce a unified ranking.

        Each method's scores are rank-normalized to [0, 1], then averaged
        to create a consensus importance ranking.
        """
        logger.info(f"Running full feature importance analysis ({len(feature_names)} features)...")

        # Run all analyses
        self.correlation_importance(X_test, y_test, feature_names)
        self.mutual_information_importance(X_test, y_test, feature_names)
        self.permutation_importance_analysis(
            X_train, y_train, X_test, y_test, feature_names
        )
        self.tree_importance(X_train, y_train, feature_names)

        shap_df = None
        if HAS_SHAP:
            shap_df, _ = self.shap_importance(
                X_train, y_train, X_test, feature_names
            )

        # Build consensus ranking
        consensus = pd.DataFrame({"feature": feature_names})

        # Rank-normalize each method (higher rank = more important)
        n = len(feature_names)
        methods = {
            "correlation": "correlation_score",
            "mutual_info": "mutual_information",
            "permutation": "importance_mean",
            "tree": "rf_importance",
        }

        if HAS_SHAP and shap_df is not None and len(shap_df) > 0:
            methods["shap"] = "mean_abs_shap"

        for method_key, score_col in methods.items():
            df = self._results[method_key]
            merged = consensus.merge(
                df[["feature", score_col]], on="feature", how="left"
            )
            # Rank normalize to [0, 1]
            ranks = merged[score_col].rank(ascending=True)
            consensus[f"rank_{method_key}"] = ranks / n

        # Average rank across all methods
        rank_cols = [c for c in consensus.columns if c.startswith("rank_")]
        consensus["consensus_score"] = consensus[rank_cols].mean(axis=1)
        consensus = consensus.sort_values(
            "consensus_score", ascending=False
        ).reset_index(drop=True)
        consensus["rank"] = range(1, len(consensus) + 1)

        self._results["consensus"] = consensus
        logger.info("Consensus ranking computed")
        return consensus

    def plot_summary(
        self,
        consensus: Optional[pd.DataFrame] = None,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a multi-panel feature importance visualization.

        Panels:
        1. Consensus ranking (horizontal bar)
        2. Method comparison heatmap
        3. Individual method bar charts
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not installed, skipping plot")
            return

        if consensus is None:
            consensus = self._results.get("consensus")
        if consensus is None:
            raise ValueError("Run full_analysis() first")

        top = consensus.head(top_n)
        rank_cols = [c for c in top.columns if c.startswith("rank_")]
        method_labels = [c.replace("rank_", "").replace("_", " ").title() for c in rank_cols]

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("Feature Importance Analysis — ATLAS Stock ML",
                      fontsize=16, fontweight="bold", y=1.02)

        # Panel 1: Consensus ranking
        ax = axes[0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))[::-1]
        bars = ax.barh(
            range(top_n), top["consensus_score"].values[::-1],
            color=colors, edgecolor="white", linewidth=0.5
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"].values[::-1], fontsize=10)
        ax.set_xlabel("Consensus Score (normalized)", fontsize=11)
        ax.set_title(f"Top {top_n} Features — Consensus Ranking", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Panel 2: Method comparison heatmap
        ax = axes[1]
        heatmap_data = top[rank_cols].values
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"].values, fontsize=10)
        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels, fontsize=10, rotation=45, ha="right")
        ax.set_title("Method Agreement Heatmap", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Normalized Rank", shrink=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def get_top_features(self, n: int = 10) -> List[str]:
        """Return the top N features by consensus ranking."""
        consensus = self._results.get("consensus")
        if consensus is None:
            raise ValueError("Run full_analysis() first")
        return consensus.head(n)["feature"].tolist()

    def to_markdown_table(self, top_n: int = 15) -> str:
        """Export consensus ranking as a markdown table for documentation."""
        consensus = self._results.get("consensus")
        if consensus is None:
            raise ValueError("Run full_analysis() first")

        top = consensus.head(top_n)
        lines = [
            "| Rank | Feature | Consensus Score | Top Method |",
            "|------|---------|-----------------|------------|",
        ]
        rank_cols = [c for c in top.columns if c.startswith("rank_")]
        for _, row in top.iterrows():
            best_method = max(rank_cols, key=lambda c: row[c])
            best_method_name = best_method.replace("rank_", "").replace("_", " ").title()
            lines.append(
                f"| {int(row['rank'])} | {row['feature']} | "
                f"{row['consensus_score']:.3f} | {best_method_name} |"
            )
        return "\n".join(lines)


def main():
    """Demo feature importance analysis with synthetic data."""
    print("=" * 60)
    print("Feature Importance Analysis Demo")
    print("=" * 60)

    np.random.seed(42)

    # Simulate LSTM-style input (samples, timesteps, features)
    n_train, n_test = 400, 100
    timesteps, n_features = 30, 25
    feature_names = [
        "close", "Log_Volume",
        "MA_10", "MA_20", "MA_50", "MA_200",
        "Price_to_MA50", "Price_to_MA200", "MA_50_200_Cross",
        "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Width", "BB_Position",
        "Volume_MA_20", "Volume_ROC", "Volume_Ratio",
        "Daily_Return", "Momentum_14", "Momentum_30", "ROC_10",
        "Volatility_14", "Volatility_30", "ATR_14",
    ]

    X_train = np.random.randn(n_train, timesteps, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, timesteps, n_features).astype(np.float32)

    # Target has signal from a few features (momentum, RSI, volatility)
    y_train = (
        0.3 * X_train[:, -1, 18]   # Daily_Return
        + 0.2 * X_train[:, -1, 9]  # RSI (scaled)
        + 0.1 * X_train[:, -1, 22] # Volatility_14
        + np.random.randn(n_train) * 0.05
    )
    y_test = (
        0.3 * X_test[:, -1, 18]
        + 0.2 * X_test[:, -1, 9]
        + 0.1 * X_test[:, -1, 22]
        + np.random.randn(n_test) * 0.05
    )

    analyzer = FeatureImportanceAnalyzer()
    consensus = analyzer.full_analysis(X_train, y_train, X_test, y_test, feature_names)

    print("\nTop 10 Features by Consensus Ranking:")
    print(consensus.head(10)[["rank", "feature", "consensus_score"]].to_string(index=False))

    print("\n" + analyzer.to_markdown_table(top_n=10))

    # Save plot
    fig = analyzer.plot_summary(save_path="results/feature_importance.png")
    if fig:
        print("\nPlot saved to results/feature_importance.png")


if __name__ == "__main__":
    main()
