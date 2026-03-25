"""
Backtest Tearsheet Generator for ATLAS Stock ML Pipeline

Produces a professional-grade performance report similar to what quantitative
hedge funds and asset managers use for strategy evaluation. Generates both
a visual PDF-style tearsheet and a structured metrics dictionary.

Metrics include:
- Return statistics: Total, CAGR, annualized volatility
- Risk-adjusted: Sharpe, Sortino, Calmar ratios
- Drawdown analysis: Max drawdown, duration, recovery
- Distribution: Skewness, kurtosis, best/worst periods
- Rolling analysis: 30/60/90-day rolling Sharpe, returns
- Monthly return heatmap

Usage:
    from ml.backtest_tearsheet import BacktestTearsheet
    ts = BacktestTearsheet(portfolio_values, dates, benchmark_values)
    ts.generate_report(save_path="results/tearsheet.png")
    metrics = ts.compute_metrics()
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mtick
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BacktestTearsheet:
    """Generate quantitative tearsheets from portfolio return series.

    Args:
        portfolio_values: Array of daily portfolio values (NAV).
        dates: Corresponding date array/index.
        benchmark_values: Optional benchmark NAV for comparison (e.g., SPY).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        name: Strategy name for report title.
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        portfolio_values: np.ndarray,
        dates: pd.DatetimeIndex,
        benchmark_values: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.04,
        name: str = "ATLAS ML Strategy",
    ):
        self.name = name
        self.risk_free_rate = risk_free_rate

        # Build returns series
        self.portfolio = pd.Series(portfolio_values, index=dates, name="Strategy")
        self.returns = self.portfolio.pct_change().dropna()

        if benchmark_values is not None:
            self.benchmark = pd.Series(benchmark_values, index=dates, name="Benchmark")
            self.bench_returns = self.benchmark.pct_change().dropna()
        else:
            self.benchmark = None
            self.bench_returns = None

    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        r = self.returns
        n_days = len(r)
        n_years = n_days / self.TRADING_DAYS_PER_YEAR

        # Basic returns
        total_return = (self.portfolio.iloc[-1] / self.portfolio.iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

        # Volatility
        ann_vol = r.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        downside_vol = r[r < 0].std() * np.sqrt(self.TRADING_DAYS_PER_YEAR) if len(r[r < 0]) > 0 else 1e-6

        # Risk-adjusted ratios
        daily_rf = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        excess_returns = r - daily_rf
        sharpe = excess_returns.mean() / r.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR) if r.std() > 0 else 0
        sortino = excess_returns.mean() / downside_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR) if downside_vol > 0 else 0

        # Drawdown analysis
        cummax = self.portfolio.cummax()
        drawdown = (self.portfolio - cummax) / cummax
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

        # Drawdown duration
        in_dd = drawdown < 0
        dd_groups = (~in_dd).cumsum()[in_dd]
        max_dd_duration = dd_groups.value_counts().max() if len(dd_groups) > 0 else 0

        # Distribution
        skewness = float(r.skew())
        kurtosis = float(r.kurtosis())

        # Win rate
        win_rate = (r > 0).sum() / len(r) if len(r) > 0 else 0
        profit_factor = abs(r[r > 0].sum() / r[r < 0].sum()) if r[r < 0].sum() != 0 else float("inf")

        # Best / worst
        best_day = r.max()
        worst_day = r.min()
        best_month = r.resample("ME").sum().max() if n_days > 20 else best_day
        worst_month = r.resample("ME").sum().min() if n_days > 20 else worst_day

        metrics = {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_duration_days": int(max_dd_duration),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "best_day": best_day,
            "worst_day": worst_day,
            "best_month": best_month,
            "worst_month": worst_month,
            "trading_days": n_days,
            "years": n_years,
        }

        # Benchmark comparison
        if self.bench_returns is not None:
            bench_total = (self.benchmark.iloc[-1] / self.benchmark.iloc[0]) - 1
            bench_vol = self.bench_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            bench_sharpe = (self.bench_returns.mean() - daily_rf) / self.bench_returns.std() * np.sqrt(
                self.TRADING_DAYS_PER_YEAR
            ) if self.bench_returns.std() > 0 else 0
            active_return = total_return - bench_total
            tracking_error = (r - self.bench_returns).std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            information_ratio = active_return / tracking_error if tracking_error > 0 else 0

            # Beta and Alpha
            cov = np.cov(r, self.bench_returns)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
            alpha = cagr - (self.risk_free_rate + beta * (bench_total / max(n_years, 1e-6) - self.risk_free_rate))

            metrics.update({
                "benchmark_return": bench_total,
                "benchmark_sharpe": bench_sharpe,
                "active_return": active_return,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "beta": beta,
                "alpha": alpha,
            })

        return metrics

    def _monthly_returns_table(self) -> pd.DataFrame:
        """Compute monthly returns as a pivot table (Year x Month)."""
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        table = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        pivot = table.pivot_table(values="Return", index="Year", columns="Month", aggfunc="sum")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
        return pivot

    def generate_report(self, save_path: Optional[str] = None) -> Optional[object]:
        """Generate a full tearsheet visualization.

        Layout:
        - Row 1: Cumulative returns + drawdown
        - Row 2: Monthly heatmap + distribution
        - Row 3: Rolling Sharpe + metrics table
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not installed, cannot generate tearsheet")
            return None

        metrics = self.compute_metrics()

        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.25)

        fig.suptitle(
            f"{self.name} — Performance Tearsheet",
            fontsize=20, fontweight="bold", y=0.98,
        )
        fig.text(
            0.5, 0.96,
            f"Period: {self.portfolio.index[0].strftime('%Y-%m-%d')} to "
            f"{self.portfolio.index[-1].strftime('%Y-%m-%d')} | "
            f"{metrics['trading_days']} trading days",
            ha="center", fontsize=12, color="gray",
        )

        # ── Panel 1: Cumulative Returns ──────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        cum_returns = (1 + self.returns).cumprod()
        ax1.plot(cum_returns.index, cum_returns.values, color="#00d4ff",
                 linewidth=2, label=self.name)
        if self.bench_returns is not None:
            bench_cum = (1 + self.bench_returns).cumprod()
            ax1.plot(bench_cum.index, bench_cum.values, color="#888888",
                     linewidth=1.5, linestyle="--", label="Benchmark")
            ax1.legend(fontsize=11)
        ax1.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Growth of $1", fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.2f"))

        # ── Panel 2: Underwater (Drawdown) Chart ─────────────────────
        ax2 = fig.add_subplot(gs[1, :])
        cummax = self.portfolio.cummax()
        drawdown = (self.portfolio - cummax) / cummax
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color="#ff4444", alpha=0.4)
        ax2.plot(drawdown.index, drawdown.values, color="#ff4444", linewidth=1)
        ax2.set_title("Underwater Plot (Drawdown)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Drawdown %", fontsize=11)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.grid(alpha=0.3)

        # ── Panel 3: Monthly Returns Heatmap ─────────────────────────
        ax3 = fig.add_subplot(gs[2, 0])
        try:
            monthly_table = self._monthly_returns_table()
            if not monthly_table.empty:
                im = ax3.imshow(monthly_table.values, cmap="RdYlGn", aspect="auto",
                                vmin=-0.10, vmax=0.10)
                ax3.set_yticks(range(len(monthly_table.index)))
                ax3.set_yticklabels(monthly_table.index.astype(int))
                ax3.set_xticks(range(len(monthly_table.columns)))
                ax3.set_xticklabels(monthly_table.columns, fontsize=9)
                # Annotate cells
                for i in range(len(monthly_table.index)):
                    for j in range(len(monthly_table.columns)):
                        val = monthly_table.iloc[i, j]
                        if not np.isnan(val):
                            ax3.text(j, i, f"{val:.1%}", ha="center", va="center",
                                     fontsize=8, fontweight="bold",
                                     color="white" if abs(val) > 0.05 else "black")
                fig.colorbar(im, ax=ax3, label="Monthly Return", shrink=0.8,
                             format=mtick.PercentFormatter(1.0))
        except Exception as e:
            ax3.text(0.5, 0.5, f"Insufficient data\nfor monthly heatmap\n({e})",
                     ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")

        # ── Panel 4: Return Distribution ─────────────────────────────
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(self.returns.values, bins=50, color="#00d4ff", alpha=0.7,
                 edgecolor="white", linewidth=0.5, density=True)
        ax4.axvline(self.returns.mean(), color="#ff9900", linestyle="--",
                     linewidth=2, label=f"Mean: {self.returns.mean():.4f}")
        ax4.axvline(0, color="white", linestyle="-", linewidth=1, alpha=0.5)
        ax4.set_title("Daily Return Distribution", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Daily Return", fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)

        # ── Panel 5: Rolling Sharpe Ratio ────────────────────────────
        ax5 = fig.add_subplot(gs[3, 0])
        for window, color, label in [(30, "#00d4ff", "30-day"), (90, "#ff9900", "90-day")]:
            if len(self.returns) >= window:
                daily_rf = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
                rolling_mean = (self.returns - daily_rf).rolling(window).mean()
                rolling_std = self.returns.rolling(window).std()
                rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
                ax5.plot(rolling_sharpe.index, rolling_sharpe.values,
                         color=color, linewidth=1.5, label=label)
        ax5.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax5.axhline(1, color="green", linewidth=1, linestyle=":", alpha=0.5, label="Sharpe = 1")
        ax5.set_title("Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
        ax5.set_ylabel("Sharpe Ratio", fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(alpha=0.3)

        # ── Panel 6: Key Metrics Table ───────────────────────────────
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.axis("off")
        table_data = [
            ["Total Return", f"{metrics['total_return']:+.2%}"],
            ["CAGR", f"{metrics['cagr']:+.2%}"],
            ["Annualized Vol", f"{metrics['annualized_volatility']:.2%}"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
            ["Sortino Ratio", f"{metrics['sortino_ratio']:.2f}"],
            ["Calmar Ratio", f"{metrics['calmar_ratio']:.2f}"],
            ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
            ["Max DD Duration", f"{metrics['max_drawdown_duration_days']} days"],
            ["Win Rate", f"{metrics['win_rate']:.1%}"],
            ["Profit Factor", f"{metrics['profit_factor']:.2f}"],
            ["Best Day", f"{metrics['best_day']:+.2%}"],
            ["Worst Day", f"{metrics['worst_day']:+.2%}"],
            ["Skewness", f"{metrics['skewness']:.2f}"],
            ["Kurtosis", f"{metrics['kurtosis']:.2f}"],
        ]

        if "beta" in metrics:
            table_data.extend([
                ["Beta", f"{metrics['beta']:.2f}"],
                ["Alpha", f"{metrics['alpha']:+.2%}"],
                ["Info Ratio", f"{metrics['information_ratio']:.2f}"],
            ])

        table = ax6.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            colWidths=[0.5, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)

        # Style header
        for j in range(2):
            table[0, j].set_facecolor("#1a1a2e")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Color code return values
        for i, (_, val_str) in enumerate(table_data, start=1):
            if "%" in val_str and val_str.startswith("+"):
                table[i, 1].set_text_props(color="green", fontweight="bold")
            elif "%" in val_str and val_str.startswith("-"):
                table[i, 1].set_text_props(color="red", fontweight="bold")

        ax6.set_title("Key Performance Metrics", fontsize=14, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            logger.info(f"Tearsheet saved to {save_path}")

        return fig

    def to_markdown(self) -> str:
        """Export metrics as markdown for README or documentation."""
        m = self.compute_metrics()
        lines = [
            f"## {self.name} — Performance Summary\n",
            f"**Period:** {self.portfolio.index[0].strftime('%Y-%m-%d')} to "
            f"{self.portfolio.index[-1].strftime('%Y-%m-%d')} "
            f"({m['trading_days']} trading days)\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Return | {m['total_return']:+.2%} |",
            f"| CAGR | {m['cagr']:+.2%} |",
            f"| Annualized Volatility | {m['annualized_volatility']:.2%} |",
            f"| Sharpe Ratio | {m['sharpe_ratio']:.2f} |",
            f"| Sortino Ratio | {m['sortino_ratio']:.2f} |",
            f"| Calmar Ratio | {m['calmar_ratio']:.2f} |",
            f"| Max Drawdown | {m['max_drawdown']:.2%} |",
            f"| Win Rate | {m['win_rate']:.1%} |",
            f"| Profit Factor | {m['profit_factor']:.2f} |",
        ]

        if "alpha" in m:
            lines.extend([
                f"| Alpha | {m['alpha']:+.2%} |",
                f"| Beta | {m['beta']:.2f} |",
                f"| Information Ratio | {m['information_ratio']:.2f} |",
            ])

        return "\n".join(lines)


def main():
    """Demo tearsheet with simulated portfolio data."""
    print("=" * 60)
    print("Backtest Tearsheet Demo")
    print("=" * 60)

    np.random.seed(42)

    # Simulate 2 years of daily returns
    n_days = 504  # ~2 years
    dates = pd.bdate_range(end="2026-03-20", periods=n_days)

    # Strategy: slight positive drift with realistic volatility
    daily_returns = np.random.normal(0.0004, 0.012, n_days)  # ~10% annual, 19% vol
    portfolio_values = 10000 * np.cumprod(1 + daily_returns)

    # Benchmark: buy-and-hold market
    bench_returns = np.random.normal(0.0003, 0.011, n_days)
    benchmark_values = 10000 * np.cumprod(1 + bench_returns)

    ts = BacktestTearsheet(
        portfolio_values=portfolio_values,
        dates=dates,
        benchmark_values=benchmark_values,
        name="ATLAS LSTM Strategy",
    )

    # Print metrics
    metrics = ts.compute_metrics()
    print("\nKey Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Generate tearsheet
    import os
    os.makedirs("results", exist_ok=True)
    fig = ts.generate_report(save_path="results/tearsheet_demo.png")
    if fig:
        print("\nTearsheet saved to results/tearsheet_demo.png")

    # Markdown export
    print("\n" + ts.to_markdown())


if __name__ == "__main__":
    main()
