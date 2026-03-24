#!/usr/bin/env python3
"""
Walk-Forward Backtesting Engine for ATLAS Stock ML Intelligence System.

Implements production-quality walk-forward validation for time-series financial
data. Unlike random train/test splits, walk-forward uses expanding windows that
respect the temporal ordering of market data, avoiding look-ahead bias.

Calculates institutional-grade performance metrics: Sharpe, Sortino, Calmar,
max drawdown, alpha/beta, profit factor, and per-symbol attribution.

Usage:
    python tests/integration/walk_forward_backtest.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup so we can import from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.stock_api import StockAPI  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single executed trade."""
    date: pd.Timestamp
    symbol: str
    side: str           # "BUY" or "SELL"
    quantity: float
    price: float
    value: float
    commission: float
    pnl: float = 0.0   # realised P&L (filled on sell)


@dataclass
class WindowResult:
    """Performance snapshot for one walk-forward window."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy_return: float
    benchmark_return: float
    trades: List[Trade] = field(default_factory=list)


@dataclass
class DrawdownInfo:
    """Peak-to-trough drawdown detail."""
    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _annualized_return(equity_series: pd.Series) -> float:
    """Compute annualized return from a daily equity curve."""
    if len(equity_series) < 2:
        return 0.0
    total_return = equity_series.iloc[-1] / equity_series.iloc[0]
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    if n_days <= 0:
        return 0.0
    return float(total_return ** (365.25 / n_days) - 1.0)


def _annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * np.sqrt(252))


def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio."""
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualized Sortino ratio (penalises downside volatility only)."""
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = float(np.sqrt((downside ** 2).mean())) * np.sqrt(252)
    ann_excess = float(excess.mean() * 252)
    return ann_excess / downside_std


def _max_drawdown(equity_series: pd.Series) -> DrawdownInfo:
    """Compute maximum drawdown with peak/trough dates."""
    if len(equity_series) < 2:
        ts = equity_series.index[0] if len(equity_series) else pd.Timestamp.now()
        return DrawdownInfo(0.0, ts, ts, ts)

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max

    trough_idx = drawdown.idxmin()
    peak_idx = equity_series.loc[:trough_idx].idxmax()

    # Recovery: first date after trough where equity >= peak value
    peak_val = equity_series.loc[peak_idx]
    post_trough = equity_series.loc[trough_idx:]
    recovered = post_trough[post_trough >= peak_val]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None

    return DrawdownInfo(
        max_drawdown=float(abs(drawdown.min())),
        peak_date=peak_idx,
        trough_date=trough_idx,
        recovery_date=recovery_date,
    )


def _calmar_ratio(ann_return: float, max_dd: float) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    if max_dd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return ann_return / max_dd


def _win_rate(trade_pnls: pd.Series) -> float:
    """Percentage of trades with positive P&L."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def _profit_factor(trade_pnls: pd.Series) -> float:
    """Sum of gains divided by sum of losses."""
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _beta_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> Tuple[float, float]:
    """CAPM beta and Jensen's alpha against a benchmark."""
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 0.0

    # Align on common dates
    combined = pd.DataFrame({
        "strat": strategy_returns,
        "bench": benchmark_returns,
    }).dropna()

    if len(combined) < 2 or combined["bench"].std() == 0:
        return 0.0, 0.0

    cov = combined.cov()
    beta = float(cov.loc["strat", "bench"] / cov.loc["bench", "bench"])

    ann_strat = float(combined["strat"].mean() * 252)
    ann_bench = float(combined["bench"].mean() * 252)
    alpha = ann_strat - (risk_free_rate + beta * (ann_bench - risk_free_rate))
    return beta, alpha


def _monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns (rows=year, cols=month)."""
    monthly = equity_series.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = table.pivot_table(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][:len(pivot.columns)]
    return pivot


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------

class BacktestReport:
    """Generates and stores comprehensive backtest analytics.

    Attributes:
        metrics: Dict of scalar performance metrics.
        equity_curve: Daily equity series.
        drawdown_series: Daily drawdown series.
        monthly_table: Monthly returns pivot table.
        symbol_attribution: Per-symbol P&L breakdown.
        window_results: Per walk-forward-window results.
        trades: Complete trade log.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series,
        trades: List[Trade],
        window_results: List[WindowResult],
        symbols: List[str],
        risk_free_rate: float = 0.04,
        commission_rate: float = 0.0,
    ) -> None:
        self.equity_curve = equity_curve
        self.benchmark_curve = benchmark_curve
        self.trades = trades
        self.window_results = window_results
        self.risk_free_rate = risk_free_rate
        self.commission_rate = commission_rate

        # Compute daily returns
        self.daily_returns = equity_curve.pct_change().dropna()
        self.bench_returns = benchmark_curve.pct_change().dropna()

        # Drawdown
        running_max = equity_curve.cummax()
        self.drawdown_series = (equity_curve - running_max) / running_max

        # Monthly
        self.monthly_table = _monthly_returns(equity_curve)

        # Per-symbol attribution
        self.symbol_attribution = self._compute_attribution(trades, symbols)

        # Scalar metrics
        self.metrics = self._compute_metrics()

    # ---- internal ---------------------------------------------------------

    def _compute_attribution(
        self, trades: List[Trade], symbols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate P&L, trade count, and win rate per symbol."""
        attr: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            sym_trades = [t for t in trades if t.symbol == sym and t.side == "SELL"]
            pnls = pd.Series([t.pnl for t in sym_trades], dtype=float)
            attr[sym] = {
                "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
                "trade_count": len(sym_trades),
                "win_rate": _win_rate(pnls) if len(pnls) else 0.0,
                "avg_pnl": float(pnls.mean()) if len(pnls) else 0.0,
            }
        return attr

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute all scalar performance metrics."""
        ann_ret = _annualized_return(self.equity_curve)
        ann_vol = _annualized_volatility(self.daily_returns)
        dd_info = _max_drawdown(self.equity_curve)

        sell_pnls = pd.Series(
            [t.pnl for t in self.trades if t.side == "SELL"], dtype=float
        )

        beta, alpha = _beta_alpha(
            self.daily_returns, self.bench_returns, self.risk_free_rate
        )

        total_commission = sum(t.commission for t in self.trades)

        return {
            "total_return": float(
                self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
            ),
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": _sharpe_ratio(self.daily_returns, self.risk_free_rate),
            "sortino_ratio": _sortino_ratio(self.daily_returns, self.risk_free_rate),
            "max_drawdown": dd_info.max_drawdown,
            "max_drawdown_peak": str(dd_info.peak_date.date()),
            "max_drawdown_trough": str(dd_info.trough_date.date()),
            "calmar_ratio": _calmar_ratio(ann_ret, dd_info.max_drawdown),
            "win_rate": _win_rate(sell_pnls),
            "profit_factor": _profit_factor(sell_pnls),
            "beta": beta,
            "alpha": alpha,
            "total_trades": len(self.trades),
            "total_commission": total_commission,
            "benchmark_return": float(
                self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1
            ),
            "risk_free_rate": self.risk_free_rate,
        }

    # ---- public -----------------------------------------------------------

    def summary(self) -> str:
        """Return formatted summary statistics table."""
        m = self.metrics
        lines = [
            "",
            "=" * 64,
            "  WALK-FORWARD BACKTEST REPORT",
            "=" * 64,
            "",
            "  Performance Metrics",
            "  " + "-" * 50,
            f"  Total Return:            {m['total_return']:>+10.2%}",
            f"  Annualized Return:       {m['annualized_return']:>+10.2%}",
            f"  Annualized Volatility:   {m['annualized_volatility']:>10.2%}",
            f"  Sharpe Ratio:            {m['sharpe_ratio']:>10.3f}",
            f"  Sortino Ratio:           {m['sortino_ratio']:>10.3f}",
            f"  Max Drawdown:            {m['max_drawdown']:>10.2%}",
            f"    Peak:                  {m['max_drawdown_peak']:>10s}",
            f"    Trough:                {m['max_drawdown_trough']:>10s}",
            f"  Calmar Ratio:            {m['calmar_ratio']:>10.3f}",
            "",
            "  Risk Metrics",
            "  " + "-" * 50,
            f"  Beta (vs SPY):           {m['beta']:>10.3f}",
            f"  Alpha (annualized):      {m['alpha']:>+10.2%}",
            f"  Win Rate:                {m['win_rate']:>10.2%}",
            f"  Profit Factor:           {m['profit_factor']:>10.3f}",
            "",
            "  Execution",
            "  " + "-" * 50,
            f"  Total Trades:            {m['total_trades']:>10d}",
            f"  Total Commission:        ${m['total_commission']:>9.2f}",
            f"  Benchmark Return (SPY):  {m['benchmark_return']:>+10.2%}",
            f"  Risk-Free Rate:          {m['risk_free_rate']:>10.2%}",
            "",
        ]

        # Walk-forward window breakdown
        if self.window_results:
            lines.append("  Walk-Forward Windows")
            lines.append("  " + "-" * 50)
            for w in self.window_results:
                lines.append(
                    f"    Window {w.window_id}: "
                    f"test {str(w.test_start.date())} -> {str(w.test_end.date())}  "
                    f"strat={w.strategy_return:+.2%}  bench={w.benchmark_return:+.2%}"
                )
            lines.append("")

        # Per-symbol attribution
        if self.symbol_attribution:
            lines.append("  Per-Symbol Attribution")
            lines.append("  " + "-" * 50)
            lines.append(f"  {'Symbol':<8} {'P&L':>10} {'Trades':>8} {'Win%':>8}")
            for sym, attr in sorted(
                self.symbol_attribution.items(),
                key=lambda x: x[1]["total_pnl"],
                reverse=True,
            ):
                lines.append(
                    f"  {sym:<8} ${attr['total_pnl']:>9.2f} "
                    f"{attr['trade_count']:>8d} "
                    f"{attr['win_rate']:>7.1%}"
                )
            lines.append("")

        # Monthly returns
        if not self.monthly_table.empty:
            lines.append("  Monthly Returns")
            lines.append("  " + "-" * 50)
            for year in self.monthly_table.index:
                row = self.monthly_table.loc[year]
                vals = "  ".join(
                    f"{v:+.1%}" if pd.notna(v) else "   --  "
                    for v in row.values
                )
                lines.append(f"  {year}  {vals}")
            lines.append("")

        lines.append("=" * 64)
        return "\n".join(lines)

    def save_results(self, filepath: str) -> None:
        """Save metrics, equity curve, and attribution to a JSON file.

        Args:
            filepath: Destination path for the JSON output.
        """
        output: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "metrics": self.metrics,
            "equity_curve": {
                str(d.date()): round(v, 2)
                for d, v in self.equity_curve.items()
            },
            "drawdown_series": {
                str(d.date()): round(v, 6)
                for d, v in self.drawdown_series.items()
            },
            "symbol_attribution": self.symbol_attribution,
            "window_results": [
                {
                    "window_id": w.window_id,
                    "train_start": str(w.train_start.date()),
                    "train_end": str(w.train_end.date()),
                    "test_start": str(w.test_start.date()),
                    "test_end": str(w.test_end.date()),
                    "strategy_return": round(w.strategy_return, 6),
                    "benchmark_return": round(w.benchmark_return, 6),
                }
                for w in self.window_results
            ],
            "monthly_returns": (
                self.monthly_table.to_dict() if not self.monthly_table.empty else {}
            ),
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to {filepath}")


# ---------------------------------------------------------------------------
# WalkForwardBacktest
# ---------------------------------------------------------------------------

class WalkForwardBacktest:
    """Walk-forward backtesting engine with expanding-window validation.

    Walk-forward validation splits time-series data into N sequential windows.
    For each window the training set expands from the start of the data up to
    the window boundary, and the out-of-sample test set is the next segment.
    This prevents look-ahead bias inherent in random cross-validation.

    The strategy is equal-weight rebalancing with configurable frequency,
    transaction costs, and risk parameters.

    Args:
        initial_capital: Starting portfolio value in USD.
        commission_rate: Per-trade commission as a fraction of trade value
                         (0.001 = 10 bps).
        risk_free_rate:  Annualized risk-free rate for Sharpe/Sortino.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.risk_free_rate = risk_free_rate
        self.api = StockAPI()

    # ---- data fetching ----------------------------------------------------

    def _fetch_data(
        self, symbols: List[str], period: str
    ) -> Dict[str, pd.DataFrame]:
        """Download historical OHLCV data for all symbols.

        Returns:
            Mapping of symbol -> DataFrame with 'timestamp' index and OHLCV cols.
        """
        data: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            print(f"  Fetching {sym} ({period})...", end=" ")
            df = self.api.get_historical_data(sym, period=period)
            if df is None or df.empty:
                print("FAILED")
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            data[sym] = df
            print(f"{len(df)} rows  [{df.index[0].date()} -> {df.index[-1].date()}]")
        return data

    def _build_close_matrix(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Build an aligned close-price matrix from per-symbol DataFrames."""
        frames = {sym: df["close"].rename(sym) for sym, df in data.items()}
        close = pd.concat(frames, axis=1).sort_index()
        close.ffill(inplace=True)
        close.dropna(inplace=True)
        return close

    # ---- walk-forward splits ----------------------------------------------

    @staticmethod
    def _split_windows(
        dates: pd.DatetimeIndex, n_windows: int
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Create expanding-window train/test splits.

        The total date range is divided into (n_windows + 1) equal segments.
        Window *i* trains on segments 0..i and tests on segment i+1.

        Returns:
            List of (train_dates, test_dates) tuples.
        """
        n_total = len(dates)
        seg_size = n_total // (n_windows + 1)
        if seg_size < 5:
            raise ValueError(
                f"Not enough data ({n_total} rows) for {n_windows} windows. "
                f"Reduce walk_forward_windows or use a longer start_period."
            )

        splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        for i in range(n_windows):
            train_end_idx = seg_size * (i + 1)
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + seg_size, n_total)
            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            if len(test_dates) > 0:
                splits.append((train_dates, test_dates))
        return splits

    # ---- simulation -------------------------------------------------------

    def _simulate_window(
        self,
        close: pd.DataFrame,
        test_dates: pd.DatetimeIndex,
        symbols: List[str],
        rebalance_freq: int,
    ) -> Tuple[pd.Series, List[Trade]]:
        """Simulate the equal-weight rebalancing strategy over test_dates.

        Args:
            close: Full close-price DataFrame.
            test_dates: Dates for this out-of-sample window.
            symbols: Tradeable symbols (subset of close columns).
            rebalance_freq: Trading days between rebalances.

        Returns:
            (equity_series, trades_list) for the window.
        """
        available = [s for s in symbols if s in close.columns]
        test_close = close.loc[test_dates, available]

        cash = self.initial_capital
        holdings: Dict[str, float] = {s: 0.0 for s in available}
        equity: Dict[pd.Timestamp, float] = {}
        trades: List[Trade] = []

        # Cost basis tracking for P&L
        cost_basis: Dict[str, float] = {s: 0.0 for s in available}

        for day_idx, (date, row) in enumerate(test_close.iterrows()):
            # Rebalance on the first day and then every rebalance_freq days
            if day_idx % rebalance_freq == 0:
                portfolio_value = cash
                for s in available:
                    portfolio_value += holdings[s] * row[s]

                target_weight = 1.0 / len(available) if available else 0.0

                for s in available:
                    price = row[s]
                    if pd.isna(price) or price <= 0:
                        continue

                    target_value = portfolio_value * target_weight
                    current_value = holdings[s] * price
                    delta_value = target_value - current_value
                    delta_shares = delta_value / price

                    if abs(delta_value) < 50:  # minimum trade threshold
                        continue

                    commission = abs(delta_value) * self.commission_rate

                    if delta_shares > 0:
                        total_cost = delta_value + commission
                        if total_cost > cash:
                            # Buy what we can afford
                            delta_value = cash - commission
                            if delta_value <= 0:
                                continue
                            delta_shares = delta_value / price
                            commission = abs(delta_value) * self.commission_rate

                        cash -= delta_shares * price + commission
                        cost_basis[s] = (
                            (cost_basis[s] * holdings[s] + delta_shares * price)
                            / (holdings[s] + delta_shares)
                            if (holdings[s] + delta_shares) > 0
                            else price
                        )
                        holdings[s] += delta_shares

                        trades.append(Trade(
                            date=date,
                            symbol=s,
                            side="BUY",
                            quantity=delta_shares,
                            price=price,
                            value=delta_shares * price,
                            commission=commission,
                        ))
                    else:
                        shares_to_sell = abs(delta_shares)
                        shares_to_sell = min(shares_to_sell, holdings[s])
                        if shares_to_sell <= 0:
                            continue
                        proceeds = shares_to_sell * price
                        commission = proceeds * self.commission_rate
                        realised_pnl = (price - cost_basis[s]) * shares_to_sell

                        cash += proceeds - commission
                        holdings[s] -= shares_to_sell

                        trades.append(Trade(
                            date=date,
                            symbol=s,
                            side="SELL",
                            quantity=shares_to_sell,
                            price=price,
                            value=proceeds,
                            commission=commission,
                            pnl=realised_pnl,
                        ))

            # Record daily equity
            port_val = cash
            for s in available:
                port_val += holdings[s] * row[s]
            equity[date] = port_val

        equity_series = pd.Series(equity, dtype=float)
        equity_series.index = pd.DatetimeIndex(equity_series.index)
        return equity_series, trades

    # ---- main entry point -------------------------------------------------

    def run(
        self,
        symbols: Optional[List[str]] = None,
        start_period: str = "2y",
        rebalance_freq: int = 21,
        walk_forward_windows: int = 5,
    ) -> BacktestReport:
        """Execute walk-forward backtest.

        Args:
            symbols: List of tickers to trade. Defaults to a diversified set.
            start_period: yfinance period string for data download.
            rebalance_freq: Trading days between portfolio rebalances.
            walk_forward_windows: Number of walk-forward out-of-sample windows.

        Returns:
            BacktestReport with all computed metrics and series data.
        """
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "XOM"]

        benchmark_sym = "SPY"
        all_symbols = list(set(symbols + [benchmark_sym]))

        print("\n" + "=" * 64)
        print("  ATLAS Walk-Forward Backtest Engine")
        print("=" * 64)
        print(f"  Capital:     ${self.initial_capital:,.0f}")
        print(f"  Symbols:     {', '.join(symbols)}")
        print(f"  Period:      {start_period}")
        print(f"  Rebalance:   every {rebalance_freq} trading days")
        print(f"  Windows:     {walk_forward_windows}")
        print(f"  Commission:  {self.commission_rate:.2%} per trade")
        print(f"  Risk-free:   {self.risk_free_rate:.2%}")
        print("=" * 64)

        # 1. Fetch data
        print("\n  [1/3] Downloading historical data...")
        raw_data = self._fetch_data(all_symbols, start_period)
        if len(raw_data) < 2:
            raise RuntimeError(
                f"Insufficient data: only got {list(raw_data.keys())}. "
                "Need at least 2 symbols (including SPY benchmark)."
            )

        close = self._build_close_matrix(raw_data)
        print(f"\n  Aligned price matrix: {close.shape[0]} days x {close.shape[1]} symbols")
        print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}")

        # 2. Walk-forward splits
        print(f"\n  [2/3] Creating {walk_forward_windows} walk-forward windows...")
        splits = self._split_windows(close.index, walk_forward_windows)

        all_equity_segments: List[pd.Series] = []
        all_bench_segments: List[pd.Series] = []
        all_trades: List[Trade] = []
        window_results: List[WindowResult] = []

        for i, (train_dates, test_dates) in enumerate(splits):
            print(
                f"\n  Window {i + 1}/{len(splits)}: "
                f"train [{train_dates[0].date()} -> {train_dates[-1].date()}] "
                f"({len(train_dates)} days) | "
                f"test [{test_dates[0].date()} -> {test_dates[-1].date()}] "
                f"({len(test_dates)} days)"
            )

            # Strategy equity
            eq, trades = self._simulate_window(
                close, test_dates, symbols, rebalance_freq
            )

            # Benchmark equity (buy-and-hold SPY over same window)
            if benchmark_sym in close.columns:
                spy_prices = close.loc[test_dates, benchmark_sym]
                bench_eq = spy_prices / spy_prices.iloc[0] * self.initial_capital
            else:
                bench_eq = pd.Series(
                    self.initial_capital, index=test_dates, dtype=float
                )

            strat_ret = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
            bench_ret = float(bench_eq.iloc[-1] / bench_eq.iloc[0] - 1) if len(bench_eq) > 1 else 0.0

            window_results.append(WindowResult(
                window_id=i + 1,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                strategy_return=strat_ret,
                benchmark_return=bench_ret,
                trades=trades,
            ))

            all_equity_segments.append(eq)
            all_bench_segments.append(bench_eq)
            all_trades.extend(trades)

            print(
                f"    Strategy: {strat_ret:+.2%}  |  "
                f"Benchmark: {bench_ret:+.2%}  |  "
                f"Trades: {len(trades)}"
            )

        # 3. Stitch equity curves (chain-link: each window starts where
        #    the prior ended, preserving compounding)
        print(f"\n  [3/3] Computing performance metrics...")
        full_equity = self._chain_equity(all_equity_segments)
        full_bench = self._chain_equity(all_bench_segments)

        report = BacktestReport(
            equity_curve=full_equity,
            benchmark_curve=full_bench,
            trades=all_trades,
            window_results=window_results,
            symbols=symbols,
            risk_free_rate=self.risk_free_rate,
            commission_rate=self.commission_rate,
        )
        return report

    @staticmethod
    def _chain_equity(segments: List[pd.Series]) -> pd.Series:
        """Chain-link multiple equity segments into one continuous curve.

        Each segment is rescaled so it starts at the ending value of the
        previous segment, preserving compounded growth across windows.
        """
        if not segments:
            return pd.Series(dtype=float)

        chained = segments[0].copy()
        for seg in segments[1:]:
            if seg.empty or chained.empty:
                continue
            scale = chained.iloc[-1] / seg.iloc[0]
            scaled = seg * scale
            # Drop the first point if it overlaps
            if scaled.index[0] in chained.index:
                scaled = scaled.iloc[1:]
            chained = pd.concat([chained, scaled])

        return chained


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a demonstration walk-forward backtest and print results."""
    print("\n" + "#" * 64)
    print("#  ATLAS - Walk-Forward Backtesting Engine")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 64)

    engine = WalkForwardBacktest(
        initial_capital=100_000,
        commission_rate=0.001,   # 10 bps
        risk_free_rate=0.04,
    )

    report = engine.run(
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "XOM"],
        start_period="2y",
        rebalance_freq=21,
        walk_forward_windows=5,
    )

    print(report.summary())

    # Save results
    results_dir = os.path.join(_PROJECT_ROOT, "tests", "integration", "results")
    results_path = os.path.join(results_dir, "walk_forward_results.json")
    report.save_results(results_path)

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
