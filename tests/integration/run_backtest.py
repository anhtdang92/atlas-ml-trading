#!/usr/bin/env python3
"""
Local Stock ML Strategy Backtest
Simulates an equal-weight rebalancing strategy using historical Yahoo Finance data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

try:
    from data.stock_api import StockAPI
except ImportError:
    print("Error: Could not import StockAPI")
    sys.exit(1)


class LocalBacktest:
    """Simple backtest engine for stock strategy."""

    def __init__(self, initial_capital=25000, symbols=None):
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'JNJ', 'TLT', 'GLD']
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.symbols = symbols
        self.holdings = {s: 0.0 for s in symbols}
        self.portfolio_history = []
        self.trade_history = []
        self.stock_api = StockAPI()

        # Risk parameters
        self.max_position_size = 0.15
        self.min_position_size = 0.02
        self.min_trade_value = 100
        self.trading_fee = 0.0  # Zero-commission

        print(f"Initializing Backtest")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Fee Model: Zero-commission")

    def fetch_historical_data(self, symbol, days=90):
        """Fetch historical OHLC data via yfinance."""
        print(f"\n   Fetching {days} days of data for {symbol}...")

        period = "6mo" if days <= 180 else "1y" if days <= 365 else "2y"
        df = self.stock_api.get_historical_data(symbol, period=period)

        if df is None or df.empty:
            print(f"   Failed to fetch data for {symbol}")
            return None

        df = df.tail(days).reset_index(drop=True)
        print(f"   Loaded {len(df)} days of data")
        print(f"   Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        return df

    def calculate_equal_weight_allocation(self):
        return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def rebalance(self, prices, date):
        target_allocations = self.calculate_equal_weight_allocation()

        portfolio_value = self.cash
        for symbol, quantity in self.holdings.items():
            portfolio_value += quantity * prices.get(symbol, 0)

        print(f"\n   Rebalancing on {date} | Portfolio: ${portfolio_value:,.2f}")

        trades_executed = 0
        for symbol, target_weight in target_allocations.items():
            price = prices.get(symbol)
            if price is None or price == 0:
                continue

            target_value = portfolio_value * target_weight
            current_value = self.holdings[symbol] * price
            target_quantity = target_value / price
            current_quantity = self.holdings[symbol]
            quantity_diff = target_quantity - current_quantity
            trade_value = abs(quantity_diff * price)

            if trade_value < self.min_trade_value:
                continue

            if quantity_diff > 0:
                cost = quantity_diff * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[symbol] += quantity_diff
                    self.trade_history.append({
                        'date': date, 'symbol': symbol, 'action': 'BUY',
                        'quantity': quantity_diff, 'price': price,
                        'value': cost, 'fee': 0.0
                    })
                    trades_executed += 1
            elif quantity_diff < 0:
                quantity_to_sell = abs(quantity_diff)
                proceeds = quantity_to_sell * price
                self.cash += proceeds
                self.holdings[symbol] -= quantity_to_sell
                self.trade_history.append({
                    'date': date, 'symbol': symbol, 'action': 'SELL',
                    'quantity': quantity_to_sell, 'price': price,
                    'value': proceeds, 'fee': 0.0
                })
                trades_executed += 1

        final_value = self.cash
        for symbol, quantity in self.holdings.items():
            final_value += quantity * prices.get(symbol, 0)

        self.portfolio_history.append({
            'date': date, 'value': final_value,
            'cash': self.cash, 'holdings': self.holdings.copy()
        })

    def run_backtest(self, days=90, rebalance_frequency=5):
        """Run the backtest over historical data."""
        print(f"\n{'='*60}")
        print(f"Running Backtest: {days} days, rebalance every {rebalance_frequency} days")
        print(f"{'='*60}")

        data = {}
        for symbol in self.symbols:
            df = self.fetch_historical_data(symbol, days)
            if df is not None:
                data[symbol] = df

        if not data:
            print("\nNo data available for backtesting")
            return

        dates = data[self.symbols[0]]['timestamp'].values

        for i, date in enumerate(dates):
            if i % rebalance_frequency == 0:
                prices = {}
                for symbol in self.symbols:
                    if symbol in data:
                        idx = (data[symbol]['timestamp'] == date).idxmax()
                        prices[symbol] = float(data[symbol].loc[idx, 'close'])
                self.rebalance(prices, pd.Timestamp(date).date())

        print(f"\n{'='*60}")
        print(f"Backtest Complete!")
        print(f"{'='*60}")
        self.print_results()

    def print_results(self):
        if not self.portfolio_history:
            print("No portfolio history to display")
            return

        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]['value']
        total_return = ((final_value - initial_value) / initial_value) * 100

        values = [p['value'] for p in self.portfolio_history]
        returns = np.diff(values) / values[:-1]

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/5) if len(returns) > 1 and np.std(returns) > 0 else 0

        cummax = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - cummax) / cummax
        max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        num_trades = len(self.trade_history)

        print(f"\nPerformance Summary:")
        print(f"{'='*60}")
        print(f"   Initial Capital:    ${initial_value:,.2f}")
        print(f"   Final Value:        ${final_value:,.2f}")
        print(f"   Total Return:       {total_return:+.2f}%")
        print(f"   Sharpe Ratio:       {sharpe:.2f}")
        print(f"   Max Drawdown:       {max_drawdown:.2f}%")
        print(f"   Total Trades:       {num_trades}")
        print(f"   Rebalances:         {len(self.portfolio_history)}")

        print(f"\nFinal Holdings:")
        final_holdings = self.portfolio_history[-1]['holdings']
        for symbol, quantity in final_holdings.items():
            if quantity > 0:
                print(f"   {symbol}: {quantity:.2f} shares")
        print(f"   CASH: ${self.portfolio_history[-1]['cash']:,.2f}")


def main():
    print("="*60)
    print("ATLAS Stock ML Strategy - Local Backtest")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    backtest = LocalBacktest(
        initial_capital=25000,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY', 'JPM']
    )

    # Run backtest on last 90 trading days with weekly rebalancing
    backtest.run_backtest(days=90, rebalance_frequency=5)

    print(f"\n{'='*60}")
    print("Backtest Complete!")
    print(f"{'='*60}")
    print("\nNote: This is a simplified backtest using recent Yahoo Finance data.")
    print("   For ML-enhanced results, train LSTM models first with ./bin/train_now.sh")


if __name__ == "__main__":
    main()
