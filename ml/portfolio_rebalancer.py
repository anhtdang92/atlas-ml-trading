"""
Portfolio Rebalancer for Stock ML Trading Dashboard

Handles portfolio rebalancing logic for stock investments:
- Calculates target allocations based on ML predictions
- Implements risk controls and position limits
- Generates buy/sell orders
- Handles paper trading vs live trading modes

Strategy: Equal-weight base + ML-enhanced weighting across sectors
Risk: Max 15% per position, Min 2%, $100 minimum trade
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from .prediction_service import PredictionService, DEFAULT_SYMBOLS
from data.stock_api import StockAPI, get_stock_info

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Portfolio rebalancing service with ML-enhanced allocation for stocks."""

    SUPPORTED_SYMBOLS = DEFAULT_SYMBOLS
    BASE_ALLOCATION = 1.0 / len(SUPPORTED_SYMBOLS)

    # Risk controls
    MAX_POSITION_WEIGHT = 0.15   # 15% max per position
    MIN_POSITION_WEIGHT = 0.02   # 2% min per position
    MIN_TRADE_SIZE = 100.0       # Min $100 trade
    TRADING_FEE = 0.0            # Most brokers are zero-commission now

    # ML enhancement
    ML_WEIGHT_FACTOR = 0.3
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, paper_trading: bool = True, config_file: str = "config/rebalancing_config.json"):
        self.paper_trading = paper_trading
        self.config_file = config_file
        self.prediction_service = PredictionService(provider="local")
        self.stock_api = StockAPI()

        self.current_portfolio = {}
        self.target_portfolio = {}
        self.rebalancing_orders = []

        self.config = self._load_config()
        logger.info(f"Portfolio Rebalancer initialized (Paper Trading: {paper_trading})")

    def _load_config(self) -> Dict:
        default_config = {
            "max_position_weight": self.MAX_POSITION_WEIGHT,
            "min_position_weight": self.MIN_POSITION_WEIGHT,
            "min_trade_size": self.MIN_TRADE_SIZE,
            "trading_fee": self.TRADING_FEE,
            "ml_weight_factor": self.ML_WEIGHT_FACTOR,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "rebalancing_schedule": "monthly",
            "last_rebalancing": None
        }

        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            else:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")

        return default_config

    def save_config(self) -> bool:
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Could not save config: {e}")
            return False

    def get_current_portfolio(self) -> Dict[str, float]:
        """Get current portfolio allocation (mock for now)."""
        n = len(self.SUPPORTED_SYMBOLS)
        mock_portfolio = {}
        for i, symbol in enumerate(self.SUPPORTED_SYMBOLS):
            # Simulate slightly uneven allocation
            mock_portfolio[symbol] = self.BASE_ALLOCATION + np.random.uniform(-0.03, 0.03)

        # Normalize to sum to 1
        total = sum(mock_portfolio.values())
        mock_portfolio = {k: v/total for k, v in mock_portfolio.items()}

        self.current_portfolio = mock_portfolio
        return self.current_portfolio

    def get_target_allocation(self, use_ml: bool = True) -> Dict[str, float]:
        if not use_ml:
            target = {s: self.BASE_ALLOCATION for s in self.SUPPORTED_SYMBOLS}
        else:
            target = self._calculate_ml_enhanced_allocation()

        target = self._apply_risk_controls(target)
        self.target_portfolio = target
        return self.target_portfolio

    def _calculate_ml_enhanced_allocation(self) -> Dict[str, float]:
        if hasattr(self.prediction_service, 'get_all_predictions'):
            import inspect
            sig = inspect.signature(self.prediction_service.get_all_predictions)
            if 'symbols' in sig.parameters:
                predictions_dict = self.prediction_service.get_all_predictions(
                    symbols=self.SUPPORTED_SYMBOLS, days_ahead=21
                )
                predictions = [predictions_dict[s] for s in self.SUPPORTED_SYMBOLS if s in predictions_dict]
            else:
                predictions = self.prediction_service.get_all_predictions(days_ahead=21)
        else:
            predictions = []

        base_weights = {s: self.BASE_ALLOCATION for s in self.SUPPORTED_SYMBOLS}

        ml_adjustments = {}
        for pred in predictions:
            symbol = pred['symbol']
            predicted_return = pred['predicted_return']
            confidence = pred['confidence']

            if confidence >= self.CONFIDENCE_THRESHOLD:
                adjustment = predicted_return * confidence * self.ML_WEIGHT_FACTOR
                ml_adjustments[symbol] = adjustment
            else:
                ml_adjustments[symbol] = 0.0

        for symbol in self.SUPPORTED_SYMBOLS:
            base_weights[symbol] += ml_adjustments.get(symbol, 0.0)

        total = sum(base_weights.values())
        if total > 0:
            base_weights = {k: v/total for k, v in base_weights.items()}

        return base_weights

    def _apply_risk_controls(self, allocation: Dict[str, float]) -> Dict[str, float]:
        controlled = {}
        for symbol, weight in allocation.items():
            weight = max(self.MIN_POSITION_WEIGHT, min(weight, self.MAX_POSITION_WEIGHT))
            controlled[symbol] = weight

        total = sum(controlled.values())
        if total > 0:
            controlled = {k: v/total for k, v in controlled.items()}
        return controlled

    def calculate_rebalancing_orders(self, portfolio_value: float = 25000.0) -> List[Dict]:
        current = self.get_current_portfolio()
        target = self.get_target_allocation(use_ml=True)

        orders = []
        for symbol in self.SUPPORTED_SYMBOLS:
            current_weight = current.get(symbol, 0.0)
            target_weight = target.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            dollar_diff = weight_diff * portfolio_value

            if abs(dollar_diff) >= self.MIN_TRADE_SIZE:
                order_type = "BUY" if dollar_diff > 0 else "SELL"
                order_size = abs(dollar_diff)
                fee = order_size * self.TRADING_FEE

                orders.append({
                    'symbol': symbol,
                    'type': order_type,
                    'amount_usd': order_size,
                    'net_amount_usd': order_size - fee,
                    'fee_usd': fee,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'weight_change': weight_diff,
                    'timestamp': datetime.now().isoformat()
                })

        self.rebalancing_orders = orders
        return orders

    def get_rebalancing_summary(self) -> Dict:
        current = self.get_current_portfolio()
        target = self.get_target_allocation(use_ml=True)
        orders = self.calculate_rebalancing_orders()

        total_fees = sum(o['fee_usd'] for o in orders)
        buy_orders = [o for o in orders if o['type'] == 'BUY']
        sell_orders = [o for o in orders if o['type'] == 'SELL']

        drift_metrics = {}
        for symbol in self.SUPPORTED_SYMBOLS:
            c = current.get(symbol, 0.0)
            t = target.get(symbol, 0.0)
            drift_metrics[symbol] = {
                'current': c, 'target': t,
                'drift': abs(t - c),
                'status': 'OVERWEIGHT' if c > t else 'UNDERWEIGHT'
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            'current_allocation': current,
            'target_allocation': target,
            'orders': orders,
            'metrics': {
                'total_trades': len(orders),
                'total_fees': total_fees,
                'buy_orders': len(buy_orders),
                'sell_orders': len(sell_orders),
                'max_drift': max(d['drift'] for d in drift_metrics.values()),
                'avg_drift': np.mean([d['drift'] for d in drift_metrics.values()])
            },
            'drift_analysis': drift_metrics,
            'recommendations': self._generate_recommendations(drift_metrics, orders)
        }

    def _generate_recommendations(self, drift_analysis: Dict, orders: List[Dict]) -> List[str]:
        recommendations = []

        max_drift = max(d['drift'] for d in drift_analysis.values())
        if max_drift > 0.03:
            recommendations.append("Significant allocation drift detected - rebalancing recommended")

        small_trades = [o for o in orders if o['amount_usd'] < 200]
        if len(small_trades) > 5:
            recommendations.append("Many small trades - consider consolidating")

        if not self.paper_trading:
            recommendations.append("LIVE TRADING MODE - review all orders before execution")

        if not recommendations:
            recommendations.append("Portfolio is well-balanced - no immediate rebalancing needed")

        return recommendations

    def execute_rebalancing(self, portfolio_value: float = 25000.0) -> Dict:
        if self.paper_trading:
            return self._execute_paper_trading(portfolio_value)
        else:
            return self._execute_live_trading(portfolio_value)

    def _execute_paper_trading(self, portfolio_value: float) -> Dict:
        orders = self.calculate_rebalancing_orders(portfolio_value)
        executed = []
        for order in orders:
            executed_order = order.copy()
            executed_order['status'] = 'EXECUTED'
            executed_order['execution_time'] = datetime.now().isoformat()
            executed_order['paper_trading'] = True
            executed.append(executed_order)

        return {
            'status': 'success',
            'mode': 'paper_trading',
            'orders_executed': len(executed),
            'total_fees': sum(o['fee_usd'] for o in executed),
            'execution_time': datetime.now().isoformat(),
            'orders': executed
        }

    def _execute_live_trading(self, portfolio_value: float) -> Dict:
        return {
            'status': 'error',
            'message': 'Live trading not implemented yet - use paper trading mode',
            'mode': 'live_trading'
        }
