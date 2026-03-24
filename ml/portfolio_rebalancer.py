"""
Portfolio Rebalancer for Stock ML Trading Dashboard

Handles portfolio rebalancing logic for stock investments:
- Calculates target allocations based on ML predictions
- Implements risk controls: position limits, sector limits, speculative caps
- Position-level stop-loss (8%) and portfolio drawdown circuit breaker (12%)
- Cash reserve (5%) always held back from allocation
- Generates buy/sell orders
- Handles paper trading vs live trading modes

Strategy: Equal-weight base + ML-enhanced weighting across sectors
Risk v2.0: Max 10% per position, sector limits, speculative caps, stop-loss, drawdown halt
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from .prediction_service import PredictionService, DEFAULT_SYMBOLS
from data.stock_api import StockAPI, get_stock_info, SECTOR_MAP, SPECULATIVE_CAPS

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Portfolio rebalancing service with ML-enhanced allocation for stocks."""

    SUPPORTED_SYMBOLS = DEFAULT_SYMBOLS
    BASE_ALLOCATION = 1.0 / len(SUPPORTED_SYMBOLS)

    # Risk controls (v2.0 — tightened)
    MAX_POSITION_WEIGHT = 0.10   # 10% max per position (was 15%)
    MIN_POSITION_WEIGHT = 0.02   # 2% min per position
    MIN_TRADE_SIZE = 100.0       # Min $100 trade
    TRADING_FEE = 0.0            # Most brokers are zero-commission now
    CASH_RESERVE_PCT = 0.05      # Hold 5% cash at all times

    # ML enhancement (v2.0 — reduced until model is validated)
    ML_WEIGHT_FACTOR = 0.05      # Was 0.3 — reduced to 5% until LSTM is proven
    CONFIDENCE_THRESHOLD = 0.6

    # Stop-loss and drawdown
    STOP_LOSS_PCT = -0.08        # Sell if position drops 8% from purchase
    MAX_PORTFOLIO_DRAWDOWN = -0.12  # Halt trading if portfolio drops 12% from peak

    # Sector exposure limits (prevent overconcentration)
    SECTOR_LIMITS = {
        'Technology': 0.25,
        'Financials': 0.15,
        'Healthcare': 0.15,
        'Energy': 0.12,
        'Consumer Staples': 0.15,
        'Consumer Disc.': 0.10,
        'Industrials': 0.10,
        'Utilities': 0.08,
        'Real Estate': 0.08,
        'Materials': 0.08,
        'Cybersecurity': 0.08,
        'Cloud Data': 0.05,
        'Fintech': 0.08,
        'Bond ETF': 0.15,
        'Commodity ETF': 0.10,
        'Small Cap ETF': 0.12,
        'Sector ETF': 0.20,
        'Index ETF': 0.10,
        'Growth ETF': 0.05,
    }

    def __init__(self, paper_trading: bool = True, config_file: str = "config/rebalancing_config.json"):
        self.paper_trading = paper_trading
        self.config_file = config_file
        self.prediction_service = PredictionService(provider="local")
        self.stock_api = StockAPI()

        self.current_portfolio = {}
        self.target_portfolio = {}
        self.rebalancing_orders = []

        # Drawdown tracking
        self.peak_portfolio_value = 0.0
        self.trading_halted = False
        self.halt_reason = None

        # Position tracking for stop-loss
        self.purchase_prices = {}  # symbol -> avg purchase price

        self.config = self._load_config()
        self._apply_config_overrides()
        logger.info(f"Portfolio Rebalancer v2.0 initialized (Paper Trading: {paper_trading})")

    def _load_config(self) -> Dict:
        default_config = {
            "max_position_weight": self.MAX_POSITION_WEIGHT,
            "min_position_weight": self.MIN_POSITION_WEIGHT,
            "min_trade_size": self.MIN_TRADE_SIZE,
            "trading_fee": self.TRADING_FEE,
            "ml_weight_factor": self.ML_WEIGHT_FACTOR,
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "rebalancing_schedule": "monthly",
            "last_rebalancing": None,
            "cash_reserve_pct": self.CASH_RESERVE_PCT,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "max_portfolio_drawdown": self.MAX_PORTFOLIO_DRAWDOWN,
            "sector_limits": self.SECTOR_LIMITS,
            "speculative_caps": dict(SPECULATIVE_CAPS),
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

    def _apply_config_overrides(self):
        """Apply loaded config values to instance attributes."""
        self.MAX_POSITION_WEIGHT = self.config.get("max_position_weight", self.MAX_POSITION_WEIGHT)
        self.MIN_POSITION_WEIGHT = self.config.get("min_position_weight", self.MIN_POSITION_WEIGHT)
        self.ML_WEIGHT_FACTOR = self.config.get("ml_weight_factor", self.ML_WEIGHT_FACTOR)
        self.CONFIDENCE_THRESHOLD = self.config.get("confidence_threshold", self.CONFIDENCE_THRESHOLD)
        self.CASH_RESERVE_PCT = self.config.get("cash_reserve_pct", self.CASH_RESERVE_PCT)
        self.STOP_LOSS_PCT = self.config.get("stop_loss_pct", self.STOP_LOSS_PCT)
        self.MAX_PORTFOLIO_DRAWDOWN = self.config.get("max_portfolio_drawdown", self.MAX_PORTFOLIO_DRAWDOWN)

        loaded_sectors = self.config.get("sector_limits", {})
        if loaded_sectors:
            self.SECTOR_LIMITS.update(loaded_sectors)

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

    # ------------------------------------------------------------------
    # Drawdown circuit breaker
    # ------------------------------------------------------------------

    def check_drawdown(self, current_portfolio_value: float) -> Dict:
        """Check portfolio-level drawdown against the circuit breaker.

        Returns status dict with 'halted' flag and reason.
        """
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value

        if self.peak_portfolio_value <= 0:
            return {'halted': False, 'drawdown': 0.0}

        drawdown = (current_portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value

        if drawdown <= self.MAX_PORTFOLIO_DRAWDOWN:
            self.trading_halted = True
            self.halt_reason = (
                f"Portfolio drawdown {drawdown:.1%} exceeds limit {self.MAX_PORTFOLIO_DRAWDOWN:.1%}. "
                f"Peak: ${self.peak_portfolio_value:,.2f}, Current: ${current_portfolio_value:,.2f}. "
                f"All trading halted — manual review required."
            )
            logger.critical(self.halt_reason)
            return {'halted': True, 'drawdown': drawdown, 'reason': self.halt_reason}

        return {'halted': False, 'drawdown': drawdown}

    def reset_drawdown_halt(self):
        """Manually reset the drawdown halt after review."""
        self.trading_halted = False
        self.halt_reason = None
        logger.info("Drawdown halt manually reset")

    # ------------------------------------------------------------------
    # Position-level stop-loss
    # ------------------------------------------------------------------

    def evaluate_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check all positions for stop-loss triggers.

        Returns list of stop-loss sell orders for positions that breached the threshold.
        """
        stop_loss_orders = []
        for symbol, purchase_price in self.purchase_prices.items():
            if symbol not in current_prices or purchase_price <= 0:
                continue
            current_price = current_prices[symbol]
            loss_pct = (current_price - purchase_price) / purchase_price

            if loss_pct <= self.STOP_LOSS_PCT:
                stop_loss_orders.append({
                    'symbol': symbol,
                    'type': 'SELL',
                    'reason': 'STOP_LOSS',
                    'purchase_price': purchase_price,
                    'current_price': current_price,
                    'loss_pct': loss_pct,
                    'timestamp': datetime.now().isoformat(),
                })
                logger.warning(
                    f"STOP-LOSS triggered for {symbol}: "
                    f"bought ${purchase_price:.2f}, now ${current_price:.2f} ({loss_pct:.1%})"
                )

        return stop_loss_orders

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

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
        target = self._apply_sector_limits(target)
        target = self._apply_speculative_caps(target)
        target = self._apply_cash_reserve(target)
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
            prediction_source = pred.get('prediction_source', '')

            # Only apply ML adjustments from real ML models, never from mocks
            if prediction_source in ('local_ml', 'vertex_ai_ml') and confidence >= self.CONFIDENCE_THRESHOLD:
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
        """Apply per-position min/max weight limits."""
        controlled = {}
        for symbol, weight in allocation.items():
            weight = max(self.MIN_POSITION_WEIGHT, min(weight, self.MAX_POSITION_WEIGHT))
            controlled[symbol] = weight

        total = sum(controlled.values())
        if total > 0:
            controlled = {k: v/total for k, v in controlled.items()}
        return controlled

    def _apply_sector_limits(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Enforce sector-level concentration limits.

        If a sector's aggregate weight exceeds its limit, proportionally scale
        down all positions in that sector and redistribute to other sectors.
        """
        sector_weights = {}
        for symbol, weight in allocation.items():
            sector = SECTOR_MAP.get(symbol, 'Unknown')
            sector_weights.setdefault(sector, []).append((symbol, weight))

        adjusted = dict(allocation)
        excess_total = 0.0

        for sector, holdings in sector_weights.items():
            limit = self.SECTOR_LIMITS.get(sector, 0.20)  # default 20%
            sector_total = sum(w for _, w in holdings)

            if sector_total > limit:
                scale_factor = limit / sector_total
                for symbol, weight in holdings:
                    old = adjusted[symbol]
                    adjusted[symbol] = weight * scale_factor
                    excess_total += old - adjusted[symbol]

        # Redistribute excess proportionally to under-limit sectors
        if excess_total > 0:
            under_limit_symbols = []
            for symbol, weight in adjusted.items():
                sector = SECTOR_MAP.get(symbol, 'Unknown')
                limit = self.SECTOR_LIMITS.get(sector, 0.20)
                sector_total = sum(adjusted[s] for s, _ in sector_weights.get(sector, []))
                if sector_total < limit:
                    under_limit_symbols.append(symbol)

            if under_limit_symbols:
                per_symbol_add = excess_total / len(under_limit_symbols)
                for symbol in under_limit_symbols:
                    adjusted[symbol] += per_symbol_add

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}

        return adjusted

    def _apply_speculative_caps(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Enforce hard caps on speculative/high-risk symbols."""
        adjusted = dict(allocation)
        excess = 0.0

        for symbol, cap in SPECULATIVE_CAPS.items():
            if symbol in adjusted and adjusted[symbol] > cap:
                excess += adjusted[symbol] - cap
                adjusted[symbol] = cap

        # Redistribute excess to non-speculative symbols
        if excess > 0:
            non_spec = [s for s in adjusted if s not in SPECULATIVE_CAPS]
            if non_spec:
                per_symbol_add = excess / len(non_spec)
                for symbol in non_spec:
                    adjusted[symbol] += per_symbol_add

        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}

        return adjusted

    def _apply_cash_reserve(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Scale all allocations down to reserve cash.

        With 5% cash reserve, all weights are multiplied by 0.95 so that
        5% of portfolio value remains uninvested.
        """
        invest_pct = 1.0 - self.CASH_RESERVE_PCT
        return {k: v * invest_pct for k, v in allocation.items()}

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    def calculate_rebalancing_orders(self, portfolio_value: float = 25000.0) -> List[Dict]:
        # Check drawdown before generating any orders
        drawdown_status = self.check_drawdown(portfolio_value)
        if drawdown_status['halted']:
            logger.critical("Trading halted — no orders generated")
            return []

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

        # Sector exposure summary
        sector_exposure = {}
        for symbol, weight in target.items():
            sector = SECTOR_MAP.get(symbol, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        return {
            'timestamp': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'cash_reserve_pct': self.CASH_RESERVE_PCT,
            'current_allocation': current,
            'target_allocation': target,
            'sector_exposure': sector_exposure,
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
            'risk_controls': {
                'max_position_weight': self.MAX_POSITION_WEIGHT,
                'ml_weight_factor': self.ML_WEIGHT_FACTOR,
                'stop_loss_pct': self.STOP_LOSS_PCT,
                'max_drawdown': self.MAX_PORTFOLIO_DRAWDOWN,
                'sector_limits': self.SECTOR_LIMITS,
                'speculative_caps': dict(SPECULATIVE_CAPS),
            },
            'recommendations': self._generate_recommendations(drift_metrics, orders, sector_exposure)
        }

    def _generate_recommendations(self, drift_analysis: Dict, orders: List[Dict],
                                  sector_exposure: Dict = None) -> List[str]:
        recommendations = []

        if self.trading_halted:
            recommendations.append(
                f"TRADING HALTED: {self.halt_reason}. Manual review required before resuming."
            )
            return recommendations

        max_drift = max(d['drift'] for d in drift_analysis.values())
        if max_drift > 0.03:
            recommendations.append("Significant allocation drift detected - rebalancing recommended")

        small_trades = [o for o in orders if o['amount_usd'] < 200]
        if len(small_trades) > 5:
            recommendations.append("Many small trades - consider consolidating")

        # Warn about sector overweight
        if sector_exposure:
            for sector, weight in sector_exposure.items():
                limit = self.SECTOR_LIMITS.get(sector, 0.20)
                if weight > limit * 0.9:  # Warn at 90% of limit
                    recommendations.append(
                        f"Sector '{sector}' near limit: {weight:.1%} (limit {limit:.1%})"
                    )

        if not self.paper_trading:
            recommendations.append("LIVE TRADING MODE - review all orders before execution")

        if not recommendations:
            recommendations.append("Portfolio is well-balanced - no immediate rebalancing needed")

        return recommendations

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_rebalancing(self, portfolio_value: float = 25000.0) -> Dict:
        if self.trading_halted:
            return {
                'status': 'halted',
                'message': self.halt_reason,
                'mode': 'paper_trading' if self.paper_trading else 'live_trading'
            }

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
            'cash_reserve_pct': self.CASH_RESERVE_PCT,
            'execution_time': datetime.now().isoformat(),
            'orders': executed
        }

    def _execute_live_trading(self, portfolio_value: float) -> Dict:
        return {
            'status': 'error',
            'message': 'Live trading not implemented yet - use paper trading mode',
            'mode': 'live_trading'
        }
