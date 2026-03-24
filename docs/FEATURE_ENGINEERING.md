# Feature Engineering: Technical Indicators & Financial Rationale

This document explains each of the 25 technical indicators used in the LSTM model, why they were chosen, and their expected predictive value for 21-day position trading.

## Feature Summary

| # | Feature | Category | Financial Rationale | Expected Signal |
|---|---------|----------|--------------------|--------------------|
| 1 | `close` | Price | Raw price level, captures regime and scale | Base reference for all other features |
| 2 | `volume` | Volume | Trading activity level; high volume confirms price moves | Volume precedes price; breakouts on high volume are more reliable |
| 3 | `MA_10` | Trend | 2-week moving average; captures short-term trend | Fast-moving trend indicator; crossovers with slower MAs signal direction changes |
| 4 | `MA_20` | Trend | 1-month moving average; intermediate trend | Standard institutional trading reference period |
| 5 | `MA_50` | Trend | ~2.5 month average; medium-term trend | Key support/resistance level watched by traders |
| 6 | `MA_200` | Trend | ~10 month average; long-term trend | Institutional trend indicator; price above MA200 = bullish regime |
| 7 | `Price_to_MA50` | Trend | Price relative to 50-day MA | >1.0 = above trend (bullish), <1.0 = below trend (bearish); mean-reverting |
| 8 | `Price_to_MA200` | Trend | Price relative to 200-day MA | Regime classifier: above = bull market, below = bear market |
| 9 | `MA_50_200_Cross` | Trend | Golden Cross / Death Cross ratio | MA50/MA200 > 1.0 = Golden Cross (bullish), < 1.0 = Death Cross (bearish) |
| 10 | `RSI` | Momentum | 14-day Relative Strength Index | <30 = oversold (buy signal), >70 = overbought (sell signal); mean-reverting for position trading |
| 11 | `MACD` | Momentum | Moving Average Convergence/Divergence (12/26) | Measures trend strength and direction; positive = bullish momentum |
| 12 | `MACD_Signal` | Momentum | 9-day EMA of MACD | MACD crossing above signal = bullish entry; below = bearish |
| 13 | `MACD_Histogram` | Momentum | MACD minus Signal line | Rate of change of momentum; shrinking histogram = momentum weakening |
| 14 | `BB_Width` | Volatility | Bollinger Band width (upper-lower)/middle | Narrow bands = low volatility (breakout imminent); wide = high volatility |
| 15 | `BB_Position` | Volatility | Price position within Bollinger Bands (0-1) | Near 0 = near lower band (potential reversal up); near 1 = near upper band |
| 16 | `Volume_MA_20` | Volume | 20-day average volume | Baseline volume level; deviations from average signal unusual activity |
| 17 | `Volume_ROC` | Volume | 10-day volume rate of change | Sudden volume increases often precede large price moves |
| 18 | `Volume_Ratio` | Volume | Current volume / 20-day average | >2.0 = volume spike (confirms breakout); <0.5 = low conviction |
| 19 | `Daily_Return` | Returns | 1-day percentage change | Short-term price momentum; captures immediate market reaction |
| 20 | `Momentum_14` | Momentum | 14-day price change % | 2-week momentum; positive = uptrend, negative = downtrend |
| 21 | `Momentum_30` | Momentum | 30-day price change % | 1-month momentum; aligns with prediction horizon (21 days) |
| 22 | `ROC_10` | Momentum | 10-day Rate of Change | Price velocity over 2 weeks; extreme values suggest overextension |
| 23 | `Volatility_14` | Risk | 14-day rolling standard deviation of returns | Short-term risk measure; high volatility = higher uncertainty in predictions |
| 24 | `Volatility_30` | Risk | 30-day rolling standard deviation of returns | Medium-term risk; used for position sizing and confidence scoring |
| 25 | `ATR_14` | Risk | 14-day Average True Range | Range-based volatility; accounts for gaps; used in stop-loss placement |

## Feature Categories

### Trend Features (7 features)
Moving averages at multiple timeframes (10, 20, 50, 200 days) plus relative price positions. These capture the directional bias at different scales. The MA50/MA200 cross (Golden Cross/Death Cross) is one of the most widely followed institutional signals.

**Why multiple timeframes?** Different market participants operate on different horizons. Short-term traders watch MA10/MA20, swing traders watch MA50, and institutions watch MA200. Price behavior relative to each MA carries different information.

### Momentum Features (6 features)
RSI, MACD (with signal and histogram), and momentum at 14/30-day periods. These measure the speed and strength of price changes.

**Why RSI for position trading?** RSI extremes (<30 oversold, >70 overbought) are mean-reverting on the 21-day horizon. Academic research shows RSI has predictive power for 1-4 week returns, making it well-suited for our prediction horizon.

### Volatility/Risk Features (5 features)
Bollinger Band metrics, rolling volatility at two periods, and ATR. These capture market uncertainty and risk regime.

**Why include volatility?** Volatility is a strong predictor of future returns through the leverage effect (declining prices increase volatility) and the volatility risk premium. Including both return-based (std dev) and range-based (ATR) volatility captures different aspects of market risk.

### Volume Features (3 features)
Volume moving average, rate of change, and ratio to average. Volume confirms price moves and signals conviction.

**Why volume matters:** Price moves on high volume are more likely to persist than moves on low volume. Volume spikes often precede major price moves by 1-3 days, providing a leading signal for our 21-day horizon.

### Return Features (4 features)
Daily return, close price, and two momentum periods. These provide the raw price dynamics the model builds upon.

## Normalization

All features are normalized to [0, 1] range using min-max scaling per symbol:

```
normalized = (value - min) / (max - min)
```

**Why per-symbol normalization?** Different stocks have vastly different price ranges (NVDA ~$800 vs COIN ~$200). Normalizing per-symbol ensures the LSTM treats each stock's features on an equal scale.

**Why [0, 1] not z-score?** LSTM cells use sigmoid and tanh activations internally. Input values in [0, 1] avoid saturation of these activation functions, leading to more stable gradients during training.

## Sequence Construction

```
Input:  30 trading days × 25 features = 750 values per sample
Output: 21-day forward return (single scalar)
```

**Why 30-day lookback?** This captures approximately 6 weeks of trading history, enough to include one full business cycle of technical patterns (support/resistance tests, moving average crossovers) while keeping the sequence short enough for efficient LSTM training.

**Why 21-day prediction horizon?** 21 trading days ≈ 1 calendar month. This is the sweet spot for position trading: long enough for fundamental trends to play out, short enough for technical indicators to have predictive power. Academic literature shows diminishing returns beyond 1-month horizons for technical analysis.
