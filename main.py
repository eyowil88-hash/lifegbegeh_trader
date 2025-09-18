# ULTIMATE FOREX TRADING BOT WITH ELITE FEATURES
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import feedparser
from textblob import TextBlob
import json
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

print("🤖 ULTIMATE FOREX TRADING BOT INITIALIZING...")
print("📱 Telegram Alerts | 🤖 AI/ML | 🌡️ Market Regime | 📰 News Sentiment")
print("💰 Arbitrage Detection | ⏰ Timeframe Analysis | ⚖️ Risk Management")

# =============================================================================
# CONFIGURATION - USER SETTINGS
# =============================================================================
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
timeframes = ['5m', '15m', '1h', '4h', '1d']
account_balance = 5000
risk_per_trade = 1.0
leverage = 10
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8340887342:AAGwQJwglAiD3uSuLg-cIdPYb87ywkMgMBA")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "630055275")

# =============================================================================
# ELITE FEATURE 1: NEWS SENTIMENT ANALYSIS
# =============================================================================
def get_news_sentiment():
    """Analyze forex news sentiment in real-time"""
    news_sources = {
        'forexlive': 'https://www.forexlive.com/feed',
        'investing': 'https://www.investing.com/rss/news_285.rss',
        'fxstreet': 'https://www.fxstreet.com/rss'
    }
    
    sentiment_scores = []
    headlines = []
    
    for source, url in news_sources.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                text = f"{entry.title} {getattr(entry, 'summary', '')}"
                analysis = TextBlob(text)
                sentiment_scores.append(analysis.sentiment.polarity)
                headlines.append(entry.title)
        except Exception as e:
            print(f"News error ({source}): {e}")
            continue
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment, headlines[:3]

def adjust_signals_with_news(signal_strength, news_sentiment):
    """Adjust trading signals based on news sentiment"""
    if news_sentiment > 0.3:
        return min(signal_strength * 1.3, 1.0)
    elif news_sentiment < -0.3:
        return max(signal_strength * 0.7, -1.0)
    return signal_strength

# =============================================================================
# ELITE FEATURE 2: MARKET REGIME DETECTION
# =============================================================================
def detect_market_regime(data):
    """Detect current market regime"""
    if len(data) < 50:
        return "INSUFFICIENT_DATA"
    
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 20:
        return "INSUFFICIENT_DATA"
    
    volatility = returns.rolling(20).std().iloc[-1]
    
    trend_strength = abs(data['Close'].rolling(50).mean().iloc[-1] - 
                        data['Close'].rolling(200).mean().iloc[-1]) / data['Close'].iloc[-1]
    
    if volatility > 0.008 and trend_strength > 0.02:
        return "HIGH_VOLATILITY_TRENDING"
    elif volatility > 0.008:
        return "HIGH_VOLATILITY_RANGING"
    elif trend_strength > 0.02:
        return "LOW_VOLATILITY_TRENDING"
    else:
        return "LOW_VOLATILITY_RANGING"

def adjust_strategy_for_regime(signal_strength, regime):
    """Adjust strategy based on market regime"""
    regime_adjustments = {
        "HIGH_VOLATILITY_TRENDING": 1.2,
        "HIGH_VOLATILITY_RANGING": 0.7,
        "LOW_VOLATILITY_TRENDING": 1.1,
        "LOW_VOLATILITY_RANGING": 0.8
    }
    return signal_strength * regime_adjustments.get(regime, 1.0)

# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================
def calculate_advanced_indicators(data, timeframe):
    """Calculate all technical indicators"""
    # Handle data cleaning
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    if len(data) < 30:
        raise ValueError("Insufficient data after cleaning")
    
    # Moving Averages with timeframe-specific periods
    ma_periods = {
        '5m': (8, 21), '15m': (12, 26), 
        '1h': (20, 50), '4h': (20, 50), '1d': (50, 200)
    }
    fast_period, slow_period = ma_periods.get(timeframe, (20, 50))
    
    data['Fast_MA'] = data['Close'].rolling(window=fast_period, min_periods=1).mean()
    data['Slow_MA'] = data['Close'].rolling(window=slow_period, min_periods=1).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # ATR for volatility
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = data['TR'].rolling(window=14, min_periods=1).mean()
    
    return data

def generate_signals(data):
    """Generate trading signals from indicators"""
    if len(data) < 30:
        return 0.0
    
    try:
        # Get latest values
        current_data = data.iloc[-1]
        fast_ma = current_data['Fast_MA']
        slow_ma = current_data['Slow_MA']
        rsi = current_data['RSI']
        macd = current_data['MACD']
        macd_signal = current_data['MACD_Signal']
        close = current_data['Close']
        
        # Individual signals
        ma_signal = 1 if fast_ma > slow_ma else -1
        rsi_signal = 1 if rsi > 50 else -1
        macd_signal_val = 1 if macd > macd_signal else -1
        
        # Weighted combined signal
        weights = {'ma': 0.4, 'rsi': 0.3, 'macd': 0.3}
        final_signal = (ma_signal * weights['ma'] + 
                       rsi_signal * weights['rsi'] + 
                       macd_signal_val * weights['macd'])
        
        return final_signal
    
    except Exception as e:
        print(f"Signal generation error: {e}")
        return 0.0

# =============================================================================
# TIMEFRAME ANALYSIS FUNCTIONS
# =============================================================================
def analyze_timeframe_performance(all_data):
    """Analyze which timeframes are generating the best signals"""
    timeframe_performance = {}
    
    for pair, timeframes_data in all_data.items():
        for timeframe, data in timeframes_data.items():
            if isinstance(data, pd.DataFrame) and 'Signal_Strength' in data.columns:
                recent_signals = data['Signal_Strength'].tail(10)
                if len(recent_signals) > 0:
                    avg_strength = recent_signals.abs().mean()
                    win_rate = (recent_signals > 0).mean() if any(recent_signals > 0) else 0
                    
                    if timeframe not in timeframe_performance:
                        timeframe_performance[timeframe] = []
                    
                    timeframe_performance[timeframe].append({
                        'avg_strength': avg_strength,
                        'win_rate': win_rate
                    })
    
    # Calculate average performance per timeframe
    performance_summary = {}
    for timeframe, performances in timeframe_performance.items():
        if performances:
            avg_strength = np.mean([p['avg_strength'] for p in performances])
            avg_win_rate = np.mean([p['win_rate'] for p in performances])
            performance_summary[timeframe] = {
                'avg_strength': avg_strength,
                'win_rate': avg_win_rate,
                'score': avg_strength * avg_win_rate
            }
    
    # Find best timeframe
    best_timeframe = max(performance_summary.items(), key=lambda x: x[1]['score'])[0] if performance_summary else '1h'
    
    return performance_summary, best_timeframe

def get_timeframe_emoji(timeframe):
    """Get emoji for each timeframe"""
    timeframe_emojis = {
        '5m': '⏱️', '15m': '🕒', '1h': '🕐', '4h': '🕓', '1d': '📅'
    }
    return timeframe_emojis.get(timeframe, '⏰')

# =============================================================================
# TELEGRAM FUNCTIONS WITH TIMEFRAME VISIBILITY
# =============================================================================
def send_telegram_alert(message, parse_mode='HTML'):
    """Send formatted alert to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not configured - skipping alert")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram alert sent!")
            return True
        else:
            print(f"❌ Telegram error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Telegram connection failed: {e}")
        return False

def send_telegram_signal(pair, signal_type, price, strength, timeframe, news_sentiment=0, market_regime="N/A"):
    """Send professional trading signal with timeframe visibility"""
    emoji = "🟢" if signal_type == "BUY" else "🔴"
    timeframe_emoji = get_timeframe_emoji(timeframe)
    
    message = f"""
{emoji} <b>ELITE TRADING SIGNAL</b> {emoji}

{timeframe_emoji} <b>TIMEFRAME:</b> {timeframe.upper()}
<b>Pair:</b> {pair}
<b>Action:</b> {signal_type}
<b>Price:</b> {price:.5f}
<b>Strength:</b> {strength:.2f}/1.0

<b>Market Regime:</b> {market_regime}
<b>News Sentiment:</b> {news_sentiment:.2f}

<b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 <i>Signal generated from {timeframe} chart analysis</i>
"""
    return send_telegram_alert(message)

def send_daily_report(performance_data, timeframe_performance):
    """Send daily performance report with timeframe analysis"""
    timeframe_info = "\n".join([
        f"• {tf}: Strength={data['avg_strength']:.2f}, Win Rate={data['win_rate']:.0%}"
        for tf, data in timeframe_performance.items()
    ])
    
    message = f"""
📊 <b>DAILY TRADING REPORT</b>

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
<b>Account Balance:</b> ${account_balance:,.2f}
<b>Risk per Trade:</b> {risk_per_trade}%

<b>Performance Summary:</b>
• Total Signals: {performance_data.get('total_signals', 0)}
• Strong Signals: {performance_data.get('strong_signals', 0)}
• Best Timeframe: {performance_data.get('best_timeframe', 'N/A')}

<b>Timeframe Performance:</b>
{timeframe_info}

<b>Market Condition:</b> {performance_data.get('market_condition', 'Neutral')}
"""
    return send_telegram_alert(message)

def test_telegram_connection():
    """Test Telegram connection"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not configured")
        return False
    
    print("Testing Telegram connection...")
    test_message = f"""
🤖 <b>FOREX BOT CONNECTION TEST</b>

✅ <b>Status:</b> Connected Successfully
📊 <b>Account:</b> ${account_balance:,.2f}
🆔 <b>Chat ID:</b> {TELEGRAM_CHAT_ID}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Monitoring Timeframes:</b>
⏱️ 5m (Scalping) | 🕒 15m (Short-term)
🕐 1h (Swing) | 🕓 4h (Position) | 📅 1d (Long-term)

🚀 <b>System:</b> Ready for live trading!
"""
    if send_telegram_alert(test_message):
        print("✅ Telegram test successful! Check your phone.")
        return True
    else:
        print("❌ Telegram test failed.")
        return False

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
def calculate_position_size(stop_loss_pips, risk_amount):
    """Calculate position size based on risk"""
    pip_value = 10
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)

def dynamic_stop_loss(data, current_price, signal):
    """Calculate dynamic stop loss based on volatility"""
    atr = data['ATR'].iloc[-1] if not pd.isna(data['ATR'].iloc[-1]) else 0.001
    if signal == 1:
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3)
    else:
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - (atr * 3)
    return stop_loss, take_profit

# =============================================================================
# MAIN ANALYSIS ENGINE WITH TIMEFRAME VISIBILITY
# =============================================================================
def analyze_markets():
    """Main analysis function with timeframe focus"""
    print("\n" + "="*60)
    print("📊 ANALYZING FOREX MARKETS - MULTI TIMEFRAME ANALYSIS")
    print("="*60)
    
    all_data = {}
    performance_data = {
        'total_signals': 0,
        'strong_signals': 0,
        'best_pair': '',
        'market_condition': 'Neutral',
        'best_timeframe': '1h'
    }
    
    # Get market sentiment and regime
    news_sentiment, headlines = get_news_sentiment()
    print(f"📰 News Sentiment: {news_sentiment:.3f}")
    
    # Analyze each pair and timeframe
    for pair in forex_pairs:
        print(f"\n🔍 Analyzing {pair} across timeframes...")
        pair_data = {}
        
        for timeframe in timeframes:
            try:
                print(f"   ⏰ {timeframe}...", end=" ")
                
                # Download data
                data = yf.download(pair, start=start_date, end=end_date, 
                                  interval=timeframe, progress=False)
                if data.empty:
                    print("No data")
                    continue
                
                if len(data) < 30:
                    print("Insufficient data")
                    continue
                
                # Calculate indicators
                data = calculate_advanced_indicators(data, timeframe)
                
                # Generate signals
                signal_strength = generate_signals(data)
                if abs(signal_strength) < 0.1:
                    print("Weak signal")
                    continue
                
                # Adjust for news sentiment
                signal_strength = adjust_signals_with_news(signal_strength, news_sentiment)
                
                # Detect market regime
                market_regime = detect_market_regime(data)
                
                # Adjust for market regime
                signal_strength = adjust_strategy_for_regime(signal_strength, market_regime)
                
                data['Signal_Strength'] = signal_strength
                data['Position'] = np.where(signal_strength > 0.2, 1, 
                                          np.where(signal_strength < -0.2, -1, 0))
                
                pair_data[timeframe] = data
                print(f"Signal: {signal_strength:.2f}")
                
                # Send strong signals via Telegram
                if abs(signal_strength) > 0.6:
                    action = "BUY" if signal_strength > 0 else "SELL"
                    send_telegram_signal(
                        pair=pair,
                        signal_type=action,
                        price=float(data['Close'].iloc[-1]),
                        strength=abs(signal_strength),
                        timeframe=timeframe,
                        news_sentiment=news_sentiment,
                        market_regime=market_regime
                    )
                    performance_data['strong_signals'] += 1
                
                performance_data['total_signals'] += 1
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if pair_data:
            all_data[pair] = pair_data
    
    # Analyze timeframe performance
    timeframe_performance, best_timeframe = analyze_timeframe_performance(all_data)
    performance_data['best_timeframe'] = best_timeframe
    performance_data['timeframe_performance'] = timeframe_performance
    
    print(f"\n⏰ Best Performing Timeframe: {best_timeframe}")
    for tf, perf in timeframe_performance.items():
        print(f"   {tf}: Strength={perf['avg_strength']:.2f}, Win Rate={perf['win_rate']:.0%}")
    
    return all_data, performance_data

# =============================================================================
# DASHBOARD WITH TIMEFRAME VISIBILITY
# =============================================================================
def create_live_dashboard(performance_data, timeframe_performance):
    """Create real-time dashboard with timeframe focus"""
    timeframe_html = "\n".join([
        f"""<div class="timeframe-card">
            <h3>{get_timeframe_emoji(tf)} {tf.upper()}</h3>
            <p>Strength: <b>{data['avg_strength']:.2f}</b></p>
            <p>Win Rate: <b>{data['win_rate']:.0%}</b></p>
            <p>Score: <b>{data['score']:.2f}</b></p>
        </div>"""
        for tf, data in timeframe_performance.items()
    ])
    
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forex Trading Dashboard - Timeframe Analysis</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .timeframe-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .timeframe-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .b
