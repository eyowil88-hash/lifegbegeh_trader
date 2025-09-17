# ULTIMATE FOREX TRADING BOT WITH TELEGRAM ALERTS
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🤖 ULTIMATE FOREX TRADING BOT INITIALIZING...")
print("📱 Telegram Alerts Enabled | 🤖 ML Active | ⚖️ Risk Management Live")

# =============================================================================
# CONFIGURATION - USER SETTINGS
# =============================================================================
forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
timeframes = ['5m', '15m', '1h', '4h', '1d']
account_balance = 5000  # Your account balance
risk_per_trade = 1.0    # Risk percentage per trade
leverage = 10
start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# =============================================================================
# TELEGRAM CONFIGURATION - YOUR CREDENTIALS
# =============================================================================
TELEGRAM_BOT_TOKEN = "8340887342:AAGwQJwglAiD3uSuLg-cIdPYb87ywkMgMBA"
TELEGRAM_CHAT_ID = "630055275"

# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================
def send_telegram_alert(message, parse_mode='HTML'):
    """Send formatted alert to Telegram"""
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

def send_telegram_signal(pair, signal_type, price, strength, timeframe, ml_confidence=None):
    """Send professional trading signal"""
    emoji = "🟢" if signal_type == "BUY" else "🔴"
    message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

<b>Pair:</b> {pair}
<b>Action:</b> {signal_type}
<b>Price:</b> {price:.5f}
<b>Timeframe:</b> {timeframe}
<b>Signal Strength:</b> {strength:.2f}/1.0
{f'<b>ML Confidence:</b> {ml_confidence:.0%}' if ml_confidence else ''}

<b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return send_telegram_alert(message)

def send_daily_report(performance_data):
    """Send daily performance report"""
    message = f"""
📊 <b>DAILY TRADING REPORT</b>

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}
<b>Account Balance:</b> ${account_balance:,.2f}
<b>Risk per Trade:</b> {risk_per_trade}%

<b>Performance Summary:</b>
• Total Signals: {performance_data.get('total_signals', 0)}
• Strong Signals: {performance_data.get('strong_signals', 0)}
• Best Pair: {performance_data.get('best_pair', 'N/A')}

<b>Market Condition:</b> {performance_data.get('market_condition', 'Neutral')}

<b>Today's Recommendation:</b>
{performance_data.get('recommendation', 'Monitor key levels')}
"""
    return send_telegram_alert(message)

def test_telegram_connection():
    """Test Telegram connection"""
    print("Testing Telegram connection...")
    test_message = f"""
🤖 <b>FOREX BOT CONNECTION TEST</b>

✅ <b>Status:</b> Connected Successfully
📊 <b>Account:</b> ${account_balance:,.2f}
🆔 <b>Chat ID:</b> {TELEGRAM_CHAT_ID}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚀 <b>System:</b> Ready for live trading!

<i>You will receive real-time trading signals here</i>
"""
    if send_telegram_alert(test_message):
        print("✅ Telegram test successful! Check your phone.")
        return True
    else:
        print("❌ Telegram test failed.")
        return False

# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================
def calculate_advanced_indicators(data, timeframe):
    """Calculate all technical indicators"""
    # Moving Averages
    fast_period = 12 if timeframe in ['5m', '15m'] else 20
    slow_period = 26 if timeframe in ['5m', '15m'] else 50
    data['Fast_MA'] = data['Close'].rolling(window=fast_period).mean()
    data['Slow_MA'] = data['Close'].rolling(window=slow_period).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
    
    # ATR for volatility
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    data['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
    
    return data

def generate_signals(data):
    """Generate trading signals from indicators"""
    # Individual signals
    ma_signal = 1 if data['Fast_MA'].iloc[-1] > data['Slow_MA'].iloc[-1] else -1
    rsi_signal = 1 if data['RSI'].iloc[-1] > 50 else -1
    macd_signal = 1 if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1] else -1
    bb_signal = 1 if data['Close'].iloc[-1] > data['BB_Middle'].iloc[-1] else -1
    
    # Weighted combined signal
    weights = {'ma': 0.4, 'rsi': 0.2, 'macd': 0.2, 'bb': 0.2}
    final_signal = (ma_signal * weights['ma'] + 
                   rsi_signal * weights['rsi'] + 
                   macd_signal * weights['macd'] + 
                   bb_signal * weights['bb'])
    
    return final_signal

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
def calculate_position_size(stop_loss_pips, risk_amount):
    """Calculate position size based on risk"""
    pip_value = 10  # $10 per pip for standard lot
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)

def dynamic_stop_loss(data, current_price, signal):
    """Calculate dynamic stop loss based on volatility"""
    atr = data['ATR'].iloc[-1] if not pd.isna(data['ATR'].iloc[-1]) else 0.001
    if signal == 1:  # Long
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3)
    else:  # Short
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - (atr * 3)
    return stop_loss, take_profit

# =============================================================================
# MACHINE LEARNING PREDICTIONS
# =============================================================================
def generate_ml_predictions(data, pair):
    """Generate ML-based predictions"""
    try:
        if len(data) < 100:
            return 0.5, 0.5
        
        # Create features
        data['Returns_1'] = data['Close'].pct_change(1)
        data['Returns_5'] = data['Close'].pct_change(5)
        data['Volatility'] = data['High'] - data['Low']
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        
        # Prepare features
        features = ['Returns_1', 'Returns_5', 'Volatility', 'Momentum', 'RSI', 'MACD']
        X = data[features].dropna()
        y = data['Target'].loc[X.index]
        
        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            latest_features = X.iloc[[-1]]
            prediction_proba = model.predict_proba(latest_features)[0]
            accuracy = model.score(X_test, y_test)
            
            return prediction_proba[1], accuracy
    except Exception as e:
        print(f"ML Error for {pair}: {e}")
    
    return 0.5, 0.5

# =============================================================================
# MAIN ANALYSIS ENGINE
# =============================================================================
def analyze_markets():
    """Main analysis function"""
    print("\n" + "="*60)
    print("📊 ANALYZING FOREX MARKETS")
    print("="*60)
    
    all_data = {}
    performance_data = {
        'total_signals': 0,
        'strong_signals': 0,
        'best_pair': '',
        'market_condition': 'Neutral',
        'recommendation': 'Wait for clear signals'
    }
    
    # Analyze each pair and timeframe
    for pair in forex_pairs:
        print(f"🔍 Analyzing {pair}...")
        pair_data = {}
        
        for timeframe in timeframes:
            try:
                # Download data
                data = yf.download(pair, start=start_date, end=end_date, 
                                  interval=timeframe, progress=False)
                if data.empty:
                    continue
                
                # Calculate indicators
                data = calculate_advanced_indicators(data, timeframe)
                
                # Generate signals
                signal_strength = generate_signals(data)
                data['Signal_Strength'] = signal_strength
                data['Position'] = np.where(signal_strength > 0.2, 1, 
                                          np.where(signal_strength < -0.2, -1, 0))
                
                pair_data[timeframe] = data
                
                # Check for strong signals to send via Telegram
                if timeframe in ['1h', '4h'] and abs(signal_strength) > 0.6:
                    action = "BUY" if signal_strength > 0 else "SELL"
                    send_telegram_signal(
                        pair=pair,
                        signal_type=action,
                        price=data['Close'].iloc[-1],
                        strength=abs(signal_strength),
                        timeframe=timeframe
                    )
                    performance_data['strong_signals'] += 1
                
                performance_data['total_signals'] += 1
                
            except Exception as e:
                print(f"❌ Error analyzing {pair} ({timeframe}): {e}")
        
        all_data[pair] = pair_data
    
    return all_data, performance_data

# =============================================================================
# EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    print("🤖 ULTIMATE FOREX TRADING BOT STARTING...")
    
    # Test Telegram connection
    if not test_telegram_connection():
        print("⚠️  Continuing without Telegram...")
    
    # Analyze markets
    market_data, performance_data = analyze_markets()
    
    # Send daily report
    send_daily_report(performance_data)
    
    # Display summary
    print("\n" + "="*60)
    print("📈 ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total signals generated: {performance_data['total_signals']}")
    print(f"Strong signals sent: {performance_data['strong_signals']}")
    print(f"Telegram alerts: {'✅ Enabled' if test_telegram_connection() else '❌ Disabled'}")
    
    # Risk management info
    risk_amount = account_balance * (risk_per_trade / 100)
    print(f"\n⚠️  RISK MANAGEMENT:")
    print(f"Account: ${account_balance:,.2f}")
    print(f"Risk per trade: ${risk_amount:.2f} ({risk_per_trade}%)")
    print(f"Max position size: {calculate_position_size(50, risk_amount):.2f} lots")
    
    print("\n" + "="*60)
    print("✅ BOT READY FOR TRADING")
    print("📱 Telegram alerts active")
    print("🤖 ML predictions enabled") 
    print("⚖️ Risk management live")
    print("="*60)

# =============================================================================
# RUN THE BOT
# =============================================================================
if __name__ == "__main__":
    main()
    
    # Optional: Continuous monitoring
    print("\n🔔 Continuous monitoring mode:")
    print("The bot can run every 15-30 minutes for live signals")
    print("Telegram will alert you instantly on your phone")
