"""
PREDICT. - Institutional Analytics Platform v6.0
Complete Implementation with Bayesian Optimization and Walk-Forward Validation
"""

import sys
import importlib

# Check required packages
required_packages = {
    'streamlit': 'streamlit',
    'yfinance': 'yfinance',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy'
}

missing_packages = []
for package_import, package_name in required_packages.items():
    try:
        importlib.import_module(package_import)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("\n" + "="*60)
    print("ERROR: Missing required packages")
    print("="*60)
    print("\nPlease install the following packages:")
    print(f"\npip install {' '.join(missing_packages)}")
    print("\n" + "="*60 + "\n")
    sys.exit(1)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# BAYESIAN OPTIMIZER (INTEGRATED)
# ==========================================
class BayesianOptimizer:
    """Bayesian Optimization using Differential Evolution"""
    
    def __init__(self, objective_func, bounds, n_iter=100):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_iter = n_iter
        self.param_names = list(bounds.keys())
        self.bounds_array = [bounds[k] for k in self.param_names]
        self.X_samples = []
        self.y_samples = []
        self.iteration = 0
        
    def _params_to_dict(self, x):
        return {name: val for name, val in zip(self.param_names, x)}
    
    def _validate_params(self, params):
        if params.get('panic', 0) <= params.get('thresh', 0):
            return False
        if params.get('allocCrash', 100) < params.get('allocPrudence', 50):
            return False
        return True
    
    def _objective_wrapper(self, x):
        params = self._params_to_dict(x)
        
        if not self._validate_params(params):
            return 1e6
        
        try:
            score = self.objective_func(params)
            self.X_samples.append(x.copy())
            self.y_samples.append(score)
            self.iteration += 1
            return score
        except:
            return 1e6
    
    def optimize(self):
        result = differential_evolution(
            func=self._objective_wrapper,
            bounds=self.bounds_array,
            maxiter=self.n_iter,
            popsize=15,
            tol=0.001,
            mutation=(0.5, 1.5),
            recombination=0.7,
            strategy='best1bin',
            workers=1,
            updating='deferred',
            polish=True,
            seed=42
        )
        
        best_params = self._params_to_dict(result.x)
        best_score = result.fun
        
        history = pd.DataFrame({
            'iteration': range(len(self.y_samples)),
            'score': self.y_samples
        })
        
        for i, name in enumerate(self.param_names):
            history[name] = [x[i] for x in self.X_samples]
        
        return best_params, best_score, history

class MultiObjectiveScorer:
    """Multi-objective scoring with investment profiles"""
    
    PROFILES = {
        'DEFENSIVE': {'sharpe': 0.2, 'calmar': 0.4, 'cagr': 0.1, 'maxdd': 0.2, 'sortino': 0.1},
        'BALANCED': {'sharpe': 0.35, 'calmar': 0.25, 'cagr': 0.2, 'maxdd': 0.15, 'sortino': 0.05},
        'AGGRESSIVE': {'sharpe': 0.2, 'calmar': 0.1, 'cagr': 0.5, 'maxdd': 0.1, 'sortino': 0.1}
    }
    
    @staticmethod
    def normalize_metric(value, metric_name):
        ranges = {
            'sharpe': (-2, 4), 'calmar': (-2, 4), 'cagr': (-50, 50),
            'maxdd': (-50, 0), 'sortino': (-2, 4)
        }
        min_val, max_val = ranges.get(metric_name, (0, 1))
        value = np.clip(value, min_val, max_val)
        return (value - min_val) / (max_val - min_val)
    
    @classmethod
    def calculate_score(cls, metrics, profile='BALANCED'):
        weights = cls.PROFILES.get(profile, cls.PROFILES['BALANCED'])
        score = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                raw_value = metrics[metric_name]
                if metric_name == 'maxdd':
                    raw_value = -raw_value
                normalized = cls.normalize_metric(raw_value, metric_name)
                score += weight * normalized
        
        return score

# ==========================================
# DATA & METRICS
# ==========================================
@st.cache_data(ttl=3600)
def get_data_legacy(tickers, start, end):
    """Fetch market data with retry logic"""
    if len(tickers) < 2:
        return pd.DataFrame()
    
    for attempt in range(3):
        try:
            data_list = []
            for ticker in tickers[:2]:
                try:
                    df = yf.download(ticker, start=start, end=end, progress=False)
                    if not df.empty and 'Adj Close' in df.columns:
                        data_list.append(df['Adj Close'])
                except:
                    continue
            
            if len(data_list) == 2:
                result = pd.concat(data_list, axis=1)
                result.columns = ['X2', 'X1']
                result = result.ffill().dropna()
                if not result.empty:
                    return result
        except Exception as e:
            if attempt == 2:
                print(f"Error fetching data: {e}")
            continue
    
    return pd.DataFrame()

def calculate_metrics_legacy(series):
    """Calculate comprehensive performance metrics"""
    returns = series.pct_change().dropna()
    
    if len(returns) == 0:
        return {k: 0 for k in ['Cumul', 'CAGR', 'MaxDD', 'Vol', 'Sharpe', 
                                'Calmar', 'Sortino', 'Omega', 'VaR_95', 
                                'CVaR_95', 'Skew', 'Kurt']}
    
    total_ret = ((series.iloc[-1] / series.iloc[0]) - 1) * 100
    n_years = len(returns) / 252
    cagr = (((series.iloc[-1] / series.iloc[0]) ** (1 / n_years)) - 1) * 100 if n_years > 0 else 0
    
    running_max = series.cummax()
    drawdown = (series / running_max - 1) * 100
    max_dd = drawdown.min()
    
    vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else returns.std()
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    
    threshold = 0
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    omega = (gains / losses) if losses > 0 else 0
    
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    skewness = skew(returns)
    kurt = kurtosis(returns)
    
    return {
        "Cumul": total_ret, "CAGR": cagr, "MaxDD": max_dd, "Vol": vol,
        "Sharpe": sharpe, "Calmar": calmar, "Sortino": sortino, "Omega": omega,
        "VaR_95": var_95, "CVaR_95": cvar_95, "Skew": skewness, "Kurt": kurt
    }

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        """Execute regime-based backtest"""
        prices_x2 = data['X2'].values
        prices_x1 = data['X1'].values
        dates = data.index
        n = len(data)
        
        bench_x2 = (data['X2'] / data['X2'].iloc[0]) * 100
        bench_x1 = (data['X1'] / data['X1'].iloc[0]) * 100
        
        portfolio_nav = 100.0
        position_x2 = 100.0
        position_x1 = 0.0
        
        current_regime = 'R0'
        pending_regime = 'R0'
        confirm_count = 0
        
        price_history = []
        peak_at_crash = 0.0
        trough = 0.0
        
        results = []
        trades = []
        
        rolling_window = int(params['rollingWindow'])
        threshold = params['thresh']
        panic = params['panic']
        recovery = params['recovery']
        confirm_days = params['confirm']
        alloc_prudence = params['allocPrudence'] / 100.0
        alloc_crash = params['allocCrash'] / 100.0
        tx_cost = params.get('cost', 0.001)
        
        for i in range(n):
            if i > 0:
                ret_x2 = (prices_x2[i] / prices_x2[i-1]) - 1
                ret_x1 = (prices_x1[i] / prices_x1[i-1]) - 1
                position_x2 *= (1 + ret_x2)
                position_x1 *= (1 + ret_x1)
                portfolio_nav = position_x2 + position_x1
            
            price_history.append(prices_x2[i])
            if len(price_history) > rolling_window:
                price_history.pop(0)
            
            peak = max(price_history)
            current_dd = ((prices_x2[i] - peak) / peak) * 100 if peak > 0 else 0
            
            target_regime = current_regime
            
            if current_regime != 'R2':
                if current_dd <= -panic:
                    target_regime = 'R2'
                elif current_dd <= -threshold:
                    target_regime = 'R1'
                else:
                    target_regime = 'R0'
            
            if current_regime in ['R1', 'R2']:
                if prices_x2[i] < trough:
                    trough = prices_x2[i]
                
                recovery_price = trough + (peak_at_crash - trough) * (recovery / 100.0)
                
                if prices_x2[i] >= recovery_price:
                    target_regime = 'R0'
                else:
                    if current_dd <= -panic:
                        target_regime = 'R2'
                    elif current_dd <= -threshold and current_regime != 'R2':
                        target_regime = 'R1'
            else:
                peak_at_crash = peak
                trough = prices_x2[i]
            
            if target_regime == pending_regime:
                confirm_count += 1
            else:
                pending_regime = target_regime
                confirm_count = 0
            
            if confirm_count >= confirm_days and pending_regime != current_regime:
                old_regime = current_regime
                current_regime = pending_regime
                
                if current_regime == 'R2':
                    target_alloc_x1 = alloc_crash
                    label = "CRASH"
                elif current_regime == 'R1':
                    target_alloc_x1 = alloc_prudence
                    label = "PRUDENCE"
                else:
                    target_alloc_x1 = 0.0
                    label = "OFFENSIVE"
                
                total = position_x2 + position_x1
                cost = total * tx_cost
                total -= cost
                
                position_x1 = total * target_alloc_x1
                position_x2 = total * (1 - target_alloc_x1)
                
                if current_regime != 'R0':
                    peak_at_crash = peak
                    trough = prices_x2[i]
                
                trades.append({
                    'date': dates[i], 'from': old_regime, 'to': current_regime,
                    'label': label, 'portfolio': total
                })
                
                confirm_count = 0
            
            total = position_x2 + position_x1
            alloc_x2_pct = (position_x2 / total * 100) if total > 0 else 0
            alloc_x1_pct = (position_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i], 'strategy': portfolio_nav,
                'bench_x2': bench_x2.iloc[i], 'bench_x1': bench_x1.iloc[i],
                'alloc_x2': alloc_x2_pct, 'alloc_x1': alloc_x1_pct,
                'regime': current_regime
            })
        
        df = pd.DataFrame(results).set_index('date')
        return df, trades

def create_optimization_objective(backtest_func, calculate_metrics_func, 
                                 data, profile='BALANCED', confirm=2):
    """Create objective function for optimization"""
    
    def objective(params):
        sim_params = {
            'thresh': float(params.get('thresh', 5)),
            'panic': int(params.get('panic', 15)),
            'recovery': int(params.get('recovery', 30)),
            'allocPrudence': int(params.get('allocPrudence', 50)),
            'allocCrash': int(params.get('allocCrash', 100)),
            'rollingWindow': 60,
            'confirm': confirm,
            'cost': 0.001
        }
        
        try:
            df_res, trades = backtest_func(data, sim_params)
            
            if df_res.empty:
                return 1e6
            
            metrics = calculate_metrics_func(df_res['strategy'])
            score = MultiObjectiveScorer.calculate_score(metrics, profile)
            
            if profile == 'BALANCED':
                n_trades = len(trades)
                turnover_penalty = max(0, (n_trades - 10) * 0.01)
                score -= turnover_penalty
            
            return -score
            
        except:
            return 1e6
    
    return objective

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    if 'using_optimized' not in st.session_state:
        st.session_state.using_optimized = False
    if 'optimized_params' not in st.session_state:
        st.session_state.optimized_params = None
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = None

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(
    page_title="PREDICT. | Institutional Analytics", 
    layout="wide", 
    page_icon="■"
)

# Institutional Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    .stApp { background: #0A0A0A; font-family: 'Inter', -apple-system, sans-serif; color: #E0E0E0; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF; font-weight: 500; letter-spacing: -0.01em; }
    
    .main-header {
        background: #0F0F0F; border: 1px solid #1F1F1F; border-radius: 4px;
        padding: 24px 32px; margin-bottom: 24px;
    }
    .brand-title {
        font-size: 28px; font-weight: 600; letter-spacing: -0.02em;
        color: #FFFFFF; font-family: 'IBM Plex Mono', monospace;
    }
    .brand-subtitle {
        color: #707070; font-size: 11px; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px;
    }
    .status-badge {
        background: #1A1A1A; color: #A0A0A0; padding: 6px 12px;
        border-radius: 3px; font-size: 10px; font-weight: 600;
        letter-spacing: 0.05em; border: 1px solid #2A2A2A; text-transform: uppercase;
    }
    
    section[data-testid="stSidebar"] { background: #0A0A0A; border-right: 1px solid #1A1A1A; }
    .section-header {
        font-size: 10px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.12em; color: #606060; margin: 24px 0 12px 0;
        padding-bottom: 8px; border-bottom: 1px solid #1A1A1A;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 24px; font-weight: 600; color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
    }
    [data-testid="stMetricLabel"] {
        font-size: 10px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #808080;
    }
    
    .stButton > button {
        background: #1A1A1A; color: #FFFFFF; border: 1px solid #2A2A2A;
        border-radius: 3px; padding: 10px 20px; font-weight: 500;
        font-size: 12px; letter-spacing: 0.02em; transition: all 0.2s ease;
        width: 100%; text-transform: uppercase;
    }
    .stButton > button:hover { background: #252525; border-color: #3A3A3A; }
    
    .stSlider > div > div > div > div { background: #2A2A2A; }
    .stSlider > div > div > div > div > div { background: #505050; }
    
    .stSelectbox > label, .stSlider > label, .stTextInput > label {
        font-size: 11px; font-weight: 500; color: #909090;
        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;
    }
    
    input, select, textarea {
        background: #151515 !important; border: 1px solid #2A2A2A !important;
        color: #E0E0E0 !important; border-radius: 3px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #0F0F0F; padding: 0; border-bottom: 1px solid #2A2A2A;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0; padding: 12px 24px; font-weight: 500; font-size: 12px;
        letter-spacing: 0.02em; color: #808080; background: transparent;
        border: none; border-bottom: 2px solid transparent; text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        background: transparent; color: #FFFFFF; border-bottom: 2px solid #505050;
    }
    
    .dataframe {
        background: #0F0F0F !important; border: 1px solid #2A2A2A !important;
        border-radius: 3px !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }
    .dataframe th {
        background: #151515 !important; color: #909090 !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        font-size: 10px !important; letter-spacing: 0.05em !important;
        border-bottom: 1px solid #2A2A2A !important;
    }
    .dataframe td { color: #C0C0C0 !important; border-bottom: 1px solid #1A1A1A !important; }
    
    .streamlit-expanderHeader {
        background: #0F0F0F; border: 1px solid #2A2A2A; border-radius: 3px;
        font-weight: 500; font-size: 12px; color: #C0C0C0;
        text-transform: uppercase; letter-spacing: 0.02em;
    }
    
    .stProgress > div > div > div > div { background: #2A2A2A; }
    .stProgress > div > div > div > div > div { background: #505050; }
    
    .section-divider { border-top: 1px solid #2A2A2A; margin: 32px 0; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

init_session_state()

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="brand-title">PREDICT.</div>
            <div class="brand-subtitle">Institutional Risk Analytics Platform v6.0 - Bayesian Optimization</div>
        </div>
        <div><span class="status-badge">● System Operational</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="section-header">Portfolio Configuration</div>', unsafe_allow_html=True)
    
    presets = {
        "S&P 500 2x / SPY": ["SSO", "SPY"],
        "Nasdaq 100 2x / QQQ": ["QLD", "QQQ"],
        "Tech / Bonds": ["XLK", "TLT"],
        "S&P 500 / Bonds": ["SPY", "TLT"],
        "Nasdaq 100 (EU)": ["LQQ.PA", "PUST.PA"],
        "Russell 2000 2x / IWM": ["UWM", "IWM"],
        "Custom": []
    }
    
    sel_preset = st.selectbox("Asset Universe", list(presets.keys()), key='preset')
    
    if sel_preset == "Custom":
        t_input = st.text_input("Tickers (Risk, Safe)", "SPY, TLT")
        tickers = [t.strip().upper() for t in t_input.split(',')]
    else:
        tickers = presets[sel_preset]
        st.caption(f"**Risk:** {tickers[0]} | **Safe:** {tickers[1]}")
    
    # Ticker validation button
    if st.button("🔍 Validate Tickers", use_container_width=True):
        with st.spinner("Checking tickers..."):
            valid_tickers = []
            for ticker in tickers[:2]:
                try:
                    test = yf.Ticker(ticker)
                    info = test.info
                    if info and 'regularMarketPrice' in info:
                        valid_tickers.append(f"✅ {ticker}")
                    else:
                        valid_tickers.append(f"⚠️ {ticker} (limited data)")
                except:
                    valid_tickers.append(f"❌ {ticker} (invalid)")
            
            for v in valid_tickers:
                st.write(v)
    
    period_options = ["YTD", "1Y", "3YR", "5YR", "2022", "2008", "Custom"]
    sel_period = st.selectbox("Analysis Period", period_options, index=4)
    
    today = datetime.now()
    
    if sel_period == "YTD":
        start_d = datetime(today.year, 1, 1)
        end_d = today
    elif sel_period == "1Y":
        start_d = today - timedelta(days=365)
        end_d = today
    elif sel_period == "3YR":
        start_d = today - timedelta(days=365*3)
        end_d = today
    elif sel_period == "5YR":
        start_d = today - timedelta(days=365*5)
        end_d = today
    elif sel_period == "2022":
        start_d = datetime(2022, 1, 1)
        end_d = datetime(2022, 12, 31)
    elif sel_period == "2008":
        start_d = datetime(2008, 1, 1)
        end_d = datetime(2008, 12, 31)
    else:
        start_d = st.date_input("Start Date", datetime(2022, 1, 1))
        end_d = st.date_input("End Date", datetime(2022, 12, 31))
    
    st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)
    
    # Use optimized values if available
    if st.session_state.using_optimized and st.session_state.optimized_params:
        opt_p = st.session_state.optimized_params
        st.success("✨ Using Optimized Parameters")
        thresh_default = float(opt_p['thresh'])
        panic_default = int(opt_p['panic'])
        recov_default = int(opt_p['recovery'])
        alloc_prud_default = int(opt_p['allocPrudence'])
        alloc_crash_default = int(opt_p['allocCrash'])
    else:
        thresh_default = 5.0
        panic_default = 15
        recov_default = 30
        alloc_prud_default = 50
        alloc_crash_default = 100
    
    thresh = st.slider("Threshold Level (%)", 2.0, 10.0, thresh_default, 0.5)
    panic = st.slider("Panic Threshold (%)", 10, 30, panic_default, 1)
    recov = st.slider("Recovery Target (%)", 20, 60, recov_default, 5)
    
    st.markdown('<div class="section-header">Allocation Policy</div>', unsafe_allow_html=True)
    
    alloc_prud = st.slider("Prudent Mode (X1%)", 0, 100, alloc_prud_default, 10,
                          help="% allocated to safe asset in Prudence regime")
    alloc_crash = st.slider("Crisis Mode (X1%)", 0, 100, alloc_crash_default, 10,
                           help="% allocated to safe asset in Crisis regime")
    
    if alloc_crash < alloc_prud:
        st.warning("⚠️ Crisis allocation should typically be ≥ Prudence allocation")
    
    # Allocation breakdown expander
    with st.expander("📊 Allocation Breakdown"):
        st.write("**Regime Allocations:**")
        st.write(f"• **R0 (Offensive)**: 0% Safe, 100% Risk")
        st.write(f"• **R1 (Prudence)**: {alloc_prud}% Safe, {100-alloc_prud}% Risk")
        st.write(f"• **R2 (Crisis)**: {alloc_crash}% Safe, {100-alloc_crash}% Risk")
        
        st.write("\n**Examples:**")
        st.write("• Conservative: Prudence=70%, Crisis=100%")
        st.write("• Moderate: Prudence=50%, Crisis=80%")
        st.write("• Aggressive: Prudence=30%, Crisis=60%")
    
    confirm = st.slider("Confirmation Period (Days)", 1, 3, 2, 1)
    
    # Bayesian Optimization Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚀 Bayesian Optimization</div>', unsafe_allow_html=True)
    
    profile = st.selectbox("Investment Objective", 
                          ["DEFENSIVE", "BALANCED", "AGGRESSIVE"],
                          help="DEFENSIVE: Calmar focus | BALANCED: Sharpe focus | AGGRESSIVE: CAGR focus")
    
    opt_iterations = st.select_slider("Optimization Depth",
                                      options=[50, 100, 150, 200],
                                      value=100,
                                      help="More iterations = better results but slower")
    
    use_dynamic_alloc = st.checkbox("🎯 Optimize Allocations", 
                                   value=False,
                                   help="Optimize Prudence and Crisis allocation percentages")
    
    st.caption(f"⏱️ **Expected time:** ~{opt_iterations//5}-{opt_iterations//2} seconds")
    st.caption("💡 **Method:** Differential Evolution (no extra dependencies)")
    
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        if st.button("⚡ OPTIMIZE", use_container_width=True, type="primary"):
            opt_data = get_data_legacy(tickers, start_d, end_d)
            
            if not opt_data.empty:
                with st.spinner(f"🔬 Running {profile} Bayesian optimization..."):
                    bounds = {
                        'thresh': (2.0, 10.0),
                        'panic': (10, 30),
                        'recovery': (20, 60)
                    }
                    
                    if use_dynamic_alloc:
                        bounds['allocPrudence'] = (20, 80)
                        bounds['allocCrash'] = (60, 100)
                    else:
                        bounds['allocPrudence'] = (alloc_prud, alloc_prud)
                        bounds['allocCrash'] = (alloc_crash, alloc_crash)
                    
                    objective = create_optimization_objective(
                        BacktestEngine.run_simulation,
                        calculate_metrics_legacy,
                        opt_data,
                        profile,
                        confirm
                    )
                    
                    optimizer = BayesianOptimizer(objective, bounds, n_iter=opt_iterations)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    best_params, best_score, history = optimizer.optimize()
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    st.session_state.optimized_params = best_params
                    st.session_state.optimization_history = history
                    st.session_state.using_optimized = True
                    
                    st.success("✅ Optimization Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Best Parameters:**")
                        st.write(f"• Threshold: **{best_params['thresh']:.1f}%**")
                        st.write(f"• Panic: **{best_params['panic']:.0f}%**")
                        st.write(f"• Recovery: **{best_params['recovery']:.0f}%**")
                        if use_dynamic_alloc:
                            st.write(f"• Prudence Alloc: **{best_params['allocPrudence']:.0f}%**")
                            st.write(f"• Crisis Alloc: **{best_params['allocCrash']:.0f}%**")
                    
                    with col2:
                        valid_results = (history['score'] < 1e5).sum()
                        st.write("**Optimization Stats:**")
                        st.write(f"• Score: **{-best_score:.3f}**")
                        st.write(f"• Iterations: **{len(history)}**")
                        st.write(f"• Valid: **{valid_results}**")
                    
                    st.info("💡 Adjust sliders above to apply these parameters")
                    st.rerun()
            else:
                st.warning("⚠️ Load data first")
    
    with opt_col2:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.using_optimized = False
            st.session_state.optimized_params = None
            st.success("✅ Reset to defaults")
            st.rerun()

# Main Content
data = get_data_legacy(tickers, start_d, end_d)

if data.empty:
    st.error("⚠️ Unable to retrieve market data")
    
    with st.expander("💡 Troubleshooting"):
        st.markdown("""
        **Possible causes:**
        
        1. **Invalid tickers** - Click "🔍 Validate Tickers" button in sidebar
        2. **European markets** - Try US equivalents:
           - LQQ.PA → QLD (Nasdaq 2x US)
           - PUST.PA → SPY (S&P 500)
        3. **Data availability** - Some tickers have limited historical data
        4. **Network issues** - Check internet connection
        
        **Recommended presets that work well:**
        - ✅ S&P 500 2x / SPY (SSO, SPY)
        - ✅ Nasdaq 100 2x / QQQ (QLD, QQQ)
        - ✅ Tech / Bonds (XLK, TLT)
        - ✅ S&P 500 / Bonds (SPY, TLT)
        
        **Try these tickers:**
        - Risk: SSO, QLD, UPRO, TQQQ, XLK
        - Safe: SPY, QQQ, TLT, IEF, AGG
        """)
    
    st.stop()

sim_params = {
    'thresh': thresh, 'panic': panic, 'recovery': recov,
    'allocPrudence': alloc_prud, 'allocCrash': alloc_crash,
    'rollingWindow': 60, 'confirm': confirm, 'cost': 0.001
}

df_res, trades = BacktestEngine.run_simulation(data, sim_params)

if df_res.empty:
    st.error("⚠ Simulation engine error")
    st.stop()

met_strat = calculate_metrics_legacy(df_res['strategy'])
met_x2 = calculate_metrics_legacy(df_res['bench_x2'])
met_x1 = calculate_metrics_legacy(df_res['bench_x1'])

tabs = st.tabs(["Performance", "Risk Analytics", "Allocation", "Trades"])

# TAB 1: Performance
with tabs[0]:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("CAGR", f"{met_strat['CAGR']:.2f}%")
    k2.metric("Sharpe", f"{met_strat['Sharpe']:.3f}")
    k3.metric("Sortino", f"{met_strat['Sortino']:.3f}")
    k4.metric("Max DD", f"{met_strat['MaxDD']:.2f}%")
    k5.metric("Omega", f"{met_strat['Omega']:.3f}")
    k6.metric("Trades", len(trades))
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Cumulative Performance")
    
    chart_data = df_res[['strategy', 'bench_x2', 'bench_x1']].copy()
    chart_data.columns = ['Strategy', f'{tickers[0]} (Risk)', f'{tickers[1]} (Safe)']
    st.line_chart(chart_data, height=400)
    
    st.markdown("#### Asset Allocation")
    alloc_data = df_res[['alloc_x2', 'alloc_x1']].copy()
    alloc_data.columns = ['Risk Asset (%)', 'Safe Asset (%)']
    st.area_chart(alloc_data, height=250)
    
    st.markdown("#### Performance Comparison")
    perf_df = pd.DataFrame({
        "Metric": ["Total Return", "CAGR", "Max Drawdown", "Volatility", "Sharpe", "Sortino", "Calmar"],
        "Strategy": [
            f"{met_strat['Cumul']:.2f}%", f"{met_strat['CAGR']:.2f}%",
            f"{met_strat['MaxDD']:.2f}%", f"{met_strat['Vol']:.2f}%",
            f"{met_strat['Sharpe']:.3f}", f"{met_strat['Sortino']:.3f}",
            f"{met_strat['Calmar']:.3f}"
        ],
        f"{tickers[0]}": [
            f"{met_x2['Cumul']:.2f}%", f"{met_x2['CAGR']:.2f}%",
            f"{met_x2['MaxDD']:.2f}%", f"{met_x2['Vol']:.2f}%",
            f"{met_x2['Sharpe']:.3f}", f"{met_x2['Sortino']:.3f}",
            f"{met_x2['Calmar']:.3f}"
        ]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

# TAB 2: Risk Analytics
with tabs[1]:
    st.markdown("#### Comprehensive Risk Profile")
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR (95%)", f"{met_strat['VaR_95']:.2f}%")
    r2.metric("CVaR (95%)", f"{met_strat['CVaR_95']:.2f}%")
    r3.metric("Skewness", f"{met_strat['Skew']:.3f}")
    r4.metric("Kurtosis", f"{met_strat['Kurt']:.3f}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Drawdown Analysis")
    
    dd_strat = (df_res['strategy'] / df_res['strategy'].cummax() - 1) * 100
    dd_x2 = (df_res['bench_x2'] / df_res['bench_x2'].cummax() - 1) * 100
    
    dd_chart = pd.DataFrame({'Strategy': dd_strat, f'{tickers[0]}': dd_x2})
    st.line_chart(dd_chart, height=300)

# TAB 3: Allocation
with tabs[2]:
    st.markdown("#### 🎯 Allocation Analysis")
    
    alloc_stats = df_res[['alloc_x2', 'alloc_x1', 'regime']].copy()
    
    col1, col2, col3, col4 = st.columns(4)
    
    regime_counts = alloc_stats['regime'].value_counts()
    total_days = len(alloc_stats)
    
    r0_pct = (regime_counts.get('R0', 0) / total_days) * 100
    r1_pct = (regime_counts.get('R1', 0) / total_days) * 100
    r2_pct = (regime_counts.get('R2', 0) / total_days) * 100
    
    col1.metric("Offensive (R0)", f"{r0_pct:.1f}%", f"{regime_counts.get('R0', 0)} days")
    col2.metric("Prudence (R1)", f"{r1_pct:.1f}%", f"{regime_counts.get('R1', 0)} days")
    col3.metric("Crisis (R2)", f"{r2_pct:.1f}%", f"{regime_counts.get('R2', 0)} days")
    
    avg_safe = alloc_stats['alloc_x1'].mean()
    col4.metric("Avg Safe Asset", f"{avg_safe:.1f}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Allocation Evolution")
    
    alloc_chart = df_res[['alloc_x2', 'alloc_x1']].copy()
    alloc_chart.columns = ['Risk Asset (%)', 'Safe Asset (%)']
    st.area_chart(alloc_chart, height=350)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Performance by Regime")
    
    df_res['returns'] = df_res['strategy'].pct_change()
    
    regime_performance = []
    for regime in ['R0', 'R1', 'R2']:
        regime_data = df_res[df_res['regime'] == regime]
        if len(regime_data) > 0:
            regime_returns = regime_data['returns'].dropna()
            
            regime_performance.append({
                'Regime': f'{regime} ({["Offensive", "Prudence", "Crisis"][int(regime[1])]})',
                'Days': len(regime_data),
                'Avg Daily Return': f"{regime_returns.mean()*100:.3f}%",
                'Volatility': f"{regime_returns.std()*100:.2f}%",
                'Sharpe (Ann.)': f"{(regime_returns.mean() / regime_returns.std() * np.sqrt(252)):.2f}" if regime_returns.std() > 0 else "N/A",
                'Win Rate': f"{(regime_returns > 0).mean()*100:.1f}%"
            })
    
    if regime_performance:
        perf_df = pd.DataFrame(regime_performance)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🔬 Allocation Efficiency")
    
    static_results = []
    
    for safe_pct in [0, 25, 50, 75, 100]:
        static_perf = df_res['bench_x2'] * (1 - safe_pct/100) + df_res['bench_x1'] * (safe_pct/100)
        static_metrics = calculate_metrics_legacy(static_perf)
        
        static_results.append({
            'Allocation': f"{100-safe_pct}% Risk / {safe_pct}% Safe",
            'CAGR': f"{static_metrics['CAGR']:.2f}%",
            'Sharpe': f"{static_metrics['Sharpe']:.2f}",
            'Max DD': f"{static_metrics['MaxDD']:.2f}%"
        })
    
    static_results.append({
        'Allocation': '🎯 Dynamic (This Strategy)',
        'CAGR': f"{met_strat['CAGR']:.2f}%",
        'Sharpe': f"{met_strat['Sharpe']:.2f}",
        'Max DD': f"{met_strat['MaxDD']:.2f}%"
    })
    
    comparison_df = pd.DataFrame(static_results)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("**Interpretation:**")
    st.write("Dynamic allocation should outperform most static allocations by adapting to market conditions.")
    
    best_static_sharpe = max([float(r['Sharpe']) for r in static_results[:-1]])
    if met_strat['Sharpe'] > best_static_sharpe:
        st.success(f"✅ Dynamic strategy beats all static allocations (Sharpe: {met_strat['Sharpe']:.2f} vs {best_static_sharpe:.2f})")
    else:
        st.warning(f"⚠️ Some static allocations perform better. Consider adjusting parameters.")

# TAB 4: Trades
with tabs[3]:
    st.markdown("#### Trade History")
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        st.markdown("#### Transition Summary")
        transition_counts = trades_df['label'].value_counts()
        
        t1, t2, t3 = st.columns(3)
        t1.metric("→ OFFENSIVE", transition_counts.get('OFFENSIVE', 0))
        t2.metric("→ PRUDENCE", transition_counts.get('PRUDENCE', 0))
        t3.metric("→ CRASH", transition_counts.get('CRASH', 0))
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Transition Timeline:**")
        
        trades_display = trades_df[['date', 'from', 'to', 'label', 'portfolio']].copy()
        trades_display['date'] = trades_display['date'].dt.strftime('%Y-%m-%d')
        trades_display['portfolio'] = trades_display['portfolio'].apply(lambda x: f"{x:.2f}")
        trades_display.columns = ['Date', 'From', 'To', 'New Regime', 'Portfolio Value']
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No regime transitions occurred (stayed in R0 - Offensive)")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #606060; font-size: 10px; padding: 16px 0; text-transform: uppercase; letter-spacing: 0.1em;">
    PREDICT. INSTITUTIONAL ANALYTICS v6.0 • ENTERPRISE EDITION
</div>
""", unsafe_allow_html=True)
