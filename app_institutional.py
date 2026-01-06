"""
PREDICT. - Institutional Analytics Platform
Enhanced with Bayesian Optimization and Walk-Forward Validation
"""

# Check for required packages
import sys
import importlib

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
    print("\nOr install all requirements:")
    print("pip install -r requirements.txt")
    print("\n" + "="*60 + "\n")
    sys.exit(1)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import optimizer modules
try:
    from optimizer import (BayesianOptimizer, WalkForwardValidator, 
                          MultiObjectiveScorer, create_optimization_objective)
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("Warning: Optimizer module not available")

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PREDICT. | Institutional Analytics", 
    layout="wide", 
    page_icon="■"
)

# ==========================================
# INSTITUTIONAL STYLING - PROFESSIONAL BLACK/GREY
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    /* BASE CONFIGURATION */
    .stApp { 
        background: #0A0A0A;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #E0E0E0;
    }
    
    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6 { 
        color: #FFFFFF;
        font-weight: 500;
        letter-spacing: -0.01em;
    }
    
    h1 { font-size: 24px; }
    h2 { font-size: 20px; }
    h3 { font-size: 18px; }
    h4 { font-size: 16px; }
    
    /* HEADER */
    .main-header {
        background: #0F0F0F;
        border: 1px solid #1F1F1F;
        border-radius: 4px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    
    .brand-title {
        font-size: 28px;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .brand-subtitle {
        color: #707070;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }
    
    .status-badge {
        background: #1A1A1A;
        color: #A0A0A0;
        padding: 6px 12px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.05em;
        border: 1px solid #2A2A2A;
        text-transform: uppercase;
    }
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #0A0A0A;
        border-right: 1px solid #1A1A1A;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 24px 16px;
    }
    
    .section-header {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #606060;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1A1A1A;
    }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
        color: #FFFFFF;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #808080;
    }
    
    /* CONTROLS */
    .stButton > button {
        background: #1A1A1A;
        color: #FFFFFF;
        border: 1px solid #2A2A2A;
        border-radius: 3px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 12px;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        width: 100%;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: #252525;
        border-color: #3A3A3A;
    }
    
    .stButton > button:active {
        background: #151515;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: #2A2A2A;
    }
    
    .stSlider > div > div > div > div > div {
        background: #505050;
    }
    
    /* Labels */
    .stSelectbox > label, .stSlider > label, .stTextInput > label {
        font-size: 11px;
        font-weight: 500;
        color: #909090;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    /* Input fields */
    input, select, textarea {
        background: #151515 !important;
        border: 1px solid #2A2A2A !important;
        color: #E0E0E0 !important;
        border-radius: 3px !important;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0F0F0F;
        padding: 0;
        border-bottom: 1px solid #2A2A2A;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 12px;
        letter-spacing: 0.02em;
        color: #808080;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        text-transform: uppercase;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #FFFFFF;
        border-bottom: 2px solid #505050;
    }
    
    /* TABLES */
    .dataframe {
        background: #0F0F0F !important;
        border: 1px solid #2A2A2A !important;
        border-radius: 3px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }
    
    .dataframe th {
        background: #151515 !important;
        color: #909090 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 10px !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid #2A2A2A !important;
    }
    
    .dataframe td {
        color: #C0C0C0 !important;
        border-bottom: 1px solid #1A1A1A !important;
    }
    
    /* EXPANDERS */
    .streamlit-expanderHeader {
        background: #0F0F0F;
        border: 1px solid #2A2A2A;
        border-radius: 3px;
        font-weight: 500;
        font-size: 12px;
        color: #C0C0C0;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    
    /* CHARTS */
    [data-testid="stLineChart"], 
    [data-testid="stAreaChart"], 
    [data-testid="stBarChart"] {
        background: #0F0F0F;
        border: 1px solid #2A2A2A;
        border-radius: 3px;
        padding: 16px;
    }
    
    /* DIVIDERS */
    hr {
        border-color: #1A1A1A;
        margin: 20px 0;
    }
    
    /* ALERTS */
    .stAlert {
        background: #0F0F0F;
        border: 1px solid #2A2A2A;
        border-radius: 3px;
        color: #C0C0C0;
    }
    
    /* PROGRESS BARS */
    .stProgress > div > div > div > div {
        background: #2A2A2A;
    }
    
    .stProgress > div > div > div > div > div {
        background: #505050;
    }
    
    /* CUSTOM COMPONENTS */
    .metric-container {
        background: #0F0F0F;
        border: 1px solid #2A2A2A;
        border-radius: 3px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .section-divider {
        border-top: 1px solid #2A2A2A;
        margin: 32px 0;
    }
    
    /* Remove default elements */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def get_data_legacy(tickers, start, end):
    """Fetch data with retry logic"""
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
    """Calculate comprehensive metrics"""
    returns = series.pct_change().dropna()
    
    if len(returns) == 0:
        return {k: 0 for k in ['Cumul', 'CAGR', 'MaxDD', 'Vol', 'Sharpe', 
                                'Calmar', 'Sortino', 'Omega', 'VaR_95', 
                                'CVaR_95', 'Skew', 'Kurt']}
    
    # Basic metrics
    total_ret = ((series.iloc[-1] / series.iloc[0]) - 1) * 100
    n_years = len(returns) / 252
    cagr = (((series.iloc[-1] / series.iloc[0]) ** (1 / n_years)) - 1) * 100 if n_years > 0 else 0
    
    # Drawdown
    running_max = series.cummax()
    drawdown = (series / running_max - 1) * 100
    max_dd = drawdown.min()
    
    # Volatility
    vol = returns.std() * np.sqrt(252) * 100
    
    # Sharpe
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else returns.std()
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    
    # Omega
    threshold = 0
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    omega = (gains / losses) if losses > 0 else 0
    
    # VaR and CVaR
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # Skewness and Kurtosis
    from scipy.stats import skew, kurtosis
    skewness = skew(returns)
    kurt = kurtosis(returns)
    
    return {
        "Cumul": total_ret,
        "CAGR": cagr,
        "MaxDD": max_dd,
        "Vol": vol,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "Sortino": sortino,
        "Omega": omega,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Skew": skewness,
        "Kurt": kurt
    }

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    @staticmethod
    def run_simulation(data, params):
        """Run regime-based simulation"""
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
                    'date': dates[i],
                    'from': old_regime,
                    'to': current_regime,
                    'label': label,
                    'portfolio': total
                })
                
                confirm_count = 0
            
            total = position_x2 + position_x1
            alloc_x2_pct = (position_x2 / total * 100) if total > 0 else 0
            alloc_x1_pct = (position_x1 / total * 100) if total > 0 else 0
            
            results.append({
                'date': dates[i],
                'strategy': portfolio_nav,
                'bench_x2': bench_x2.iloc[i],
                'bench_x1': bench_x1.iloc[i],
                'alloc_x2': alloc_x2_pct,
                'alloc_x1': alloc_x1_pct,
                'regime': current_regime
            })
        
        df = pd.DataFrame(results).set_index('date')
        return df, trades

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
def init_session_state():
    """Initialize session state variables"""
    if 'default_params' not in st.session_state:
        st.session_state.default_params = {
            'thresh': 5.0,
            'panic': 15,
            'recovery': 30,
            'allocPrudence': 50,
            'allocCrash': 100
        }
    
    if 'optimized_params' not in st.session_state:
        st.session_state.optimized_params = None
    
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None

init_session_state()

# ==========================================
# UI - HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="brand-title">PREDICT.</div>
            <div class="brand-subtitle">Institutional Risk Analytics Platform v6.0 - Bayesian Optimization</div>
        </div>
        <div>
            <span class="status-badge">● System Operational</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not OPTIMIZER_AVAILABLE:
    st.warning("⚠️ Optimizer module not loaded. Using legacy mode.")

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown('<div class="section-header">Portfolio Configuration</div>', unsafe_allow_html=True)
    
    presets = {
        "S&P 500 2x / SPY": ["SSO", "SPY"],
        "Nasdaq 100 2x / QQQ": ["QLD", "QQQ"],
        "Tech / Bonds": ["XLK", "TLT"],
        "S&P 500 / Bonds": ["SPY", "TLT"],
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
    
    # Period selection
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
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Strategy Parameters</div>', unsafe_allow_html=True)
    
    # Use optimized params if available, otherwise use defaults
    current_params = (st.session_state.optimized_params 
                     if st.session_state.optimized_params 
                     else st.session_state.default_params)
    
    # Show optimization status
    if st.session_state.optimized_params:
        st.success("✨ Using Optimized Parameters")
        if st.session_state.optimization_results:
            st.caption(f"Score: {st.session_state.optimization_results.get('score', 0):.3f}")
    
    thresh = st.slider("Threshold Level (%)", 2.0, 10.0, 
                      float(current_params['thresh']), 0.5)
    panic = st.slider("Panic Threshold (%)", 10, 30, 
                     int(current_params['panic']), 1)
    recov = st.slider("Recovery Target (%)", 20, 60, 
                     int(current_params['recovery']), 5)
    
    st.markdown('<div class="section-header">Allocation Policy</div>', unsafe_allow_html=True)
    
    alloc_prud = st.slider("Prudent Mode (X1%)", 0, 100, 
                          int(current_params['allocPrudence']), 10,
                          help="% allocated to safe asset in Prudence regime")
    alloc_crash = st.slider("Crisis Mode (X1%)", 0, 100, 
                           int(current_params['allocCrash']), 10,
                           help="% allocated to safe asset in Crisis regime")
    
    confirm = st.slider("Confirmation Period (Days)", 1, 3, 2, 1)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Bayesian Optimization</div>', unsafe_allow_html=True)
    
    profile = st.selectbox("Investment Objective", 
                          ["DEFENSIVE", "BALANCED", "AGGRESSIVE"])
    
    opt_iterations = st.select_slider("Optimization Depth",
                                      options=[50, 100, 150, 200],
                                      value=100,
                                      help="More iterations = better results but slower")
    
    use_dynamic_alloc = st.checkbox("🎯 Optimize Allocations", 
                                   value=False,
                                   help="Optimize allocation percentages")
    
    st.caption(f"**Expected runtime:** ~{opt_iterations//5}-{opt_iterations//2} seconds")
    
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        if st.button("⚡ OPTIMIZE", use_container_width=True, type="primary"):
            if not OPTIMIZER_AVAILABLE:
                st.error("Optimizer module not available")
            else:
                opt_data = get_data_legacy(tickers, start_d, end_d)
                
                if not opt_data.empty:
                    with st.spinner(f"Running {profile} Bayesian optimization..."):
                        # Define parameter bounds
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
                        
                        # Create objective function
                        objective = create_optimization_objective(
                            BacktestEngine.run_simulation,
                            calculate_metrics_legacy,
                            opt_data,
                            profile,
                            confirm
                        )
                        
                        # Run optimization
                        optimizer = BayesianOptimizer(
                            objective,
                            bounds,
                            n_iter=opt_iterations
                        )
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        best_params, best_score, history = optimizer.optimize()
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        # Store results
                        st.session_state.optimized_params = best_params
                        st.session_state.optimization_results = {
                            'score': -best_score,  # Convert back to positive
                            'history': history,
                            'iterations': len(history)
                        }
                        
                        st.success(f"✅ Optimization Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Best Parameters:**")
                            st.write(f"• Threshold: **{best_params['thresh']:.1f}%**")
                            st.write(f"• Panic: **{best_params['panic']:.0f}%**")
                            st.write(f"• Recovery: **{best_params['recovery']:.0f}%**")
                            if use_dynamic_alloc:
                                st.write(f"• Prudence: **{best_params['allocPrudence']:.0f}%**")
                                st.write(f"• Crash: **{best_params['allocCrash']:.0f}%**")
                        
                        with col2:
                            st.write("**Optimization Stats:**")
                            st.write(f"• Score: **{-best_score:.3f}**")
                            st.write(f"• Iterations: **{len(history)}**")
                            st.write(f"• Valid: **{(history['score'] < 1e5).sum()}**")
                        
                        st.info("✨ **Parameters applied** - refresh page to use them")
                        st.rerun()
                else:
                    st.warning("⚠️ Load data first")
    
    with opt_col2:
        if st.button("📊 RESET", use_container_width=True):
            st.session_state.optimized_params = None
            st.session_state.optimization_results = None
            st.success("✅ Reset to defaults")
            st.rerun()

# ==========================================
# MAIN CONTENT
# ==========================================
data = get_data_legacy(tickers, start_d, end_d)

if data.empty:
    st.error("⚠️ Unable to retrieve market data")
    st.stop()

# Build simulation parameters
sim_params = {
    'thresh': thresh,
    'panic': panic,
    'recovery': recov,
    'allocPrudence': alloc_prud,
    'allocCrash': alloc_crash,
    'rollingWindow': 60,
    'confirm': confirm,
    'cost': 0.001
}

df_res, trades = BacktestEngine.run_simulation(data, sim_params)

if df_res.empty:
    st.error("⚠ Simulation engine error")
    st.stop()

# Calculate metrics
met_strat = calculate_metrics_legacy(df_res['strategy'])
met_x2 = calculate_metrics_legacy(df_res['bench_x2'])
met_x1 = calculate_metrics_legacy(df_res['bench_x1'])

# ==========================================
# TABS
# ==========================================
tabs = st.tabs(["Performance", "Risk Analytics", "Optimization", "Walk-Forward", "Allocation"])

# TAB 1: PERFORMANCE
with tabs[0]:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("CAGR", f"{met_strat['CAGR']:.2f}%")
    k2.metric("Sharpe", f"{met_strat['Sharpe']:.3f}")
    k3.metric("Sortino", f"{met_strat['Sortino']:.3f}")
    k4.metric("Max DD", f"{met_strat['MaxDD']:.2f}%")
    k5.metric("Calmar", f"{met_strat['Calmar']:.3f}")
    k6.metric("Trades", len(trades))
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Show optimization improvement
    if st.session_state.optimization_results:
        st.markdown("#### 🎯 Optimization Impact")
        
        opt_score = st.session_state.optimization_results['score']
        current_score = MultiObjectiveScorer.calculate_score(
            {
                'sharpe': met_strat['Sharpe'],
                'cagr': met_strat['CAGR'],
                'maxdd': met_strat['MaxDD'],
                'calmar': met_strat['Calmar'],
                'sortino': met_strat['Sortino']
            },
            profile
        )
        
        improvement = ((current_score - opt_score) / abs(opt_score) * 100) if opt_score != 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimized Score", f"{opt_score:.3f}")
        col2.metric("Current Score", f"{current_score:.3f}")
        col3.metric("Improvement", f"{improvement:+.1f}%",
                   delta_color="normal" if improvement >= 0 else "inverse")
        
        if improvement < -5:
            st.warning("⚠️ Current parameters underperform optimized")
        elif improvement > 5:
            st.success("✅ Current parameters outperform optimization!")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("#### Cumulative Performance")
    chart_data = df_res[['strategy', 'bench_x2', 'bench_x1']].copy()
    chart_data.columns = ['Strategy', f'{tickers[0]} (Risk)', f'{tickers[1]} (Safe)']
    st.line_chart(chart_data, height=400)
    
    st.markdown("#### Performance Comparison")
    perf_df = pd.DataFrame({
        "Metric": ["Total Return", "CAGR", "Max Drawdown", "Volatility", "Sharpe", "Sortino", "Calmar"],
        "Strategy": [
            f"{met_strat['Cumul']:.2f}%",
            f"{met_strat['CAGR']:.2f}%",
            f"{met_strat['MaxDD']:.2f}%",
            f"{met_strat['Vol']:.2f}%",
            f"{met_strat['Sharpe']:.3f}",
            f"{met_strat['Sortino']:.3f}",
            f"{met_strat['Calmar']:.3f}"
        ],
        f"{tickers[0]}": [
            f"{met_x2['Cumul']:.2f}%",
            f"{met_x2['CAGR']:.2f}%",
            f"{met_x2['MaxDD']:.2f}%",
            f"{met_x2['Vol']:.2f}%",
            f"{met_x2['Sharpe']:.3f}",
            f"{met_x2['Sortino']:.3f}",
            f"{met_x2['Calmar']:.3f}"
        ]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

# TAB 2: RISK
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

# TAB 3: OPTIMIZATION
with tabs[2]:
    if not st.session_state.optimization_results:
        st.info("ℹ️ Run optimization to see results")
    else:
        st.markdown("#### Optimization History")
        
        history = st.session_state.optimization_results['history']
        
        # Plot convergence
        st.markdown("**Score Convergence**")
        
        # Filter valid results
        valid_history = history[history['score'] < 1e5].copy()
        valid_history['best_so_far'] = valid_history['score'].cummin()
        
        conv_chart = pd.DataFrame({
            'Score': -valid_history['score'],  # Convert back to positive
            'Best So Far': -valid_history['best_so_far']
        })
        st.line_chart(conv_chart, height=300)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Parameter evolution
        st.markdown("#### Parameter Evolution")
        
        param_cols = ['thresh', 'panic', 'recovery']
        if 'allocPrudence' in valid_history.columns:
            param_cols.extend(['allocPrudence', 'allocCrash'])
        
        for param in param_cols:
            if param in valid_history.columns:
                st.markdown(f"**{param.capitalize()}**")
                st.line_chart(valid_history[param], height=150)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Top 10 results
        st.markdown("#### Top 10 Configurations")
        
        top_10 = valid_history.nsmallest(10, 'score')[['thresh', 'panic', 'recovery', 'score']].copy()
        top_10['score'] = -top_10['score']
        top_10['rank'] = range(1, len(top_10) + 1)
        top_10 = top_10[['rank', 'thresh', 'panic', 'recovery', 'score']]
        
        st.dataframe(top_10, use_container_width=True, hide_index=True)

# TAB 4: WALK-FORWARD VALIDATION
with tabs[3]:
    st.markdown("#### Walk-Forward Validation")
    st.info("Out-of-sample testing to detect overfitting")
    
    if OPTIMIZER_AVAILABLE and st.button("🔬 Run Walk-Forward Validation", use_container_width=True, type="primary"):
        with st.spinner("Running walk-forward analysis..."):
            
            # Create validator
            validator = WalkForwardValidator(
                data,
                train_window=252,  # 1 year
                test_window=63,    # 3 months
                step_size=21       # Monthly
            )
            
            # Helper function for optimization
            def optimize_on_data(train_data):
                bounds = {
                    'thresh': (2.0, 10.0),
                    'panic': (10, 30),
                    'recovery': (20, 60),
                    'allocPrudence': (alloc_prud, alloc_prud),
                    'allocCrash': (alloc_crash, alloc_crash)
                }
                
                objective = create_optimization_objective(
                    BacktestEngine.run_simulation,
                    calculate_metrics_legacy,
                    train_data,
                    profile,
                    confirm
                )
                
                optimizer = BayesianOptimizer(objective, bounds, n_iter=50)
                best_params, best_score, _ = optimizer.optimize()
                
                return best_params, best_score
            
            # Run validation
            results = validator.validate(
                BacktestEngine.run_simulation,
                optimize_on_data,
                calculate_metrics_legacy
            )
            
            if 'error' in results:
                st.error(f"❌ {results['error']}")
            else:
                # Display results
                st.success("✅ Walk-Forward Validation Complete")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Train Score", f"{results['avg_train_score']:.3f}")
                col2.metric("Avg Test Sharpe", f"{results['avg_test_sharpe']:.3f}")
                col3.metric("Degradation", f"{results['degradation']*100:.1f}%",
                           delta_color="inverse")
                col4.metric("Splits", results['n_splits'])
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Overfitting flags
                st.markdown("#### Overfitting Diagnostics")
                
                if len(results['overfitting_flags']) == 0:
                    st.success("✅ No overfitting detected")
                else:
                    st.warning(f"⚠️ {len(results['overfitting_flags'])} flag(s) detected:")
                    for flag in results['overfitting_flags']:
                        st.write(f"• {flag}")
                
                st.info(results['recommended_action'])
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Parameter stability
                st.markdown("#### Parameter Stability")
                
                stability_df = pd.DataFrame(results['param_stability']).T
                stability_df = stability_df.round(3)
                st.dataframe(stability_df, use_container_width=True)
                
                st.caption("**CV** = Coefficient of Variation (lower is more stable)")
                
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                
                # Split results
                st.markdown("#### Results by Split")
                st.dataframe(results['results'], use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        **Walk-Forward Validation Process:**
        
        1. **Split data** into training and testing windows
        2. **Optimize** parameters on training data
        3. **Test** on unseen data
        4. **Roll forward** and repeat
        
        **Metrics Tracked:**
        - Train/Test performance gap
        - Parameter stability across splits
        - Test performance consistency
        
        **Benefits:**
        - Detects overfitting
        - Validates parameter robustness
        - Provides realistic performance expectations
        """)

# TAB 5: ALLOCATION
with tabs[4]:
    st.markdown("#### Allocation Analysis")
    
    # Time in each regime
    alloc_stats = df_res[['alloc_x2', 'alloc_x1', 'regime']].copy()
    regime_counts = alloc_stats['regime'].value_counts()
    total_days = len(alloc_stats)
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    # Regime transitions
    st.markdown("#### Regime Transitions")
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        transition_counts = trades_df['label'].value_counts()
        
        trans_col1, trans_col2, trans_col3 = st.columns(3)
        trans_col1.metric("→ OFFENSIVE", transition_counts.get('OFFENSIVE', 0))
        trans_col2.metric("→ PRUDENCE", transition_counts.get('PRUDENCE', 0))
        trans_col3.metric("→ CRASH", transition_counts.get('CRASH', 0))
        
        st.markdown("**Transition Timeline:**")
        trades_display = trades_df[['date', 'from', 'to', 'label']].copy()
        trades_display['date'] = trades_display['date'].dt.strftime('%Y-%m-%d')
        trades_display.columns = ['Date', 'From', 'To', 'New Regime']
        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No regime transitions occurred")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #606060; font-size: 10px; padding: 16px 0; text-transform: uppercase; letter-spacing: 0.1em;">
    PREDICT. INSTITUTIONAL ANALYTICS v6.0 • BAYESIAN OPTIMIZATION EDITION
</div>
""", unsafe_allow_html=True)
