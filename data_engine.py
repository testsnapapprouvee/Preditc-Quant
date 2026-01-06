"""
PREDICT. - Multi-Asset Data Engine
Phase 1.1: Enterprise-grade data pipeline with institutional validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataEngine:
    """Enterprise-grade multi-asset data pipeline"""
    
    def __init__(self, tickers: List[str], start: datetime, end: datetime):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.universe = {}
        self.quality_report = {}
        
    def fetch_multi_asset(self) -> pd.DataFrame:
        """Download and validate multi-asset universe"""
        data_dict = {}
        
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start, end=self.end, progress=False)
                
                if not df.empty:
                    # Always use Adj Close for accurate returns
                    if 'Adj Close' in df.columns:
                        series = df['Adj Close']
                    elif isinstance(df.columns, pd.MultiIndex):
                        series = df[('Adj Close', ticker)]
                    else:
                        series = df['Close']
                    
                    # Store volume for quality checks
                    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(index=df.index)
                    
                    data_dict[ticker] = {
                        'price': series,
                        'volume': volume
                    }
                    
            except Exception as e:
                print(f"Error fetching {ticker}: {str(e)}")
                continue
        
        if not data_dict:
            return pd.DataFrame()
        
        # Combine prices into single DataFrame
        prices = pd.DataFrame({ticker: data['price'] for ticker, data in data_dict.items()})
        volumes = pd.DataFrame({ticker: data['volume'] for ticker, data in data_dict.items()})
        
        self.universe = {'prices': prices, 'volumes': volumes}
        return prices
    
    def clean_data(self) -> pd.DataFrame:
        """Institutional data cleaning with quality metrics"""
        if not self.universe:
            return pd.DataFrame()
        
        prices = self.universe['prices'].copy()
        volumes = self.universe['volumes'].copy()
        
        initial_rows = len(prices)
        quality_metrics = {}
        
        for ticker in prices.columns:
            # Quality checks
            total_days = len(prices[ticker])
            missing_days = prices[ticker].isna().sum()
            zero_volume_days = (volumes[ticker] == 0).sum() if ticker in volumes.columns else 0
            
            # Detect outliers (>5 sigma daily moves)
            returns = prices[ticker].pct_change()
            mean_ret = returns.mean()
            std_ret = returns.std()
            outliers = ((returns - mean_ret).abs() > 5 * std_ret).sum()
            
            # Detect suspensions (5+ consecutive missing days)
            suspensions = self._detect_suspensions(prices[ticker])
            
            quality_metrics[ticker] = {
                'total_days': total_days,
                'missing_days': missing_days,
                'missing_pct': (missing_days / total_days * 100) if total_days > 0 else 0,
                'zero_volume_days': zero_volume_days,
                'outliers': outliers,
                'suspensions': suspensions
            }
        
        # Forward fill missing data (institutional standard)
        prices = prices.ffill()
        
        # Remove rows where all assets are NaN
        prices = prices.dropna(how='all')
        
        # Final quality report
        final_rows = len(prices)
        self.quality_report = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': initial_rows - final_rows,
            'tickers': quality_metrics
        }
        
        self.universe['prices_clean'] = prices
        return prices
    
    def _detect_suspensions(self, series: pd.Series, min_consecutive: int = 5) -> int:
        """Detect trading suspensions (consecutive missing data)"""
        is_missing = series.isna()
        suspensions = 0
        current_streak = 0
        
        for missing in is_missing:
            if missing:
                current_streak += 1
            else:
                if current_streak >= min_consecutive:
                    suspensions += 1
                current_streak = 0
        
        return suspensions
    
    def calculate_returns(self) -> Dict[str, pd.DataFrame]:
        """Return calculation engine with multiple return types"""
        if 'prices_clean' not in self.universe:
            self.clean_data()
        
        prices = self.universe['prices_clean']
        
        # Log returns (for analysis)
        log_returns = np.log(prices / prices.shift(1))
        
        # Arithmetic returns (for reporting)
        arithmetic_returns = prices.pct_change()
        
        # Excess returns vs first asset (benchmark)
        benchmark = prices.iloc[:, 0]
        excess_returns = prices.div(benchmark, axis=0).pct_change()
        
        return {
            'log_returns': log_returns,
            'arithmetic_returns': arithmetic_returns,
            'excess_returns': excess_returns,
            'prices': prices
        }
    
    def get_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        if not self.quality_report:
            self.clean_data()
        
        return self.quality_report
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate data meets institutional standards"""
        report = self.get_quality_report()
        
        validation = {}
        for ticker, metrics in report['tickers'].items():
            # Institutional thresholds
            passes_missing = metrics['missing_pct'] < 5.0  # <5% missing
            passes_outliers = metrics['outliers'] < len(self.universe['prices_clean']) * 0.01  # <1% outliers
            passes_suspensions = metrics['suspensions'] == 0  # No suspensions
            
            validation[ticker] = {
                'passes_quality': passes_missing and passes_outliers and passes_suspensions,
                'missing_check': passes_missing,
                'outlier_check': passes_outliers,
                'suspension_check': passes_suspensions
            }
        
        return validation


class AdvancedMetrics:
    """Path-dependent and advanced performance metrics (Phase 4.1)"""
    
    @staticmethod
    def calculate_comprehensive_metrics(series: pd.Series) -> Dict:
        """Calculate 20+ institutional metrics"""
        if series.empty or len(series) < 2:
            return {}
        
        # Basic returns
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        days = len(series)
        years = days / 252
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) if years > 0 else 0
        
        # Drawdown analysis
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        max_dd = drawdown.min()
        
        # Drawdown duration analysis
        dd_duration_stats = AdvancedMetrics._analyze_drawdown_durations(drawdown)
        
        # Returns analysis
        returns = series.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else vol
        
        # Risk metrics
        sharpe = cagr / vol if vol > 0 else 0
        sortino = cagr / downside_std if downside_std > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Tail risk
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        var_99 = returns.quantile(0.01)
        cvar_99 = returns[returns <= var_99].mean()
        
        # Higher moments
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        # Advanced ratios
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns <= 0].sum())
        omega = gains / losses if losses > 0 else 0
        
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        upi = cagr / ulcer_index if ulcer_index > 0 else 0
        
        # Streak analysis
        streak_stats = AdvancedMetrics._analyze_streaks(returns)
        
        # Conditional returns
        conditional_stats = AdvancedMetrics._conditional_returns(returns)
        
        return {
            # Core metrics
            'total_return': total_return * 100,
            'cagr': cagr * 100,
            'volatility': vol * 100,
            
            # Drawdown metrics
            'max_drawdown': max_dd * 100,
            'avg_drawdown': dd_duration_stats['avg_dd'] * 100,
            'max_dd_duration': dd_duration_stats['max_duration'],
            'avg_dd_duration': dd_duration_stats['avg_duration'],
            'recovery_time': dd_duration_stats['avg_recovery'],
            'underwater_pct': dd_duration_stats['underwater_pct'],
            
            # Risk-adjusted returns
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'omega': omega,
            'upi': upi,
            
            # Tail risk
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'var_99': var_99 * 100,
            'cvar_99': cvar_99 * 100,
            'tail_ratio': tail_ratio,
            
            # Distribution
            'skewness': skew,
            'kurtosis': kurt,
            
            # Streaks
            'max_win_streak': streak_stats['max_win_streak'],
            'max_loss_streak': streak_stats['max_loss_streak'],
            'avg_win_streak': streak_stats['avg_win_streak'],
            'avg_loss_streak': streak_stats['avg_loss_streak'],
            'win_rate': streak_stats['win_rate'],
            
            # Conditional returns
            'return_after_gain': conditional_stats['after_gain'],
            'return_after_loss': conditional_stats['after_loss'],
            'up_capture': conditional_stats['up_capture'],
            'down_capture': conditional_stats['down_capture']
        }
    
    @staticmethod
    def _analyze_drawdown_durations(drawdown: pd.Series) -> Dict:
        """Comprehensive drawdown duration analysis"""
        durations = []
        recoveries = []
        current_dd_length = 0
        in_drawdown = False
        dd_start = None
        underwater_days = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                underwater_days += 1
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                current_dd_length += 1
            else:
                if in_drawdown:
                    durations.append(current_dd_length)
                    if dd_start is not None:
                        recoveries.append(i - dd_start)
                    in_drawdown = False
                    current_dd_length = 0
        
        return {
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'avg_recovery': np.mean(recoveries) if recoveries else 0,
            'num_drawdowns': len(durations),
            'underwater_pct': (underwater_days / len(drawdown)) * 100,
            'avg_dd': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        }
    
    @staticmethod
    def _analyze_streaks(returns: pd.Series) -> Dict:
        """Win/loss streak analysis"""
        wins = returns > 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        win_streaks = []
        loss_streaks = []
        
        for win in wins:
            if win:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
            
            max_win_streak = max(max_win_streak, current_win_streak)
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'win_rate': (wins.sum() / len(wins)) * 100
        }
    
    @staticmethod
    def _conditional_returns(returns: pd.Series) -> Dict:
        """Context-dependent performance analysis"""
        # Returns after large gains (>1%)
        after_gain = returns[returns.shift(1) > 0.01].mean() if (returns.shift(1) > 0.01).any() else 0
        
        # Returns after large losses (<-1%)
        after_loss = returns[returns.shift(1) < -0.01].mean() if (returns.shift(1) < -0.01).any() else 0
        
        # Up/down capture (vs median)
        median_return = returns.median()
        up_days = returns > median_return
        down_days = returns <= median_return
        
        up_capture = returns[up_days].mean() if up_days.any() else 0
        down_capture = returns[down_days].mean() if down_days.any() else 0
        
        return {
            'after_gain': after_gain * 100,
            'after_loss': after_loss * 100,
            'up_capture': up_capture * 100,
            'down_capture': down_capture * 100
        }
