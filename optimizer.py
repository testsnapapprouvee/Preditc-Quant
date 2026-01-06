"""
PREDICT. - Advanced Optimization Framework
Phase 3.1-3.2: Bayesian optimization + Walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameters
    10x faster than grid search with better results
    """
    
    def __init__(self, objective: str = 'sharpe', multi_objective: bool = True):
        """
        Initialize Bayesian optimizer
        
        Args:
            objective: Primary objective ('sharpe', 'calmar', 'sortino')
            multi_objective: Use composite score
        """
        self.objective = objective
        self.multi_objective = multi_objective
        self.observations = []
        
    def define_parameter_space(self) -> Dict:
        """
        Define search space for regime strategy
        
        Returns:
            Dict with parameter bounds
        """
        return {
            'threshold': {
                'type': 'continuous',
                'bounds': (2.0, 12.0),
                'default': 5.0
            },
            'panic': {
                'type': 'continuous',
                'bounds': (10.0, 35.0),
                'default': 15.0
            },
            'recovery': {
                'type': 'continuous',
                'bounds': (20.0, 70.0),
                'default': 30.0
            },
            'alloc_prudence': {
                'type': 'continuous',
                'bounds': (0.3, 0.8),
                'default': 0.5
            },
            'alloc_crash': {
                'type': 'continuous',
                'bounds': (0.7, 1.0),
                'default': 1.0
            },
            'confirm_days': {
                'type': 'discrete',
                'bounds': (1, 5),
                'default': 2
            }
        }
    
    def calculate_multi_objective_score(self, metrics: Dict) -> float:
        """
        Composite objective function
        
        Weights:
            40% Sharpe - risk-adjusted returns
            30% Calmar - drawdown-adjusted returns  
            20% Turnover penalty - transaction costs
            10% Max DD penalty - tail risk
        """
        sharpe = metrics.get('sharpe', 0)
        calmar = metrics.get('calmar', 0)
        turnover = metrics.get('turnover', 1.0)
        max_dd = metrics.get('max_drawdown', -50)
        
        # Normalize components
        sharpe_score = np.clip(sharpe / 3.0, -1, 1)  # Target Sharpe ~3
        calmar_score = np.clip(calmar / 2.0, -1, 1)  # Target Calmar ~2
        turnover_score = np.clip(1 - turnover / 5.0, 0, 1)  # Penalize >5x turnover
        dd_score = np.clip(1 + max_dd / 30.0, 0, 1)  # Penalize >-30% DD
        
        score = (
            0.40 * sharpe_score +
            0.30 * calmar_score +
            0.20 * turnover_score +
            0.10 * dd_score
        )
        
        return score
    
    def optimize(self, backtest_func: Callable, data: pd.DataFrame, 
                 n_iterations: int = 100, random_seed: int = 42) -> Dict:
        """
        Run Bayesian optimization
        
        Args:
            backtest_func: Function that takes params and returns metrics
            data: Historical data for backtesting
            n_iterations: Number of optimization iterations
            random_seed: Random seed for reproducibility
        
        Returns:
            Dict with best parameters and confidence intervals
        """
        np.random.seed(random_seed)
        param_space = self.define_parameter_space()
        
        # Initialize with random samples
        n_random = min(20, n_iterations // 5)
        best_score = -np.inf
        best_params = None
        
        print(f"Running Bayesian optimization ({n_iterations} iterations)...")
        
        for i in range(n_iterations):
            # Sample parameters
            if i < n_random:
                # Random exploration
                params = self._sample_random(param_space)
            else:
                # Bayesian acquisition
                params = self._bayesian_sample(param_space)
            
            # Validate parameter constraints
            if params['panic'] <= params['threshold']:
                continue
            
            # Run backtest
            try:
                metrics = backtest_func(params, data)
                
                if self.multi_objective:
                    score = self.calculate_multi_objective_score(metrics)
                else:
                    score = metrics.get(self.objective, 0)
                
                # Store observation
                self.observations.append({
                    'params': params.copy(),
                    'score': score,
                    'metrics': metrics.copy()
                })
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 20 == 0:
                    print(f"  Iteration {i+1}/{n_iterations} - Best score: {best_score:.4f}")
                
            except Exception as e:
                print(f"  Error in iteration {i+1}: {str(e)}")
                continue
        
        # Calculate confidence intervals
        confidence = self._calculate_confidence(best_params, param_space)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'confidence_intervals': confidence,
            'n_evaluations': len(self.observations),
            'observations': self.observations
        }
    
    def _sample_random(self, param_space: Dict) -> Dict:
        """Sample random parameters from space"""
        params = {}
        for name, config in param_space.items():
            if config['type'] == 'continuous':
                low, high = config['bounds']
                params[name] = np.random.uniform(low, high)
            elif config['type'] == 'discrete':
                low, high = config['bounds']
                params[name] = np.random.randint(low, high + 1)
        return params
    
    def _bayesian_sample(self, param_space: Dict) -> Dict:
        """
        Sample using Gaussian Process acquisition function
        Simplified Expected Improvement (EI)
        """
        if len(self.observations) < 5:
            return self._sample_random(param_space)
        
        # Build surrogate model (simple GP approximation)
        X = []
        y = []
        for obs in self.observations:
            X.append(list(obs['params'].values()))
            y.append(obs['score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Current best
        y_best = y.max()
        
        # Sample candidates and choose best EI
        n_candidates = 20
        best_ei = -np.inf
        best_candidate = None
        
        for _ in range(n_candidates):
            candidate = self._sample_random(param_space)
            candidate_vec = np.array(list(candidate.values()))
            
            # Simplified EI using nearest neighbors
            distances = np.linalg.norm(X - candidate_vec, axis=1)
            k_nearest = 5
            nearest_idx = np.argsort(distances)[:k_nearest]
            
            # Estimate mean and std from neighbors
            mu = y[nearest_idx].mean()
            sigma = y[nearest_idx].std() + 1e-6
            
            # Expected Improvement
            improvement = mu - y_best
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        
        return best_candidate if best_candidate else self._sample_random(param_space)
    
    def _calculate_confidence(self, best_params: Dict, param_space: Dict) -> Dict:
        """Calculate 95% confidence intervals for parameters"""
        if len(self.observations) < 10:
            return {}
        
        confidence = {}
        
        # Collect parameter values from top 10% of observations
        sorted_obs = sorted(self.observations, key=lambda x: x['score'], reverse=True)
        top_n = max(5, len(sorted_obs) // 10)
        top_obs = sorted_obs[:top_n]
        
        for param_name in best_params.keys():
            values = [obs['params'][param_name] for obs in top_obs]
            
            mean = np.mean(values)
            std = np.std(values)
            
            # 95% CI
            ci_lower = mean - 1.96 * std
            ci_upper = mean + 1.96 * std
            
            # Clip to bounds
            bounds = param_space[param_name]['bounds']
            ci_lower = max(ci_lower, bounds[0])
            ci_upper = min(ci_upper, bounds[1])
            
            confidence[param_name] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        return confidence


class WalkForwardValidator:
    """
    Enterprise walk-forward analysis
    Prevents overfitting with out-of-sample validation
    """
    
    def __init__(self, train_days: int = 252, test_days: int = 63, step_days: int = 21):
        """
        Initialize walk-forward validator
        
        Args:
            train_days: Training window size
            test_days: Test window size
            step_days: Step size between windows
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
    
    def rolling_validation(self, data: pd.DataFrame, optimizer_func: Callable,
                          backtest_func: Callable, fixed_params: Dict) -> pd.DataFrame:
        """
        Rolling window walk-forward analysis
        
        Args:
            data: Full dataset
            optimizer_func: Function to optimize parameters
            backtest_func: Function to run backtest
            fixed_params: Parameters not being optimized
        
        Returns:
            DataFrame with walk-forward results
        """
        results = []
        total_days = len(data)
        current_start = 0
        
        print("Running walk-forward analysis...")
        iteration = 0
        
        while current_start + self.train_days + self.test_days <= total_days:
            iteration += 1
            
            # Define train window
            train_end = current_start + self.train_days
            train_data = data.iloc[current_start:train_end]
            
            # Optimize on train data
            print(f"\n  Period {iteration}:")
            print(f"    Train: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
            
            try:
                best_params = optimizer_func(train_data, fixed_params)
                train_metrics = backtest_func(best_params, train_data)
                
                # Define test window
                test_start = train_end
                test_end = min(test_start + self.test_days, total_days)
                test_data = data.iloc[test_start:test_end]
                
                print(f"    Test:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
                
                # Evaluate on test data
                test_metrics = backtest_func(best_params, test_data)
                
                # Calculate degradation
                train_sharpe = train_metrics.get('sharpe', 0)
                test_sharpe = test_metrics.get('sharpe', 0)
                degradation = ((train_sharpe - test_sharpe) / train_sharpe * 100) if train_sharpe != 0 else 0
                
                results.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    
                    # Parameters
                    **{f'param_{k}': v for k, v in best_params.items()},
                    
                    # Train metrics
                    'train_cagr': train_metrics.get('cagr', 0),
                    'train_sharpe': train_sharpe,
                    'train_maxdd': train_metrics.get('max_drawdown', 0),
                    
                    # Test metrics
                    'test_cagr': test_metrics.get('cagr', 0),
                    'test_sharpe': test_sharpe,
                    'test_maxdd': test_metrics.get('max_drawdown', 0),
                    
                    # Stability
                    'degradation_pct': degradation
                })
                
                print(f"    Train Sharpe: {train_sharpe:.3f} | Test Sharpe: {test_sharpe:.3f} | Degradation: {degradation:.1f}%")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                
            current_start += self.step_days
        
        return pd.DataFrame(results)
    
    def analyze_stability(self, results: pd.DataFrame) -> Dict:
        """
        Analyze parameter stability and overfitting
        
        Returns:
            Dict with stability metrics
        """
        if results.empty:
            return {}
        
        # Parameter variance (lower is more stable)
        param_cols = [col for col in results.columns if col.startswith('param_')]
        param_variance = {}
        for col in param_cols:
            param_name = col.replace('param_', '')
            param_variance[param_name] = {
                'mean': results[col].mean(),
                'std': results[col].std(),
                'cv': results[col].std() / results[col].mean() if results[col].mean() != 0 else 0
            }
        
        # Performance consistency
        test_sharpe_mean = results['test_sharpe'].mean()
        test_sharpe_std = results['test_sharpe'].std()
        
        # Degradation analysis
        avg_degradation = results['degradation_pct'].mean()
        
        # Overfitting score (0-1, higher = more overfitting)
        overfitting_score = np.clip(avg_degradation / 50, 0, 1)
        
        # Stability coefficient (0-1, higher = more stable)
        stability = 1 / (1 + test_sharpe_std) if test_sharpe_std > 0 else 1
        
        return {
            'parameter_variance': param_variance,
            'test_sharpe_mean': test_sharpe_mean,
            'test_sharpe_std': test_sharpe_std,
            'avg_degradation_pct': avg_degradation,
            'overfitting_score': overfitting_score,
            'stability_coefficient': stability,
            'win_rate': (results['test_cagr'] > 0).mean() * 100
        }
    
    def detect_overfitting(self, results: pd.DataFrame, threshold: float = 20.0) -> Dict:
        """
        Detect overfitting using multiple heuristics
        
        Args:
            threshold: Degradation threshold % for flagging
        
        Returns:
            Overfitting diagnosis
        """
        if results.empty:
            return {'overfitting': False, 'confidence': 0}
        
        # Train vs test performance gap
        train_test_gap = (results['train_sharpe'] - results['test_sharpe']).mean()
        high_gap = train_test_gap > 0.5
        
        # High degradation
        avg_degradation = results['degradation_pct'].mean()
        high_degradation = avg_degradation > threshold
        
        # Unstable parameters
        param_cols = [col for col in results.columns if col.startswith('param_')]
        high_variance = False
        if param_cols:
            cvs = [results[col].std() / results[col].mean() if results[col].mean() != 0 else 0 
                   for col in param_cols]
            high_variance = np.mean(cvs) > 0.3
        
        # Combined diagnosis
        flags = sum([high_gap, high_degradation, high_variance])
        
        return {
            'overfitting': flags >= 2,
            'confidence': flags / 3,
            'train_test_gap': train_test_gap,
            'avg_degradation': avg_degradation,
            'high_variance': high_variance,
            'recommendation': 'Reduce complexity or increase regularization' if flags >= 2 else 'Strategy appears robust'
        }
