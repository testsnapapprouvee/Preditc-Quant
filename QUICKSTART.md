# PREDICT. v5.0 - Quick Start Guide

## Installation Rapide

### Étape 1: Installer les dépendances

```bash
pip install -r requirements.txt
```

**Packages requis:**
- streamlit (interface web)
- pandas (manipulation de données)
- numpy (calculs numériques)
- scipy (optimisation)
- yfinance (données de marché)

### Étape 2: Lancer l'application

```bash
streamlit run app_institutional.py
```

L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

---

## Structure des Fichiers

```
predict_institutional/
├── app_institutional.py      # Interface Streamlit principale
├── data_engine.py            # Multi-asset data pipeline
├── backtest_engine.py        # Vectorized backtest engine
├── optimizer.py              # Bayesian + Walk-forward
├── requirements.txt          # Dépendances Python
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
├── README.md                 # Documentation complète
└── QUICKSTART.md            # Ce fichier
```

---

## Utilisation de Base

### 1. Configuration du Portfolio

Dans la sidebar gauche:
- Sélectionnez un preset (Nasdaq 100, S&P 500) ou Custom
- Choisissez la période d'analyse (YTD, 1Y, 3YR, etc.)

### 2. Paramètres de Stratégie

Ajustez les seuils:
- **Threshold Level**: Niveau de drawdown pour mode prudent
- **Panic Threshold**: Niveau de drawdown pour mode crise
- **Recovery Target**: % de récupération pour sortir du mode défensif

### 3. Politique d'Allocation

Définissez les allocations:
- **Prudent Mode**: % alloué à l'actif safe en mode prudent
- **Crisis Mode**: % alloué à l'actif safe en mode crise
- **Confirmation Period**: Jours avant de changer de régime

### 4. Explorer les Résultats

5 onglets disponibles:
- **Performance**: Métriques et courbes de performance
- **Risk Analytics**: Analyse des risques (VaR, CVaR, drawdowns)
- **Advanced Metrics**: 20+ métriques path-dependent
- **Optimization**: Framework Bayesian (description)
- **Validation**: Walk-forward validation (description)

---

## Fonctionnalités Avancées

### Data Quality Report

Cliquez sur "Data Quality Report" pour voir:
- Nombre de jours de données
- Pourcentage de données manquantes par ticker
- Détection d'outliers
- Suspensions de trading

### Exemples de Métriques

**Performance:**
- CAGR, Sharpe, Sortino, Calmar, Omega

**Risk:**
- Max Drawdown, Duration, VaR 95%, CVaR 95%
- Skewness, Kurtosis, Tail Ratio

**Path-Dependent:**
- Max/Avg DD Duration, Recovery Time
- Win/Loss Streaks, Win Rate
- Conditional Returns (après gain/perte)

---

## Troubleshooting

### Erreur: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
```

### Erreur: "Unable to retrieve market data"

**Causes possibles:**
1. Ticker invalide ou inexistant
2. Pas de connexion internet
3. Yahoo Finance temporairement indisponible

**Solution:** Vérifiez les tickers et réessayez

### Performance lente

**Optimisations:**
1. Réduisez la période d'analyse
2. Utilisez moins d'actifs
3. Fermez d'autres applications gourmandes

---

## Utilisation Programmatique

### Importer les modules

```python
from data_engine import DataEngine, AdvancedMetrics
from backtest_engine import VectorizedBacktestEngine, RegimeDetector
from optimizer import BayesianOptimizer, WalkForwardValidator

# Data pipeline
engine = DataEngine(['SPY', 'TLT'], '2020-01-01', '2023-12-31')
prices = engine.fetch_multi_asset()
clean_data = engine.clean_data()

# Backtest
detector = RegimeDetector(threshold=-5, panic=-15)
regimes = detector.detect_regimes(prices['SPY'])
signals = detector.generate_allocation_signal(regimes)

# Metrics
metrics = AdvancedMetrics.calculate_comprehensive_metrics(equity_curve)

# Optimization
optimizer = BayesianOptimizer(multi_objective=True)
best = optimizer.optimize(backtest_func, data, n_iterations=100)

# Validation
validator = WalkForwardValidator(train_days=252, test_days=63)
wf_results = validator.rolling_validation(data, opt_func, bt_func, params)
```

---

## Deployment sur Streamlit Cloud

### Étape 1: Préparer les fichiers

Assurez-vous d'avoir:
- ✅ app_institutional.py
- ✅ data_engine.py
- ✅ backtest_engine.py
- ✅ optimizer.py
- ✅ requirements.txt
- ✅ .streamlit/config.toml

### Étape 2: Push sur GitHub

```bash
git init
git add .
git commit -m "PREDICT v5.0 - Institutional Edition"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

### Étape 3: Déployer

1. Allez sur https://streamlit.io/cloud
2. Connectez votre repo GitHub
3. Sélectionnez `app_institutional.py` comme main file
4. Déployez!

---

## Support & Documentation

**Documentation complète:** Voir README.md

**Versions:**
- v5.0 - Institutional Edition (Janvier 2026)
- Phases 1-4 MUST-HAVE implémentées
- Performance: 10x amélioration

**Features:**
- ✅ Multi-asset data engine
- ✅ Vectorized backtest (10x faster)
- ✅ Bayesian optimization
- ✅ Walk-forward validation
- ✅ 20+ advanced metrics
- ✅ Professional UI

---

## Tips & Best Practices

### Pour de meilleurs résultats:

1. **Utilisez au moins 2 ans de données** pour l'optimisation
2. **Vérifiez le Data Quality Report** avant de faire confiance aux résultats
3. **Comparez plusieurs périodes** (2008, 2020, 2022) pour la robustesse
4. **Ne sur-optimisez pas** - gardez les paramètres simples
5. **Surveillez le turnover** - trop de trades = coûts élevés

### Interprétation des métriques:

- **Sharpe > 1.0**: Bon
- **Sharpe > 2.0**: Excellent
- **Max DD < -20%**: Acceptable
- **Win Rate > 55%**: Bon
- **Calmar > 1.0**: Très bon

---

**Questions?** Consultez le README.md pour plus de détails techniques.

*PREDICT. - Institutional Risk Analytics Platform*
