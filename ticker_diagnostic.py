"""
PREDICT. - Ticker Diagnostic Tool
Test if tickers are valid and have sufficient data
"""

import yfinance as yf
from datetime import datetime, timedelta
import sys

def test_ticker(ticker, start_date, end_date):
    """Test if a ticker is valid and has data"""
    print(f"\n{'='*60}")
    print(f"Testing: {ticker}")
    print(f"{'='*60}")
    
    try:
        # Fetch data
        print(f"Downloading data from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"❌ No data returned for {ticker}")
            return False
        
        # Check columns
        print(f"\n✓ Data downloaded: {len(df)} days")
        print(f"Columns: {list(df.columns)}")
        
        # Check for Adj Close
        if 'Adj Close' in df.columns:
            prices = df['Adj Close']
            print(f"✓ Adj Close available")
        elif 'Close' in df.columns:
            prices = df['Close']
            print(f"⚠ Using Close (no Adj Close)")
        else:
            print(f"❌ No price data found")
            return False
        
        # Statistics
        print(f"\nPrice Statistics:")
        print(f"  First: {prices.iloc[0]:.2f} ({df.index[0].strftime('%Y-%m-%d')})")
        print(f"  Last:  {prices.iloc[-1]:.2f} ({df.index[-1].strftime('%Y-%m-%d')})")
        print(f"  Min:   {prices.min():.2f}")
        print(f"  Max:   {prices.max():.2f}")
        
        # Check for missing data
        missing = prices.isna().sum()
        missing_pct = (missing / len(prices)) * 100
        print(f"\nData Quality:")
        print(f"  Missing days: {missing} ({missing_pct:.1f}%)")
        
        if missing_pct > 10:
            print(f"  ⚠ High missing data percentage")
        elif missing_pct > 5:
            print(f"  ⚠ Moderate missing data")
        else:
            print(f"  ✓ Good data quality")
        
        # Check volume
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].mean()
            print(f"  Avg Volume: {avg_volume:,.0f}")
            
            if avg_volume < 100000:
                print(f"  ⚠ Low liquidity (volume < 100k)")
            else:
                print(f"  ✓ Good liquidity")
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        print(f"\nReturn Statistics:")
        print(f"  Daily Mean: {returns.mean()*100:.3f}%")
        print(f"  Daily Std:  {returns.std()*100:.2f}%")
        print(f"  Total:      {((prices.iloc[-1] / prices.iloc[0]) - 1)*100:.1f}%")
        
        # Check for outliers
        outliers = ((returns - returns.mean()).abs() > 5 * returns.std()).sum()
        print(f"\nOutliers (>5σ): {outliers}")
        
        if outliers > len(returns) * 0.01:
            print(f"  ⚠ High number of outliers")
        
        print(f"\n{'='*60}")
        print(f"✅ {ticker} is VALID and has sufficient data")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing {ticker}:")
        print(f"   {str(e)}")
        print(f"{'='*60}")
        return False


def test_ticker_pair(ticker1, ticker2, start_date, end_date):
    """Test a pair of tickers"""
    print(f"\n{'#'*60}")
    print(f"# TESTING TICKER PAIR")
    print(f"# Risk Asset: {ticker1}")
    print(f"# Safe Asset: {ticker2}")
    print(f"{'#'*60}")
    
    result1 = test_ticker(ticker1, start_date, end_date)
    result2 = test_ticker(ticker2, start_date, end_date)
    
    print(f"\n{'='*60}")
    print(f"PAIR TEST RESULTS")
    print(f"{'='*60}")
    print(f"{ticker1}: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"{ticker2}: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 and result2:
        print(f"\n✅ PAIR IS VALID - Ready to use in PREDICT.")
    else:
        print(f"\n❌ PAIR HAS ISSUES - Consider different tickers")
    
    print(f"{'='*60}\n")
    
    return result1 and result2


if __name__ == "__main__":
    # Default test configuration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    print("\n" + "="*60)
    print("PREDICT. - Ticker Diagnostic Tool")
    print("="*60)
    
    if len(sys.argv) >= 3:
        # Test specific pair from command line
        ticker1 = sys.argv[1]
        ticker2 = sys.argv[2]
        test_ticker_pair(ticker1, ticker2, start_date, end_date)
    else:
        # Test common presets
        print("\nTesting common ticker pairs...\n")
        
        pairs = [
            ("SSO", "SPY", "S&P 500 2x / SPY"),
            ("QLD", "QQQ", "Nasdaq 100 2x / QQQ"),
            ("XLK", "TLT", "Tech / Bonds"),
            ("SPY", "TLT", "S&P 500 / Bonds"),
            ("LQQ.PA", "PUST.PA", "Nasdaq 100 (EU)"),
        ]
        
        results = {}
        
        for ticker1, ticker2, name in pairs:
            print(f"\n{'#'*60}")
            print(f"# {name}")
            print(f"{'#'*60}")
            result = test_ticker_pair(ticker1, ticker2, start_date, end_date)
            results[name] = result
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY - RECOMMENDED PRESETS")
        print("="*60)
        
        for name, result in results.items():
            status = "✅ RECOMMENDED" if result else "❌ NOT RECOMMENDED"
            print(f"{status}: {name}")
        
        print("="*60)
        print("\nUsage: python ticker_diagnostic.py TICKER1 TICKER2")
        print("Example: python ticker_diagnostic.py SSO SPY")
        print("="*60 + "\n")
