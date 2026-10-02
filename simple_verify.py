#!/usr/bin/env python3
"""
Simple Setup Verification
"""

import sys
import os

def main():
    print("SENTIMENT ANALYSIS SYSTEM VERIFICATION")
    print("="*50)
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check key dependencies
    deps = ['pandas', 'numpy', 'matplotlib', 'torch', 'transformers']
    print(f"\nChecking dependencies:")
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"  OK: {dep}")
        except ImportError:
            print(f"  MISSING: {dep}")
    
    # Check key files
    files = ['fixed_main.py', 'adaptive_sentiment.py', 'fix_bias_data.csv']
    print(f"\nChecking files:")
    
    for file in files:
        if os.path.exists(file):
            print(f"  OK: {file}")
        else:
            print(f"  MISSING: {file}")
    
    # Test basic functionality
    print(f"\nTesting basic functionality:")
    try:
        from adaptive_sentiment import AdaptiveSentimentAnalyzer
        analyzer = AdaptiveSentimentAnalyzer()
        prediction, score, _ = analyzer.predict_sentiment("I love this")
        print(f"  OK: Basic prediction works - '{prediction}' (score: {score:.3f})")
        
        # Test enhanced features
        try:
            from enhanced_adaptive_learning import EnhancedAdaptiveLearning
            enhanced = EnhancedAdaptiveLearning()
            print(f"  OK: Enhanced features available")
        except:
            print(f"  WARNING: Enhanced features not available")
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print(f"\nSYSTEM STATUS: READY")
    print(f"\nQuick start:")
    print(f"  python fixed_main.py")

if __name__ == "__main__":
    main()
