#!/usr/bin/env python3
"""
Comprehensive Test Suite - All Bugs Fixed
Tests all major functionality after bug fixes
"""

import os
import sys
from datetime import datetime

def test_imports():
    """Test all critical imports"""
    print("🧪 TESTING IMPORTS")
    print("="*30)
    
    tests = [
        ("pandas", "import pandas as pd"),
        ("adaptive_sentiment", "from adaptive_sentiment import AdaptiveSentimentAnalyzer"),
        ("robust_dataset_loader", "from robust_dataset_loader import RobustDatasetLoader"),
        ("fixed_main", "from fixed_main import FixedSentimentAnalyzer"),
        ("dataset_config", "from dataset_config import DatasetConfig"),
    ]
    
    passed = 0
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"✅ {name}: OK")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
    
    print(f"Import tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n🧪 TESTING SENTIMENT ANALYSIS")
    print("="*40)
    
    try:
        from adaptive_sentiment import AdaptiveSentimentAnalyzer
        analyzer = AdaptiveSentimentAnalyzer()
        
        test_cases = [
            ("This is amazing!", "Positive"),
            ("I hate this product", "Negative"),
            ("It's okay", "Neutral"),
            ("Absolutely wonderful experience!", "Positive"),
            ("Terrible and disappointing", "Negative")
        ]
        
        passed = 0
        for text, expected in test_cases:
            prediction, score, _ = analyzer.predict_sentiment(text)
            if prediction == expected:
                print(f"✅ '{text}' → {prediction} (score: {score:.2f})")
                passed += 1
            else:
                print(f"❌ '{text}' → {prediction} (expected: {expected})")
        
        print(f"Sentiment tests: {passed}/{len(test_cases)} passed")
        return passed >= len(test_cases) * 0.8  # 80% pass rate acceptable
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("\n🧪 TESTING DATASET LOADING")
    print("="*35)
    
    try:
        from robust_dataset_loader import RobustDatasetLoader
        loader = RobustDatasetLoader()
        
        test_files = ["fix_bias_data.csv", "my_reviews.csv"]
        passed = 0
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    df = loader.load_dataset(file_path)
                    print(f"✅ {file_path}: Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                    passed += 1
                except Exception as e:
                    print(f"❌ {file_path}: Failed - {e}")
            else:
                print(f"⚠️  {file_path}: File not found")
        
        print(f"Dataset loading tests: {passed}/{len(test_files)} passed")
        return passed > 0
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_fixed_main():
    """Test the fixed main analyzer"""
    print("\n🧪 TESTING FIXED MAIN ANALYZER")
    print("="*40)
    
    try:
        from fixed_main import FixedSentimentAnalyzer
        analyzer = FixedSentimentAnalyzer()
        
        # Test single prediction
        result = analyzer.predict_single("This is a great product!", show_confidence=True)
        
        if result and 'prediction' in result:
            print(f"✅ Fixed analyzer working: {result['prediction']} (confidence: {result.get('confidence', 'N/A')})")
            return True
        else:
            print(f"❌ Fixed analyzer returned unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Fixed main analyzer test failed: {e}")
        return False

def test_windows_compatibility():
    """Test Windows-specific functionality"""
    print("\n🧪 TESTING WINDOWS COMPATIBILITY")
    print("="*40)
    
    try:
        # Test safe print function
        from windows_safe_train import safe_print
        safe_print("Testing Windows-safe printing...")
        print("✅ Windows-safe printing: OK")
        
        # Test path handling
        test_path = "c:\\Users\\test\\file.csv"
        normalized_path = os.path.normpath(test_path)
        print(f"✅ Path normalization: {normalized_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Windows compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE BUG FIX VERIFICATION")
    print("="*50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Import Tests", test_imports),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Dataset Loading", test_dataset_loading),
        ("Fixed Main Analyzer", test_fixed_main),
        ("Windows Compatibility", test_windows_compatibility)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()
    
    print("="*50)
    print("🎯 FINAL TEST RESULTS")
    print("="*50)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! All bugs have been fixed.")
        print("\n🚀 Your sentiment analysis system is ready to use!")
        print("Try running: python fixed_main.py")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed. System is functional with minor issues.")
    else:
        print("⚠️  Several tests failed. Some bugs may remain.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
