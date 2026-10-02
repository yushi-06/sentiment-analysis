#!/usr/bin/env python3
"""
Quick verification that bias fix is still working
Run this anytime to check if your model is working correctly
"""

from adaptive_sentiment import AdaptiveSentimentAnalyzer

def quick_verification():
    print("🔍 QUICK BIAS CHECK")
    print("="*25)
    
    analyzer = AdaptiveSentimentAnalyzer()
    
    # Test the most problematic case from your original issue
    test_cases = [
        ("this is a bad product", "Negative"),
        ("terrible quality", "Negative"), 
        ("I love this", "Positive"),
        ("it's okay", "Neutral")
    ]
    
    all_correct = True
    
    for text, expected in test_cases:
        prediction, score, _ = analyzer.predict_sentiment(text)
        is_correct = prediction == expected
        status = "✅" if is_correct else "❌"
        
        print(f"{status} '{text}' → {prediction} (expected: {expected})")
        
        if not is_correct:
            all_correct = False
    
    print(f"\n{'🎉 BIAS FIX WORKING!' if all_correct else '⚠️ BIAS DETECTED!'}")
    return all_correct

if __name__ == "__main__":
    quick_verification()
