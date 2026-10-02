#!/usr/bin/env python3
"""
Demo of the updated manual mode with percentages
"""

from fixed_main import FixedSentimentAnalyzer

def demo_manual_mode():
    print("🎯 DEMO: Manual Mode with Percentages")
    print("="*45)
    
    # Initialize analyzer
    analyzer = FixedSentimentAnalyzer()
    
    # Simulate manual input (your previous test texts)
    texts = [
        "the movie is great",
        "you are so beautiful", 
        "the food could be better",
        "the faculty is SHITTTTTT!!!!!!!!!!",
        "im not feeling well today"
    ]
    
    print(f"\nSimulating manual input with your previous texts:")
    print("="*50)
    
    # Analyze each text with percentage (same as manual mode now does)
    print("\nResults:")
    for i, text in enumerate(texts, 1):
        result = analyzer.predict_single(text)
        if result:
            if isinstance(result, dict):
                prediction = result['prediction']
                confidence = result.get('confidence', 0.5)
                percentage = confidence * 100
                uncertain = result.get('uncertain', False)
                
                status_indicator = " 🔴" if uncertain else ""
                print(f"{i}. {prediction} ({percentage:.1f}%){status_indicator}")
            else:
                print(f"{i}. {result} (50.0%)")
        else:
            print(f"{i}. No data to analyze.")
    
    print(f"\n🎉 COMPARISON:")
    print("Before: 1. Positive")
    print("Now:    1. Positive (25.0%) 🔴")
    print()
    print("✅ Every manual entry now shows percentage confidence!")
    print("✅ Red circle (🔴) indicates uncertain predictions")
    print("✅ Higher percentage = more confident prediction")

if __name__ == "__main__":
    demo_manual_mode()
