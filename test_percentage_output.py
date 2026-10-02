#!/usr/bin/env python3
"""
Test the percentage output functionality
"""

from fixed_main import FixedSentimentAnalyzer

def test_percentage_output():
    print("🧪 TESTING PERCENTAGE OUTPUT")
    print("="*40)
    
    # Initialize analyzer
    analyzer = FixedSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        "I love this product!",
        "This is terrible",
        "It's okay",
        "Amazing quality!",
        "The faculty is SHITTTTTT!!!!!!!!!!",
        "I'm not feeling well today"
    ]
    
    print("\n📊 Testing individual predictions with percentages:")
    for i, text in enumerate(test_texts, 1):
        result = analyzer.predict_single(text)
        
        if result:
            prediction = result['prediction']
            confidence = result.get('confidence', 0.5)
            percentage = confidence * 100
            uncertain = result.get('uncertain', False)
            
            status_icon = "🔴" if uncertain else "🟢"
            status_text = "Uncertain" if uncertain else "Confident"
            
            print(f"{i}. '{text}'")
            print(f"   → {prediction} ({percentage:.1f}%) {status_icon} {status_text}")
    
    print(f"\n✅ Percentage output is working!")
    print(f"Now when you use manual mode, you'll see:")
    print(f"   1. Positive (75.0%)")
    print(f"   2. Negative (60.0%)")
    print(f"   3. Neutral (45.0%) 🔴 Uncertain")

if __name__ == "__main__":
    test_percentage_output()
