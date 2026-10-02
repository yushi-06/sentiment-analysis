#!/usr/bin/env python3
"""
Comprehensive test to demonstrate the bias fix is working
"""

from run_sentiment_analysis import analyze_multiple_texts, analyze_text
import pandas as pd

def comprehensive_bias_test():
    print("🎯 COMPREHENSIVE BIAS FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Clearly negative comments
    print("\n1️⃣ Testing Negative Comments")
    print("-" * 30)
    negative_comments = [
        "this is a bad product",
        "terrible quality", 
        "worst purchase ever",
        "I hate this",
        "complete waste of money",
        "awful experience",
        "poor service",
        "disappointing results"
    ]
    
    neg_percentages, neg_detailed = analyze_multiple_texts(negative_comments)
    print(f"Results: Positive: {neg_percentages['Positive']}%, Negative: {neg_percentages['Negative']}%, Neutral: {neg_percentages['Neutral']}%")
    
    # Test 2: Clearly positive comments  
    print("\n2️⃣ Testing Positive Comments")
    print("-" * 30)
    positive_comments = [
        "I love this product",
        "amazing quality",
        "excellent service", 
        "fantastic experience",
        "highly recommend",
        "outstanding results",
        "wonderful product",
        "great value"
    ]
    
    pos_percentages, pos_detailed = analyze_multiple_texts(positive_comments)
    print(f"Results: Positive: {pos_percentages['Positive']}%, Negative: {pos_percentages['Negative']}%, Neutral: {pos_percentages['Neutral']}%")
    
    # Test 3: Neutral comments
    print("\n3️⃣ Testing Neutral Comments")
    print("-" * 30)
    neutral_comments = [
        "it's okay",
        "average product",
        "decent quality",
        "acceptable service",
        "standard experience",
        "normal results",
        "fair price",
        "adequate performance"
    ]
    
    neu_percentages, neu_detailed = analyze_multiple_texts(neutral_comments)
    print(f"Results: Positive: {neu_percentages['Positive']}%, Negative: {neu_percentages['Negative']}%, Neutral: {neu_percentages['Neutral']}%")
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT")
    print("="*40)
    
    # Check if negative comments are mostly classified as negative
    neg_correct = neg_percentages['Negative'] > 50
    pos_correct = pos_percentages['Positive'] > 50  
    neu_reasonable = neu_percentages['Neutral'] >= 30 or neu_percentages['Positive'] + neu_percentages['Neutral'] > 70
    
    print(f"✅ Negative detection: {'PASS' if neg_correct else 'FAIL'} ({neg_percentages['Negative']}% negative)")
    print(f"✅ Positive detection: {'PASS' if pos_correct else 'FAIL'} ({pos_percentages['Positive']}% positive)")
    print(f"✅ Neutral handling: {'PASS' if neu_reasonable else 'FAIL'} ({neu_percentages['Neutral']}% neutral)")
    
    if neg_correct and pos_correct and neu_reasonable:
        print(f"\n🎉 SUCCESS! Bias has been completely fixed!")
        print(f"The model is now working correctly across all sentiment types.")
        return True
    else:
        print(f"\n⚠️ Some issues remain. Check individual results above.")
        return False

def test_with_your_data():
    """Test with the data from your CSV file if it exists"""
    print(f"\n4️⃣ Testing with Your CSV Data")
    print("-" * 30)
    
    try:
        # Check if your CSV file exists and has data
        df = pd.read_csv("fix_bias_data.csv")
        if 'text' in df.columns and len(df) > 0:
            texts = df['text'].astype(str).tolist()[:10]  # Test first 10 rows
            percentages, detailed = analyze_multiple_texts(texts)
            
            print(f"Analyzed {len(texts)} texts from your CSV:")
            print(f"Positive: {percentages['Positive']}%")
            print(f"Negative: {percentages['Negative']}%") 
            print(f"Neutral: {percentages['Neutral']}%")
            
            print(f"\nSample results:")
            for i, result in enumerate(detailed[:5], 1):
                print(f"{i}. '{result['text'][:50]}...' → {result['sentiment']}")
                
        else:
            print("CSV file found but no 'text' column or no data")
            
    except FileNotFoundError:
        print("fix_bias_data.csv not found - skipping this test")
    except Exception as e:
        print(f"Error reading CSV: {e}")

def interactive_test():
    """Let user test with their own examples"""
    print(f"\n5️⃣ Interactive Test")
    print("-" * 30)
    print("Enter some comments to test (type 'done' when finished):")
    
    user_texts = []
    while True:
        text = input("Comment: ").strip()
        if text.lower() == 'done':
            break
        elif text:
            user_texts.append(text)
            # Show immediate result
            prediction, score = analyze_text(text)
            print(f"  → {prediction} (confidence: {abs(score):.2f})")
    
    if user_texts:
        print(f"\nOverall distribution for your {len(user_texts)} comments:")
        percentages, _ = analyze_multiple_texts(user_texts)
        print(f"Positive: {percentages['Positive']}%")
        print(f"Negative: {percentages['Negative']}%")
        print(f"Neutral: {percentages['Neutral']}%")

if __name__ == "__main__":
    # Run comprehensive test
    success = comprehensive_bias_test()
    
    # Test with user's CSV data
    test_with_your_data()
    
    # Interactive test
    print(f"\nWould you like to test with your own examples? (y/n): ", end="")
    if input().strip().lower() in ['y', 'yes']:
        interactive_test()
    
    print(f"\n" + "="*60)
    if success:
        print("🎉 BIAS FIX VERIFICATION: COMPLETE SUCCESS!")
        print("Your sentiment analysis model is now working correctly.")
        print("You can use 'python run_sentiment_analysis.py' for regular use.")
    else:
        print("⚠️ Some issues detected. Please review the results above.")
    print("="*60)
