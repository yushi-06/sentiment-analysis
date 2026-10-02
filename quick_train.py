#!/usr/bin/env python3
"""
Quick Dataset Training
Simple script to quickly train from a dataset file
"""

import sys
import os
import pandas as pd
from adaptive_sentiment import AdaptiveSentimentAnalyzer

# Try to import enhanced features
try:
    from enhanced_adaptive_learning import EnhancedAdaptiveLearning
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False

def quick_train_from_csv(file_path, text_col="text", sentiment_col="sentiment"):
    """Quick training from CSV file"""
    try:
        print(f"🎓 QUICK TRAINING FROM: {os.path.basename(file_path)}")
        print("="*50)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(f"QUICK TRAINING FROM: {os.path.basename(file_path)}")
        print("="*50)
    
    # Load analyzer
    if ENHANCED_FEATURES:
        print("Loading enhanced learning system...")
        enhanced_learner = EnhancedAdaptiveLearning()
        analyzer = enhanced_learner.analyzer
        print("✅ Enhanced system loaded!")
    else:
        print("Loading basic adaptive system...")
        analyzer = AdaptiveSentimentAnalyzer()
        enhanced_learner = None
        print("✅ Basic system loaded!")
    
    # Load dataset with robust loader
    try:
        print(f"\n📂 Loading dataset...")
        from robust_dataset_loader import RobustDatasetLoader
        loader = RobustDatasetLoader()
        df = loader.load_dataset(file_path)
        print(f"📊 Loaded {len(df)} rows")
        
        # Check columns
        if text_col not in df.columns:
            print(f"❌ Text column '{text_col}' not found")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        if sentiment_col not in df.columns:
            print(f"❌ Sentiment column '{sentiment_col}' not found")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Filter valid data
        valid_df = df[[text_col, sentiment_col]].dropna()
        print(f"✅ Valid training examples: {len(valid_df)}")
        
        # Show distribution
        print(f"\n📊 Sentiment distribution:")
        print(valid_df[sentiment_col].value_counts())
        
        # Start training
        print(f"\n🎓 Starting training...")
        successful = 0
        failed = 0
        
        for idx, row in valid_df.iterrows():
            text = str(row[text_col]).strip()
            sentiment = str(row[sentiment_col]).strip()
            
            if len(text) < 3:
                failed += 1
                continue
            
            try:
                if ENHANCED_FEATURES and enhanced_learner:
                    # Enhanced learning
                    pred, score, conf, uncertain, _ = enhanced_learner.predict_with_confidence(text)
                    enhanced_learner.adaptive_learn_from_feedback(text, pred, sentiment, conf)
                else:
                    # Basic learning
                    if hasattr(analyzer, 'learn_from_correction'):
                        current_pred, _, _ = analyzer.predict_sentiment(text)
                        analyzer.learn_from_correction(text, current_pred, sentiment)
                
                successful += 1
                
                # Progress update
                if successful % 100 == 0:
                    progress = (successful + failed) / len(valid_df) * 100
                    print(f"📈 Progress: {progress:.1f}% ({successful} successful)")
                
            except Exception as e:
                failed += 1
                if failed <= 5:  # Show first few errors
                    print(f"❌ Error: {e}")
        
        print(f"\n✅ Training complete!")
        print(f"📊 Results: {successful} successful, {failed} failed")
        
        # Save model
        if ENHANCED_FEATURES and enhanced_learner:
            enhanced_learner.save_model()
            print("💾 Enhanced model saved!")
        elif hasattr(analyzer, 'save_model'):
            analyzer.save_model()
            print("💾 Model saved!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🎓 QUICK DATASET TRAINING")
        print("="*30)
        print("Usage:")
        print("  python quick_train.py <dataset_file>")
        print("  python quick_train.py <dataset_file> <text_column> <sentiment_column>")
        print()
        print("Examples:")
        print("  python quick_train.py my_data.csv")
        print("  python quick_train.py reviews.csv review sentiment")
        print("  python quick_train.py fix_bias_data.csv text sentiment")
        
        # Interactive mode
        file_path = input("\nEnter dataset file path: ").strip().strip('"')
        if not file_path:
            return
        
        text_col = input("Enter text column name (default: text): ").strip() or "text"
        sentiment_col = input("Enter sentiment column name (default: sentiment): ").strip() or "sentiment"
        
    else:
        file_path = sys.argv[1]
        text_col = sys.argv[2] if len(sys.argv) > 2 else "text"
        sentiment_col = sys.argv[3] if len(sys.argv) > 3 else "sentiment"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Train
    success = quick_train_from_csv(file_path, text_col, sentiment_col)
    
    if success:
        print(f"\n🎉 Training successful! Your model has been updated.")
        print(f"🚀 Test it with: python fixed_main.py")
    else:
        print(f"\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
