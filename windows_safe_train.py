#!/usr/bin/env python3
"""
Windows-Safe Dataset Training
Compatible with Windows console encoding
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

def safe_print(text):
    """Print text safely on Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emojis and special characters for Windows console
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)

def windows_safe_train(file_path, text_col="text", sentiment_col="sentiment"):
    """Windows-safe training from CSV file"""
    safe_print(f"QUICK TRAINING FROM: {os.path.basename(file_path)}")
    safe_print("="*50)
    
    # Load analyzer
    if ENHANCED_FEATURES:
        safe_print("Loading enhanced learning system...")
        enhanced_learner = EnhancedAdaptiveLearning()
        analyzer = enhanced_learner.analyzer
        safe_print("Enhanced system loaded!")
    else:
        safe_print("Loading basic adaptive system...")
        analyzer = AdaptiveSentimentAnalyzer()
        enhanced_learner = None
        safe_print("Basic system loaded!")
    
    # Load dataset with robust loader
    try:
        safe_print("\nLoading dataset...")
        from robust_dataset_loader import RobustDatasetLoader
        loader = RobustDatasetLoader()
        df = loader.load_dataset(file_path)
        safe_print(f"Loaded {len(df)} rows")
        
        # Check columns
        if text_col not in df.columns:
            safe_print(f"ERROR: Text column '{text_col}' not found")
            safe_print(f"Available columns: {list(df.columns)}")
            return False
        
        if sentiment_col not in df.columns:
            safe_print(f"ERROR: Sentiment column '{sentiment_col}' not found")
            safe_print(f"Available columns: {list(df.columns)}")
            return False
        
        # Filter valid data
        valid_df = df[[text_col, sentiment_col]].dropna()
        safe_print(f"Valid training examples: {len(valid_df)}")
        
        # Show distribution
        safe_print("\nSentiment distribution:")
        distribution = valid_df[sentiment_col].value_counts()
        for sentiment, count in distribution.items():
            safe_print(f"  {sentiment}: {count}")
        
        # Start training
        safe_print("\nStarting training...")
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
                if successful % 10 == 0:
                    progress = (successful + failed) / len(valid_df) * 100
                    safe_print(f"Progress: {progress:.1f}% ({successful} successful)")
                
            except Exception as e:
                failed += 1
                if failed <= 3:  # Show first few errors
                    safe_print(f"Error: {e}")
        
        safe_print("\nTraining complete!")
        safe_print(f"Results: {successful} successful, {failed} failed")
        
        # Save model
        if ENHANCED_FEATURES and enhanced_learner:
            enhanced_learner.save_model()
            safe_print("Enhanced model saved!")
        elif hasattr(analyzer, 'save_model'):
            analyzer.save_model()
            safe_print("Model saved!")
        
        return True
        
    except Exception as e:
        safe_print(f"Error: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        safe_print("WINDOWS-SAFE DATASET TRAINING")
        safe_print("="*35)
        safe_print("Usage:")
        safe_print("  python windows_safe_train.py <dataset_file>")
        safe_print("  python windows_safe_train.py <dataset_file> <text_column> <sentiment_column>")
        safe_print("")
        safe_print("Examples:")
        safe_print("  python windows_safe_train.py my_data.csv")
        safe_print("  python windows_safe_train.py reviews.csv review sentiment")
        safe_print("  python windows_safe_train.py fix_bias_data.csv text sentiment")
        
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
        safe_print(f"File not found: {file_path}")
        return
    
    # Train
    success = windows_safe_train(file_path, text_col, sentiment_col)
    
    if success:
        safe_print("\nTraining successful! Your model has been updated.")
        safe_print("Test it with: python fixed_main.py")
    else:
        safe_print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
