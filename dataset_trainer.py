#!/usr/bin/env python3
"""
Dataset Training System
Train the sentiment analysis model from various datasets
"""

import os
import pandas as pd
from datetime import datetime
from adaptive_sentiment import AdaptiveSentimentAnalyzer

# Try to import enhanced features
try:
    from enhanced_adaptive_learning import EnhancedAdaptiveLearning
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False

class DatasetTrainer:
    def __init__(self):
        print("🎓 DATASET TRAINING SYSTEM")
        print("="*40)
        print("Loading sentiment analyzer for training...")
        
        if ENHANCED_FEATURES:
            self.enhanced_learner = EnhancedAdaptiveLearning()
            self.analyzer = self.enhanced_learner.analyzer
            print("✅ Enhanced learning system loaded!")
            print("🎯 Features: Adaptive learning + Confidence scoring")
        else:
            self.analyzer = AdaptiveSentimentAnalyzer()
            self.enhanced_learner = None
            print("✅ Basic adaptive system loaded!")
        
        # Training statistics
        self.training_stats = {
            'total_examples': 0,
            'successful_training': 0,
            'failed_training': 0,
            'datasets_processed': 0,
            'training_start': datetime.now()
        }
    
    def load_dataset(self, file_path, text_column=None, sentiment_column=None):
        """Load dataset from CSV or Excel file"""
        print(f"\n📂 Loading dataset: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        try:
            # Load file based on extension
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print("❌ Unsupported file format. Please use CSV or Excel files.")
                return None
            
            print(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"📋 Available columns: {', '.join(df.columns)}")
            
            # Auto-detect or ask for column names
            text_col = self._find_text_column(df, text_column)
            sentiment_col = self._find_sentiment_column(df, sentiment_column)
            
            if not text_col or not sentiment_col:
                return None
            
            # Filter valid data
            valid_df = df[[text_col, sentiment_col]].dropna()
            print(f"✅ Valid training examples: {len(valid_df)}")
            
            return valid_df, text_col, sentiment_col
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def _find_text_column(self, df, suggested_column):
        """Find or ask for text column"""
        # Common text column names
        text_candidates = ['text', 'review', 'comment', 'message', 'content', 'sentence']
        
        if suggested_column and suggested_column in df.columns:
            print(f"📝 Using text column: '{suggested_column}'")
            return suggested_column
        
        # Auto-detect
        for candidate in text_candidates:
            if candidate in df.columns:
                print(f"📝 Auto-detected text column: '{candidate}'")
                return candidate
        
        # Ask user
        print(f"\n📝 Text column not found. Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        while True:
            choice = input("Enter text column name or number: ").strip()
            
            # Check if it's a number
            try:
                col_index = int(choice) - 1
                if 0 <= col_index < len(df.columns):
                    return df.columns[col_index]
            except ValueError:
                pass
            
            # Check if it's a column name
            if choice in df.columns:
                return choice
            
            print("❌ Invalid choice. Please try again.")
    
    def _find_sentiment_column(self, df, suggested_column):
        """Find or ask for sentiment column"""
        # Common sentiment column names
        sentiment_candidates = ['sentiment', 'label', 'emotion', 'polarity', 'class', 'target']
        
        if suggested_column and suggested_column in df.columns:
            print(f"💭 Using sentiment column: '{suggested_column}'")
            return suggested_column
        
        # Auto-detect
        for candidate in sentiment_candidates:
            if candidate in df.columns:
                print(f"💭 Auto-detected sentiment column: '{candidate}'")
                return candidate
        
        # Ask user
        print(f"\n💭 Sentiment column not found. Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        while True:
            choice = input("Enter sentiment column name or number: ").strip()
            
            # Check if it's a number
            try:
                col_index = int(choice) - 1
                if 0 <= col_index < len(df.columns):
                    return df.columns[col_index]
            except ValueError:
                pass
            
            # Check if it's a column name
            if choice in df.columns:
                return choice
            
            print("❌ Invalid choice. Please try again.")
    
    def normalize_sentiment_labels(self, df, sentiment_col):
        """Normalize sentiment labels to standard format"""
        print(f"\n🔄 Normalizing sentiment labels...")
        
        # Show unique values
        unique_values = df[sentiment_col].unique()
        print(f"📊 Found sentiment values: {unique_values}")
        
        # Create mapping
        label_mapping = {}
        
        for value in unique_values:
            value_str = str(value).lower().strip()
            
            # Auto-map common values
            if value_str in ['positive', 'pos', '1', 'good', 'happy', 'like']:
                label_mapping[value] = 'Positive'
            elif value_str in ['negative', 'neg', '0', 'bad', 'sad', 'hate']:
                label_mapping[value] = 'Negative'
            elif value_str in ['neutral', 'neu', '2', 'ok', 'okay', 'mixed']:
                label_mapping[value] = 'Neutral'
            else:
                # Ask user for mapping
                print(f"\n❓ How should '{value}' be mapped?")
                print("1. Positive")
                print("2. Negative") 
                print("3. Neutral")
                print("4. Skip (ignore this label)")
                
                while True:
                    choice = input(f"Choice for '{value}': ").strip()
                    if choice == '1':
                        label_mapping[value] = 'Positive'
                        break
                    elif choice == '2':
                        label_mapping[value] = 'Negative'
                        break
                    elif choice == '3':
                        label_mapping[value] = 'Neutral'
                        break
                    elif choice == '4':
                        label_mapping[value] = None  # Skip
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # Apply mapping
        df_mapped = df.copy()
        df_mapped[sentiment_col] = df_mapped[sentiment_col].map(label_mapping)
        
        # Remove unmapped values
        df_mapped = df_mapped.dropna(subset=[sentiment_col])
        
        print(f"✅ Label mapping complete. Valid examples: {len(df_mapped)}")
        print(f"📊 Final distribution:")
        print(df_mapped[sentiment_col].value_counts())
        
        return df_mapped
    
    def train_from_dataset(self, df, text_col, sentiment_col, batch_size=100):
        """Train the model from dataset"""
        print(f"\n🎓 Starting training from dataset...")
        print(f"📊 Training examples: {len(df)}")
        
        successful = 0
        failed = 0
        
        # Process in batches for better progress tracking
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} examples)")
            
            for idx, row in batch.iterrows():
                text = str(row[text_col]).strip()
                sentiment = str(row[sentiment_col]).strip()
                
                if not text or len(text) < 3:
                    failed += 1
                    continue
                
                try:
                    if ENHANCED_FEATURES and self.enhanced_learner:
                        # Use enhanced learning
                        # First get current prediction
                        pred, score, conf, uncertain, _ = self.enhanced_learner.predict_with_confidence(text)
                        
                        # Learn from the correct label
                        self.enhanced_learner.adaptive_learn_from_feedback(text, pred, sentiment, conf)
                    else:
                        # Use basic adaptive learning
                        if hasattr(self.analyzer, 'learn_from_correction'):
                            # Get current prediction first
                            current_pred, _, _ = self.analyzer.predict_sentiment(text)
                            self.analyzer.learn_from_correction(text, current_pred, sentiment)
                        else:
                            print("⚠️  Learning method not available")
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"❌ Error training on: '{text[:50]}...' - {e}")
                    failed += 1
            
            # Show progress
            progress = ((i + len(batch)) / len(df)) * 100
            print(f"📈 Progress: {progress:.1f}% | Successful: {successful} | Failed: {failed}")
        
        # Update stats
        self.training_stats['total_examples'] += len(df)
        self.training_stats['successful_training'] += successful
        self.training_stats['failed_training'] += failed
        self.training_stats['datasets_processed'] += 1
        
        print(f"\n✅ Training complete!")
        print(f"📊 Results: {successful} successful, {failed} failed")
        
        return successful, failed
    
    def train_from_file(self, file_path, text_column=None, sentiment_column=None):
        """Complete training pipeline from file"""
        print(f"\n🎯 TRAINING FROM FILE: {os.path.basename(file_path)}")
        print("="*60)
        
        # Load dataset
        dataset_info = self.load_dataset(file_path, text_column, sentiment_column)
        if not dataset_info:
            return False
        
        df, text_col, sentiment_col = dataset_info
        
        # Normalize labels
        df_normalized = self.normalize_sentiment_labels(df, sentiment_col)
        
        if len(df_normalized) == 0:
            print("❌ No valid training examples after normalization")
            return False
        
        # Train
        successful, failed = self.train_from_dataset(df_normalized, text_col, sentiment_col)
        
        # Save progress
        self.save_training_progress()
        
        return successful > 0
    
    def save_training_progress(self):
        """Save training progress"""
        if ENHANCED_FEATURES and self.enhanced_learner:
            # Save enhanced model
            self.enhanced_learner.save_model()
            print("💾 Enhanced model saved!")
        elif hasattr(self.analyzer, 'save_model'):
            self.analyzer.save_model()
            print("💾 Model saved!")
        else:
            print("⚠️  Model saving not available")
    
    def show_training_stats(self):
        """Show training statistics"""
        duration = datetime.now() - self.training_stats['training_start']
        
        print(f"\n📊 TRAINING STATISTICS")
        print("="*30)
        print(f"Training duration: {duration}")
        print(f"Datasets processed: {self.training_stats['datasets_processed']}")
        print(f"Total examples: {self.training_stats['total_examples']}")
        print(f"Successful training: {self.training_stats['successful_training']}")
        print(f"Failed training: {self.training_stats['failed_training']}")
        
        if self.training_stats['total_examples'] > 0:
            success_rate = (self.training_stats['successful_training'] / self.training_stats['total_examples']) * 100
            print(f"Success rate: {success_rate:.1f}%")

def main():
    """Main training interface"""
    trainer = DatasetTrainer()
    
    print(f"\n🎯 DATASET TRAINING OPTIONS")
    print("="*35)
    print("1. Train from single dataset file")
    print("2. Train from multiple files")
    print("3. Train from your existing bias correction data")
    print("4. Show training statistics")
    print("5. Exit")
    
    while True:
        choice = input(f"\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Single file training
            file_path = input("Enter dataset file path: ").strip().strip('"')
            if file_path:
                trainer.train_from_file(file_path)
        
        elif choice == '2':
            # Multiple files training
            print("Enter file paths (one per line, empty line to finish):")
            files = []
            while True:
                file_path = input("File path: ").strip().strip('"')
                if not file_path:
                    break
                files.append(file_path)
            
            for file_path in files:
                print(f"\n{'='*60}")
                trainer.train_from_file(file_path)
        
        elif choice == '3':
            # Train from existing bias correction data
            bias_data_path = "fix_bias_data.csv"
            if os.path.exists(bias_data_path):
                print(f"Training from existing bias correction data...")
                trainer.train_from_file(bias_data_path, "text", "sentiment")
            else:
                print(f"❌ Bias correction data not found: {bias_data_path}")
        
        elif choice == '4':
            # Show statistics
            trainer.show_training_stats()
        
        elif choice == '5':
            # Exit
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")
    
    # Final statistics
    trainer.show_training_stats()
    print(f"\n👋 Training session complete!")

if __name__ == "__main__":
    main()
