import pandas as pd
import os
import re
from collections import defaultdict, Counter
from adaptive_sentiment import AdaptiveSentimentAnalyzer
import numpy as np
from datetime import datetime

class AutoTrainer:
    def __init__(self, analyzer=None):
        self.analyzer = analyzer if analyzer else AdaptiveSentimentAnalyzer()
        self.training_stats = {
            'datasets_processed': 0,
            'texts_learned': 0,
            'words_learned': 0,
            'patterns_discovered': 0,
            'accuracy_improvements': []
        }
    
    def preprocess_dataset(self, df, text_column, sentiment_column):
        """Clean and prepare dataset for training"""
        print(f"📋 Preprocessing dataset...")
        
        # Remove missing values
        df = df.dropna(subset=[text_column, sentiment_column])
        
        # Clean sentiment labels
        df[sentiment_column] = df[sentiment_column].astype(str).str.strip().str.lower()
        
        # Standardize sentiment labels
        sentiment_mapping = {
            'positive': 'Positive', 'pos': 'Positive', '1': 'Positive', 'good': 'Positive',
            'negative': 'Negative', 'neg': 'Negative', '0': 'Negative', 'bad': 'Negative', 
            'neutral': 'Neutral', 'neut': 'Neutral', '2': 'Neutral', 'ok': 'Neutral'
        }
        
        df['clean_sentiment'] = df[sentiment_column].map(sentiment_mapping)
        df = df.dropna(subset=['clean_sentiment'])
        
        # Clean text
        df['clean_text'] = df[text_column].astype(str).str.strip()
        df = df[df['clean_text'].str.len() > 3]  # Remove very short texts
        
        print(f"✅ Dataset cleaned: {len(df)} valid samples")
        print(f"   Positive: {sum(df['clean_sentiment'] == 'Positive')}")
        print(f"   Negative: {sum(df['clean_sentiment'] == 'Negative')}")
        print(f"   Neutral: {sum(df['clean_sentiment'] == 'Neutral')}")
        
        return df[['clean_text', 'clean_sentiment']]
    
    def analyze_word_patterns(self, texts, sentiments):
        """Analyze word patterns in the dataset"""
        print(f"🔍 Analyzing word patterns...")
        
        word_sentiment_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
        bigram_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
        
        for text, sentiment in zip(texts, sentiments):
            words = self.analyzer.preprocess_text(text)
            
            # Analyze individual words
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_sentiment_stats[word][sentiment.lower()] += 1
                    word_sentiment_stats[word]['total'] += 1
            
            # Analyze bigrams (word pairs)
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:
                    bigram_stats[bigram][sentiment.lower()] += 1
                    bigram_stats[bigram]['total'] += 1
        
        return word_sentiment_stats, bigram_stats
    
    def discover_sentiment_words(self, word_stats, min_occurrences=3):
        """Discover new sentiment words from patterns"""
        print(f"🎯 Discovering sentiment words...")
        
        discovered_positive = set()
        discovered_negative = set()
        discovered_neutral = set()
        
        for word, stats in word_stats.items():
            if stats['total'] >= min_occurrences:
                pos_ratio = stats['positive'] / stats['total']
                neg_ratio = stats['negative'] / stats['total']
                neu_ratio = stats['neutral'] / stats['total']
                
                # Strong positive words (>70% positive)
                if pos_ratio > 0.7 and stats['positive'] >= min_occurrences:
                    discovered_positive.add(word)
                
                # Strong negative words (>70% negative)
                elif neg_ratio > 0.7 and stats['negative'] >= min_occurrences:
                    discovered_negative.add(word)
                
                # Strong neutral words (>70% neutral)
                elif neu_ratio > 0.7 and stats['neutral'] >= min_occurrences:
                    discovered_neutral.add(word)
        
        print(f"✅ Discovered sentiment words:")
        print(f"   Positive: {len(discovered_positive)} words")
        print(f"   Negative: {len(discovered_negative)} words")
        print(f"   Neutral: {len(discovered_neutral)} words")
        
        return discovered_positive, discovered_negative, discovered_neutral
    
    def train_from_dataset(self, file_path, text_column='text', sentiment_column='sentiment', 
                          sample_size=None, auto_discover=True):
        """Train the model from a dataset file"""
        print(f"🚀 Starting dataset training: {file_path}")
        print("="*60)
        
        try:
            # Load dataset
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            print(f"📂 Loaded dataset: {len(df)} rows")
            
            # Check columns
            if text_column not in df.columns:
                available_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'comment' in col.lower()]
                if available_cols:
                    text_column = available_cols[0]
                    print(f"🔄 Using text column: {text_column}")
                else:
                    raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
            
            if sentiment_column not in df.columns:
                available_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'label' in col.lower() or 'emotion' in col.lower()]
                if available_cols:
                    sentiment_column = available_cols[0]
                    print(f"🔄 Using sentiment column: {sentiment_column}")
                else:
                    raise ValueError(f"Sentiment column '{sentiment_column}' not found. Available: {list(df.columns)}")
            
            # Preprocess dataset
            clean_df = self.preprocess_dataset(df, text_column, sentiment_column)
            
            # Sample if requested
            if sample_size and len(clean_df) > sample_size:
                clean_df = clean_df.sample(n=sample_size, random_state=42)
                print(f"📊 Using sample: {len(clean_df)} texts")
            
            # Auto-discover patterns if enabled
            if auto_discover:
                word_stats, bigram_stats = self.analyze_word_patterns(
                    clean_df['clean_text'], clean_df['clean_sentiment'])
                
                pos_words, neg_words, neu_words = self.discover_sentiment_words(word_stats)
                
                # Add discovered words to analyzer
                self.analyzer.positive_words.update(pos_words)
                self.analyzer.negative_words.update(neg_words)  
                self.analyzer.neutral_words.update(neu_words)
                
                self.training_stats['words_learned'] += len(pos_words) + len(neg_words) + len(neu_words)
                self.training_stats['patterns_discovered'] += len([w for w in word_stats if word_stats[w]['total'] >= 3])
            
            # Train on each example
            print(f"📚 Training on examples...")
            trained_count = 0
            
            for idx, row in clean_df.iterrows():
                text = row['clean_text']
                correct_sentiment = row['clean_sentiment']
                
                # Get current prediction
                prediction, score, _ = self.analyzer.predict_sentiment(text)
                
                # Learn from this example
                if prediction != correct_sentiment:
                    self.analyzer.learn_from_correction(text, prediction, correct_sentiment)
                    trained_count += 1
                
                # Update word sentiment counts for all examples
                words = self.analyzer.preprocess_text(text)
                for word in words:
                    self.analyzer.word_sentiment_counts[word][correct_sentiment.lower()] += 1
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(clean_df)} texts...")
            
            # Save updated knowledge
            self.analyzer.save_knowledge()
            
            # Update training stats
            self.training_stats['datasets_processed'] += 1
            self.training_stats['texts_learned'] += len(clean_df)
            
            print(f"🎉 Training completed!")
            print(f"   Examples processed: {len(clean_df)}")
            print(f"   Corrections made: {trained_count}")
            print(f"   Accuracy: {((len(clean_df) - trained_count) / len(clean_df) * 100):.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def batch_train_from_multiple_datasets(self, dataset_paths, auto_discover=True):
        """Train from multiple datasets"""
        print(f"🔥 BATCH TRAINING FROM MULTIPLE DATASETS")
        print("="*60)
        
        success_count = 0
        total_datasets = len(dataset_paths)
        
        for i, path in enumerate(dataset_paths, 1):
            print(f"\n📂 Dataset {i}/{total_datasets}: {os.path.basename(path)}")
            
            if self.train_from_dataset(path, auto_discover=auto_discover):
                success_count += 1
                print(f"✅ Success")
            else:
                print(f"❌ Failed")
        
        print(f"\n🏆 BATCH TRAINING COMPLETE!")
        print(f"   Successful: {success_count}/{total_datasets} datasets")
        self.show_training_stats()
    
    def continuous_learning_mode(self, dataset_path, check_interval_minutes=30):
        """Continuously monitor and learn from a dataset"""
        print(f"🔄 CONTINUOUS LEARNING MODE")
        print(f"   Monitoring: {dataset_path}")
        print(f"   Check interval: {check_interval_minutes} minutes")
        print("   Press Ctrl+C to stop")
        
        last_modified = 0
        
        try:
            while True:
                if os.path.exists(dataset_path):
                    current_modified = os.path.getmtime(dataset_path)
                    
                    if current_modified > last_modified:
                        print(f"\n📱 Dataset updated! Retraining...")
                        if self.train_from_dataset(dataset_path):
                            print(f"✅ Retrained successfully")
                            last_modified = current_modified
                        else:
                            print(f"❌ Retraining failed")
                
                # Wait for next check
                import time
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n👋 Continuous learning stopped")
    
    def evaluate_on_test_set(self, test_file, text_column='text', sentiment_column='sentiment'):
        """Evaluate current model performance on test set"""
        print(f"📊 EVALUATING MODEL PERFORMANCE")
        print("="*40)
        
        try:
            # Load test data
            if test_file.lower().endswith('.csv'):
                df = pd.read_csv(test_file)
            else:
                df = pd.read_excel(test_file)
            
            clean_df = self.preprocess_dataset(df, text_column, sentiment_column)
            
            correct = 0
            total = len(clean_df)
            confusion_matrix = defaultdict(lambda: defaultdict(int))
            
            print(f"🧪 Testing on {total} examples...")
            
            for idx, row in clean_df.iterrows():
                text = row['clean_text']
                true_sentiment = row['clean_sentiment']
                
                prediction, score, _ = self.analyzer.predict_sentiment(text)
                
                if prediction == true_sentiment:
                    correct += 1
                
                confusion_matrix[true_sentiment][prediction] += 1
            
            accuracy = (correct / total) * 100
            
            print(f"\n📈 RESULTS:")
            print(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
            
            print(f"\n🔍 Confusion Matrix:")
            sentiments = ['Positive', 'Negative', 'Neutral']
            print(f"{'True\\Pred':<10} {'Pos':<6} {'Neg':<6} {'Neu':<6}")
            for true_sent in sentiments:
                row_str = f"{true_sent:<10}"
                for pred_sent in sentiments:
                    row_str += f" {confusion_matrix[true_sent][pred_sent]:<5}"
                print(row_str)
            
            # Calculate per-class metrics
            print(f"\n📊 Per-Class Performance:")
            for sentiment in sentiments:
                tp = confusion_matrix[sentiment][sentiment]
                total_true = sum(confusion_matrix[sentiment].values())
                total_pred = sum(confusion_matrix[s][sentiment] for s in sentiments)
                
                precision = tp / total_pred if total_pred > 0 else 0
                recall = tp / total_true if total_true > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"   {sentiment}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            self.training_stats['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'test_samples': total
            })
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0
    
    def show_training_stats(self):
        """Show comprehensive training statistics"""
        print(f"\n📊 TRAINING STATISTICS")
        print("="*40)
        
        stats = self.training_stats
        print(f"Datasets processed: {stats['datasets_processed']}")
        print(f"Texts learned from: {stats['texts_learned']}")
        print(f"Words discovered: {stats['words_learned']}")
        print(f"Patterns found: {stats['patterns_discovered']}")
        
        # Current model stats
        print(f"\n🧠 Current Model Knowledge:")
        print(f"Positive words: {len(self.analyzer.positive_words)}")
        print(f"Negative words: {len(self.analyzer.negative_words)}")
        print(f"Neutral words: {len(self.analyzer.neutral_words)}")
        print(f"User corrections: {len(self.analyzer.user_corrections)}")
        print(f"Word patterns: {len(self.analyzer.word_sentiment_counts)}")
        
        if stats['accuracy_improvements']:
            latest_acc = stats['accuracy_improvements'][-1]['accuracy']
            print(f"\n🎯 Latest Accuracy: {latest_acc:.1f}%")

def main():
    """Interactive training interface"""
    print("🚀 AUTO-TRAINER FOR SENTIMENT ANALYSIS")
    print("="*50)
    
    trainer = AutoTrainer()
    
    while True:
        print(f"\nChoose training mode:")
        print("1. Train from single dataset")
        print("2. Batch train from multiple datasets")
        print("3. Evaluate on test set")
        print("4. Show training statistics")
        print("5. Continuous learning mode")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            file_path = input("Enter dataset file path: ").strip().strip('"')
            text_col = input("Text column name (default: text): ").strip() or 'text'
            sentiment_col = input("Sentiment column name (default: sentiment): ").strip() or 'sentiment'
            sample_str = input("Sample size (press Enter for all): ").strip()
            sample_size = int(sample_str) if sample_str else None
            
            trainer.train_from_dataset(file_path, text_col, sentiment_col, sample_size)
            
        elif choice == '2':
            paths_str = input("Enter dataset paths (comma-separated): ").strip()
            paths = [p.strip().strip('"') for p in paths_str.split(',')]
            trainer.batch_train_from_multiple_datasets(paths)
            
        elif choice == '3':
            test_file = input("Enter test file path: ").strip().strip('"')
            trainer.evaluate_on_test_set(test_file)
            
        elif choice == '4':
            trainer.show_training_stats()
            
        elif choice == '5':
            dataset_path = input("Enter dataset path to monitor: ").strip().strip('"')
            interval = input("Check interval in minutes (default: 30): ").strip()
            interval = int(interval) if interval else 30
            trainer.continuous_learning_mode(dataset_path, interval)
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
