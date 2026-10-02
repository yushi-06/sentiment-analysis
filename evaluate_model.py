import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import logging

# Set up paths for local imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import MODEL_DIR, MAX_LENGTH, LABEL_MAPPING

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try to import Transformers
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers or torch library not found. Transformer evaluation will be skipped.")

# Import Adaptive Systems
try:
    from adaptive_sentiment import AdaptiveSentimentAnalyzer
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    logger.warning("AdaptiveSentimentAnalyzer not found in adaptive_sentiment.py.")

def load_and_combine_data():
    files = [
        ('sample_data.csv', 'text', 'sentiment'),
        ('sample_movie_reviews.csv', 'review_text', 'sentiment_label'),
        ('sample_product_reviews.csv', 'text', 'sentiment'),
        ('sample_social_media.csv', 'post', 'emotion'),
        ('my_reviews.csv', 'text', 'sentiment')
    ]
    
    combined_df = pd.DataFrame(columns=['text', 'sentiment'])
    
    for filename, text_col, sent_col in files:
        filepath = PROJECT_ROOT / filename
        if not filepath.exists():
            logger.info(f"File {filename} not found, skipping.")
            continue
                
        try:
            df = pd.read_csv(filepath)
            if text_col in df.columns and sent_col in df.columns:
                temp_df = pd.DataFrame()
                temp_df['text'] = df[text_col].astype(str)
                temp_df['sentiment'] = df[sent_col].astype(str).str.strip().str.title()
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
                logger.info(f"Loaded {len(df)} rows from {filename}")
            else:
                logger.warning(f"Columns {text_col}, {sent_col} not found in {filename}")
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            
    # Normalize sentiments to ['Positive', 'Negative', 'Neutral']
    valid_labels = ['Positive', 'Negative', 'Neutral']
    
    # Map common variations
    label_map = {
        'Pos': 'Positive',
        'Neg': 'Negative',
        'Neu': 'Neutral',
        'Joy': 'Positive',
        'Happy': 'Positive',
        'Sad': 'Negative',
        'Anger': 'Negative',
        '1': 'Positive',
        '0': 'Negative',
        '-1': 'Negative'
    }
    
    combined_df['sentiment'] = combined_df['sentiment'].replace(label_map)
    
    # Filter to only valid labels
    combined_df = combined_df[combined_df['sentiment'].isin(valid_labels)]
    
    # Drop NAs
    combined_df = combined_df.dropna()
    
    return combined_df

def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_str = "\nConfusion Matrix:\n"
    cm_str += f"{'True \\ Pred':>15} | " + " | ".join([f"{l[:8]:>8}" for l in labels]) + "\n"
    cm_str += "-" * (18 + 11 * len(labels)) + "\n"
    for i, true_label in enumerate(labels):
        row = [f"{v:>8}" for v in cm[i]]
        cm_str += f"{true_label[:15]:>15} | " + " | ".join(row) + "\n"
    return cm_str

def evaluate_predictions(y_true, y_pred, system_name, file_handle):
    labels = ['Positive', 'Negative', 'Neutral']
    
    # Clean up predictions to match expected labels exactly
    y_pred = [p.title() if p.title() in labels else 'Neutral' for p in y_pred]
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"EVALUATION RESULTS: {system_name}")
    output.append(f"{'='*60}\n")
    
    acc = accuracy_score(y_true, y_pred)
    output.append(f"Overall Accuracy: {acc:.4f}\n")
    
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    output.append("Classification Report:")
    output.append(report)
    
    cm_str = print_confusion_matrix(y_true, y_pred, labels)
    output.append(cm_str)
    
    text_output = "\n".join(output)
    print(text_output)
    file_handle.write(text_output + "\n")

def main():
    print("Starting Scientific Evaluation Script...")
    
    df = load_and_combine_data()
    if len(df) == 0:
        print("Error: No data loaded. Please ensure sample data files are present in the project directory.")
        return
        
    print("\nData loaded successfully.")
    print(f"Total valid samples: {len(df)}")
    
    dist = df['sentiment'].value_counts()
    print("\nClass Distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count} ({(count/len(df))*100:.1f}%)")
        
    texts = df['text'].tolist()
    y_true = df['sentiment'].tolist()
    
    results_file = PROJECT_ROOT / 'evaluation_results.txt'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== SENTIMENT ANALYSIS EVALUATION REPORT ===\n\n")
        f.write(f"Total Test Samples: {len(df)}\n")
        f.write("Class Distribution:\n")
        for label, count in dist.items():
            f.write(f"  {label}: {count} ({(count/len(df))*100:.1f}%)\n")
        
        # 1. Evaluate Adaptive System
        if ADAPTIVE_AVAILABLE:
            print("\nEvaluating AdaptiveSentimentAnalyzer (Rule-based)...")
            try:
                analyzer = AdaptiveSentimentAnalyzer()
                y_pred_adaptive = []
                for t in texts:
                    try:
                        res = analyzer.analyze(t)
                        y_pred_adaptive.append(res.get('sentiment', 'Neutral'))
                    except Exception:
                        y_pred_adaptive.append('Neutral')
                        
                evaluate_predictions(y_true, y_pred_adaptive, "Adaptive Lexicon System (Rule-Based)", f)
            except Exception as e:
                msg = f"Error during Adaptive system evaluation: {e}"
                print(msg)
                f.write(f"\n{msg}\n")
        else:
            print("\nSkipping Adaptive system evaluation (not found).")
            f.write("\nSkipping Adaptive system evaluation (module not found).\n")
        
        # 2. Evaluate Transformer Model
        if TRANSFORMERS_AVAILABLE:
            print("\nEvaluating Fine-Tuned XLM-RoBERTa Model...")
            model_path = MODEL_DIR
            if not model_path.exists() or not (model_path / 'config.json').exists():
                msg = f"Transformer model not found at {model_path} (Missing config.json). Skipping."
                print(msg)
                f.write(f"\n{msg}\n")
            else:
                try:
                    sentiment_pipeline = pipeline(
                        'sentiment-analysis', 
                        model=str(model_path), 
                        tokenizer=str(model_path),
                        truncation=True,
                        max_length=512,
                        device='cpu' # Using CPU by default for safety in script
                    )
                    y_pred_transformer = []
                    
                    for t in texts:
                        try:
                            # Safely handle sequence length via pipeline arguments, but cast to str just in case
                            result = sentiment_pipeline(str(t))
                            if isinstance(result, list):
                                result = result[0]
                                
                            label = result.get('label', '')
                            
                            if label in ['LABEL_0', '0']:
                                mapped = LABEL_MAPPING.get(0, 'Negative')
                            elif label in ['LABEL_1', '1']:
                                mapped = LABEL_MAPPING.get(1, 'Neutral')
                            elif label in ['LABEL_2', '2']:
                                mapped = LABEL_MAPPING.get(2, 'Positive')
                            elif label.title() in ['Positive', 'Negative', 'Neutral']:
                                mapped = label.title()
                            else:
                                mapped = 'Neutral'
                                
                            y_pred_transformer.append(mapped)
                        except Exception as e:
                            logger.debug(f"Error predicting sample: {e}")
                            y_pred_transformer.append('Neutral')
                            
                    evaluate_predictions(y_true, y_pred_transformer, "Fine-Tuned XLM-RoBERTa Model (Transformer)", f)
                except Exception as e:
                    msg = f"Failed to load/run transformer model: {e}"
                    print(msg)
                    f.write(f"\n{msg}\n")
        else:
            print("\nSkipping Transformer evaluation (transformers/torch not found).")
            f.write("\nSkipping Transformer evaluation (transformers/torch not found).\n")

    print(f"\nEvaluation complete. Comprehensive report saved to {results_file}")

if __name__ == '__main__':
    main()
