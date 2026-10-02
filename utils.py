#!/usr/bin/env python3
"""
Utility functions for sentiment analysis project
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import *

def setup_logging(level: str = LOG_LEVEL) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler('sentiment_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_file_path(file_path: str) -> Path:
    """Validate and return Path object for file"""
    if not file_path or not file_path.strip():
        raise ValueError("File path cannot be empty")
    
    path = Path(file_path.strip().strip('"'))
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format. Supported: {SUPPORTED_FORMATS}")
    
    return path

def validate_text_input(text: str) -> str:
    """Validate and clean text input"""
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    cleaned_text = text.strip()
    
    if len(cleaned_text) < MIN_TEXT_LENGTH:
        raise ValueError(f"Text too short (minimum {MIN_TEXT_LENGTH} characters)")
    
    if len(cleaned_text) > MAX_TEXT_LENGTH:
        cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]
        logging.warning(f"Text truncated to {MAX_TEXT_LENGTH} characters")
    
    return cleaned_text

def validate_sentiment_label(sentiment: str) -> str:
    """Validate sentiment label"""
    if not sentiment or not isinstance(sentiment, str):
        raise ValueError("Sentiment must be a non-empty string")
    
    sentiment = sentiment.strip().title()
    
    valid_sentiments = ['Positive', 'Negative', 'Neutral']
    if sentiment not in valid_sentiments:
        raise ValueError(f"Invalid sentiment. Must be one of: {valid_sentiments}")
    
    return sentiment

def load_dataset(file_path: str, text_column: str = None, sentiment_column: str = None) -> pd.DataFrame:
    """Load and validate dataset from file"""
    path = validate_file_path(file_path)
    
    try:
        # Load based on file extension
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path, encoding='utf-8')
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate dataframe
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Auto-detect or validate columns
        text_col = text_column or auto_detect_column(df, 'text')
        sentiment_col = sentiment_column or auto_detect_column(df, 'sentiment')
        
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found. Available: {list(df.columns)}")
        
        if sentiment_col not in df.columns:
            raise ValueError(f"Sentiment column '{sentiment_col}' not found. Available: {list(df.columns)}")
        
        # Clean and validate data
        df = df.dropna(subset=[text_col, sentiment_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df[sentiment_col] = df[sentiment_col].astype(str).str.strip()
        
        # Remove empty rows
        df = df[df[text_col].str.len() >= MIN_TEXT_LENGTH]
        
        if df.empty:
            raise ValueError("No valid data found after cleaning")
        
        logging.info(f"Loaded dataset: {len(df)} rows, columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def auto_detect_column(df: pd.DataFrame, column_type: str) -> str:
    """Auto-detect column names based on common patterns"""
    columns = df.columns.str.lower()
    
    if column_type == 'text':
        text_patterns = ['text', 'review', 'comment', 'content', 'message', 'description']
        for pattern in text_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                return df.columns[columns.get_loc(matches[0])]
        return DEFAULT_TEXT_COLUMN
    
    elif column_type == 'sentiment':
        sentiment_patterns = ['sentiment', 'label', 'emotion', 'rating', 'polarity']
        for pattern in sentiment_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                return df.columns[columns.get_loc(matches[0])]
        return DEFAULT_SENTIMENT_COLUMN
    
    raise ValueError(f"Unknown column type: {column_type}")

def normalize_sentiment_labels(df: pd.DataFrame, sentiment_column: str) -> pd.DataFrame:
    """Normalize sentiment labels to standard format"""
    sentiment_mapping = {
        # Positive variations
        'positive': 'Positive', 'pos': 'Positive', '1': 'Positive', 
        'good': 'Positive', 'like': 'Positive', 'love': 'Positive',
        
        # Negative variations  
        'negative': 'Negative', 'neg': 'Negative', '0': 'Negative',
        'bad': 'Negative', 'hate': 'Negative', 'dislike': 'Negative',
        
        # Neutral variations
        'neutral': 'Neutral', 'neut': 'Neutral', '2': 'Neutral',
        'ok': 'Neutral', 'okay': 'Neutral', 'average': 'Neutral'
    }
    
    df[sentiment_column] = df[sentiment_column].str.lower().map(sentiment_mapping)
    df = df.dropna(subset=[sentiment_column])
    
    return df

def calculate_class_distribution(df: pd.DataFrame, sentiment_column: str) -> Dict[str, float]:
    """Calculate percentage distribution of sentiment classes"""
    counts = df[sentiment_column].value_counts()
    total = len(df)
    
    distribution = {}
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = counts.get(sentiment, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        distribution[sentiment] = round(percentage, 2)
    
    return distribution

def safe_file_operation(operation, *args, **kwargs):
    """Safely execute file operations with proper error handling"""
    try:
        return operation(*args, **kwargs)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except PermissionError as e:
        logging.error(f"Permission denied: {e}")
        raise
    except Exception as e:
        logging.error(f"File operation failed: {e}")
        raise

def format_accuracy_report(correct: int, total: int, class_results: Dict = None) -> str:
    """Format accuracy report for display"""
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    report = f"📊 ACCURACY REPORT\n"
    report += f"{'='*30}\n"
    report += f"Overall: {accuracy:.1f}% ({correct}/{total})\n"
    
    if class_results:
        report += f"\nPer-Class Results:\n"
        for class_name, metrics in class_results.items():
            report += f"  {class_name}: {metrics}\n"
    
    return report

# Initialize logging
logger = setup_logging()
