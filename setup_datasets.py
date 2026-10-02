#!/usr/bin/env python3
"""
Setup Sample Datasets
Automatically configure sample datasets for training
"""

from dataset_config import DatasetConfig
import os

def setup_sample_datasets():
    """Setup sample datasets in configuration"""
    print("🔧 SETTING UP SAMPLE DATASETS")
    print("="*40)
    
    config = DatasetConfig()
    
    # Define sample datasets
    datasets = [
        {
            "name": "bias_correction",
            "file_path": "fix_bias_data.csv",
            "text_column": "text",
            "sentiment_column": "sentiment",
            "description": "Your original bias correction training data"
        },
        {
            "name": "movie_reviews",
            "file_path": "sample_movie_reviews.csv",
            "text_column": "review_text",
            "sentiment_column": "sentiment_label",
            "description": "Sample movie reviews dataset"
        },
        {
            "name": "product_reviews",
            "file_path": "sample_product_reviews.csv",
            "text_column": "text",
            "sentiment_column": "sentiment",
            "description": "Sample product reviews dataset"
        },
        {
            "name": "social_media",
            "file_path": "sample_social_media.csv",
            "text_column": "post",
            "sentiment_column": "emotion",
            "description": "Sample social media posts dataset"
        }
    ]
    
    # Add each dataset
    for dataset in datasets:
        config.add_dataset(
            dataset["name"],
            dataset["file_path"],
            dataset["text_column"],
            dataset["sentiment_column"],
            dataset["description"]
        )
    
    print(f"\n✅ Setup complete! {len(datasets)} datasets configured.")
    
    # Show configured datasets
    print(f"\n📂 Available datasets:")
    config.list_datasets()
    
    print(f"\n🚀 Quick training commands:")
    print("="*30)
    for dataset in datasets:
        if os.path.exists(dataset["file_path"]):
            print(f"python quick_train.py {dataset['file_path']} {dataset['text_column']} {dataset['sentiment_column']}")
    
    print(f"\n🎯 Or use the dataset manager:")
    print("python dataset_config.py")

if __name__ == "__main__":
    setup_sample_datasets()
