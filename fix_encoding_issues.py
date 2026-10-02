#!/usr/bin/env python3
"""
Fix Encoding Issues and Create Proper Dataset
"""

import pandas as pd
import os
from dataset_config import DatasetConfig

def fix_dataset_encoding():
    """Fix encoding issues and create proper datasets"""
    print("🔧 FIXING DATASET ENCODING ISSUES")
    print("="*45)
    
    # Check if the problematic file exists
    problem_file = "C:\\Users\\Sahil\\OneDrive\\Desktop\\sentiment-analysis\\my_reviews.csv.xlsx"
    
    if os.path.exists(problem_file):
        print(f"📂 Found problematic file: {os.path.basename(problem_file)}")
        
        try:
            # Try to read as Excel file
            print("🔄 Attempting to read as Excel file...")
            df = pd.read_excel(problem_file)
            
            print(f"✅ Successfully read Excel file!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"\n📝 First few rows:")
            print(df.head())
            
            # Save as proper CSV
            csv_file = "my_reviews_fixed.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"\n💾 Saved as proper CSV: {csv_file}")
            
            # Update dataset configuration
            config = DatasetConfig()
            
            # Remove the problematic entry
            config.remove_dataset("my_reviews.csv")
            
            # Add the fixed version
            text_col = df.columns[0] if len(df.columns) > 0 else "text"
            sentiment_col = df.columns[1] if len(df.columns) > 1 else "sentiment"
            
            config.add_dataset(
                "my_reviews_fixed",
                csv_file,
                text_col,
                sentiment_col,
                "Fixed version of my_reviews dataset"
            )
            
            print(f"✅ Updated dataset configuration!")
            
        except Exception as e:
            print(f"❌ Error reading Excel file: {e}")
            
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    print(f"🔄 Trying encoding: {encoding}")
                    df = pd.read_csv(problem_file, encoding=encoding)
                    
                    print(f"✅ Success with {encoding}!")
                    print(f"📊 Shape: {df.shape}")
                    print(f"📋 Columns: {list(df.columns)}")
                    
                    # Save with proper UTF-8 encoding
                    csv_file = "my_reviews_fixed.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    print(f"💾 Saved as proper CSV: {csv_file}")
                    
                    break
                    
                except Exception as enc_error:
                    print(f"❌ Failed with {encoding}: {enc_error}")
            
    else:
        print(f"❌ File not found: {problem_file}")
    
    # Create a sample my_reviews.csv for demonstration
    create_sample_my_reviews()

def create_sample_my_reviews():
    """Create a sample my_reviews.csv file"""
    print(f"\n📝 Creating sample my_reviews.csv...")
    
    sample_data = {
        'text': [
            "This product is amazing! I love it so much.",
            "Terrible quality, completely disappointed.",
            "It's okay, nothing special but works fine.",
            "Best purchase I've made this year!",
            "Poor customer service and late delivery.",
            "Average product, meets basic expectations.",
            "Excellent value for money, highly recommend!",
            "Waste of money, doesn't work as advertised.",
            "Decent quality for the price point.",
            "Outstanding performance and great design!",
            "Very frustrated with this purchase.",
            "It's fine, does what it's supposed to do.",
            "Incredible features and easy to use!",
            "Cheap materials and poor build quality.",
            "Fair product with reasonable functionality.",
            "Love the design and user interface!",
            "Difficult to set up and confusing manual.",
            "Standard product, nothing extraordinary.",
            "Superb quality and excellent durability!",
            "Not worth the price, many better options available."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive',
            'negative', 'neutral', 'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral', 'positive', 'negative'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('my_reviews.csv', index=False, encoding='utf-8')
    
    print(f"✅ Created my_reviews.csv with {len(df)} examples")
    print(f"📊 Distribution:")
    print(df['sentiment'].value_counts())
    
    # Update configuration
    config = DatasetConfig()
    config.add_dataset(
        "my_reviews",
        "my_reviews.csv",
        "text",
        "sentiment",
        "Sample reviews dataset for training"
    )
    
    print(f"✅ Added to dataset configuration!")

def main():
    """Main function"""
    fix_dataset_encoding()
    
    print(f"\n🎯 NEXT STEPS:")
    print("="*20)
    print("1. Use the training menu:")
    print("   python train_menu.py")
    print()
    print("2. Or train directly:")
    print("   python quick_train.py my_reviews.csv text sentiment")
    print()
    print("3. Test your improved model:")
    print("   python fixed_main.py")

if __name__ == "__main__":
    main()
