#!/usr/bin/env python3
"""
Training Menu
Easy access to dataset training
"""

import os
import subprocess
from dataset_config import DatasetConfig

def show_training_menu():
    """Show training menu with available datasets"""
    print("🎓 DATASET TRAINING MENU")
    print("="*35)
    
    config = DatasetConfig()
    config.update_dataset_status()
    
    # Get available datasets
    datasets = []
    if "datasets" in config.datasets:
        for name, dataset_config in config.datasets["datasets"].items():
            if dataset_config.get("exists", False):
                datasets.append({
                    "name": name,
                    "config": dataset_config
                })
    
    if not datasets:
        print("❌ No datasets available for training")
        print("\n🔧 Setup sample datasets first:")
        print("python setup_datasets.py")
        return
    
    # Show available datasets
    print("📂 Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        config_data = dataset["config"]
        print(f"{i}. {dataset['name']}")
        print(f"   📄 {os.path.basename(config_data['file_path'])}")
        print(f"   📝 {config_data.get('description', 'No description')}")
        print()
    
    print(f"{len(datasets) + 1}. Add new dataset")
    print(f"{len(datasets) + 2}. Exit")
    
    # Get user choice
    while True:
        try:
            choice = int(input(f"\nEnter your choice (1-{len(datasets) + 2}): "))
            
            if 1 <= choice <= len(datasets):
                # Train from selected dataset
                selected = datasets[choice - 1]
                train_from_dataset(selected)
                break
            elif choice == len(datasets) + 1:
                # Add new dataset
                add_new_dataset(config)
                break
            elif choice == len(datasets) + 2:
                # Exit
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(datasets) + 2}")
        
        except ValueError:
            print("❌ Please enter a valid number")

def train_from_dataset(dataset):
    """Train from selected dataset"""
    name = dataset["name"]
    config_data = dataset["config"]
    
    print(f"\n🎓 Training from: {name}")
    print("="*40)
    print(f"📄 File: {config_data['file_path']}")
    print(f"📝 Text column: {config_data['text_column']}")
    print(f"💭 Sentiment column: {config_data['sentiment_column']}")
    
    confirm = input(f"\nStart training? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Training cancelled")
        return
    
    # Run training
    cmd = [
        "python", "quick_train.py",
        config_data["file_path"],
        config_data["text_column"],
        config_data["sentiment_column"]
    ]
    
    try:
        print(f"\n🚀 Starting training...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\n📊 Training output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print(result.stderr)
    
    except Exception as e:
        print(f"❌ Error running training: {e}")

def add_new_dataset(config):
    """Add new dataset interactively"""
    print(f"\n📝 Add New Dataset")
    print("-" * 20)
    
    name = input("Dataset name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
        return
    
    file_path = input("File path: ").strip().strip('"')
    if not file_path:
        print("❌ File path cannot be empty")
        return
    
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found: {file_path}")
        continue_anyway = input("Add anyway? (y/n): ").strip().lower()
        if continue_anyway not in ['y', 'yes']:
            return
    
    text_col = input("Text column name (default: text): ").strip() or "text"
    sentiment_col = input("Sentiment column name (default: sentiment): ").strip() or "sentiment"
    description = input("Description (optional): ").strip()
    
    config.add_dataset(name, file_path, text_col, sentiment_col, description)
    print(f"✅ Dataset '{name}' added!")

def main():
    """Main menu function"""
    while True:
        show_training_menu()
        
        continue_choice = input(f"\nReturn to menu? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("👋 Training menu closed!")

if __name__ == "__main__":
    main()
