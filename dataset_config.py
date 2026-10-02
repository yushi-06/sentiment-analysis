#!/usr/bin/env python3
"""
Dataset Configuration System
Manage dataset paths and configurations
"""

import os
import json
from datetime import datetime

class DatasetConfig:
    def __init__(self):
        self.config_file = "dataset_paths.json"
        self.datasets = self.load_config()
    
    def load_config(self):
        """Load dataset configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default configuration
        return {
            "datasets": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def save_config(self):
        """Save dataset configuration to file"""
        self.datasets["last_updated"] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)
    
    def add_dataset(self, name, file_path, text_column="text", sentiment_column="sentiment", description=""):
        """Add a dataset configuration"""
        if "datasets" not in self.datasets:
            self.datasets["datasets"] = {}
        
        self.datasets["datasets"][name] = {
            "file_path": file_path,
            "text_column": text_column,
            "sentiment_column": sentiment_column,
            "description": description,
            "added_date": datetime.now().isoformat(),
            "exists": os.path.exists(file_path)
        }
        
        self.save_config()
        print(f"✅ Dataset '{name}' added to configuration")
    
    def list_datasets(self):
        """List all configured datasets"""
        if "datasets" not in self.datasets or not self.datasets["datasets"]:
            print("📂 No datasets configured yet")
            return []
        
        print("📂 CONFIGURED DATASETS")
        print("="*40)
        
        dataset_list = []
        for name, config in self.datasets["datasets"].items():
            exists_icon = "✅" if config.get("exists", False) else "❌"
            print(f"{exists_icon} {name}")
            print(f"   Path: {config['file_path']}")
            print(f"   Columns: {config['text_column']} → {config['sentiment_column']}")
            if config.get("description"):
                print(f"   Description: {config['description']}")
            print()
            
            dataset_list.append({
                "name": name,
                "config": config
            })
        
        return dataset_list
    
    def get_dataset(self, name):
        """Get dataset configuration by name"""
        if "datasets" in self.datasets and name in self.datasets["datasets"]:
            return self.datasets["datasets"][name]
        return None
    
    def remove_dataset(self, name):
        """Remove dataset from configuration"""
        if "datasets" in self.datasets and name in self.datasets["datasets"]:
            del self.datasets["datasets"][name]
            self.save_config()
            print(f"✅ Dataset '{name}' removed from configuration")
            return True
        else:
            print(f"❌ Dataset '{name}' not found")
            return False
    
    def update_dataset_status(self):
        """Update existence status of all datasets"""
        if "datasets" not in self.datasets:
            return
        
        updated = 0
        for name, config in self.datasets["datasets"].items():
            old_status = config.get("exists", False)
            new_status = os.path.exists(config["file_path"])
            
            if old_status != new_status:
                config["exists"] = new_status
                updated += 1
        
        if updated > 0:
            self.save_config()
            print(f"✅ Updated status for {updated} datasets")

def main():
    """Interactive dataset configuration"""
    config = DatasetConfig()
    
    print("📂 DATASET CONFIGURATION MANAGER")
    print("="*40)
    
    while True:
        print("\n🎯 Options:")
        print("1. Add new dataset")
        print("2. List all datasets")
        print("3. Remove dataset")
        print("4. Update dataset status")
        print("5. Train from configured dataset")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Add dataset
            print("\n📝 Add New Dataset")
            print("-" * 20)
            
            name = input("Dataset name: ").strip()
            if not name:
                print("❌ Name cannot be empty")
                continue
            
            file_path = input("File path: ").strip().strip('"')
            if not file_path:
                print("❌ File path cannot be empty")
                continue
            
            text_col = input("Text column name (default: text): ").strip() or "text"
            sentiment_col = input("Sentiment column name (default: sentiment): ").strip() or "sentiment"
            description = input("Description (optional): ").strip()
            
            config.add_dataset(name, file_path, text_col, sentiment_col, description)
        
        elif choice == '2':
            # List datasets
            config.update_dataset_status()
            config.list_datasets()
        
        elif choice == '3':
            # Remove dataset
            datasets = config.list_datasets()
            if not datasets:
                continue
            
            name = input("Enter dataset name to remove: ").strip()
            config.remove_dataset(name)
        
        elif choice == '4':
            # Update status
            config.update_dataset_status()
            print("✅ Dataset status updated")
        
        elif choice == '5':
            # Train from dataset
            datasets = config.list_datasets()
            if not datasets:
                continue
            
            name = input("Enter dataset name to train from: ").strip()
            dataset_config = config.get_dataset(name)
            
            if not dataset_config:
                print(f"❌ Dataset '{name}' not found")
                continue
            
            if not dataset_config.get("exists", False):
                print(f"❌ Dataset file not found: {dataset_config['file_path']}")
                continue
            
            # Run training with Windows-safe version
            print(f"\nTraining from dataset: {name}")
            import subprocess
            cmd = [
                "python", "windows_safe_train.py",
                dataset_config["file_path"],
                dataset_config["text_column"],
                dataset_config["sentiment_column"]
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Training failed: {e}")
        
        elif choice == '6':
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")
    
    print("👋 Dataset configuration complete!")

if __name__ == "__main__":
    main()
