#!/usr/bin/env python3
"""
Reset Knowledge Base
Clears learned patterns and starts fresh if predictions seem off
"""

import os
import shutil
from datetime import datetime

def reset_knowledge_base():
    """Reset the sentiment knowledge base"""
    print("🔄 RESETTING SENTIMENT KNOWLEDGE BASE")
    print("="*45)
    
    knowledge_files = [
        "sentiment_knowledge.pkl",
        "sentiment_knowledge_backup.pkl"
    ]
    
    backup_folder = "knowledge_backups"
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    # Backup existing files
    for file_name in knowledge_files:
        if os.path.exists(file_name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{backup_folder}/{file_name}_{timestamp}"
            shutil.move(file_name, backup_name)
            print(f"📦 Backed up: {file_name} -> {backup_name}")
    
    print("✅ Knowledge base reset complete!")
    print("📚 The system will start with fresh knowledge on next run")
    
    # Test the reset
    print("\n🧪 Testing fresh system...")
    try:
        from adaptive_sentiment import AdaptiveSentimentAnalyzer
        analyzer = AdaptiveSentimentAnalyzer()
        
        test_cases = [
            "This is amazing!",
            "Great product", 
            "I love it",
            "Terrible quality",
            "It's okay"
        ]
        
        print("\nFresh predictions:")
        for text in test_cases:
            pred, score, _ = analyzer.predict_sentiment(text)
            print(f"  '{text}' -> {pred} (score: {score:.3f})")
            
    except Exception as e:
        print(f"❌ Error testing: {e}")

if __name__ == "__main__":
    reset_knowledge_base()
