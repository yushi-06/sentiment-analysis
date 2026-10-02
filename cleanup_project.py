#!/usr/bin/env python3
"""
Project Cleanup Script
Removes unwanted files and organizes the sentiment analysis project
"""

import os
import shutil
from datetime import datetime

class ProjectCleanup:
    def __init__(self, project_path):
        self.project_path = project_path
        
        # Files to keep (core functionality)
        self.keep_files = {
            # Core adaptive learning system
            'adaptive_sentiment.py',
            'enhanced_adaptive_learning.py',
            'online_learning_system.py',
            'ensemble_adaptive_learning.py',
            'learning_analytics_dashboard.py',
            'critical_analysis_system.py',
            'unified_adaptive_interface.py',
            
            # Main interfaces
            'fixed_main.py',  # Your bias-fixed main
            'auto_trainer.py',
            
            # Essential utilities
            'utils.py',
            'config.py',
            
            # Data files
            'fix_bias_data.csv',
            'sample_data.csv',
            'requirements.txt',
            
            # Documentation
            'ADAPTIVE_LEARNING_README.md',
            'QUICK_START_GUIDE.md',
            'README.md',
            
            # Knowledge bases
            'sentiment_knowledge_backup_20250929_052219.pkl',
            
            # Testing
            'comprehensive_test.py',
            'verify_fix.py'
        }
        
        # Files to remove (redundant/outdated)
        self.remove_files = {
            # Redundant main files
            'main.py',  # Original biased version
            'main_adaptive.py',  # Superseded by enhanced system
            'main_backup.py',
            'run_sentiment_analysis.py',
            
            # Redundant training files
            'train.py',  # Superseded by auto_trainer
            'train_sentiment.py',
            'train_negative.py',
            
            # Redundant test files
            'test.py',  # Basic test, superseded by comprehensive_test
            'simple_test.py',
            'quick_test.py',
            'debug_test.py',
            'final_test.py',
            'interactive_test.py',
            'test_bias_fix.py',
            'test_negative.py',
            
            # Redundant fix files
            'fix_positive_bias.py',  # Functionality integrated
            'fresh_start_fix.py',
            'immediate_bias_fix.py',
            
            # Redundant documentation
            'BIAS_FIX_README.md',  # Integrated into main docs
            'CRITICAL_ANALYSIS_REPORT.md',  # Outdated
            'ENHANCED_TRAINING_README.md',  # Superseded
            'ISSUES_FOUND_AND_FIXES.md',  # Historical
            'README_ADAPTIVE.md',  # Superseded by ADAPTIVE_LEARNING_README
            
            # Redundant utilities
            'infer_sentiment.py',  # Functionality integrated
            'comprehensive_issue_check.py'  # One-time use
        }
        
        # Directories to clean
        self.clean_directories = {
            '.venv',
            '.venve', 
            'venv',
            'tmp_trainer',
            '__pycache__'
        }
        
        # Archive directory for removed files
        self.archive_dir = os.path.join(project_path, 'archived_files')
    
    def analyze_project(self):
        """Analyze current project structure"""
        print("🔍 ANALYZING PROJECT STRUCTURE")
        print("="*40)
        
        all_files = []
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), self.project_path)
                all_files.append(rel_path)
        
        # Categorize files
        keep_count = 0
        remove_count = 0
        unknown_count = 0
        
        print("📊 File Analysis:")
        
        for file in all_files:
            filename = os.path.basename(file)
            if filename in self.keep_files:
                keep_count += 1
            elif filename in self.remove_files:
                remove_count += 1
            else:
                unknown_count += 1
        
        print(f"  ✅ Files to keep: {keep_count}")
        print(f"  🗑️  Files to remove: {remove_count}")
        print(f"  ❓ Unknown files: {unknown_count}")
        
        # Show directory sizes
        print(f"\n📁 Directory Analysis:")
        for item in os.listdir(self.project_path):
            item_path = os.path.join(self.project_path, item)
            if os.path.isdir(item_path):
                size = self._get_dir_size(item_path)
                print(f"  {item}: {size:.1f} MB")
        
        return {
            'total_files': len(all_files),
            'keep_count': keep_count,
            'remove_count': remove_count,
            'unknown_count': unknown_count
        }
    
    def cleanup_project(self, create_archive=True):
        """Perform project cleanup"""
        print(f"\n🧹 STARTING PROJECT CLEANUP")
        print("="*40)
        
        # Create archive directory if requested
        if create_archive and not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
            print(f"📁 Created archive directory: {self.archive_dir}")
        
        removed_count = 0
        archived_count = 0
        
        # Remove/archive files
        for filename in self.remove_files:
            file_path = os.path.join(self.project_path, filename)
            if os.path.exists(file_path):
                if create_archive:
                    # Move to archive
                    archive_path = os.path.join(self.archive_dir, filename)
                    shutil.move(file_path, archive_path)
                    print(f"📦 Archived: {filename}")
                    archived_count += 1
                else:
                    # Delete permanently
                    os.remove(file_path)
                    print(f"🗑️  Removed: {filename}")
                    removed_count += 1
        
        # Clean directories
        for dirname in self.clean_directories:
            dir_path = os.path.join(self.project_path, dirname)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                if create_archive:
                    archive_path = os.path.join(self.archive_dir, dirname)
                    shutil.move(dir_path, archive_path)
                    print(f"📦 Archived directory: {dirname}")
                    archived_count += 1
                else:
                    shutil.rmtree(dir_path)
                    print(f"🗑️  Removed directory: {dirname}")
                    removed_count += 1
        
        # Clean results directory (keep structure, remove old files)
        results_dir = os.path.join(self.project_path, 'results')
        if os.path.exists(results_dir):
            self._clean_results_directory(results_dir)
        
        print(f"\n✅ CLEANUP COMPLETED")
        if create_archive:
            print(f"📦 Files archived: {archived_count}")
        else:
            print(f"🗑️  Files removed: {removed_count}")
    
    def _clean_results_directory(self, results_dir):
        """Clean old files from results directory"""
        print(f"\n🧹 Cleaning results directory...")
        
        # Keep only recent files (last 7 days)
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 60 * 60)
        cleaned_count = 0
        
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            if os.path.isfile(file_path):
                file_time = os.path.getmtime(file_path)
                if file_time < cutoff_time:
                    os.remove(file_path)
                    cleaned_count += 1
        
        print(f"🧹 Cleaned {cleaned_count} old result files")
    
    def _get_dir_size(self, directory):
        """Get directory size in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, FileNotFoundError):
            pass
        return total_size / (1024 * 1024)  # Convert to MB
    
    def create_project_structure_doc(self):
        """Create documentation of the cleaned project structure"""
        doc_content = f"""# 📁 Cleaned Project Structure

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Core System Files

### Adaptive Learning System
- `adaptive_sentiment.py` - Base adaptive sentiment analyzer
- `enhanced_adaptive_learning.py` - Enhanced learning with confidence scoring
- `online_learning_system.py` - Real-time continuous learning
- `ensemble_adaptive_learning.py` - Multiple model ensemble system
- `learning_analytics_dashboard.py` - Analytics and visualization
- `critical_analysis_system.py` - Critical query analysis with knowledge DB

### Main Interfaces
- `unified_adaptive_interface.py` - **MAIN ENTRY POINT** - Unified interface for all systems
- `fixed_main.py` - Bias-corrected sentiment analysis interface
- `auto_trainer.py` - Automated training system

### Utilities & Configuration
- `utils.py` - Utility functions
- `config.py` - Configuration settings

## 📊 Data Files
- `fix_bias_data.csv` - Bias correction training data
- `sample_data.csv` - Sample dataset for testing
- `sentiment_knowledge_backup_20250929_052219.pkl` - Trained model knowledge

## 🧪 Testing & Verification
- `comprehensive_test.py` - Comprehensive testing suite
- `verify_fix.py` - Bias verification tools

## 📚 Documentation
- `ADAPTIVE_LEARNING_README.md` - Complete adaptive learning documentation
- `QUICK_START_GUIDE.md` - Quick start guide
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies

## 🗂️ Directories
- `sentiment_model/` - Model files and checkpoints
- `results/` - Analysis results and outputs (cleaned of old files)
- `archived_files/` - Archived redundant files

## 🚀 How to Use

### Quick Start
```bash
python unified_adaptive_interface.py
```

### For Specific Features
```bash
# Enhanced adaptive learning
python enhanced_adaptive_learning.py

# Critical analysis system
python critical_analysis_system.py

# Learning analytics
python learning_analytics_dashboard.py
```

## 🎯 Recommended Workflow
1. Start with `unified_adaptive_interface.py`
2. Initialize Enhanced Adaptive System
3. Use Interactive Learning mode
4. Monitor progress with Analytics Dashboard
5. Use Critical Analysis for complex queries

---
*This structure provides a clean, organized, and efficient sentiment analysis system with advanced adaptive learning capabilities.*
"""
        
        doc_path = os.path.join(self.project_path, 'PROJECT_STRUCTURE.md')
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        print(f"📄 Created project structure documentation: PROJECT_STRUCTURE.md")
    
    def interactive_cleanup(self):
        """Interactive cleanup interface"""
        print("🧹 PROJECT CLEANUP TOOL")
        print("="*30)
        
        # Analyze first
        analysis = self.analyze_project()
        
        print(f"\n🎯 CLEANUP OPTIONS:")
        print("1. Full cleanup with archive (recommended)")
        print("2. Full cleanup without archive (permanent deletion)")
        print("3. Custom cleanup")
        print("4. Just create project structure documentation")
        print("5. Cancel")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            self.cleanup_project(create_archive=True)
            self.create_project_structure_doc()
            
        elif choice == '2':
            confirm = input("⚠️  This will permanently delete files. Continue? (yes/no): ").strip().lower()
            if confirm == 'yes':
                self.cleanup_project(create_archive=False)
                self.create_project_structure_doc()
            else:
                print("❌ Cleanup cancelled.")
                
        elif choice == '3':
            self._custom_cleanup()
            
        elif choice == '4':
            self.create_project_structure_doc()
            
        elif choice == '5':
            print("❌ Cleanup cancelled.")
        else:
            print("❌ Invalid choice.")
    
    def _custom_cleanup(self):
        """Custom cleanup with user selection"""
        print(f"\n🎯 CUSTOM CLEANUP")
        print("Select files to remove:")
        
        selected_files = []
        
        print(f"\n📄 Files marked for removal:")
        for i, filename in enumerate(sorted(self.remove_files), 1):
            file_path = os.path.join(self.project_path, filename)
            exists = "✅" if os.path.exists(file_path) else "❌"
            print(f"{i:2d}. {exists} {filename}")
        
        selection = input(f"\nEnter file numbers to remove (comma-separated, or 'all'): ").strip()
        
        if selection.lower() == 'all':
            selected_files = list(self.remove_files)
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                file_list = sorted(self.remove_files)
                selected_files = [file_list[i] for i in indices if 0 <= i < len(file_list)]
            except (ValueError, IndexError):
                print("❌ Invalid selection.")
                return
        
        if selected_files:
            create_archive = input("Create archive? (y/n): ").strip().lower() == 'y'
            
            # Create temporary remove set
            temp_remove_files = self.remove_files.copy()
            self.remove_files = set(selected_files)
            
            self.cleanup_project(create_archive=create_archive)
            
            # Restore original remove set
            self.remove_files = temp_remove_files

def main():
    """Main cleanup interface"""
    project_path = os.path.dirname(os.path.abspath(__file__))
    cleanup = ProjectCleanup(project_path)
    cleanup.interactive_cleanup()

if __name__ == "__main__":
    main()
