#!/usr/bin/env python3
"""
Startup Script for Intelligent Sentiment Analysis System
Main entry point with system selection and initialization
"""

import os
import sys
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_modules = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("⚠️  Missing dependencies:")
        for module in missing_modules:
            print(f"  - {module}")
        print(f"\nInstall with: pip install {' '.join(missing_modules)}")
        return False
    
    return True

def show_system_overview():
    """Show system overview and capabilities"""
    print("🧠 INTELLIGENT SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    print("A comprehensive adaptive learning system that:")
    print("  🔍 Critically analyzes your queries")
    print("  💭 Provides intelligent sentiment analysis")
    print("  🎓 Learns and adapts from your feedback")
    print("  📊 Tracks learning progress with analytics")
    print("  🧠 Builds a knowledge database of insights")
    print("  ⚡ Offers multiple learning approaches")

def show_system_options():
    """Show available system options"""
    print(f"\n🎯 SYSTEM OPTIONS:")
    print("="*25)
    
    print("🚀 RECOMMENDED (New Users):")
    print("  1. Intelligent Main System - Complete system with critical analysis")
    print("  2. Demo Mode - Interactive demonstration of all features")
    
    print("\n🎛️ ADVANCED OPTIONS:")
    print("  3. Enhanced Adaptive Learning - Advanced learning with confidence scoring")
    print("  4. Online Learning System - Real-time continuous learning")
    print("  5. Ensemble Learning - Multiple models for highest accuracy")
    print("  6. Unified Interface - Access all systems from one interface")
    
    print("\n🛠️ UTILITIES:")
    print("  7. Learning Analytics Dashboard - Visualize learning progress")
    print("  8. Critical Analysis Only - Query analysis without sentiment")
    print("  9. Bias-Fixed Main (Legacy) - Your original bias-corrected system")
    
    print("\n📊 MAINTENANCE:")
    print("  10. System Status Check")
    print("  11. Clean Project Files")
    print("  12. Exit")

def launch_system(choice):
    """Launch the selected system"""
    try:
        if choice == '1':
            print("🚀 Launching Intelligent Main System...")
            from intelligent_main import main
            main()
            
        elif choice == '2':
            print("🎮 Launching Demo Mode...")
            from demo_intelligent_system import main
            main()
            
        elif choice == '3':
            print("🎓 Launching Enhanced Adaptive Learning...")
            from enhanced_adaptive_learning import main
            main()
            
        elif choice == '4':
            print("🌊 Launching Online Learning System...")
            from online_learning_system import main
            main()
            
        elif choice == '5':
            print("🎭 Launching Ensemble Learning...")
            from ensemble_adaptive_learning import main
            main()
            
        elif choice == '6':
            print("🎯 Launching Unified Interface...")
            from unified_adaptive_interface import main
            main()
            
        elif choice == '7':
            print("📊 Launching Analytics Dashboard...")
            from learning_analytics_dashboard import main
            main()
            
        elif choice == '8':
            print("🧠 Launching Critical Analysis System...")
            from critical_analysis_system import main
            main()
            
        elif choice == '9':
            print("🔧 Launching Bias-Fixed Main (Legacy)...")
            from fixed_main import main
            main()
            
        elif choice == '10':
            show_system_status()
            
        elif choice == '11':
            print("🧹 Launching Project Cleanup...")
            from cleanup_project import main
            main()
            
        elif choice == '12':
            print("👋 Goodbye!")
            return False
        else:
            print("❌ Invalid choice. Please try again.")
            
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        print("Please check your installation.")
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        print("Please try again or contact support.")
    
    return True

def show_system_status():
    """Show current system status"""
    print(f"\n📊 SYSTEM STATUS CHECK")
    print("="*30)
    
    # Check core files
    core_files = [
        'adaptive_sentiment.py',
        'enhanced_adaptive_learning.py',
        'critical_analysis_system.py',
        'intelligent_main.py',
        'fix_bias_data.csv'
    ]
    
    print("📁 Core Files:")
    for file in core_files:
        exists = "✅" if os.path.exists(file) else "❌"
        print(f"  {exists} {file}")
    
    # Check knowledge files
    knowledge_files = [
        'sentiment_knowledge_backup_20250929_052219.pkl',
        'critical_analysis_db.pkl'
    ]
    
    print(f"\n🧠 Knowledge Files:")
    for file in knowledge_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            print(f"  ❓ {file} (will be created on first use)")
    
    # Check directories
    directories = ['results', 'sentiment_model', 'archived_files']
    
    print(f"\n📂 Directories:")
    for dir_name in directories:
        if os.path.exists(dir_name):
            count = len(os.listdir(dir_name)) if os.path.isdir(dir_name) else 0
            print(f"  ✅ {dir_name}/ ({count} items)")
        else:
            print(f"  ❓ {dir_name}/ (missing)")
    
    # System recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not os.path.exists('critical_analysis_db.pkl'):
        print("  • Run the system once to initialize knowledge database")
    
    if not os.path.exists('archived_files'):
        print("  • Consider running project cleanup to organize files")
    
    print("  • For best results, start with option 1 (Intelligent Main System)")
    print("  • Use option 2 (Demo Mode) to explore all features")

def show_quick_start_guide():
    """Show quick start guide"""
    print(f"\n🚀 QUICK START GUIDE")
    print("="*25)
    print("""
For New Users:
1. Choose option 1 (Intelligent Main System)
2. Try these example queries:
   • "I love this new product!"
   • "How does sentiment analysis work?"
   • "This is okay but could be better"
3. Provide feedback when asked to help the system learn

For Advanced Users:
• Option 3: Enhanced learning with confidence scoring
• Option 4: Real-time online learning
• Option 5: Ensemble of multiple models
• Option 7: Analytics dashboard for insights

For Developers:
• All systems are modular and can be imported
• Knowledge databases persist between sessions
• Analytics data can be exported for analysis
    """)

def main():
    """Main startup interface"""
    print("🎯 INTELLIGENT SENTIMENT ANALYSIS SYSTEM STARTUP")
    print("="*65)
    print(f"Initialized on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Show system overview
    show_system_overview()
    
    # Main loop
    while True:
        show_system_options()
        
        print(f"\n❓ Need help? Type 'help' for quick start guide")
        choice = input("Enter your choice (1-12) or 'help': ").strip().lower()
        
        if choice == 'help':
            show_quick_start_guide()
            continue
        
        if not launch_system(choice):
            break
        
        # Ask if user wants to continue
        if choice not in ['10', '11']:  # Don't ask after status/cleanup
            continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                print("👋 Goodbye!")
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        print("Please check your installation and try again.")
