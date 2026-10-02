#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies and components are properly installed
"""

import sys
import importlib

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 CHECKING DEPENDENCIES")
    print("="*40)
    
    # Core dependencies
    core_deps = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'torch', 'transformers', 'scikit-learn'
    ]
    
    # Optional dependencies
    optional_deps = [
        'openpyxl', 'xlrd', 'tqdm', 'requests', 'yaml'
    ]
    
    print("📦 Core Dependencies:")
    core_status = True
    for dep in core_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ❌ {dep}: Not installed")
            core_status = False
    
    print(f"\n📦 Optional Dependencies:")
    optional_status = True
    for dep in optional_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ⚠️  {dep}: Not installed (optional)")
    
    return core_status

def check_system_files():
    """Check if all system files are present"""
    print(f"\n🔍 CHECKING SYSTEM FILES")
    print("="*40)
    
    # Core system files
    core_files = [
        'fixed_main.py',
        'adaptive_sentiment.py',
        'enhanced_adaptive_learning.py',
        'critical_analysis_system.py',
        'intelligent_main.py',
        'start_system.py'
    ]
    
    # Data files
    data_files = [
        'fix_bias_data.csv',
        'requirements.txt'
    ]
    
    print("📄 Core System Files:")
    files_status = True
    for file in core_files:
        try:
            with open(file, 'r') as f:
                lines = len(f.readlines())
            print(f"  ✅ {file}: {lines} lines")
        except FileNotFoundError:
            print(f"  ❌ {file}: Missing")
            files_status = False
    
    print(f"\n📊 Data Files:")
    for file in data_files:
        try:
            with open(file, 'r') as f:
                lines = len(f.readlines())
            print(f"  ✅ {file}: {lines} lines")
        except FileNotFoundError:
            print(f"  ❌ {file}: Missing")
            files_status = False
    
    return files_status

def test_basic_functionality():
    """Test basic system functionality"""
    print(f"\n🧪 TESTING BASIC FUNCTIONALITY")
    print("="*40)
    
    try:
        # Test adaptive sentiment analyzer
        from adaptive_sentiment import AdaptiveSentimentAnalyzer
        analyzer = AdaptiveSentimentAnalyzer()
        
        # Test prediction
        test_text = "I love this product"
        prediction, score, _ = analyzer.predict_sentiment(test_text)
        print(f"  ✅ Basic sentiment analysis: '{test_text}' → {prediction} (score: {score:.3f})")
        
        # Test enhanced features if available
        try:
            from enhanced_adaptive_learning import EnhancedAdaptiveLearning
            enhanced = EnhancedAdaptiveLearning()
            pred, score, conf, uncertain, _ = enhanced.predict_with_confidence(test_text)
            print(f"  ✅ Enhanced analysis: Confidence {conf:.3f}, Uncertain: {uncertain}")
        except ImportError:
            print(f"  ⚠️  Enhanced features not available")
        
        # Test critical analysis if available
        try:
            from critical_analysis_system import CriticalAnalysisSystem
            critic = CriticalAnalysisSystem()
            analysis = critic.critically_analyze_query("How does sentiment analysis work?")
            print(f"  ✅ Critical analysis: {len(analysis['critical_insights'])} insights generated")
        except ImportError:
            print(f"  ⚠️  Critical analysis not available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing functionality: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 SENTIMENT ANALYSIS SYSTEM SETUP VERIFICATION")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Run checks
    deps_ok = check_dependencies()
    files_ok = check_system_files()
    func_ok = test_basic_functionality()
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("="*30)
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ Issues'}")
    print(f"System Files: {'✅ OK' if files_ok else '❌ Issues'}")
    print(f"Functionality: {'✅ OK' if func_ok else '❌ Issues'}")
    
    if deps_ok and files_ok and func_ok:
        print(f"\n🎉 SYSTEM READY!")
        print("="*20)
        print("Your enhanced sentiment analysis system is fully operational!")
        print("\n🚀 Quick Start Options:")
        print("  python fixed_main.py          - Enhanced bias-fixed system")
        print("  python start_system.py        - System launcher")
        print("  python intelligent_main.py    - Full intelligent interface")
        
    else:
        print(f"\n⚠️  SETUP ISSUES DETECTED")
        print("="*30)
        if not deps_ok:
            print("• Install missing dependencies with: pip install -r requirements.txt")
        if not files_ok:
            print("• Some system files are missing - check project structure")
        if not func_ok:
            print("• System functionality test failed - check error messages above")

if __name__ == "__main__":
    main()
