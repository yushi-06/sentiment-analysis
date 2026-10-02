#!/usr/bin/env python3
"""
ENHANCED BIAS-FIXED VERSION with Intelligent Analysis
Merges the original bias-fixed system with critical analysis and adaptive learning
"""

import os
import sys
from datetime import datetime
from adaptive_sentiment import AdaptiveSentimentAnalyzer

# Try to import intelligent features, fallback to basic if not available
try:
    from critical_analysis_system import CriticalAnalysisSystem
    from enhanced_adaptive_learning import EnhancedAdaptiveLearning
    INTELLIGENT_FEATURES = True
except ImportError:
    INTELLIGENT_FEATURES = False
    print("⚠️  Intelligent features not available. Using basic bias-fixed system.")

class EnhancedFixedSentimentAnalyzer:
    def __init__(self):
        print("Loading enhanced bias-fixed sentiment analyzer...")
        
        if INTELLIGENT_FEATURES:
            # Use intelligent system with critical analysis
            self.critical_analyzer = CriticalAnalysisSystem()
            self.enhanced_learner = self.critical_analyzer.enhanced_learner
            self.analyzer = self.enhanced_learner.analyzer
            print("✅ Enhanced intelligent system loaded successfully!")
            print("🧠 Features: Bias-correction + Critical Analysis + Adaptive Learning")
        else:
            # Fallback to basic adaptive system
            self.analyzer = AdaptiveSentimentAnalyzer()
            self.critical_analyzer = None
            self.enhanced_learner = None
            print("✅ Basic bias-fixed model loaded successfully!")
        
        # Session statistics
        self.session_stats = {
            'predictions_made': 0,
            'learning_events': 0,
            'queries_analyzed': 0,
            'session_start': datetime.now()
        }
    
    def predict_single(self, text, show_analysis=False):
        """Predict sentiment for a single text with optional intelligent analysis"""
        if not text.strip():
            return None
        
        self.session_stats['predictions_made'] += 1
        
        if INTELLIGENT_FEATURES and self.enhanced_learner:
            # Use enhanced prediction with confidence
            prediction, score, confidence, uncertain, contributions = self.enhanced_learner.predict_with_confidence(text, show_analysis)
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'uncertain': uncertain,
                'score': score
            }
            
            if show_analysis:
                print(f"\n📊 Enhanced Analysis for: '{text}'")
                print(f"   Prediction: {prediction}")
                print(f"   Confidence: {confidence:.3f} {'🔴 (Uncertain)' if uncertain else '🟢 (Confident)'}")
                print(f"   Raw Score: {score:.3f}")
            
            return result
        else:
            # Use basic prediction
            prediction, score, _ = self.analyzer.predict_sentiment(text)
            return {
                'prediction': prediction,
                'confidence': 0.5,  # Default confidence
                'uncertain': False,
                'score': score
            }
    
    def analyze_query_intelligently(self, query):
        """Perform intelligent analysis of the query if available"""
        if not INTELLIGENT_FEATURES or not self.critical_analyzer:
            return None
        
        self.session_stats['queries_analyzed'] += 1
        analysis = self.critical_analyzer.critically_analyze_query(query)
        
        return analysis
    
    def predict_batch(self, texts, show_details=False):
        """Predict sentiment for multiple texts with optional details"""
        if not texts or len(texts) == 0:
            return None
            
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return None
            
        results = []
        detailed_results = []
        
        for text in valid_texts:
            result = self.predict_single(text)
            if result:
                results.append(result['prediction'])
                if show_details:
                    detailed_results.append({
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'prediction': result['prediction'],
                        'confidence': result.get('confidence', 0.5),
                        'uncertain': result.get('uncertain', False)
                    })
        
        if not results:
            return None
            
        # Count results
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for result in results:
            counts[result] += 1
        
        # Calculate percentages
        total = len(results)
        percentages = {
            "Positive": round((counts["Positive"] / total) * 100, 2),
            "Negative": round((counts["Negative"] / total) * 100, 2),
            "Neutral": round((counts["Neutral"] / total) * 100, 2)
        }
        
        batch_result = {
            'percentages': percentages,
            'total_analyzed': total,
            'counts': counts
        }
        
        if show_details:
            batch_result['detailed_results'] = detailed_results
            
            # Calculate average confidence if available
            if INTELLIGENT_FEATURES:
                confidences = [r['confidence'] for r in detailed_results]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                uncertain_count = sum(1 for r in detailed_results if r['uncertain'])
                
                batch_result['avg_confidence'] = avg_confidence
                batch_result['uncertain_predictions'] = uncertain_count
        
        return batch_result
    
    def learn_from_feedback(self, text, predicted, correct):
        """Learn from user feedback"""
        if INTELLIGENT_FEATURES and self.enhanced_learner:
            # Get confidence for the prediction
            result = self.predict_single(text)
            confidence = result.get('confidence', 0.5)
            
            self.enhanced_learner.adaptive_learn_from_feedback(text, predicted, correct, confidence)
            self.session_stats['learning_events'] += 1
            print("✅ System learned from your feedback with adaptive learning!")
        elif hasattr(self.analyzer, 'learn_from_correction'):
            self.analyzer.learn_from_correction(text, predicted, correct)
            self.session_stats['learning_events'] += 1
            print("✅ System learned from your feedback!")
        else:
            print("⚠️  Learning not available in current configuration.")
    
    def show_session_stats(self):
        """Show current session statistics"""
        duration = datetime.now() - self.session_stats['session_start']
        
        print(f"\n📊 SESSION STATISTICS")
        print("="*30)
        print(f"Duration: {duration}")
        print(f"Predictions made: {self.session_stats['predictions_made']}")
        print(f"Learning events: {self.session_stats['learning_events']}")
        
        if INTELLIGENT_FEATURES:
            print(f"Queries analyzed: {self.session_stats['queries_analyzed']}")
            
            # Show system-wide stats
            if self.critical_analyzer:
                summary = self.critical_analyzer.get_analysis_summary()
                if isinstance(summary, dict):
                    print(f"\n🧠 SYSTEM-WIDE STATISTICS")
                    print(f"Total queries in database: {summary['total_queries']}")
                    print(f"Total insights generated: {summary['insights_generated']}")

def get_user_choice():
    """Get user choice between different modes"""
    print(f"\n🎯 Choose analysis mode:")
    print("1. Manual text input (basic)")
    print("2. Manual text input (with intelligent analysis)")
    print("3. Dataset analysis (basic)")
    print("4. Dataset analysis (detailed)")
    if INTELLIGENT_FEATURES:
        print("5. Interactive intelligent mode")
    print("6. Show session statistics")
    print("7. Exit")
    
    while True:
        choice = input("Enter your choice (1-7): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7']:
            return choice
        print("Please enter a valid choice (1-7).")

def manual_mode_basic(analyzer):
    """Handle basic manual text input mode"""
    print("\n📝 Manual Mode (BIAS-FIXED):")
    print("Enter text(s) to analyze. Type 'done' when finished, or 'quit' to exit.")
    
    texts = []
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() == 'quit':
            return
        elif text.lower() == 'done':
            break
        elif text:
            texts.append(text)
        else:
            print("Please enter some text or type 'done' to finish.")
    
    if not texts:
        print("No data to analyze.")
        return
    
    # Analyze each text individually
    print("\nResults:")
    for i, text in enumerate(texts, 1):
        result = analyzer.predict_single(text)
        if result:
            print(f"{i}. {result['prediction']}")
        else:
            print(f"{i}. No data to analyze.")

def manual_mode_intelligent(analyzer):
    """Handle intelligent manual text input mode"""
    print("\n🧠 Intelligent Manual Mode:")
    print("Enter text(s) to analyze with intelligent features.")
    print("Type 'done' when finished, or 'quit' to exit.")
    
    texts = []
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() == 'quit':
            return
        elif text.lower() == 'done':
            break
        elif text:
            texts.append(text)
        else:
            print("Please enter some text or type 'done' to finish.")
    
    if not texts:
        print("No data to analyze.")
        return
    
    # Analyze each text with intelligent features
    print("\n🤖 Intelligent Analysis Results:")
    print("="*40)
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Text: '{text}'")
        
        # Sentiment analysis
        result = analyzer.predict_single(text, show_analysis=True)
        
        # Critical analysis if available
        if INTELLIGENT_FEATURES and analyzer.critical_analyzer:
            analysis = analyzer.analyze_query_intelligently(text)
            if analysis and analysis['critical_insights']:
                print(f"   💡 Insights:")
                for insight in analysis['critical_insights'][:2]:  # Show first 2
                    print(f"   • {insight['insight']}")
        
        # Offer learning opportunity
        if result and result.get('uncertain'):
            feedback = input(f"   This prediction seems uncertain. Is '{result['prediction']}' correct? (y/n): ").strip().lower()
            if feedback == 'n':
                correct = input("   What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    analyzer.learn_from_feedback(text, result['prediction'], correct)

def dataset_mode_basic(analyzer):
    """Handle basic dataset file input mode"""
    print("\n📊 Dataset Mode (BIAS-FIXED):")
    
    # Get file path
    file_path = input("Enter the dataset file path: ").strip().strip('"')
    
    if not file_path:
        print("No data to analyze.")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        import pandas as pd
        
        # Load dataset
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV or Excel files.")
            return
        
        # Find text column
        column_name = "text"
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found.")
            print(f"Available columns: {', '.join(df.columns)}")
            column_name = input("Enter the correct column name: ").strip()
            
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found.")
                return
        
        # Get texts from column
        texts = df[column_name].astype(str).tolist()
        
        # Analyze texts
        result = analyzer.predict_batch(texts)
        
        if result is None:
            print("No data to analyze.")
            return
        
        # Output results
        percentages = result['percentages']
        print(f"\n📊 Analysis Results ({result['total_analyzed']} texts):")
        print(f"Positive: {percentages['Positive']}%")
        print(f"Negative: {percentages['Negative']}%")
        print(f"Neutral: {percentages['Neutral']}%")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def dataset_mode_detailed(analyzer):
    """Handle detailed dataset analysis"""
    print("\n📊 Detailed Dataset Analysis:")
    
    # Get file path
    file_path = input("Enter the dataset file path: ").strip().strip('"')
    
    if not file_path:
        print("No data to analyze.")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        import pandas as pd
        
        # Load dataset
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV or Excel files.")
            return
        
        # Find text column
        column_name = "text"
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found.")
            print(f"Available columns: {', '.join(df.columns)}")
            column_name = input("Enter the correct column name: ").strip()
            
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found.")
                return
        
        # Get texts from column (limit for detailed analysis)
        texts = df[column_name].astype(str).tolist()
        if len(texts) > 50:
            print(f"⚠️  Large dataset detected ({len(texts)} texts). Using first 50 for detailed analysis.")
            texts = texts[:50]
        
        # Analyze texts with details
        result = analyzer.predict_batch(texts, show_details=True)
        
        if result is None:
            print("No data to analyze.")
            return
        
        # Output summary results
        percentages = result['percentages']
        print(f"\n📊 Summary Results ({result['total_analyzed']} texts):")
        print(f"Positive: {percentages['Positive']}%")
        print(f"Negative: {percentages['Negative']}%")
        print(f"Neutral: {percentages['Neutral']}%")
        
        # Show detailed results if available
        if 'detailed_results' in result:
            if INTELLIGENT_FEATURES and 'avg_confidence' in result:
                print(f"\n🎯 Confidence Analysis:")
                print(f"Average confidence: {result['avg_confidence']:.3f}")
                print(f"Uncertain predictions: {result['uncertain_predictions']}")
            
            show_details = input("\nShow detailed results for each text? (y/n): ").strip().lower()
            if show_details == 'y':
                print(f"\n📝 Detailed Results:")
                for i, detail in enumerate(result['detailed_results'], 1):
                    uncertainty = "🔴" if detail.get('uncertain') else "🟢"
                    conf_str = f" (conf: {detail['confidence']:.3f})" if INTELLIGENT_FEATURES else ""
                    print(f"{i:2d}. {uncertainty} {detail['prediction']}{conf_str} - '{detail['text']}'")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def interactive_intelligent_mode(analyzer):
    """Interactive intelligent mode"""
    if not INTELLIGENT_FEATURES:
        print("❌ Intelligent features not available.")
        return
    
    print(f"\n🧠 Interactive Intelligent Mode")
    print("="*40)
    print("Ask questions, analyze text, and get intelligent insights!")
    print("Commands:")
    print("  - Type text to analyze")
    print("  - Type 'help' for more commands")
    print("  - Type 'quit' to exit")
    print("-"*40)
    
    while True:
        user_input = input("\n🎯 Enter your query: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'help':
            print("Available commands:")
            print("  - Any text for sentiment analysis")
            print("  - Questions for intelligent analysis")
            print("  - 'stats' for session statistics")
            print("  - 'quit' to exit")
            continue
        elif user_input.lower() == 'stats':
            analyzer.show_session_stats()
            continue
        elif not user_input:
            print("Please enter some text or a command.")
            continue
        
        try:
            # Perform intelligent analysis
            analysis = analyzer.analyze_query_intelligently(user_input)
            
            # Check if it needs sentiment analysis
            if analysis and any(keyword in user_input.lower() for keyword in ['sentiment', 'analyze', 'opinion', 'feeling']):
                result = analyzer.predict_single(user_input, show_analysis=True)
                
                if result and result.get('uncertain'):
                    feedback = input("This prediction seems uncertain. Provide feedback? (y/n): ").strip().lower()
                    if feedback == 'y':
                        correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                        if correct in ['Positive', 'Negative', 'Neutral']:
                            analyzer.learn_from_feedback(user_input, result['prediction'], correct)
            
            # Show critical insights if available
            if analysis and analysis['critical_insights']:
                print(f"\n💡 Insights:")
                for insight in analysis['critical_insights']:
                    print(f"• {insight['insight']}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function to run the enhanced bias-fixed sentiment analyzer"""
    print("=" * 70)
    if INTELLIGENT_FEATURES:
        print("🧠 ENHANCED BIAS-FIXED SENTIMENT ANALYSIS TOOL")
        print("   Features: Bias-correction + Critical Analysis + Adaptive Learning")
    else:
        print("🔧 BIAS-FIXED SENTIMENT ANALYSIS TOOL")
        print("   Features: Bias-correction + Basic Adaptive Learning")
    print("=" * 70)
    print("This version uses the adaptive system to avoid positive bias!")
    
    analyzer = EnhancedFixedSentimentAnalyzer()
    
    while True:
        choice = get_user_choice()
        
        if choice == '1':
            manual_mode_basic(analyzer)
        elif choice == '2':
            if INTELLIGENT_FEATURES:
                manual_mode_intelligent(analyzer)
            else:
                print("❌ Intelligent analysis not available. Using basic mode.")
                manual_mode_basic(analyzer)
        elif choice == '3':
            dataset_mode_basic(analyzer)
        elif choice == '4':
            dataset_mode_detailed(analyzer)
        elif choice == '5':
            if INTELLIGENT_FEATURES:
                interactive_intelligent_mode(analyzer)
            else:
                print("❌ Interactive intelligent mode not available.")
        elif choice == '6':
            analyzer.show_session_stats()
        elif choice == '7':
            break
        
        if choice != '6':  # Don't ask after showing stats
            continue_choice = input("\nReturn to main menu? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '']:
                break
    
    # Show final session summary
    analyzer.show_session_stats()
    print("\n💾 All learning progress has been automatically saved.")
    print("👋 Thank you for using the Enhanced Bias-Fixed Sentiment Analysis Tool!")

if __name__ == "__main__":
    main()
