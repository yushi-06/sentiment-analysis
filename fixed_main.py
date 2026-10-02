#!/usr/bin/env python3
"""
BIAS-FIXED VERSION of main.py with Enhanced Features
This uses the adaptive system instead of the biased traditional model
Now includes confidence scoring and adaptive learning capabilities
"""

import os
from datetime import datetime
from adaptive_sentiment import AdaptiveSentimentAnalyzer

# Try to import enhanced features
try:
    from enhanced_adaptive_learning import EnhancedAdaptiveLearning
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False

class FixedSentimentAnalyzer:
    def __init__(self):
        print("Loading bias-fixed sentiment analyzer...")
        
        if ENHANCED_FEATURES:
            self.enhanced_learner = EnhancedAdaptiveLearning()
            self.analyzer = self.enhanced_learner.analyzer
            print("✅ Enhanced bias-fixed model loaded successfully!")
            print("🎯 Features: Bias-correction + Confidence scoring + Adaptive learning")
        else:
            self.analyzer = AdaptiveSentimentAnalyzer()
            self.enhanced_learner = None
            print("✅ Basic bias-fixed model loaded successfully!")
        
        # Session statistics
        self.session_stats = {
            'predictions_made': 0,
            'learning_events': 0,
            'session_start': datetime.now()
        }
    
    def predict_single(self, text, show_confidence=False):
        """Predict sentiment for a single text, with optional confidence info"""
        if not text.strip():
            return None
        
        self.session_stats['predictions_made'] += 1
        
        if ENHANCED_FEATURES and self.enhanced_learner:
            prediction, score, confidence, uncertain, _ = self.enhanced_learner.predict_with_confidence(text)
            
            if show_confidence:
                conf_indicator = "🔴 (Uncertain)" if uncertain else "🟢 (Confident)"
                print(f"   Prediction: {prediction} | Confidence: {confidence:.3f} {conf_indicator}")
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'uncertain': uncertain
            }
        else:
            prediction, score, _ = self.analyzer.predict_sentiment(text)
            # Convert score to confidence (score is typically between -1 and 1)
            # Higher absolute score means higher confidence
            confidence = min(abs(score), 1.0)  # Ensure it's between 0 and 1
            uncertain = confidence < 0.4  # Mark as uncertain if confidence is low
            return {'prediction': prediction, 'confidence': confidence, 'uncertain': uncertain}
    
    def predict_batch(self, texts, show_confidence_summary=False):
        """Predict sentiment for multiple texts, returns distribution with optional confidence info"""
        if not texts or len(texts) == 0:
            return None
            
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return None
            
        results = []
        confidences = []
        uncertain_count = 0
        
        for text in valid_texts:
            result = self.predict_single(text)
            if result:
                if isinstance(result, dict):
                    results.append(result['prediction'])
                    confidences.append(result.get('confidence', 0.5))
                    if result.get('uncertain', False):
                        uncertain_count += 1
                else:
                    results.append(result)
                    confidences.append(0.5)
        
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
        
        result_data = {
            'percentages': percentages,
            'total_analyzed': total
        }
        
        if show_confidence_summary and ENHANCED_FEATURES:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            result_data['avg_confidence'] = avg_confidence
            result_data['uncertain_predictions'] = uncertain_count
        
        return result_data

def get_user_choice():
    """Get user choice between different modes"""
    print("\n🎯 Choose analysis mode:")
    print("1. Manual text input (basic)")
    if ENHANCED_FEATURES:
        print("2. Manual text input (with confidence scoring)")
    print("3. Dataset analysis (basic)")
    if ENHANCED_FEATURES:
        print("4. Dataset analysis (with confidence summary)")
        print("5. Interactive learning mode")
    print("6. Show session statistics")
    print("7. Exit")
    
    while True:
        choice = input("Enter your choice: ").strip()
        valid_choices = ['1', '3', '6', '7']
        if ENHANCED_FEATURES:
            valid_choices.extend(['2', '4', '5'])
        
        if choice in valid_choices:
            return choice
        print(f"Please enter a valid choice: {', '.join(valid_choices)}")

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
    
    # Analyze each text individually with percentage
    print("\nResults:")
    for i, text in enumerate(texts, 1):
        result = analyzer.predict_single(text)
        if result:
            if isinstance(result, dict):
                prediction = result['prediction']
                confidence = result.get('confidence', 0.5)
                percentage = confidence * 100
                print(f"{i}. {prediction} ({percentage:.1f}%)")
            else:
                print(f"{i}. {result} (50.0%)")  # Default percentage for basic mode
        else:
            print(f"{i}. No data to analyze.")

def manual_mode_enhanced(analyzer):
    """Handle enhanced manual text input mode with confidence scoring"""
    print("\n🎯 Enhanced Manual Mode (BIAS-FIXED + CONFIDENCE):")
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
    
    # Analyze each text with confidence scoring and percentage
    print("\n🎯 Enhanced Results:")
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Text: '{text}'")
        result = analyzer.predict_single(text)
        
        if result:
            prediction = result['prediction']
            confidence = result.get('confidence', 0.5)
            percentage = confidence * 100
            uncertain = result.get('uncertain', False)
            
            status_icon = "🔴" if uncertain else "🟢"
            status_text = "Uncertain" if uncertain else "Confident"
            
            print(f"   Result: {prediction} ({percentage:.1f}%) {status_icon} {status_text}")
            
            # Offer learning opportunity for uncertain predictions
            if uncertain and ENHANCED_FEATURES:
                feedback = input("   This prediction seems uncertain. Is it correct? (y/n): ").strip().lower()
                if feedback == 'n':
                    correct = input("   What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                    if correct in ['Positive', 'Negative', 'Neutral']:
                        analyzer.learn_from_feedback(text, result['prediction'], correct)

def interactive_learning_mode(analyzer):
    """Interactive learning mode with feedback"""
    if not ENHANCED_FEATURES:
        print("❌ Enhanced features not available for interactive learning.")
        return
    
    print("\n🎓 Interactive Learning Mode:")
    print("The system will learn from your feedback to improve accuracy.")
    print("Type 'quit' to exit.")
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            break
        elif not text:
            continue
        
        result = analyzer.predict_single(text)
        
        if result:
            prediction = result['prediction']
            confidence = result.get('confidence', 0.5)
            percentage = confidence * 100
            uncertain = result.get('uncertain', False)
            
            status_icon = "🔴" if uncertain else "🟢"
            status_text = "Uncertain" if uncertain else "Confident"
            
            print(f"   Prediction: {prediction} ({percentage:.1f}%) {status_icon} {status_text}")
            
            feedback = input("Is this prediction correct? (y/n) or provide correct sentiment: ").strip()
            
            if feedback.lower() in ['n', 'no']:
                correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    analyzer.learn_from_feedback(text, result['prediction'], correct)
            elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != result['prediction']:
                    analyzer.learn_from_feedback(text, result['prediction'], feedback.title())
            elif feedback.lower() in ['y', 'yes']:
                print("✅ Thank you for confirming!")

def dataset_mode_basic(analyzer):
    """Handle basic dataset file input mode"""
    print("\n📊 Dataset Mode (BIAS-FIXED):")
    
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
        percentages = result['percentages'] if isinstance(result, dict) else result
        total = result.get('total_analyzed', len(texts)) if isinstance(result, dict) else len(texts)
        
        print(f"\n📊 Results ({total} texts analyzed):")
        print(f"Positive: {percentages['Positive']}%")
        print(f"Negative: {percentages['Negative']}%")
        print(f"Neutral: {percentages['Neutral']}%")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def dataset_mode_enhanced(analyzer):
    """Handle enhanced dataset analysis with confidence summary"""
    print("\n🎯 Enhanced Dataset Mode (BIAS-FIXED + CONFIDENCE):")
    
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
        
        # Analyze texts with confidence summary
        result = analyzer.predict_batch(texts, show_confidence_summary=True)
        
        if result is None:
            print("No data to analyze.")
            return
        
        # Output results
        percentages = result['percentages']
        total = result['total_analyzed']
        
        print(f"\n🎯 Enhanced Results ({total} texts analyzed):")
        print(f"Positive: {percentages['Positive']}%")
        print(f"Negative: {percentages['Negative']}%")
        print(f"Neutral: {percentages['Neutral']}%")
        
        if 'avg_confidence' in result:
            print(f"\n📊 Confidence Analysis:")
            print(f"Average confidence: {result['avg_confidence']:.3f}")
            print(f"Uncertain predictions: {result['uncertain_predictions']}")
            
            if result['uncertain_predictions'] > 0:
                uncertainty_rate = (result['uncertain_predictions'] / total) * 100
                print(f"Uncertainty rate: {uncertainty_rate:.1f}%")
        
    except Exception as e:
        print(f"Error processing file: {e}")

# Add methods to the FixedSentimentAnalyzer class
def add_methods_to_analyzer():
    """Add learning and stats methods to the analyzer class"""
    def learn_from_feedback(self, text, predicted, correct):
        """Learn from user feedback"""
        if ENHANCED_FEATURES and self.enhanced_learner:
            # Get confidence for the prediction
            result = self.predict_single(text)
            confidence = result.get('confidence', 0.5)
            
            self.enhanced_learner.adaptive_learn_from_feedback(text, predicted, correct, confidence)
            self.session_stats['learning_events'] += 1
            print("✅ System learned from your feedback with enhanced adaptive learning!")
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
        
        if ENHANCED_FEATURES and hasattr(self, 'enhanced_learner'):
            print(f"\n🎯 Enhanced Features Active:")
            print(f"Confidence scoring: ✅")
            print(f"Adaptive learning: ✅")
            print(f"Uncertainty detection: ✅")
    
    # Add methods to the class
    FixedSentimentAnalyzer.learn_from_feedback = learn_from_feedback
    FixedSentimentAnalyzer.show_session_stats = show_session_stats

# Call the function to add methods
add_methods_to_analyzer()

def main():
    """Main function to run the enhanced bias-fixed sentiment analyzer"""
    print("=" * 70)
    if ENHANCED_FEATURES:
        print("🎯 ENHANCED BIAS-FIXED SENTIMENT ANALYSIS TOOL")
        print("   Features: Bias-correction + Confidence scoring + Adaptive learning")
    else:
        print("🔧 BIAS-FIXED SENTIMENT ANALYSIS TOOL")
    print("=" * 70)
    print("This version uses the adaptive system to avoid positive bias!")

    analyzer = FixedSentimentAnalyzer()

    while True:
        choice = get_user_choice()

        if choice == '1':
            manual_mode_basic(analyzer)
        elif choice == '2' and ENHANCED_FEATURES:
            manual_mode_enhanced(analyzer)
        elif choice == '3':
            dataset_mode_basic(analyzer)
        elif choice == '4' and ENHANCED_FEATURES:
            dataset_mode_enhanced(analyzer)
        elif choice == '5' and ENHANCED_FEATURES:
            interactive_learning_mode(analyzer)
        elif choice == '6':
            analyzer.show_session_stats()
        elif choice == '7':
            break
        else:
            print("❌ Invalid choice or feature not available.")
            continue

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
