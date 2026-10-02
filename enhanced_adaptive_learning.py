#!/usr/bin/env python3
"""
Enhanced Adaptive Learning System for Sentiment Analysis
Adds advanced learning features like confidence-based learning, active learning, and real-time feedback
"""

import json
import os
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from adaptive_sentiment import AdaptiveSentimentAnalyzer
import pandas as pd

class EnhancedAdaptiveLearning:
    def __init__(self, base_analyzer=None):
        self.analyzer = base_analyzer if base_analyzer else AdaptiveSentimentAnalyzer()
        
        # Enhanced learning parameters
        self.confidence_threshold = 0.3  # Threshold for uncertain predictions
        self.learning_rate = 1.0  # Dynamic learning rate
        self.uncertainty_buffer = deque(maxlen=100)  # Store uncertain predictions
        
        # Learning analytics
        self.learning_history = []
        self.performance_metrics = {
            'accuracy_over_time': [],
            'confidence_over_time': [],
            'learning_events': [],
            'uncertainty_scores': []
        }
        
        # Active learning queue
        self.uncertain_predictions = []
        self.feedback_queue = deque(maxlen=50)
        
        print("🚀 Enhanced Adaptive Learning System initialized!")
        print(f"📊 Confidence threshold: {self.confidence_threshold}")
        print(f"🎯 Learning rate: {self.learning_rate}")
    
    def predict_with_confidence(self, text, show_analysis=False):
        """Enhanced prediction with confidence scoring and uncertainty detection"""
        prediction, score, contributions = self.analyzer.predict_sentiment(text, show_analysis)
        
        # Calculate confidence based on score magnitude and word agreement
        confidence = abs(score)
        
        # Adjust confidence based on word agreement
        if contributions:
            positive_words = sum(1 for c in contributions if c['score'] > 0)
            negative_words = sum(1 for c in contributions if c['score'] < 0)
            total_sentiment_words = positive_words + negative_words
            
            if total_sentiment_words > 0:
                agreement_ratio = max(positive_words, negative_words) / total_sentiment_words
                confidence *= agreement_ratio
        
        # Determine if prediction is uncertain
        is_uncertain = confidence < self.confidence_threshold
        
        # Store uncertainty for active learning
        if is_uncertain:
            self.uncertainty_buffer.append({
                'text': text,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        # Update performance metrics
        self.performance_metrics['confidence_over_time'].append({
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'prediction': prediction
        })
        
        return prediction, score, confidence, is_uncertain, contributions
    
    def adaptive_learn_from_feedback(self, text, predicted_sentiment, correct_sentiment, confidence):
        """Enhanced learning with confidence-based learning rate adjustment"""
        
        # Adjust learning rate based on confidence
        # Lower confidence = higher learning rate (learn more from mistakes)
        adjusted_learning_rate = self.learning_rate * (1 + (1 - confidence))
        
        # Learn from correction with adjusted rate
        self.analyzer.learn_from_correction(text, predicted_sentiment, correct_sentiment)
        
        # Apply additional learning based on confidence
        if confidence < 0.5:  # Very uncertain prediction
            # Learn more aggressively
            words = self.analyzer.preprocess_text(text)
            for word in words:
                # Boost learning for uncertain predictions
                self.analyzer.word_sentiment_counts[word][correct_sentiment.lower()] += int(adjusted_learning_rate)
        
        # Record learning event
        learning_event = {
            'text': text,
            'predicted': predicted_sentiment,
            'correct': correct_sentiment,
            'confidence': confidence,
            'learning_rate': adjusted_learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_history.append(learning_event)
        self.performance_metrics['learning_events'].append(learning_event)
        
        print(f"📚 Enhanced learning applied (rate: {adjusted_learning_rate:.2f})")
        print(f"🎯 Confidence was: {confidence:.3f}")
    
    def get_uncertain_predictions_for_review(self, limit=10):
        """Get most uncertain predictions for human review (Active Learning)"""
        if not self.uncertainty_buffer:
            return []
        
        # Sort by confidence (lowest first) and return most uncertain
        uncertain_list = list(self.uncertainty_buffer)
        uncertain_list.sort(key=lambda x: x['confidence'])
        
        return uncertain_list[:limit]
    
    def interactive_learning_with_confidence(self):
        """Enhanced interactive learning mode with confidence display and active learning"""
        print("\n" + "="*70)
        print("🧠 ENHANCED ADAPTIVE SENTIMENT ANALYZER - CONFIDENCE LEARNING")
        print("="*70)
        print("Features:")
        print("  ✨ Real-time confidence scoring")
        print("  🎯 Adaptive learning rates")
        print("  🔍 Active learning suggestions")
        print("  📊 Learning analytics")
        print("\nCommands:")
        print("  - Type text to analyze")
        print("  - Type 'review' to review uncertain predictions")
        print("  - Type 'stats' to see learning statistics")
        print("  - Type 'analytics' to view learning analytics")
        print("  - Type 'quit' to exit")
        print("-"*70)
        
        session_start = datetime.now()
        session_predictions = 0
        session_corrections = 0
        
        while True:
            text = input("\n🔤 Enter text to analyze: ").strip()
            
            if text.lower() == 'quit':
                self._show_session_summary(session_start, session_predictions, session_corrections)
                break
                
            elif text.lower() == 'review':
                self._review_uncertain_predictions()
                continue
                
            elif text.lower() == 'stats':
                self.show_enhanced_stats()
                continue
                
            elif text.lower() == 'analytics':
                self.show_learning_analytics()
                continue
                
            elif not text:
                print("Please enter some text.")
                continue
            
            # Enhanced prediction with confidence
            prediction, score, confidence, is_uncertain, contributions = self.predict_with_confidence(text, show_analysis=True)
            session_predictions += 1
            
            # Display enhanced results
            print(f"\n🤖 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.3f} {'🔴 (Uncertain)' if is_uncertain else '🟢 (Confident)'}")
            print(f"📈 Raw Score: {score:.3f}")
            
            if is_uncertain:
                print("💡 This prediction has low confidence - consider providing feedback!")
            
            # Ask for feedback
            feedback = input("\nIs this correct? (y/n) or provide correct answer (Positive/Negative/Neutral): ").strip()
            
            if feedback.lower() in ['n', 'no']:
                correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    self.adaptive_learn_from_feedback(text, prediction, correct, confidence)
                    session_corrections += 1
                else:
                    print("Invalid sentiment. Please use: Positive, Negative, or Neutral")
                    
            elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != prediction:
                    self.adaptive_learn_from_feedback(text, prediction, feedback.title(), confidence)
                    session_corrections += 1
                else:
                    print("✅ Thanks for confirming!")
                    
            elif feedback.lower() in ['y', 'yes']:
                print("✅ Thanks for confirming!")
            else:
                print("No feedback recorded.")
    
    def _review_uncertain_predictions(self):
        """Review and get feedback on uncertain predictions"""
        uncertain = self.get_uncertain_predictions_for_review(5)
        
        if not uncertain:
            print("🎉 No uncertain predictions to review!")
            return
        
        print(f"\n🔍 REVIEWING {len(uncertain)} UNCERTAIN PREDICTIONS")
        print("="*50)
        
        for i, item in enumerate(uncertain, 1):
            print(f"\n{i}. Text: '{item['text']}'")
            print(f"   Prediction: {item['prediction']} (confidence: {item['confidence']:.3f})")
            
            feedback = input("   Correct sentiment? (Positive/Negative/Neutral or 'skip'): ").strip()
            
            if feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != item['prediction']:
                    self.adaptive_learn_from_feedback(
                        item['text'], item['prediction'], feedback.title(), item['confidence']
                    )
                    print("   ✅ Learned from correction!")
                else:
                    print("   ✅ Prediction confirmed!")
            elif feedback.lower() == 'skip':
                continue
            else:
                print("   ⏭️  Skipped")
    
    def _show_session_summary(self, start_time, predictions, corrections):
        """Show summary of the learning session"""
        duration = datetime.now() - start_time
        
        print(f"\n📊 SESSION SUMMARY")
        print("="*30)
        print(f"Duration: {duration}")
        print(f"Predictions made: {predictions}")
        print(f"Corrections received: {corrections}")
        if predictions > 0:
            print(f"Correction rate: {(corrections/predictions)*100:.1f}%")
        print(f"Total learning events: {len(self.learning_history)}")
        print("👋 Thank you for helping me learn!")
    
    def show_enhanced_stats(self):
        """Show enhanced learning statistics"""
        print(f"\n📈 ENHANCED LEARNING STATISTICS")
        print("="*40)
        
        # Basic stats
        self.analyzer.show_learning_stats()
        
        # Enhanced stats
        print(f"\n🚀 Enhanced Features:")
        print(f"Learning events: {len(self.learning_history)}")
        print(f"Uncertain predictions: {len(self.uncertainty_buffer)}")
        print(f"Current learning rate: {self.learning_rate}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Recent learning events
        if self.learning_history:
            print(f"\n🔄 Recent Learning Events:")
            for event in self.learning_history[-3:]:
                print(f"  '{event['text'][:30]}...' → {event['correct']} (conf: {event['confidence']:.3f})")
        
        # Confidence distribution
        if self.performance_metrics['confidence_over_time']:
            confidences = [m['confidence'] for m in self.performance_metrics['confidence_over_time']]
            avg_confidence = np.mean(confidences)
            print(f"\n📊 Confidence Metrics:")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Low confidence predictions: {sum(1 for c in confidences if c < self.confidence_threshold)}")
    
    def show_learning_analytics(self):
        """Display learning analytics and trends"""
        print(f"\n📊 LEARNING ANALYTICS DASHBOARD")
        print("="*50)
        
        if not self.learning_history:
            print("No learning data available yet.")
            return
        
        # Learning rate over time
        learning_rates = [event['learning_rate'] for event in self.learning_history]
        confidences = [event['confidence'] for event in self.learning_history]
        
        print(f"📈 Learning Trends:")
        print(f"  Total learning events: {len(self.learning_history)}")
        print(f"  Average learning rate: {np.mean(learning_rates):.3f}")
        print(f"  Average confidence in corrections: {np.mean(confidences):.3f}")
        
        # Recent performance
        recent_events = self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
        recent_confidences = [event['confidence'] for event in recent_events]
        
        if recent_confidences:
            print(f"\n🔄 Recent Performance (last {len(recent_events)} corrections):")
            print(f"  Average confidence: {np.mean(recent_confidences):.3f}")
            print(f"  Improving trend: {'📈 Yes' if len(recent_confidences) > 1 and recent_confidences[-1] > recent_confidences[0] else '📉 Stable/Declining'}")
        
        # Uncertainty analysis
        uncertain_count = len(self.uncertainty_buffer)
        total_predictions = len(self.performance_metrics['confidence_over_time'])
        
        if total_predictions > 0:
            uncertainty_rate = (uncertain_count / total_predictions) * 100
            print(f"\n🎯 Uncertainty Analysis:")
            print(f"  Current uncertain predictions: {uncertain_count}")
            print(f"  Uncertainty rate: {uncertainty_rate:.1f}%")
            print(f"  Recommendation: {'🟢 Good' if uncertainty_rate < 20 else '🟡 Consider more training' if uncertainty_rate < 40 else '🔴 Needs significant training'}")
    
    def save_learning_session(self, filename=None):
        """Save the current learning session data"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_session_{timestamp}.json"
        
        session_data = {
            'learning_history': self.learning_history,
            'performance_metrics': self.performance_metrics,
            'uncertainty_buffer': list(self.uncertainty_buffer),
            'session_info': {
                'confidence_threshold': self.confidence_threshold,
                'learning_rate': self.learning_rate,
                'total_events': len(self.learning_history),
                'saved_at': datetime.now().isoformat()
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"💾 Learning session saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving session: {e}")
    
    def load_learning_session(self, filename):
        """Load a previous learning session"""
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            self.learning_history.extend(session_data.get('learning_history', []))
            
            # Merge performance metrics
            for key, value in session_data.get('performance_metrics', {}).items():
                if key in self.performance_metrics:
                    self.performance_metrics[key].extend(value)
            
            # Load uncertainty buffer
            uncertainty_data = session_data.get('uncertainty_buffer', [])
            for item in uncertainty_data:
                self.uncertainty_buffer.append(item)
            
            print(f"📖 Loaded learning session from {filename}")
            print(f"📊 Added {len(session_data.get('learning_history', []))} learning events")
            
        except Exception as e:
            print(f"❌ Error loading session: {e}")

def main():
    """Main interface for enhanced adaptive learning"""
    print("🚀 ENHANCED ADAPTIVE LEARNING SYSTEM")
    print("="*50)
    
    # Initialize enhanced learning system
    enhanced_learner = EnhancedAdaptiveLearning()
    
    while True:
        print(f"\nChoose mode:")
        print("1. Interactive Learning with Confidence")
        print("2. Review Uncertain Predictions")
        print("3. Show Enhanced Statistics")
        print("4. Show Learning Analytics")
        print("5. Save Learning Session")
        print("6. Load Learning Session")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            enhanced_learner.interactive_learning_with_confidence()
            
        elif choice == '2':
            enhanced_learner._review_uncertain_predictions()
            
        elif choice == '3':
            enhanced_learner.show_enhanced_stats()
            
        elif choice == '4':
            enhanced_learner.show_learning_analytics()
            
        elif choice == '5':
            filename = input("Enter filename (or press Enter for auto): ").strip()
            enhanced_learner.save_learning_session(filename if filename else None)
            
        elif choice == '6':
            filename = input("Enter filename to load: ").strip()
            if filename:
                enhanced_learner.load_learning_session(filename)
            
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
