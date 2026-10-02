#!/usr/bin/env python3
"""
Ensemble Adaptive Learning System for Sentiment Analysis
Combines multiple adaptive models for improved accuracy and robustness
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import pickle
from adaptive_sentiment import AdaptiveSentimentAnalyzer
from enhanced_adaptive_learning import EnhancedAdaptiveLearning

class EnsembleAdaptiveLearning:
    def __init__(self, num_models=3, ensemble_strategy='voting'):
        """
        Initialize ensemble of adaptive models
        
        Args:
            num_models: Number of models in the ensemble
            ensemble_strategy: 'voting', 'weighted', 'confidence', or 'stacking'
        """
        self.num_models = num_models
        self.ensemble_strategy = ensemble_strategy
        
        # Create ensemble of enhanced adaptive learners
        self.models = []
        for i in range(num_models):
            model = EnhancedAdaptiveLearning()
            # Slightly different initialization for diversity
            model.analyzer.confidence_threshold = 0.2 + (i * 0.1)  # Vary thresholds
            model.learning_rate = 0.8 + (i * 0.1)  # Vary learning rates
            self.models.append(model)
        
        # Ensemble-specific parameters
        self.model_weights = np.ones(num_models) / num_models  # Equal weights initially
        self.model_performance = [[] for _ in range(num_models)]  # Track individual performance
        
        # Ensemble learning history
        self.ensemble_history = []
        self.consensus_threshold = 0.6  # Minimum agreement for confident prediction
        
        print(f"🎭 Ensemble Adaptive Learning initialized!")
        print(f"   Models: {num_models}")
        print(f"   Strategy: {ensemble_strategy}")
        print(f"   Consensus threshold: {self.consensus_threshold}")
    
    def predict_ensemble(self, text, show_analysis=False):
        """Make ensemble prediction using all models"""
        individual_predictions = []
        individual_confidences = []
        individual_scores = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            pred, score, conf, uncertain, contrib = model.predict_with_confidence(text, show_analysis=False)
            individual_predictions.append(pred)
            individual_confidences.append(conf)
            individual_scores.append(score)
        
        # Apply ensemble strategy
        if self.ensemble_strategy == 'voting':
            final_prediction, ensemble_confidence = self._majority_voting(
                individual_predictions, individual_confidences
            )
        elif self.ensemble_strategy == 'weighted':
            final_prediction, ensemble_confidence = self._weighted_voting(
                individual_predictions, individual_confidences, individual_scores
            )
        elif self.ensemble_strategy == 'confidence':
            final_prediction, ensemble_confidence = self._confidence_based_ensemble(
                individual_predictions, individual_confidences, individual_scores
            )
        elif self.ensemble_strategy == 'stacking':
            final_prediction, ensemble_confidence = self._stacking_ensemble(
                individual_predictions, individual_confidences, individual_scores
            )
        else:
            final_prediction, ensemble_confidence = self._majority_voting(
                individual_predictions, individual_confidences
            )
        
        # Calculate consensus
        consensus = self._calculate_consensus(individual_predictions)
        is_uncertain = ensemble_confidence < 0.3 or consensus < self.consensus_threshold
        
        if show_analysis:
            self._show_ensemble_analysis(text, individual_predictions, individual_confidences, 
                                       final_prediction, ensemble_confidence, consensus)
        
        return final_prediction, ensemble_confidence, is_uncertain, individual_predictions, consensus
    
    def _majority_voting(self, predictions, confidences):
        """Simple majority voting ensemble"""
        vote_counts = Counter(predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        
        # Calculate ensemble confidence as average confidence of agreeing models
        agreeing_confidences = [conf for pred, conf in zip(predictions, confidences) 
                               if pred == final_prediction]
        ensemble_confidence = np.mean(agreeing_confidences) if agreeing_confidences else 0.0
        
        return final_prediction, ensemble_confidence
    
    def _weighted_voting(self, predictions, confidences, scores):
        """Weighted voting based on model performance"""
        sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            weight = self.model_weights[i] * conf
            sentiment_scores[pred] += weight
        
        final_prediction = max(sentiment_scores, key=sentiment_scores.get)
        ensemble_confidence = sentiment_scores[final_prediction] / sum(sentiment_scores.values())
        
        return final_prediction, ensemble_confidence
    
    def _confidence_based_ensemble(self, predictions, confidences, scores):
        """Ensemble based on confidence weighting"""
        # Weight by confidence
        weighted_votes = defaultdict(float)
        total_confidence = sum(confidences)
        
        if total_confidence == 0:
            return self._majority_voting(predictions, confidences)
        
        for pred, conf in zip(predictions, confidences):
            weighted_votes[pred] += conf / total_confidence
        
        final_prediction = max(weighted_votes, key=weighted_votes.get)
        ensemble_confidence = weighted_votes[final_prediction]
        
        return final_prediction, ensemble_confidence
    
    def _stacking_ensemble(self, predictions, confidences, scores):
        """Meta-learning ensemble (simplified stacking)"""
        # For now, use a simple heuristic-based meta-learner
        # In a full implementation, this would be a trained meta-model
        
        # Heuristic: If high-confidence models agree, trust them more
        high_conf_predictions = [pred for pred, conf in zip(predictions, confidences) if conf > 0.5]
        
        if len(high_conf_predictions) >= 2:
            vote_counts = Counter(high_conf_predictions)
            if vote_counts.most_common(1)[0][1] >= 2:  # At least 2 high-conf models agree
                final_prediction = vote_counts.most_common(1)[0][0]
                ensemble_confidence = 0.8  # High confidence due to agreement
                return final_prediction, ensemble_confidence
        
        # Fall back to weighted voting
        return self._weighted_voting(predictions, confidences, scores)
    
    def _calculate_consensus(self, predictions):
        """Calculate consensus among models"""
        vote_counts = Counter(predictions)
        max_votes = vote_counts.most_common(1)[0][1]
        consensus = max_votes / len(predictions)
        return consensus
    
    def _show_ensemble_analysis(self, text, individual_predictions, individual_confidences, 
                               final_prediction, ensemble_confidence, consensus):
        """Show detailed ensemble analysis"""
        print(f"\n🎭 ENSEMBLE ANALYSIS")
        print(f"Text: '{text}'")
        print(f"Individual predictions:")
        for i, (pred, conf) in enumerate(zip(individual_predictions, individual_confidences)):
            print(f"  Model {i+1}: {pred} (conf: {conf:.3f}, weight: {self.model_weights[i]:.3f})")
        print(f"Final prediction: {final_prediction}")
        print(f"Ensemble confidence: {ensemble_confidence:.3f}")
        print(f"Consensus: {consensus:.3f}")
        print(f"Strategy: {self.ensemble_strategy}")
    
    def learn_ensemble(self, text, predicted_sentiment, correct_sentiment):
        """Learn from feedback across all models in ensemble"""
        individual_predictions = []
        
        # Get individual predictions for performance tracking
        for model in self.models:
            pred, _, conf, _, _ = model.predict_with_confidence(text)
            individual_predictions.append((pred, conf))
        
        # Train each model
        for i, model in enumerate(self.models):
            pred, conf = individual_predictions[i]
            model.adaptive_learn_from_feedback(text, pred, correct_sentiment, conf)
            
            # Track individual model performance
            is_correct = pred == correct_sentiment
            self.model_performance[i].append(is_correct)
        
        # Update model weights based on recent performance
        self._update_model_weights()
        
        # Record ensemble learning event
        ensemble_event = {
            'text': text,
            'individual_predictions': [pred for pred, _ in individual_predictions],
            'individual_confidences': [conf for _, conf in individual_predictions],
            'correct_sentiment': correct_sentiment,
            'timestamp': datetime.now().isoformat(),
            'model_weights': self.model_weights.tolist()
        }
        
        self.ensemble_history.append(ensemble_event)
        
        print(f"🎭 Ensemble learned from: '{text[:30]}...' → {correct_sentiment}")
        print(f"📊 Updated model weights: {[f'{w:.3f}' for w in self.model_weights]}")
    
    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        window_size = 20  # Look at last 20 predictions
        
        new_weights = []
        for i in range(self.num_models):
            if len(self.model_performance[i]) >= window_size:
                recent_performance = self.model_performance[i][-window_size:]
                accuracy = sum(recent_performance) / len(recent_performance)
            elif len(self.model_performance[i]) > 0:
                accuracy = sum(self.model_performance[i]) / len(self.model_performance[i])
            else:
                accuracy = 0.5  # Default
            
            new_weights.append(max(0.1, accuracy))  # Minimum weight of 0.1
        
        # Normalize weights
        total_weight = sum(new_weights)
        if total_weight > 0:
            self.model_weights = np.array(new_weights) / total_weight
        else:
            self.model_weights = np.ones(self.num_models) / self.num_models
    
    def interactive_ensemble_learning(self):
        """Interactive learning mode for ensemble"""
        print(f"\n🎭 ENSEMBLE ADAPTIVE LEARNING MODE")
        print("="*60)
        print("Features:")
        print("  🎯 Multiple model predictions")
        print("  📊 Consensus analysis")
        print("  ⚖️ Dynamic model weighting")
        print("  🔄 Ensemble learning")
        print("\nCommands:")
        print("  - Type text to get ensemble predictions")
        print("  - Type 'weights' to see current model weights")
        print("  - Type 'performance' to see model performance")
        print("  - Type 'strategy' to change ensemble strategy")
        print("  - Type 'stats' to see ensemble statistics")
        print("  - Type 'quit' to exit")
        print("-"*60)
        
        while True:
            text = input("\n🎭 Enter text for ensemble analysis: ").strip()
            
            if text.lower() == 'quit':
                break
            elif text.lower() == 'weights':
                self._show_model_weights()
                continue
            elif text.lower() == 'performance':
                self._show_model_performance()
                continue
            elif text.lower() == 'strategy':
                self._change_ensemble_strategy()
                continue
            elif text.lower() == 'stats':
                self._show_ensemble_stats()
                continue
            elif not text:
                print("Please enter some text.")
                continue
            
            # Get ensemble prediction
            prediction, confidence, uncertain, individual_preds, consensus = self.predict_ensemble(text, show_analysis=True)
            
            print(f"\n🎭 Ensemble Result:")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Consensus: {consensus:.3f}")
            print(f"   Status: {'🔴 Uncertain' if uncertain else '🟢 Confident'}")
            
            # Ask for feedback
            feedback = input("\nIs this correct? (y/n) or provide correct answer: ").strip()
            
            if feedback.lower() in ['n', 'no']:
                correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    self.learn_ensemble(text, prediction, correct)
                else:
                    print("Invalid sentiment.")
            elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != prediction:
                    self.learn_ensemble(text, prediction, feedback.title())
                else:
                    print("✅ Ensemble prediction confirmed!")
            elif feedback.lower() in ['y', 'yes']:
                print("✅ Ensemble prediction confirmed!")
    
    def _show_model_weights(self):
        """Show current model weights"""
        print(f"\n⚖️ CURRENT MODEL WEIGHTS")
        print("="*30)
        for i, weight in enumerate(self.model_weights):
            print(f"Model {i+1}: {weight:.3f}")
        print(f"Strategy: {self.ensemble_strategy}")
    
    def _show_model_performance(self):
        """Show individual model performance"""
        print(f"\n📊 MODEL PERFORMANCE")
        print("="*30)
        for i in range(self.num_models):
            if self.model_performance[i]:
                accuracy = sum(self.model_performance[i]) / len(self.model_performance[i])
                recent_accuracy = sum(self.model_performance[i][-10:]) / min(10, len(self.model_performance[i]))
                print(f"Model {i+1}:")
                print(f"  Overall accuracy: {accuracy:.3f}")
                print(f"  Recent accuracy: {recent_accuracy:.3f}")
                print(f"  Total predictions: {len(self.model_performance[i])}")
            else:
                print(f"Model {i+1}: No performance data")
    
    def _change_ensemble_strategy(self):
        """Change ensemble strategy"""
        print(f"\nCurrent strategy: {self.ensemble_strategy}")
        print("Available strategies:")
        print("1. voting - Simple majority voting")
        print("2. weighted - Performance-weighted voting")
        print("3. confidence - Confidence-based weighting")
        print("4. stacking - Meta-learning approach")
        
        choice = input("Enter new strategy (1-4): ").strip()
        
        strategies = {'1': 'voting', '2': 'weighted', '3': 'confidence', '4': 'stacking'}
        if choice in strategies:
            self.ensemble_strategy = strategies[choice]
            print(f"✅ Strategy changed to: {self.ensemble_strategy}")
        else:
            print("Invalid choice.")
    
    def _show_ensemble_stats(self):
        """Show ensemble statistics"""
        print(f"\n📊 ENSEMBLE STATISTICS")
        print("="*40)
        print(f"Number of models: {self.num_models}")
        print(f"Ensemble strategy: {self.ensemble_strategy}")
        print(f"Consensus threshold: {self.consensus_threshold}")
        print(f"Total ensemble events: {len(self.ensemble_history)}")
        
        if self.ensemble_history:
            # Calculate ensemble accuracy
            correct_predictions = 0
            for event in self.ensemble_history:
                # Simulate ensemble prediction for this event
                individual_preds = event['individual_predictions']
                vote_counts = Counter(individual_preds)
                ensemble_pred = vote_counts.most_common(1)[0][0]
                if ensemble_pred == event['correct_sentiment']:
                    correct_predictions += 1
            
            ensemble_accuracy = correct_predictions / len(self.ensemble_history)
            print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
            
            # Show recent consensus trends
            recent_events = self.ensemble_history[-10:]
            consensus_values = []
            for event in recent_events:
                consensus = self._calculate_consensus(event['individual_predictions'])
                consensus_values.append(consensus)
            
            if consensus_values:
                avg_consensus = np.mean(consensus_values)
                print(f"Recent average consensus: {avg_consensus:.3f}")
    
    def save_ensemble_state(self, filename=None):
        """Save ensemble state"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_state_{timestamp}.pkl"
        
        ensemble_state = {
            'num_models': self.num_models,
            'ensemble_strategy': self.ensemble_strategy,
            'model_weights': self.model_weights.tolist(),
            'model_performance': self.model_performance,
            'ensemble_history': self.ensemble_history,
            'consensus_threshold': self.consensus_threshold,
            'models': []
        }
        
        # Save individual model states
        for i, model in enumerate(self.models):
            model_state = {
                'analyzer_state': {
                    'positive_words': list(model.analyzer.positive_words),
                    'negative_words': list(model.analyzer.negative_words),
                    'neutral_words': list(model.analyzer.neutral_words),
                    'word_sentiment_counts': dict(model.analyzer.word_sentiment_counts),
                    'user_corrections': model.analyzer.user_corrections
                },
                'learning_history': model.learning_history,
                'performance_metrics': model.performance_metrics
            }
            ensemble_state['models'].append(model_state)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(ensemble_state, f)
            print(f"💾 Ensemble state saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving ensemble state: {e}")
    
    def load_ensemble_state(self, filename):
        """Load ensemble state"""
        try:
            with open(filename, 'rb') as f:
                ensemble_state = pickle.load(f)
            
            # Restore ensemble parameters
            self.ensemble_strategy = ensemble_state['ensemble_strategy']
            self.model_weights = np.array(ensemble_state['model_weights'])
            self.model_performance = ensemble_state['model_performance']
            self.ensemble_history = ensemble_state['ensemble_history']
            self.consensus_threshold = ensemble_state['consensus_threshold']
            
            # Restore individual models
            for i, model_state in enumerate(ensemble_state['models']):
                if i < len(self.models):
                    analyzer_state = model_state['analyzer_state']
                    
                    # Restore analyzer state
                    self.models[i].analyzer.positive_words.update(analyzer_state['positive_words'])
                    self.models[i].analyzer.negative_words.update(analyzer_state['negative_words'])
                    self.models[i].analyzer.neutral_words.update(analyzer_state['neutral_words'])
                    self.models[i].analyzer.word_sentiment_counts.update(analyzer_state['word_sentiment_counts'])
                    self.models[i].analyzer.user_corrections = analyzer_state['user_corrections']
                    
                    # Restore learning history
                    self.models[i].learning_history = model_state['learning_history']
                    self.models[i].performance_metrics = model_state['performance_metrics']
            
            print(f"📖 Ensemble state loaded from {filename}")
            print(f"🎭 Restored {len(ensemble_state['models'])} models")
            
        except Exception as e:
            print(f"❌ Error loading ensemble state: {e}")

def main():
    """Main interface for ensemble adaptive learning"""
    print("🎭 ENSEMBLE ADAPTIVE LEARNING SYSTEM")
    print("="*50)
    
    # Initialize ensemble
    num_models = input("Number of models in ensemble (default 3): ").strip()
    num_models = int(num_models) if num_models else 3
    
    strategy = input("Ensemble strategy (voting/weighted/confidence/stacking, default voting): ").strip()
    strategy = strategy if strategy in ['voting', 'weighted', 'confidence', 'stacking'] else 'voting'
    
    ensemble = EnsembleAdaptiveLearning(num_models=num_models, ensemble_strategy=strategy)
    
    while True:
        print(f"\nChoose mode:")
        print("1. Interactive Ensemble Learning")
        print("2. Show Ensemble Statistics")
        print("3. Show Model Performance")
        print("4. Change Ensemble Strategy")
        print("5. Save Ensemble State")
        print("6. Load Ensemble State")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            ensemble.interactive_ensemble_learning()
        elif choice == '2':
            ensemble._show_ensemble_stats()
        elif choice == '3':
            ensemble._show_model_performance()
        elif choice == '4':
            ensemble._change_ensemble_strategy()
        elif choice == '5':
            filename = input("Enter filename (or press Enter for auto): ").strip()
            ensemble.save_ensemble_state(filename if filename else None)
        elif choice == '6':
            filename = input("Enter filename to load: ").strip()
            if filename:
                ensemble.load_ensemble_state(filename)
        elif choice == '7':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
