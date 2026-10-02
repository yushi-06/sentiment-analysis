#!/usr/bin/env python3
"""
Online Learning System for Sentiment Analysis
Enables real-time model updates and continuous learning from streaming data
"""

import json
import os
import time
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta
import queue
import pandas as pd
from adaptive_sentiment import AdaptiveSentimentAnalyzer
from enhanced_adaptive_learning import EnhancedAdaptiveLearning

class OnlineLearningSystem:
    def __init__(self, base_analyzer=None, update_interval=60):
        """
        Initialize online learning system
        
        Args:
            base_analyzer: Base sentiment analyzer
            update_interval: Seconds between model updates
        """
        self.enhanced_learner = EnhancedAdaptiveLearning(base_analyzer)
        self.analyzer = self.enhanced_learner.analyzer
        
        # Online learning parameters
        self.update_interval = update_interval
        self.batch_size = 10  # Process in small batches
        self.forgetting_factor = 0.95  # For exponential decay of old patterns
        
        # Data streams and queues
        self.feedback_queue = queue.Queue()
        self.prediction_stream = deque(maxlen=1000)
        self.update_queue = queue.Queue()
        
        # Real-time statistics
        self.online_stats = {
            'predictions_made': 0,
            'feedback_received': 0,
            'model_updates': 0,
            'accuracy_trend': deque(maxlen=100),
            'last_update': None,
            'learning_velocity': 0  # Rate of learning
        }
        
        # Threading for continuous updates
        self.update_thread = None
        self.is_running = False
        
        print("🌊 Online Learning System initialized!")
        print(f"⏱️  Update interval: {update_interval} seconds")
        print(f"📦 Batch size: {self.batch_size}")
    
    def start_online_learning(self):
        """Start the online learning process"""
        if self.is_running:
            print("⚠️  Online learning is already running!")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._continuous_update_loop, daemon=True)
        self.update_thread.start()
        
        print("🚀 Online learning started!")
        print("📡 System is now learning continuously from feedback...")
    
    def stop_online_learning(self):
        """Stop the online learning process"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        print("🛑 Online learning stopped!")
    
    def predict_online(self, text, user_id=None):
        """Make prediction and add to online stream"""
        prediction, score, confidence, is_uncertain, contributions = self.enhanced_learner.predict_with_confidence(text)
        
        # Add to prediction stream
        prediction_data = {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'score': score,
            'is_uncertain': is_uncertain,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        self.prediction_stream.append(prediction_data)
        self.online_stats['predictions_made'] += 1
        
        return prediction, confidence, is_uncertain
    
    def add_feedback_online(self, text, predicted_sentiment, correct_sentiment, user_id=None):
        """Add feedback to the online learning queue"""
        feedback_data = {
            'text': text,
            'predicted': predicted_sentiment,
            'correct': correct_sentiment,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_queue.put(feedback_data)
        self.online_stats['feedback_received'] += 1
        
        print(f"📥 Feedback queued: '{text[:30]}...' → {correct_sentiment}")
    
    def _continuous_update_loop(self):
        """Continuous loop for processing updates"""
        print("🔄 Starting continuous update loop...")
        
        while self.is_running:
            try:
                # Process feedback batch
                self._process_feedback_batch()
                
                # Update model if needed
                self._update_model_online()
                
                # Calculate learning metrics
                self._update_learning_metrics()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error in update loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _process_feedback_batch(self):
        """Process a batch of feedback from the queue"""
        batch = []
        
        # Collect batch
        while len(batch) < self.batch_size and not self.feedback_queue.empty():
            try:
                feedback = self.feedback_queue.get_nowait()
                batch.append(feedback)
            except queue.Empty:
                break
        
        if not batch:
            return
        
        print(f"🔄 Processing feedback batch: {len(batch)} items")
        
        # Process each feedback item
        for feedback in batch:
            # Find corresponding prediction for accuracy calculation
            prediction_match = self._find_matching_prediction(feedback)
            
            if prediction_match:
                confidence = prediction_match.get('confidence', 0.5)
            else:
                confidence = 0.5  # Default confidence
            
            # Apply enhanced learning
            self.enhanced_learner.adaptive_learn_from_feedback(
                feedback['text'],
                feedback['predicted'],
                feedback['correct'],
                confidence
            )
            
            # Update accuracy trend
            is_correct = feedback['predicted'] == feedback['correct']
            self.online_stats['accuracy_trend'].append(1 if is_correct else 0)
        
        self.online_stats['model_updates'] += 1
        self.online_stats['last_update'] = datetime.now().isoformat()
    
    def _find_matching_prediction(self, feedback):
        """Find matching prediction in the stream"""
        for pred in reversed(self.prediction_stream):
            if (pred['text'] == feedback['text'] and 
                pred['prediction'] == feedback['predicted']):
                return pred
        return None
    
    def _update_model_online(self):
        """Apply online model updates with forgetting"""
        # Apply forgetting factor to old patterns
        if self.online_stats['model_updates'] % 10 == 0:  # Every 10 updates
            self._apply_forgetting_factor()
    
    def _apply_forgetting_factor(self):
        """Apply exponential decay to old word patterns"""
        print("🧠 Applying forgetting factor to old patterns...")
        
        for word in self.analyzer.word_sentiment_counts:
            for sentiment in self.analyzer.word_sentiment_counts[word]:
                current_count = self.analyzer.word_sentiment_counts[word][sentiment]
                decayed_count = current_count * self.forgetting_factor
                self.analyzer.word_sentiment_counts[word][sentiment] = max(1, int(decayed_count))
    
    def _update_learning_metrics(self):
        """Update learning velocity and other metrics"""
        if len(self.online_stats['accuracy_trend']) > 10:
            recent_accuracy = sum(list(self.online_stats['accuracy_trend'])[-10:]) / 10
            
            # Calculate learning velocity (improvement rate)
            if len(self.online_stats['accuracy_trend']) > 20:
                older_accuracy = sum(list(self.online_stats['accuracy_trend'])[-20:-10]) / 10
                self.online_stats['learning_velocity'] = recent_accuracy - older_accuracy
    
    def get_online_stats(self):
        """Get current online learning statistics"""
        stats = self.online_stats.copy()
        
        # Add current accuracy
        if self.online_stats['accuracy_trend']:
            stats['current_accuracy'] = sum(self.online_stats['accuracy_trend']) / len(self.online_stats['accuracy_trend'])
            stats['recent_accuracy'] = sum(list(self.online_stats['accuracy_trend'])[-10:]) / min(10, len(self.online_stats['accuracy_trend']))
        else:
            stats['current_accuracy'] = 0
            stats['recent_accuracy'] = 0
        
        # Add queue sizes
        stats['feedback_queue_size'] = self.feedback_queue.qsize()
        stats['prediction_stream_size'] = len(self.prediction_stream)
        
        return stats
    
    def show_online_dashboard(self):
        """Display real-time learning dashboard"""
        stats = self.get_online_stats()
        
        print(f"\n📊 ONLINE LEARNING DASHBOARD")
        print("="*50)
        print(f"Status: {'🟢 Running' if self.is_running else '🔴 Stopped'}")
        print(f"Predictions made: {stats['predictions_made']}")
        print(f"Feedback received: {stats['feedback_received']}")
        print(f"Model updates: {stats['model_updates']}")
        print(f"Current accuracy: {stats['current_accuracy']:.3f}")
        print(f"Recent accuracy: {stats['recent_accuracy']:.3f}")
        print(f"Learning velocity: {stats['learning_velocity']:.3f}")
        print(f"Feedback queue: {stats['feedback_queue_size']} items")
        print(f"Last update: {stats['last_update']}")
        
        # Show trend
        if stats['learning_velocity'] > 0.01:
            print("📈 Trend: Improving")
        elif stats['learning_velocity'] < -0.01:
            print("📉 Trend: Declining")
        else:
            print("📊 Trend: Stable")
    
    def simulate_data_stream(self, data_file, delay=2):
        """Simulate a data stream from a file for testing"""
        print(f"🎭 Starting data stream simulation from {data_file}")
        
        try:
            df = pd.read_csv(data_file)
            
            # Ensure we have the right columns
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            sentiment_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]
            
            print(f"📊 Loaded {len(df)} samples for simulation")
            print(f"Using columns: {text_col}, {sentiment_col}")
            
            for idx, row in df.iterrows():
                if not self.is_running:
                    break
                
                text = str(row[text_col])
                true_sentiment = str(row[sentiment_col]).title()
                
                # Make prediction
                prediction, confidence, is_uncertain = self.predict_online(text, user_id="simulator")
                
                print(f"\n📝 Simulated prediction {idx+1}:")
                print(f"   Text: '{text[:50]}...'")
                print(f"   Predicted: {prediction} (confidence: {confidence:.3f})")
                print(f"   True: {true_sentiment}")
                
                # Simulate feedback with some delay (realistic scenario)
                if idx % 3 == 0:  # Simulate that not all predictions get feedback
                    self.add_feedback_online(text, prediction, true_sentiment, user_id="simulator")
                
                time.sleep(delay)
                
                # Show dashboard every 10 predictions
                if (idx + 1) % 10 == 0:
                    self.show_online_dashboard()
        
        except Exception as e:
            print(f"❌ Error in simulation: {e}")
    
    def interactive_online_mode(self):
        """Interactive mode with online learning"""
        print(f"\n🌊 INTERACTIVE ONLINE LEARNING MODE")
        print("="*50)
        print("Commands:")
        print("  - Type text to get real-time predictions")
        print("  - After prediction, provide feedback if incorrect")
        print("  - Type 'dashboard' to see learning stats")
        print("  - Type 'start' to begin online learning")
        print("  - Type 'stop' to stop online learning")
        print("  - Type 'simulate' to run data stream simulation")
        print("  - Type 'quit' to exit")
        print("-"*50)
        
        while True:
            command = input("\n🎯 Enter text or command: ").strip()
            
            if command.lower() == 'quit':
                self.stop_online_learning()
                break
            
            elif command.lower() == 'start':
                self.start_online_learning()
                continue
            
            elif command.lower() == 'stop':
                self.stop_online_learning()
                continue
            
            elif command.lower() == 'dashboard':
                self.show_online_dashboard()
                continue
            
            elif command.lower() == 'simulate':
                file_path = input("Enter data file path: ").strip().strip('"')
                if os.path.exists(file_path):
                    delay = input("Enter delay between predictions (seconds, default 2): ").strip()
                    delay = float(delay) if delay else 2.0
                    
                    # Start simulation in a separate thread
                    sim_thread = threading.Thread(
                        target=self.simulate_data_stream, 
                        args=(file_path, delay), 
                        daemon=True
                    )
                    sim_thread.start()
                else:
                    print(f"❌ File not found: {file_path}")
                continue
            
            elif not command:
                print("Please enter some text or a command.")
                continue
            
            # Make online prediction
            prediction, confidence, is_uncertain = self.predict_online(command)
            
            print(f"\n🤖 Online Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.3f} {'🔴 (Uncertain)' if is_uncertain else '🟢 (Confident)'}")
            
            # Ask for feedback
            feedback = input("Correct? (y/n) or provide correct sentiment (Positive/Negative/Neutral): ").strip()
            
            if feedback.lower() in ['n', 'no']:
                correct = input("Correct sentiment? (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    self.add_feedback_online(command, prediction, correct)
                    print("✅ Feedback added to online learning queue!")
            
            elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != prediction:
                    self.add_feedback_online(command, prediction, feedback.title())
                    print("✅ Feedback added to online learning queue!")
                else:
                    print("✅ Prediction confirmed!")
            
            elif feedback.lower() in ['y', 'yes']:
                print("✅ Prediction confirmed!")

def main():
    """Main interface for online learning system"""
    print("🌊 ONLINE LEARNING SYSTEM FOR SENTIMENT ANALYSIS")
    print("="*60)
    
    # Initialize online learning system
    online_learner = OnlineLearningSystem(update_interval=30)  # Update every 30 seconds
    
    while True:
        print(f"\nChoose mode:")
        print("1. Interactive Online Learning")
        print("2. Start/Stop Online Learning Service")
        print("3. Show Online Dashboard")
        print("4. Simulate Data Stream")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            online_learner.interactive_online_mode()
            
        elif choice == '2':
            if online_learner.is_running:
                online_learner.stop_online_learning()
            else:
                online_learner.start_online_learning()
                print("🚀 Online learning service started!")
                print("   The system will now continuously learn from feedback.")
                print("   Use other modes to interact with the system.")
        
        elif choice == '3':
            online_learner.show_online_dashboard()
        
        elif choice == '4':
            file_path = input("Enter data file path for simulation: ").strip().strip('"')
            if os.path.exists(file_path):
                delay = input("Enter delay between predictions (seconds, default 2): ").strip()
                delay = float(delay) if delay else 2.0
                
                if not online_learner.is_running:
                    online_learner.start_online_learning()
                
                online_learner.simulate_data_stream(file_path, delay)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '5':
            online_learner.stop_online_learning()
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
