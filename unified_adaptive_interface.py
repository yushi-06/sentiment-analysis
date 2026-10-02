#!/usr/bin/env python3
"""
Unified Adaptive Learning Interface
Integrates all adaptive learning features into a single comprehensive system
"""

import os
import sys
from datetime import datetime
from adaptive_sentiment import AdaptiveSentimentAnalyzer
from enhanced_adaptive_learning import EnhancedAdaptiveLearning
from online_learning_system import OnlineLearningSystem
from learning_analytics_dashboard import LearningAnalyticsDashboard
from ensemble_adaptive_learning import EnsembleAdaptiveLearning
from auto_trainer import AutoTrainer

class UnifiedAdaptiveLearningInterface:
    def __init__(self):
        """Initialize the unified adaptive learning system"""
        self.current_system = None
        self.system_type = None
        
        # Available systems
        self.systems = {
            'basic': None,
            'enhanced': None,
            'online': None,
            'ensemble': None
        }
        
        self.analytics_dashboard = None
        self.auto_trainer = None
        
        print("🚀 UNIFIED ADAPTIVE LEARNING SYSTEM")
        print("="*60)
        print("Welcome to the comprehensive sentiment analysis learning platform!")
        print("This system integrates multiple adaptive learning approaches.")
    
    def initialize_system(self, system_type='enhanced'):
        """Initialize a specific learning system"""
        print(f"\n🔧 Initializing {system_type} learning system...")
        
        try:
            if system_type == 'basic':
                self.systems['basic'] = AdaptiveSentimentAnalyzer()
                self.current_system = self.systems['basic']
                
            elif system_type == 'enhanced':
                self.systems['enhanced'] = EnhancedAdaptiveLearning()
                self.current_system = self.systems['enhanced']
                
            elif system_type == 'online':
                self.systems['online'] = OnlineLearningSystem()
                self.current_system = self.systems['online']
                
            elif system_type == 'ensemble':
                num_models = 3  # Default
                strategy = 'weighted'  # Default
                self.systems['ensemble'] = EnsembleAdaptiveLearning(num_models, strategy)
                self.current_system = self.systems['ensemble']
            
            self.system_type = system_type
            
            # Initialize analytics dashboard
            if system_type in ['enhanced', 'online']:
                self.analytics_dashboard = LearningAnalyticsDashboard(self.current_system)
            
            # Initialize auto trainer
            if hasattr(self.current_system, 'analyzer'):
                self.auto_trainer = AutoTrainer(self.current_system.analyzer)
            elif hasattr(self.current_system, 'models'):  # Ensemble
                self.auto_trainer = AutoTrainer(self.current_system.models[0].analyzer)
            else:  # Basic
                self.auto_trainer = AutoTrainer(self.current_system)
            
            print(f"✅ {system_type.title()} system initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing {system_type} system: {e}")
            return False
        
        return True
    
    def show_main_menu(self):
        """Display the main menu"""
        print(f"\n🎯 MAIN MENU - Current System: {self.system_type.title() if self.system_type else 'None'}")
        print("="*60)
        
        # System selection
        print("🔧 SYSTEM SELECTION:")
        print("1. Initialize Basic Adaptive System")
        print("2. Initialize Enhanced Adaptive System (Recommended)")
        print("3. Initialize Online Learning System")
        print("4. Initialize Ensemble Learning System")
        
        # Core functionality
        print("\n🎯 CORE FUNCTIONALITY:")
        print("5. Interactive Prediction & Learning")
        print("6. Batch Training from Dataset")
        print("7. Model Evaluation & Testing")
        
        # Advanced features
        print("\n📊 ADVANCED FEATURES:")
        print("8. Learning Analytics Dashboard")
        print("9. Online Learning Management")
        print("10. Ensemble Model Management")
        
        # Utilities
        print("\n🛠️ UTILITIES:")
        print("11. Export/Import Learning Data")
        print("12. System Comparison")
        print("13. Help & Documentation")
        print("14. Exit")
        
        return input("\nEnter your choice (1-14): ").strip()
    
    def interactive_prediction_learning(self):
        """Interactive prediction and learning interface"""
        if not self.current_system:
            print("❌ No system initialized. Please select a system first.")
            return
        
        print(f"\n🎯 INTERACTIVE LEARNING - {self.system_type.title()} System")
        print("="*50)
        
        if self.system_type == 'basic':
            self._basic_interactive_mode()
        elif self.system_type == 'enhanced':
            self.current_system.interactive_learning_with_confidence()
        elif self.system_type == 'online':
            self.current_system.interactive_online_mode()
        elif self.system_type == 'ensemble':
            self.current_system.interactive_ensemble_learning()
    
    def _basic_interactive_mode(self):
        """Basic interactive mode for simple adaptive analyzer"""
        print("Commands: Type text to analyze, 'stats' for statistics, 'quit' to exit")
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() == 'quit':
                break
            elif text.lower() == 'stats':
                self.current_system.show_learning_stats()
                continue
            elif not text:
                continue
            
            # Make prediction
            prediction, score, contributions = self.current_system.predict_sentiment(text, show_analysis=True)
            
            # Get feedback
            feedback = input(f"\nPrediction: {prediction}. Correct? (y/n) or provide correct: ").strip()
            
            if feedback.lower() in ['n', 'no']:
                correct = input("Correct sentiment (Positive/Negative/Neutral): ").strip().title()
                if correct in ['Positive', 'Negative', 'Neutral']:
                    self.current_system.learn_from_correction(text, prediction, correct)
            elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
                if feedback.title() != prediction:
                    self.current_system.learn_from_correction(text, prediction, feedback.title())
    
    def batch_training_interface(self):
        """Batch training interface"""
        if not self.auto_trainer:
            print("❌ Auto trainer not available. Please initialize a system first.")
            return
        
        print(f"\n📚 BATCH TRAINING INTERFACE")
        print("="*40)
        print("1. Train from single dataset")
        print("2. Train from multiple datasets")
        print("3. Continuous learning from dataset")
        print("4. Back to main menu")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            file_path = input("Enter dataset file path: ").strip().strip('"')
            if os.path.exists(file_path):
                text_col = input("Text column name (default: text): ").strip() or 'text'
                sentiment_col = input("Sentiment column name (default: sentiment): ").strip() or 'sentiment'
                sample_str = input("Sample size (press Enter for all): ").strip()
                sample_size = int(sample_str) if sample_str else None
                
                self.auto_trainer.train_from_dataset(file_path, text_col, sentiment_col, sample_size)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '2':
            paths_str = input("Enter dataset paths (comma-separated): ").strip()
            paths = [p.strip().strip('"') for p in paths_str.split(',')]
            valid_paths = [p for p in paths if os.path.exists(p)]
            
            if valid_paths:
                self.auto_trainer.batch_train_from_multiple_datasets(valid_paths)
            else:
                print("❌ No valid file paths found.")
        
        elif choice == '3':
            dataset_path = input("Enter dataset path to monitor: ").strip().strip('"')
            if os.path.exists(dataset_path):
                interval = input("Check interval in minutes (default: 30): ").strip()
                interval = int(interval) if interval else 30
                self.auto_trainer.continuous_learning_mode(dataset_path, interval)
            else:
                print(f"❌ File not found: {dataset_path}")
    
    def model_evaluation_interface(self):
        """Model evaluation and testing interface"""
        if not self.current_system:
            print("❌ No system initialized. Please select a system first.")
            return
        
        print(f"\n🧪 MODEL EVALUATION INTERFACE")
        print("="*40)
        print("1. Evaluate on test dataset")
        print("2. Quick accuracy test")
        print("3. Confidence analysis")
        print("4. Cross-validation (if available)")
        print("5. Back to main menu")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            test_file = input("Enter test file path: ").strip().strip('"')
            if os.path.exists(test_file) and self.auto_trainer:
                self.auto_trainer.evaluate_on_test_set(test_file)
            else:
                print("❌ File not found or auto trainer not available.")
        
        elif choice == '2':
            self._quick_accuracy_test()
        
        elif choice == '3':
            self._confidence_analysis()
        
        elif choice == '4':
            print("Cross-validation not implemented yet.")
    
    def _quick_accuracy_test(self):
        """Quick accuracy test with predefined examples"""
        test_examples = [
            ("I love this product!", "Positive"),
            ("This is terrible", "Negative"),
            ("It's okay, nothing special", "Neutral"),
            ("Amazing quality and great service!", "Positive"),
            ("Worst experience ever", "Negative"),
            ("Average performance", "Neutral")
        ]
        
        correct = 0
        total = len(test_examples)
        
        print(f"\n🧪 Running quick accuracy test ({total} examples)...")
        
        for text, true_sentiment in test_examples:
            if self.system_type == 'basic':
                prediction, _, _ = self.current_system.predict_sentiment(text)
            elif self.system_type == 'enhanced':
                prediction, _, _, _, _ = self.current_system.predict_with_confidence(text)
            elif self.system_type == 'online':
                prediction, _, _ = self.current_system.predict_online(text)
            elif self.system_type == 'ensemble':
                prediction, _, _, _, _ = self.current_system.predict_ensemble(text)
            
            is_correct = prediction == true_sentiment
            correct += is_correct
            
            print(f"  '{text}' → {prediction} {'✅' if is_correct else '❌'} (expected: {true_sentiment})")
        
        accuracy = (correct / total) * 100
        print(f"\n📊 Quick Test Results:")
        print(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    def _confidence_analysis(self):
        """Analyze confidence patterns"""
        if self.system_type not in ['enhanced', 'online', 'ensemble']:
            print("❌ Confidence analysis only available for enhanced, online, or ensemble systems.")
            return
        
        test_texts = [
            "I absolutely love this!",
            "This is okay",
            "I hate this so much",
            "Not sure about this",
            "It's fine I guess"
        ]
        
        print(f"\n🎯 CONFIDENCE ANALYSIS")
        print("="*30)
        
        for text in test_texts:
            if self.system_type == 'enhanced':
                pred, _, conf, uncertain, _ = self.current_system.predict_with_confidence(text)
                print(f"'{text}' → {pred} (conf: {conf:.3f}) {'🔴' if uncertain else '🟢'}")
            elif self.system_type == 'online':
                pred, conf, uncertain = self.current_system.predict_online(text)
                print(f"'{text}' → {pred} (conf: {conf:.3f}) {'🔴' if uncertain else '🟢'}")
            elif self.system_type == 'ensemble':
                pred, conf, uncertain, _, consensus = self.current_system.predict_ensemble(text)
                print(f"'{text}' → {pred} (conf: {conf:.3f}, consensus: {consensus:.3f}) {'🔴' if uncertain else '🟢'}")
    
    def analytics_dashboard_interface(self):
        """Analytics dashboard interface"""
        if not self.analytics_dashboard:
            print("❌ Analytics dashboard not available for current system.")
            return
        
        self.analytics_dashboard.show_interactive_dashboard()
    
    def online_learning_management(self):
        """Online learning management interface"""
        if self.system_type != 'online':
            print("❌ Online learning management only available for online learning system.")
            return
        
        print(f"\n🌊 ONLINE LEARNING MANAGEMENT")
        print("="*40)
        print("1. Start/Stop Online Learning Service")
        print("2. Show Online Dashboard")
        print("3. Simulate Data Stream")
        print("4. Configure Online Parameters")
        print("5. Back to main menu")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            if self.current_system.is_running:
                self.current_system.stop_online_learning()
            else:
                self.current_system.start_online_learning()
        
        elif choice == '2':
            self.current_system.show_online_dashboard()
        
        elif choice == '3':
            file_path = input("Enter data file for simulation: ").strip().strip('"')
            if os.path.exists(file_path):
                delay = input("Delay between predictions (seconds, default 2): ").strip()
                delay = float(delay) if delay else 2.0
                self.current_system.simulate_data_stream(file_path, delay)
        
        elif choice == '4':
            self._configure_online_parameters()
    
    def _configure_online_parameters(self):
        """Configure online learning parameters"""
        print(f"\nCurrent parameters:")
        print(f"  Update interval: {self.current_system.update_interval} seconds")
        print(f"  Batch size: {self.current_system.batch_size}")
        print(f"  Forgetting factor: {self.current_system.forgetting_factor}")
        
        new_interval = input(f"New update interval (current: {self.current_system.update_interval}): ").strip()
        if new_interval:
            self.current_system.update_interval = int(new_interval)
        
        new_batch = input(f"New batch size (current: {self.current_system.batch_size}): ").strip()
        if new_batch:
            self.current_system.batch_size = int(new_batch)
        
        new_forgetting = input(f"New forgetting factor (current: {self.current_system.forgetting_factor}): ").strip()
        if new_forgetting:
            self.current_system.forgetting_factor = float(new_forgetting)
        
        print("✅ Parameters updated!")
    
    def ensemble_management(self):
        """Ensemble model management interface"""
        if self.system_type != 'ensemble':
            print("❌ Ensemble management only available for ensemble learning system.")
            return
        
        print(f"\n🎭 ENSEMBLE MANAGEMENT")
        print("="*30)
        print("1. Show Model Weights")
        print("2. Show Model Performance")
        print("3. Change Ensemble Strategy")
        print("4. Add/Remove Models")
        print("5. Save/Load Ensemble State")
        print("6. Back to main menu")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            self.current_system._show_model_weights()
        elif choice == '2':
            self.current_system._show_model_performance()
        elif choice == '3':
            self.current_system._change_ensemble_strategy()
        elif choice == '4':
            print("Add/Remove models not implemented yet.")
        elif choice == '5':
            action = input("Save or Load? (s/l): ").strip().lower()
            filename = input("Enter filename: ").strip()
            if action == 's':
                self.current_system.save_ensemble_state(filename)
            elif action == 'l':
                self.current_system.load_ensemble_state(filename)
    
    def export_import_interface(self):
        """Export/Import learning data interface"""
        print(f"\n💾 EXPORT/IMPORT INTERFACE")
        print("="*30)
        print("1. Export learning session")
        print("2. Import learning session")
        print("3. Export analytics report")
        print("4. Export model state")
        print("5. Import model state")
        print("6. Back to main menu")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            if hasattr(self.current_system, 'save_learning_session'):
                filename = input("Enter filename (or press Enter for auto): ").strip()
                self.current_system.save_learning_session(filename if filename else None)
        
        elif choice == '2':
            if hasattr(self.current_system, 'load_learning_session'):
                filename = input("Enter filename to load: ").strip()
                if filename and os.path.exists(filename):
                    self.current_system.load_learning_session(filename)
        
        elif choice == '3':
            if self.analytics_dashboard:
                filename = input("Enter filename (or press Enter for auto): ").strip()
                self.analytics_dashboard.export_learning_report(filename if filename else None)
        
        elif choice == '4':
            self._export_model_state()
        
        elif choice == '5':
            self._import_model_state()
    
    def _export_model_state(self):
        """Export current model state"""
        if not self.current_system:
            print("❌ No system initialized.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_state_{self.system_type}_{timestamp}.pkl"
        
        try:
            if hasattr(self.current_system, 'save_knowledge'):
                self.current_system.save_knowledge()
                print(f"✅ Model state saved!")
            elif hasattr(self.current_system, 'save_ensemble_state'):
                self.current_system.save_ensemble_state(filename)
            else:
                print("❌ Export not supported for current system.")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def _import_model_state(self):
        """Import model state"""
        filename = input("Enter filename to import: ").strip()
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return
        
        try:
            if hasattr(self.current_system, 'load_ensemble_state'):
                self.current_system.load_ensemble_state(filename)
            else:
                print("❌ Import not supported for current system.")
        except Exception as e:
            print(f"❌ Import failed: {e}")
    
    def show_help(self):
        """Show help and documentation"""
        print(f"\n📖 HELP & DOCUMENTATION")
        print("="*50)
        print("""
🎯 SYSTEM TYPES:
  • Basic: Simple adaptive learning with word-based patterns
  • Enhanced: Advanced learning with confidence scoring and active learning
  • Online: Real-time continuous learning from streaming data
  • Ensemble: Multiple models working together for better accuracy

🚀 GETTING STARTED:
  1. Choose and initialize a learning system (option 2 recommended)
  2. Use interactive learning to train the model with your feedback
  3. Optionally use batch training for large datasets
  4. Monitor progress with analytics dashboard

💡 TIPS:
  • Enhanced system provides confidence scores and uncertainty detection
  • Online system is best for continuous, real-time applications
  • Ensemble system offers highest accuracy but uses more resources
  • Use analytics dashboard to track learning progress and insights

🔧 ADVANCED FEATURES:
  • Confidence-based learning rates
  • Active learning for uncertain predictions
  • Real-time model updates
  • Comprehensive analytics and visualizations
  • Model ensemble with multiple strategies
        """)
    
    def run(self):
        """Main application loop"""
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == '1':
                    self.initialize_system('basic')
                elif choice == '2':
                    self.initialize_system('enhanced')
                elif choice == '3':
                    self.initialize_system('online')
                elif choice == '4':
                    self.initialize_system('ensemble')
                elif choice == '5':
                    self.interactive_prediction_learning()
                elif choice == '6':
                    self.batch_training_interface()
                elif choice == '7':
                    self.model_evaluation_interface()
                elif choice == '8':
                    self.analytics_dashboard_interface()
                elif choice == '9':
                    self.online_learning_management()
                elif choice == '10':
                    self.ensemble_management()
                elif choice == '11':
                    self.export_import_interface()
                elif choice == '12':
                    print("System comparison not implemented yet.")
                elif choice == '13':
                    self.show_help()
                elif choice == '14':
                    print("\n👋 Thank you for using the Unified Adaptive Learning System!")
                    print("Your learning progress has been saved automatically.")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again or contact support.")

def main():
    """Entry point for the unified adaptive learning system"""
    try:
        interface = UnifiedAdaptiveLearningInterface()
        interface.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
