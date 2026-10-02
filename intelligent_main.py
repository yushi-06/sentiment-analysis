#!/usr/bin/env python3
"""
Intelligent Main Interface with Critical Analysis
Integrates adaptive learning with critical analysis and knowledge database
"""

import os
import sys
from datetime import datetime
from critical_analysis_system import CriticalAnalysisSystem
from enhanced_adaptive_learning import EnhancedAdaptiveLearning
from learning_analytics_dashboard import LearningAnalyticsDashboard

class IntelligentSentimentSystem:
    def __init__(self):
        """Initialize the intelligent sentiment analysis system"""
        print("🧠 INTELLIGENT SENTIMENT ANALYSIS SYSTEM")
        print("="*60)
        print("Initializing advanced adaptive learning with critical analysis...")
        
        # Initialize core components
        self.critical_analyzer = CriticalAnalysisSystem()
        self.enhanced_learner = self.critical_analyzer.enhanced_learner
        self.analytics_dashboard = LearningAnalyticsDashboard(self.enhanced_learner)
        
        # System state
        self.session_stats = {
            'queries_analyzed': 0,
            'predictions_made': 0,
            'learning_events': 0,
            'session_start': datetime.now()
        }
        
        print("✅ System initialized successfully!")
        print("🎯 Features: Critical Analysis + Adaptive Learning + Analytics")
    
    def analyze_and_predict(self, query, context=None):
        """Analyze query critically and provide sentiment prediction if relevant"""
        print(f"\n🔍 Analyzing: '{query}'")
        
        # Step 1: Critical analysis
        analysis = self.critical_analyzer.critically_analyze_query(query, context)
        self.session_stats['queries_analyzed'] += 1
        
        # Step 2: Sentiment prediction if relevant
        sentiment_result = None
        if self._is_sentiment_query(query, analysis):
            prediction, confidence, uncertain, _, _ = self.enhanced_learner.predict_with_confidence(query)
            sentiment_result = {
                'prediction': prediction,
                'confidence': confidence,
                'uncertain': uncertain
            }
            self.session_stats['predictions_made'] += 1
        
        # Step 3: Generate comprehensive response
        response = self._generate_intelligent_response(query, analysis, sentiment_result)
        
        return {
            'analysis': analysis,
            'sentiment': sentiment_result,
            'response': response
        }
    
    def _is_sentiment_query(self, query, analysis):
        """Determine if query requires sentiment analysis"""
        # Check if query contains text to analyze
        sentiment_indicators = [
            'sentiment', 'emotion', 'feeling', 'opinion', 'mood',
            'positive', 'negative', 'neutral', 'analyze this',
            'what do you think', 'how does this sound'
        ]
        
        query_lower = query.lower()
        
        # Direct sentiment keywords
        if any(indicator in query_lower for indicator in sentiment_indicators):
            return True
        
        # Check if it's asking for analysis of text
        if analysis['query_analysis']['question_type'] in ['analytical', 'opinion']:
            return True
        
        # Check if query contains quoted text or expressions
        if '"' in query or "'" in query:
            return True
        
        # Check for emotional expressions
        emotional_words = ['love', 'hate', 'like', 'dislike', 'amazing', 'terrible', 'great', 'awful']
        if any(word in query_lower for word in emotional_words):
            return True
        
        return False
    
    def _generate_intelligent_response(self, query, analysis, sentiment_result):
        """Generate intelligent response based on analysis"""
        response_parts = []
        
        # Analysis summary
        qa = analysis['query_analysis']
        ca = analysis['complexity_analysis']
        da = analysis['domain_analysis']
        
        response_parts.append(f"📊 **Analysis Summary:**")
        response_parts.append(f"- Query Type: {qa['question_type'].title()}")
        response_parts.append(f"- Complexity: {ca['complexity_level'].title()}")
        response_parts.append(f"- Domain: {da['primary_domain'].title()}")
        
        # Sentiment analysis results
        if sentiment_result:
            response_parts.append(f"\n💭 **Sentiment Analysis:**")
            response_parts.append(f"- Prediction: {sentiment_result['prediction']}")
            response_parts.append(f"- Confidence: {sentiment_result['confidence']:.3f}")
            if sentiment_result['uncertain']:
                response_parts.append(f"- ⚠️ Uncertain prediction - consider providing feedback")
        
        # Critical insights
        if analysis['critical_insights']:
            response_parts.append(f"\n💡 **Critical Insights:**")
            for insight in analysis['critical_insights']:
                response_parts.append(f"- {insight['insight']}")
                response_parts.append(f"  💡 {insight['recommendation']}")
        
        # Contextual recommendations
        recommendations = self._generate_contextual_recommendations(analysis, sentiment_result)
        if recommendations:
            response_parts.append(f"\n🎯 **Recommendations:**")
            for rec in recommendations:
                response_parts.append(f"- {rec}")
        
        return "\n".join(response_parts)
    
    def _generate_contextual_recommendations(self, analysis, sentiment_result):
        """Generate contextual recommendations based on analysis"""
        recommendations = []
        
        # Based on complexity
        if analysis['complexity_analysis']['complexity_level'] == 'complex':
            recommendations.append("Consider breaking this into smaller, more specific questions")
        
        # Based on domain
        domain = analysis['domain_analysis']['primary_domain']
        if domain == 'technical':
            recommendations.append("For technical queries, provide specific examples or code snippets")
        elif domain == 'business':
            recommendations.append("Consider providing market context or business objectives")
        
        # Based on sentiment uncertainty
        if sentiment_result and sentiment_result['uncertain']:
            recommendations.append("Provide more context to improve sentiment analysis accuracy")
        
        # Based on question type
        if analysis['query_analysis']['question_type'] == 'analytical':
            recommendations.append("Provide data or evidence to support analytical conclusions")
        
        return recommendations
    
    def interactive_intelligent_mode(self):
        """Interactive mode with intelligent analysis"""
        print(f"\n🧠 INTELLIGENT INTERACTIVE MODE")
        print("="*50)
        print("This system provides:")
        print("  🔍 Critical analysis of your queries")
        print("  💭 Intelligent sentiment analysis")
        print("  📊 Learning analytics and insights")
        print("  🎯 Contextual recommendations")
        print("\nCommands:")
        print("  - Type your question or text to analyze")
        print("  - Type 'analytics' for learning dashboard")
        print("  - Type 'knowledge' to search knowledge database")
        print("  - Type 'stats' for session statistics")
        print("  - Type 'help' for detailed help")
        print("  - Type 'quit' to exit")
        print("-"*50)
        
        while True:
            user_input = input("\n🎯 Enter your query: ").strip()
            
            if user_input.lower() == 'quit':
                self._show_session_summary()
                break
            elif user_input.lower() == 'analytics':
                self.analytics_dashboard.show_interactive_dashboard()
                continue
            elif user_input.lower() == 'knowledge':
                self._knowledge_search_interface()
                continue
            elif user_input.lower() == 'stats':
                self._show_session_stats()
                continue
            elif user_input.lower() == 'help':
                self._show_detailed_help()
                continue
            elif not user_input:
                print("Please enter a query or command.")
                continue
            
            # Process the query
            try:
                result = self.analyze_and_predict(user_input)
                
                # Display results
                print(f"\n🤖 **INTELLIGENT ANALYSIS RESULTS**")
                print("="*45)
                print(result['response'])
                
                # Offer learning opportunity if sentiment was analyzed
                if result['sentiment']:
                    self._offer_learning_opportunity(user_input, result['sentiment'])
                
                # Ask for feedback on analysis quality
                self._collect_analysis_feedback(user_input, result)
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("Please try again or contact support.")
    
    def _knowledge_search_interface(self):
        """Interface for searching knowledge database"""
        print(f"\n🔍 KNOWLEDGE DATABASE SEARCH")
        print("="*35)
        
        search_query = input("Enter search terms: ").strip()
        if not search_query:
            return
        
        matches = self.critical_analyzer.search_knowledge_db(search_query)
        
        if matches:
            print(f"\n📚 Found {len(matches)} similar queries:")
            for i, match in enumerate(matches, 1):
                print(f"\n{i}. '{match['query']}'")
                print(f"   Similarity: {match['similarity']:.3f}")
                print(f"   Date: {match['timestamp'][:10]}")
                
                if match['insights']:
                    print(f"   Insights:")
                    for insight in match['insights'][:2]:  # Show first 2 insights
                        print(f"   • {insight['insight']}")
        else:
            print("🔍 No similar queries found in knowledge database.")
    
    def _show_session_stats(self):
        """Show current session statistics"""
        duration = datetime.now() - self.session_stats['session_start']
        
        print(f"\n📊 SESSION STATISTICS")
        print("="*25)
        print(f"Duration: {duration}")
        print(f"Queries analyzed: {self.session_stats['queries_analyzed']}")
        print(f"Predictions made: {self.session_stats['predictions_made']}")
        print(f"Learning events: {self.session_stats['learning_events']}")
        
        # System-wide stats
        summary = self.critical_analyzer.get_analysis_summary()
        if isinstance(summary, dict):
            print(f"\n🧠 SYSTEM-WIDE STATISTICS")
            print(f"Total queries in database: {summary['total_queries']}")
            print(f"Total insights generated: {summary['insights_generated']}")
    
    def _show_detailed_help(self):
        """Show detailed help information"""
        print(f"\n📖 DETAILED HELP")
        print("="*20)
        print("""
🎯 **QUERY TYPES SUPPORTED:**
  • Sentiment Analysis: "Analyze this text: 'I love this product'"
  • Opinion Questions: "What do you think about this?"
  • Technical Questions: "How does the algorithm work?"
  • Analytical Requests: "Compare these approaches"
  • Problem Solving: "How can I improve accuracy?"

🧠 **CRITICAL ANALYSIS FEATURES:**
  • Query structure analysis
  • Complexity assessment
  • Domain identification
  • Intent recognition
  • Contextual recommendations

💭 **SENTIMENT ANALYSIS CAPABILITIES:**
  • Confidence scoring
  • Uncertainty detection
  • Adaptive learning from feedback
  • Real-time model improvement

📊 **ANALYTICS & INSIGHTS:**
  • Learning progress tracking
  • Performance metrics
  • Knowledge database search
  • Pattern recognition

🎓 **LEARNING FEATURES:**
  • Automatic learning from corrections
  • Confidence-based learning rates
  • Active learning suggestions
  • Knowledge persistence

💡 **TIPS FOR BETTER RESULTS:**
  • Be specific in your queries
  • Provide context when relevant
  • Give feedback on predictions
  • Use quotes for text to analyze
  • Ask follow-up questions for clarity
        """)
    
    def _offer_learning_opportunity(self, query, sentiment_result):
        """Offer learning opportunity for sentiment predictions"""
        prediction = sentiment_result['prediction']
        confidence = sentiment_result['confidence']
        
        print(f"\n🎓 **LEARNING OPPORTUNITY**")
        print(f"Sentiment: {prediction} (confidence: {confidence:.3f})")
        
        if sentiment_result['uncertain']:
            print("⚠️ This prediction has low confidence - your feedback would be valuable!")
        
        feedback = input("Is this sentiment correct? (y/n) or provide correct sentiment: ").strip()
        
        if feedback.lower() in ['n', 'no']:
            correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
            if correct in ['Positive', 'Negative', 'Neutral']:
                self.enhanced_learner.adaptive_learn_from_feedback(query, prediction, correct, confidence)
                self.session_stats['learning_events'] += 1
                print("✅ Thank you! The system learned from your feedback.")
        elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
            if feedback.title() != prediction:
                self.enhanced_learner.adaptive_learn_from_feedback(query, prediction, feedback.title(), confidence)
                self.session_stats['learning_events'] += 1
                print("✅ Thank you! The system learned from your feedback.")
        elif feedback.lower() in ['y', 'yes']:
            print("✅ Thank you for confirming the prediction!")
    
    def _collect_analysis_feedback(self, query, result):
        """Collect feedback on analysis quality"""
        feedback = input("\nWas this analysis helpful? (y/n): ").strip().lower()
        
        if feedback == 'n':
            improvement = input("How could the analysis be improved? ").strip()
            if improvement:
                # Store feedback for future improvements
                feedback_record = {
                    'query': query,
                    'feedback': improvement,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_type': 'quality_feedback'
                }
                
                # Add to knowledge database
                self.critical_analyzer.knowledge_db['analysis_history'].append(feedback_record)
                self.critical_analyzer.save_knowledge_db()
                
                print("📝 Thank you for the feedback! It will help improve the system.")
    
    def _show_session_summary(self):
        """Show session summary before exit"""
        duration = datetime.now() - self.session_stats['session_start']
        
        print(f"\n📊 SESSION SUMMARY")
        print("="*25)
        print(f"Session duration: {duration}")
        print(f"Queries processed: {self.session_stats['queries_analyzed']}")
        print(f"Predictions made: {self.session_stats['predictions_made']}")
        print(f"Learning events: {self.session_stats['learning_events']}")
        
        if self.session_stats['learning_events'] > 0:
            print(f"🎓 The system learned from {self.session_stats['learning_events']} corrections!")
        
        print("\n💾 All learning progress has been automatically saved.")
        print("🧠 Knowledge database updated with new insights.")
        print("👋 Thank you for using the Intelligent Sentiment Analysis System!")

def main():
    """Main entry point for intelligent sentiment system"""
    try:
        system = IntelligentSentimentSystem()
        
        print(f"\n🚀 WELCOME TO INTELLIGENT SENTIMENT ANALYSIS")
        print("This system combines critical analysis with adaptive learning.")
        print("It will analyze your queries intelligently and learn from your feedback.")
        
        # Start interactive mode
        system.interactive_intelligent_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
