#!/usr/bin/env python3
"""
Demo Script for Intelligent Sentiment Analysis System
Demonstrates the complete adaptive learning system with critical analysis
"""

import os
import time
from intelligent_main import IntelligentSentimentSystem

def demo_critical_analysis():
    """Demonstrate critical analysis capabilities"""
    print("🎯 DEMO: Critical Analysis Capabilities")
    print("="*50)
    
    system = IntelligentSentimentSystem()
    
    # Demo queries of different types and complexities
    demo_queries = [
        {
            'query': "I love this new smartphone, it's amazing!",
            'description': "Simple sentiment analysis"
        },
        {
            'query': "How does the adaptive learning algorithm improve sentiment analysis accuracy over time?",
            'description': "Complex technical question"
        },
        {
            'query': "What do you think about this review: 'The product is okay but the service was terrible'?",
            'description': "Opinion question with mixed sentiment"
        },
        {
            'query': "Can you analyze the sentiment and explain why it's classified that way?",
            'description': "Analytical request"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n📝 Demo Query {i}: {demo['description']}")
        print(f"Query: '{demo['query']}'")
        print("-" * 40)
        
        # Analyze the query
        result = system.analyze_and_predict(demo['query'])
        
        # Display results
        print("🤖 ANALYSIS RESULTS:")
        print(result['response'])
        
        # Pause between demos
        if i < len(demo_queries):
            input("\nPress Enter to continue to next demo...")
    
    return system

def demo_learning_capabilities(system):
    """Demonstrate adaptive learning capabilities"""
    print(f"\n🎓 DEMO: Adaptive Learning Capabilities")
    print("="*50)
    
    # Demo learning scenarios
    learning_scenarios = [
        {
            'text': "This product is decent",
            'initial_prediction': "Neutral",
            'correct_sentiment': "Positive",
            'explanation': "User corrects neutral to positive - system learns"
        },
        {
            'text': "Not bad at all",
            'initial_prediction': "Negative", 
            'correct_sentiment': "Positive",
            'explanation': "Double negative case - system learns pattern"
        }
    ]
    
    for i, scenario in enumerate(learning_scenarios, 1):
        print(f"\n📚 Learning Scenario {i}: {scenario['explanation']}")
        print(f"Text: '{scenario['text']}'")
        
        # Get prediction
        prediction, confidence, uncertain, _, _ = system.enhanced_learner.predict_with_confidence(scenario['text'])
        
        print(f"Initial Prediction: {prediction} (confidence: {confidence:.3f})")
        print(f"Correct Sentiment: {scenario['correct_sentiment']}")
        
        # Simulate learning
        if prediction != scenario['correct_sentiment']:
            system.enhanced_learner.adaptive_learn_from_feedback(
                scenario['text'], 
                prediction, 
                scenario['correct_sentiment'], 
                confidence
            )
            print("✅ System learned from correction!")
            
            # Test again to show improvement
            new_prediction, new_confidence, _, _, _ = system.enhanced_learner.predict_with_confidence(scenario['text'])
            print(f"New Prediction: {new_prediction} (confidence: {new_confidence:.3f})")
        
        time.sleep(1)
    
    return system

def demo_knowledge_database(system):
    """Demonstrate knowledge database capabilities"""
    print(f"\n🧠 DEMO: Knowledge Database Capabilities")
    print("="*50)
    
    # Show analysis summary
    summary = system.critical_analyzer.get_analysis_summary()
    
    if isinstance(summary, dict):
        print("📊 Knowledge Database Summary:")
        print(f"  Total queries analyzed: {summary['total_queries']}")
        print(f"  Insights generated: {summary['insights_generated']}")
        print(f"  Average processing time: {summary['avg_processing_time']:.3f}s")
        
        if summary['question_types']:
            print(f"\n🔍 Question Types Analyzed:")
            for q_type, count in summary['question_types'].items():
                print(f"  {q_type.title()}: {count}")
        
        if summary['primary_domains']:
            print(f"\n🎓 Domains Covered:")
            for domain, count in summary['primary_domains'].items():
                print(f"  {domain.title()}: {count}")
    else:
        print(summary)
    
    # Demo search functionality
    print(f"\n🔍 Demo: Knowledge Search")
    search_query = "sentiment analysis"
    matches = system.critical_analyzer.search_knowledge_db(search_query)
    
    if matches:
        print(f"Found {len(matches)} similar queries for '{search_query}':")
        for match in matches[:3]:  # Show top 3
            print(f"  • '{match['query'][:50]}...' (similarity: {match['similarity']:.3f})")
    else:
        print("No matches found (database may be new)")

def demo_analytics_features(system):
    """Demonstrate analytics features"""
    print(f"\n📊 DEMO: Analytics Features")
    print("="*40)
    
    # Show enhanced stats
    print("🎯 Enhanced Learning Statistics:")
    system.enhanced_learner.show_enhanced_stats()
    
    # Show learning insights
    if hasattr(system, 'analytics_dashboard'):
        print(f"\n💡 Learning Pattern Analysis:")
        insights = system.analytics_dashboard.analyze_learning_patterns()
        
        print(f"  Total learning events: {insights['total_learning_events']}")
        print(f"  Average confidence: {insights['avg_confidence']:.3f}")
        print(f"  Learning trend: {insights['learning_trend']}")
        print(f"  Learning efficiency: {insights['learning_efficiency']:.3f}")
        
        if insights['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in insights['recommendations']:
                print(f"  • {rec}")

def interactive_demo():
    """Interactive demo mode"""
    print("🎮 INTERACTIVE DEMO MODE")
    print("="*30)
    print("Try these example queries:")
    print("1. 'I absolutely love this new feature!'")
    print("2. 'How does confidence scoring work in sentiment analysis?'")
    print("3. 'This product is okay but could be better'")
    print("4. 'Analyze this text and explain your reasoning'")
    print("\nOr enter your own queries!")
    
    system = IntelligentSentimentSystem()
    
    while True:
        query = input("\n🎯 Enter query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            break
        elif not query:
            continue
        
        try:
            result = system.analyze_and_predict(query)
            print(f"\n🤖 ANALYSIS:")
            print(result['response'])
            
            if result['sentiment']:
                print(f"\n💭 Sentiment: {result['sentiment']['prediction']} "
                      f"(confidence: {result['sentiment']['confidence']:.3f})")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo interface"""
    print("🚀 INTELLIGENT SENTIMENT ANALYSIS SYSTEM DEMO")
    print("="*60)
    print("This demo showcases the complete adaptive learning system with:")
    print("  🧠 Critical analysis of queries")
    print("  💭 Intelligent sentiment analysis") 
    print("  🎓 Adaptive learning from feedback")
    print("  📊 Knowledge database and analytics")
    
    print(f"\n🎯 DEMO OPTIONS:")
    print("1. Full automated demo (recommended)")
    print("2. Critical analysis demo only")
    print("3. Learning capabilities demo only")
    print("4. Interactive demo mode")
    print("5. Launch full intelligent system")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        # Full automated demo
        system = demo_critical_analysis()
        demo_learning_capabilities(system)
        demo_knowledge_database(system)
        demo_analytics_features(system)
        
        print(f"\n🎉 DEMO COMPLETED!")
        print("The system is now ready for use with learned patterns.")
        
    elif choice == '2':
        demo_critical_analysis()
        
    elif choice == '3':
        system = IntelligentSentimentSystem()
        demo_learning_capabilities(system)
        
    elif choice == '4':
        interactive_demo()
        
    elif choice == '5':
        # Launch full system
        from intelligent_main import main as intelligent_main
        intelligent_main()
        
    elif choice == '6':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
