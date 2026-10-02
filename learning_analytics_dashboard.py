#!/usr/bin/env python3
"""
Learning Analytics Dashboard for Sentiment Analysis
Provides comprehensive visualization and analysis of learning progress
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from adaptive_sentiment import AdaptiveSentimentAnalyzer
from enhanced_adaptive_learning import EnhancedAdaptiveLearning

class LearningAnalyticsDashboard:
    def __init__(self, enhanced_learner=None):
        """Initialize the learning analytics dashboard"""
        self.enhanced_learner = enhanced_learner if enhanced_learner else EnhancedAdaptiveLearning()
        self.analyzer = self.enhanced_learner.analyzer
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("📊 Learning Analytics Dashboard initialized!")
    
    def generate_comprehensive_report(self, save_html=True):
        """Generate a comprehensive learning analytics report"""
        print("📈 Generating comprehensive learning analytics report...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Learning Progress Over Time',
                'Confidence Distribution',
                'Word Learning Patterns',
                'Sentiment Distribution',
                'Accuracy Trends',
                'Learning Velocity'
            ],
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}],
                   [{"secondary_y": True}, {"type": "scatter"}]]
        )
        
        # 1. Learning Progress Over Time
        self._add_learning_progress_plot(fig, row=1, col=1)
        
        # 2. Confidence Distribution
        self._add_confidence_distribution_plot(fig, row=1, col=2)
        
        # 3. Word Learning Patterns
        self._add_word_learning_plot(fig, row=2, col=1)
        
        # 4. Sentiment Distribution
        self._add_sentiment_distribution_plot(fig, row=2, col=2)
        
        # 5. Accuracy Trends
        self._add_accuracy_trends_plot(fig, row=3, col=1)
        
        # 6. Learning Velocity
        self._add_learning_velocity_plot(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🧠 Sentiment Analysis Learning Analytics Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_analytics_report_{timestamp}.html"
            pyo.plot(fig, filename=filename, auto_open=False)
            print(f"📊 Report saved as {filename}")
        
        return fig
    
    def _add_learning_progress_plot(self, fig, row, col):
        """Add learning progress over time plot"""
        if not self.enhanced_learner.learning_history:
            return
        
        # Extract data
        timestamps = [datetime.fromisoformat(event['timestamp']) for event in self.enhanced_learner.learning_history]
        learning_rates = [event['learning_rate'] for event in self.enhanced_learner.learning_history]
        confidences = [event['confidence'] for event in self.enhanced_learner.learning_history]
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=learning_rates, name="Learning Rate", line=dict(color='blue')),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=confidences, name="Confidence", line=dict(color='red')),
            row=row, col=col, secondary_y=True
        )
    
    def _add_confidence_distribution_plot(self, fig, row, col):
        """Add confidence distribution histogram"""
        if not self.enhanced_learner.performance_metrics['confidence_over_time']:
            return
        
        confidences = [m['confidence'] for m in self.enhanced_learner.performance_metrics['confidence_over_time']]
        
        fig.add_trace(
            go.Histogram(x=confidences, name="Confidence Distribution", nbinsx=20),
            row=row, col=col
        )
    
    def _add_word_learning_plot(self, fig, row, col):
        """Add word learning patterns plot"""
        # Get top learned words
        word_counts = {}
        for word, sentiments in self.analyzer.word_sentiment_counts.items():
            total_count = sum(sentiments.values())
            if total_count > 5:  # Only words with significant learning
                word_counts[word] = total_count
        
        # Top 15 words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_words:
            words, counts = zip(*top_words)
            
            fig.add_trace(
                go.Bar(x=list(words), y=list(counts), name="Word Learning Count"),
                row=row, col=col
            )
    
    def _add_sentiment_distribution_plot(self, fig, row, col):
        """Add sentiment distribution pie chart"""
        if not self.enhanced_learner.learning_history:
            return
        
        sentiments = [event['correct'] for event in self.enhanced_learner.learning_history]
        sentiment_counts = Counter(sentiments)
        
        fig.add_trace(
            go.Pie(labels=list(sentiment_counts.keys()), values=list(sentiment_counts.values()),
                   name="Sentiment Distribution"),
            row=row, col=col
        )
    
    def _add_accuracy_trends_plot(self, fig, row, col):
        """Add accuracy trends over time"""
        if len(self.enhanced_learner.learning_history) < 10:
            return
        
        # Calculate rolling accuracy
        window_size = 10
        accuracies = []
        timestamps = []
        
        for i in range(window_size, len(self.enhanced_learner.learning_history)):
            window = self.enhanced_learner.learning_history[i-window_size:i]
            correct_predictions = sum(1 for event in window if event['predicted'] == event['correct'])
            accuracy = correct_predictions / window_size
            accuracies.append(accuracy)
            timestamps.append(datetime.fromisoformat(window[-1]['timestamp']))
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=accuracies, name="Rolling Accuracy", 
                      line=dict(color='green', width=3)),
            row=row, col=col
        )
    
    def _add_learning_velocity_plot(self, fig, row, col):
        """Add learning velocity scatter plot"""
        if not self.enhanced_learner.learning_history:
            return
        
        # Calculate learning velocity (change in confidence over time)
        velocities = []
        confidences = []
        
        for i in range(1, len(self.enhanced_learner.learning_history)):
            prev_conf = self.enhanced_learner.learning_history[i-1]['confidence']
            curr_conf = self.enhanced_learner.learning_history[i]['confidence']
            velocity = curr_conf - prev_conf
            velocities.append(velocity)
            confidences.append(curr_conf)
        
        fig.add_trace(
            go.Scatter(x=confidences, y=velocities, mode='markers',
                      name="Learning Velocity", marker=dict(size=8, opacity=0.6)),
            row=row, col=col
        )
    
    def create_word_cloud_analysis(self):
        """Create word cloud analysis for learned words"""
        print("☁️ Creating word cloud analysis...")
        
        # Collect word frequencies by sentiment
        sentiment_words = {'Positive': {}, 'Negative': {}, 'Neutral': {}}
        
        for word, sentiments in self.analyzer.word_sentiment_counts.items():
            for sentiment, count in sentiments.items():
                if count > 2:  # Minimum threshold
                    sentiment_key = sentiment.title()
                    if sentiment_key in sentiment_words:
                        sentiment_words[sentiment_key][word] = count
        
        # Create word frequency plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (sentiment, words) in enumerate(sentiment_words.items()):
            if words:
                # Get top 20 words
                top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:20]
                words_list, counts_list = zip(*top_words)
                
                axes[idx].barh(range(len(words_list)), counts_list)
                axes[idx].set_yticks(range(len(words_list)))
                axes[idx].set_yticklabels(words_list)
                axes[idx].set_title(f'Top {sentiment} Words Learned')
                axes[idx].set_xlabel('Learning Count')
            else:
                axes[idx].text(0.5, 0.5, f'No {sentiment} words learned yet', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'Top {sentiment} Words Learned')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"word_learning_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"☁️ Word analysis saved as {filename}")
        
        return sentiment_words
    
    def analyze_learning_patterns(self):
        """Analyze learning patterns and provide insights"""
        print("🔍 Analyzing learning patterns...")
        
        insights = {
            'total_learning_events': len(self.enhanced_learner.learning_history),
            'avg_confidence': 0,
            'learning_trend': 'stable',
            'most_learned_sentiment': 'unknown',
            'learning_efficiency': 0,
            'recommendations': []
        }
        
        if not self.enhanced_learner.learning_history:
            insights['recommendations'].append("No learning data available. Start using the system to generate insights.")
            return insights
        
        # Calculate average confidence
        confidences = [event['confidence'] for event in self.enhanced_learner.learning_history]
        insights['avg_confidence'] = np.mean(confidences)
        
        # Determine learning trend
        if len(confidences) > 10:
            recent_conf = np.mean(confidences[-10:])
            older_conf = np.mean(confidences[:10])
            
            if recent_conf > older_conf + 0.1:
                insights['learning_trend'] = 'improving'
            elif recent_conf < older_conf - 0.1:
                insights['learning_trend'] = 'declining'
        
        # Most learned sentiment
        sentiments = [event['correct'] for event in self.enhanced_learner.learning_history]
        sentiment_counts = Counter(sentiments)
        if sentiment_counts:
            insights['most_learned_sentiment'] = sentiment_counts.most_common(1)[0][0]
        
        # Learning efficiency (correct predictions / total learning events)
        correct_initial = sum(1 for event in self.enhanced_learner.learning_history 
                             if event['predicted'] == event['correct'])
        insights['learning_efficiency'] = correct_initial / len(self.enhanced_learner.learning_history)
        
        # Generate recommendations
        if insights['avg_confidence'] < 0.3:
            insights['recommendations'].append("Low average confidence. Consider more diverse training data.")
        
        if insights['learning_trend'] == 'declining':
            insights['recommendations'].append("Learning trend is declining. Review recent training data quality.")
        
        if insights['learning_efficiency'] < 0.5:
            insights['recommendations'].append("Low learning efficiency. The model is making many incorrect initial predictions.")
        
        if len(set(sentiments)) < 3:
            insights['recommendations'].append("Limited sentiment diversity in training. Add more varied examples.")
        
        return insights
    
    def create_performance_comparison(self, baseline_accuracy=0.7):
        """Create performance comparison visualization"""
        print("📊 Creating performance comparison...")
        
        if len(self.enhanced_learner.learning_history) < 20:
            print("⚠️ Insufficient data for performance comparison (need at least 20 learning events)")
            return
        
        # Calculate rolling accuracy over time
        window_size = 10
        accuracies = []
        timestamps = []
        
        for i in range(window_size, len(self.enhanced_learner.learning_history)):
            window = self.enhanced_learner.learning_history[i-window_size:i]
            correct = sum(1 for event in window if event['predicted'] == event['correct'])
            accuracy = correct / window_size
            accuracies.append(accuracy)
            timestamps.append(datetime.fromisoformat(window[-1]['timestamp']))
        
        # Create comparison plot
        fig = go.Figure()
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=timestamps, y=accuracies,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='blue', width=3)
        ))
        
        # Add baseline
        fig.add_hline(
            y=baseline_accuracy,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline ({baseline_accuracy:.1%})"
        )
        
        # Add improvement areas
        improvement_periods = []
        for i, acc in enumerate(accuracies):
            if acc > baseline_accuracy:
                improvement_periods.append((timestamps[i], acc))
        
        if improvement_periods:
            imp_times, imp_accs = zip(*improvement_periods)
            fig.add_trace(go.Scatter(
                x=imp_times, y=imp_accs,
                mode='markers',
                name='Above Baseline',
                marker=dict(color='green', size=10, symbol='star')
            ))
        
        fig.update_layout(
            title='📈 Model Performance vs Baseline Over Time',
            xaxis_title='Time',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.html"
        pyo.plot(fig, filename=filename, auto_open=False)
        
        print(f"📊 Performance comparison saved as {filename}")
        
        return fig
    
    def export_learning_report(self, filename=None):
        """Export comprehensive learning report as JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_report_{timestamp}.json"
        
        # Gather all analytics data
        insights = self.analyze_learning_patterns()
        word_analysis = {}
        
        # Word learning statistics
        for word, sentiments in self.analyzer.word_sentiment_counts.items():
            total_count = sum(sentiments.values())
            if total_count > 2:
                word_analysis[word] = {
                    'total_learned': total_count,
                    'sentiments': dict(sentiments),
                    'dominant_sentiment': max(sentiments.items(), key=lambda x: x[1])[0] if sentiments else 'none'
                }
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_learning_events': len(self.enhanced_learner.learning_history),
                'total_predictions': len(self.enhanced_learner.performance_metrics['confidence_over_time']),
                'analyzer_stats': {
                    'positive_words': len(self.analyzer.positive_words),
                    'negative_words': len(self.analyzer.negative_words),
                    'neutral_words': len(self.analyzer.neutral_words),
                    'learned_patterns': len(self.analyzer.word_sentiment_counts)
                }
            },
            'learning_insights': insights,
            'word_learning_analysis': word_analysis,
            'performance_metrics': self.enhanced_learner.performance_metrics,
            'recent_learning_history': self.enhanced_learner.learning_history[-50:] if self.enhanced_learner.learning_history else []
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Learning report exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
        
        return report
    
    def show_interactive_dashboard(self):
        """Show interactive dashboard in terminal"""
        print(f"\n📊 LEARNING ANALYTICS DASHBOARD")
        print("="*60)
        
        while True:
            print(f"\nChoose analysis:")
            print("1. Generate Comprehensive Report")
            print("2. Word Cloud Analysis")
            print("3. Learning Pattern Analysis")
            print("4. Performance Comparison")
            print("5. Export Learning Report")
            print("6. Show Current Statistics")
            print("7. Back to Main Menu")
            
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == '1':
                self.generate_comprehensive_report()
                
            elif choice == '2':
                self.create_word_cloud_analysis()
                
            elif choice == '3':
                insights = self.analyze_learning_patterns()
                self._display_insights(insights)
                
            elif choice == '4':
                baseline = input("Enter baseline accuracy (0-1, default 0.7): ").strip()
                baseline = float(baseline) if baseline else 0.7
                self.create_performance_comparison(baseline)
                
            elif choice == '5':
                filename = input("Enter filename (or press Enter for auto): ").strip()
                self.export_learning_report(filename if filename else None)
                
            elif choice == '6':
                self.enhanced_learner.show_enhanced_stats()
                
            elif choice == '7':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _display_insights(self, insights):
        """Display learning insights in a formatted way"""
        print(f"\n🔍 LEARNING PATTERN ANALYSIS")
        print("="*40)
        
        print(f"📊 Overview:")
        print(f"  Total learning events: {insights['total_learning_events']}")
        print(f"  Average confidence: {insights['avg_confidence']:.3f}")
        print(f"  Learning trend: {insights['learning_trend']}")
        print(f"  Most learned sentiment: {insights['most_learned_sentiment']}")
        print(f"  Learning efficiency: {insights['learning_efficiency']:.3f}")
        
        if insights['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n✅ No specific recommendations - learning is progressing well!")

def main():
    """Main interface for learning analytics dashboard"""
    print("📊 LEARNING ANALYTICS DASHBOARD")
    print("="*50)
    
    # Initialize dashboard
    dashboard = LearningAnalyticsDashboard()
    
    # Show interactive dashboard
    dashboard.show_interactive_dashboard()

if __name__ == "__main__":
    main()
