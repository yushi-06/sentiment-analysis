#!/usr/bin/env python3
"""
Critical Analysis System with Adaptive Learning
Analyzes user queries critically and stores insights in a knowledge database
"""

import json
import os
import re
import pickle
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
from enhanced_adaptive_learning import EnhancedAdaptiveLearning

class CriticalAnalysisSystem:
    def __init__(self, knowledge_db_file="critical_analysis_db.pkl"):
        """Initialize the critical analysis system"""
        self.knowledge_db_file = knowledge_db_file
        self.enhanced_learner = EnhancedAdaptiveLearning()
        
        # Critical analysis components
        self.analysis_patterns = {
            'question_types': {
                'factual': ['what', 'when', 'where', 'who', 'which'],
                'analytical': ['why', 'how', 'analyze', 'compare', 'evaluate'],
                'opinion': ['think', 'feel', 'believe', 'opinion', 'prefer'],
                'prediction': ['will', 'predict', 'forecast', 'expect', 'future'],
                'problem_solving': ['solve', 'fix', 'improve', 'optimize', 'resolve']
            },
            'complexity_indicators': {
                'simple': ['yes', 'no', 'basic', 'simple', 'easy'],
                'moderate': ['explain', 'describe', 'outline', 'summarize'],
                'complex': ['analyze', 'synthesize', 'evaluate', 'critique', 'justify']
            },
            'domain_keywords': {
                'sentiment': ['sentiment', 'emotion', 'feeling', 'mood', 'opinion'],
                'technical': ['algorithm', 'model', 'system', 'code', 'implementation'],
                'business': ['strategy', 'market', 'customer', 'revenue', 'growth'],
                'data': ['data', 'dataset', 'analysis', 'statistics', 'metrics']
            }
        }
        
        # Knowledge database structure
        self.knowledge_db = {
            'queries': [],
            'insights': defaultdict(list),
            'patterns': defaultdict(int),
            'learning_events': [],
            'critical_assessments': [],
            'domain_knowledge': defaultdict(dict),
            'user_preferences': {},
            'analysis_history': []
        }
        
        # Load existing knowledge
        self.load_knowledge_db()
        
        print("🧠 Critical Analysis System initialized!")
        print(f"📊 Knowledge entries: {len(self.knowledge_db['queries'])}")
        print(f"🎯 Insights stored: {sum(len(v) for v in self.knowledge_db['insights'].values())}")
    
    def critically_analyze_query(self, query, context=None):
        """Perform critical analysis of user query"""
        analysis_start = datetime.now()
        
        # Step 1: Parse and categorize the query
        query_analysis = self._parse_query_structure(query)
        
        # Step 2: Determine complexity and intent
        complexity_analysis = self._assess_complexity(query)
        
        # Step 3: Extract domain and context
        domain_analysis = self._identify_domain(query, context)
        
        # Step 4: Perform sentiment analysis if relevant
        sentiment_analysis = None
        if self._is_sentiment_related(query):
            prediction, confidence, uncertain, _, _ = self.enhanced_learner.predict_with_confidence(query)
            sentiment_analysis = {
                'prediction': prediction,
                'confidence': confidence,
                'uncertain': uncertain,
                'analysis_type': 'sentiment'
            }
        
        # Step 5: Generate critical insights
        critical_insights = self._generate_critical_insights(
            query, query_analysis, complexity_analysis, domain_analysis, sentiment_analysis
        )
        
        # Step 6: Store in knowledge database
        analysis_record = {
            'query': query,
            'timestamp': analysis_start.isoformat(),
            'query_analysis': query_analysis,
            'complexity_analysis': complexity_analysis,
            'domain_analysis': domain_analysis,
            'sentiment_analysis': sentiment_analysis,
            'critical_insights': critical_insights,
            'context': context,
            'processing_time': (datetime.now() - analysis_start).total_seconds()
        }
        
        self._store_analysis(analysis_record)
        
        return analysis_record
    
    def _parse_query_structure(self, query):
        """Parse the structure and type of the query"""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Identify question type
        question_type = 'statement'
        for q_type, keywords in self.analysis_patterns['question_types'].items():
            if any(keyword in query_lower for keyword in keywords):
                question_type = q_type
                break
        
        # Extract key entities and concepts
        entities = self._extract_entities(query)
        
        # Identify grammatical patterns
        has_question_mark = '?' in query
        is_imperative = query_lower.startswith(('add', 'create', 'make', 'build', 'implement'))
        is_conditional = any(word in query_lower for word in ['if', 'when', 'unless', 'provided'])
        
        return {
            'question_type': question_type,
            'entities': entities,
            'word_count': len(words),
            'has_question_mark': has_question_mark,
            'is_imperative': is_imperative,
            'is_conditional': is_conditional,
            'key_verbs': self._extract_verbs(query),
            'key_nouns': self._extract_nouns(query)
        }
    
    def _assess_complexity(self, query):
        """Assess the complexity level of the query"""
        query_lower = query.lower()
        
        complexity_score = 0
        complexity_indicators = []
        
        # Check complexity indicators
        for level, keywords in self.analysis_patterns['complexity_indicators'].items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                complexity_indicators.append((level, matches))
                if level == 'simple':
                    complexity_score += 1
                elif level == 'moderate':
                    complexity_score += 2
                elif level == 'complex':
                    complexity_score += 3
        
        # Additional complexity factors
        if len(query.split()) > 20:
            complexity_score += 1
        if query.count(',') > 2:
            complexity_score += 1
        if any(word in query_lower for word in ['however', 'nevertheless', 'furthermore', 'moreover']):
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score <= 2:
            complexity_level = 'simple'
        elif complexity_score <= 5:
            complexity_level = 'moderate'
        else:
            complexity_level = 'complex'
        
        return {
            'complexity_level': complexity_level,
            'complexity_score': complexity_score,
            'indicators': complexity_indicators,
            'requires_deep_analysis': complexity_score > 4
        }
    
    def _identify_domain(self, query, context):
        """Identify the domain and subject area of the query"""
        query_lower = query.lower()
        
        domain_scores = {}
        for domain, keywords in self.analysis_patterns['domain_keywords'].items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        # Context analysis
        context_clues = []
        if context:
            context_lower = context.lower()
            for domain, keywords in self.analysis_patterns['domain_keywords'].items():
                if any(keyword in context_lower for keyword in keywords):
                    context_clues.append(domain)
        
        return {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'context_clues': context_clues,
            'is_cross_domain': len(domain_scores) > 1
        }
    
    def _is_sentiment_related(self, query):
        """Check if query is related to sentiment analysis"""
        sentiment_keywords = ['sentiment', 'emotion', 'feeling', 'opinion', 'mood', 'positive', 'negative', 'neutral']
        return any(keyword in query.lower() for keyword in sentiment_keywords)
    
    def _generate_critical_insights(self, query, query_analysis, complexity_analysis, domain_analysis, sentiment_analysis):
        """Generate critical insights about the query"""
        insights = []
        
        # Query structure insights
        if query_analysis['question_type'] == 'analytical':
            insights.append({
                'type': 'analytical_query',
                'insight': 'This query requires analytical thinking and may need multi-step reasoning.',
                'recommendation': 'Provide structured analysis with evidence and reasoning.'
            })
        
        # Complexity insights
        if complexity_analysis['complexity_level'] == 'complex':
            insights.append({
                'type': 'high_complexity',
                'insight': 'This is a complex query that may require breaking down into sub-problems.',
                'recommendation': 'Consider decomposing into smaller, manageable parts.'
            })
        
        # Domain insights
        if domain_analysis['is_cross_domain']:
            insights.append({
                'type': 'cross_domain',
                'insight': 'This query spans multiple domains, requiring interdisciplinary knowledge.',
                'recommendation': 'Ensure comprehensive coverage of all relevant domains.'
            })
        
        # Sentiment insights
        if sentiment_analysis and sentiment_analysis['uncertain']:
            insights.append({
                'type': 'sentiment_uncertainty',
                'insight': f"Sentiment analysis shows uncertainty (confidence: {sentiment_analysis['confidence']:.3f})",
                'recommendation': 'Consider asking for clarification or additional context.'
            })
        
        # Pattern-based insights
        if query_analysis['is_imperative']:
            insights.append({
                'type': 'action_required',
                'insight': 'This query requests specific action or implementation.',
                'recommendation': 'Provide actionable steps and concrete solutions.'
            })
        
        return insights
    
    def _extract_entities(self, text):
        """Extract key entities from text (simplified NER)"""
        # Simple entity extraction - in production, use spaCy or similar
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized)
        
        # Extract technical terms
        technical_patterns = [
            r'\b\w+\.py\b',  # Python files
            r'\b\w+\.csv\b',  # CSV files
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_verbs(self, text):
        """Extract key verbs (simplified)"""
        # Simple verb extraction - in production, use POS tagging
        common_verbs = ['add', 'create', 'make', 'build', 'implement', 'analyze', 'compare', 'evaluate', 'improve', 'fix']
        words = text.lower().split()
        return [word for word in words if word in common_verbs]
    
    def _extract_nouns(self, text):
        """Extract key nouns (simplified)"""
        # Simple noun extraction
        technical_nouns = ['system', 'model', 'algorithm', 'data', 'analysis', 'learning', 'interface', 'database']
        words = text.lower().split()
        return [word for word in words if word in technical_nouns]
    
    def _store_analysis(self, analysis_record):
        """Store analysis in knowledge database"""
        # Add to queries
        self.knowledge_db['queries'].append(analysis_record)
        
        # Update patterns
        self.knowledge_db['patterns'][analysis_record['query_analysis']['question_type']] += 1
        self.knowledge_db['patterns'][analysis_record['complexity_analysis']['complexity_level']] += 1
        self.knowledge_db['patterns'][analysis_record['domain_analysis']['primary_domain']] += 1
        
        # Store insights
        for insight in analysis_record['critical_insights']:
            self.knowledge_db['insights'][insight['type']].append({
                'query': analysis_record['query'],
                'insight': insight['insight'],
                'timestamp': analysis_record['timestamp']
            })
        
        # Update domain knowledge
        domain = analysis_record['domain_analysis']['primary_domain']
        if domain not in self.knowledge_db['domain_knowledge']:
            self.knowledge_db['domain_knowledge'][domain] = {}
        
        # Store entities in domain knowledge
        for entity in analysis_record['query_analysis']['entities']:
            if entity not in self.knowledge_db['domain_knowledge'][domain]:
                self.knowledge_db['domain_knowledge'][domain][entity] = 0
            self.knowledge_db['domain_knowledge'][domain][entity] += 1
        
        # Save to file
        self.save_knowledge_db()
    
    def get_analysis_summary(self):
        """Get summary of analysis patterns"""
        total_queries = len(self.knowledge_db['queries'])
        
        if total_queries == 0:
            return "No queries analyzed yet."
        
        summary = {
            'total_queries': total_queries,
            'question_types': dict(Counter(q['query_analysis']['question_type'] for q in self.knowledge_db['queries'])),
            'complexity_levels': dict(Counter(q['complexity_analysis']['complexity_level'] for q in self.knowledge_db['queries'])),
            'primary_domains': dict(Counter(q['domain_analysis']['primary_domain'] for q in self.knowledge_db['queries'])),
            'avg_processing_time': sum(q['processing_time'] for q in self.knowledge_db['queries']) / total_queries,
            'insights_generated': sum(len(v) for v in self.knowledge_db['insights'].values())
        }
        
        return summary
    
    def search_knowledge_db(self, query, limit=5):
        """Search the knowledge database for similar queries"""
        query_lower = query.lower()
        matches = []
        
        for record in self.knowledge_db['queries']:
            # Simple similarity based on common words
            record_words = set(record['query'].lower().split())
            query_words = set(query_lower.split())
            
            common_words = record_words.intersection(query_words)
            similarity = len(common_words) / max(len(record_words), len(query_words))
            
            if similarity > 0.2:  # Threshold for relevance
                matches.append({
                    'query': record['query'],
                    'similarity': similarity,
                    'timestamp': record['timestamp'],
                    'insights': record['critical_insights']
                })
        
        # Sort by similarity and return top matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:limit]
    
    def interactive_analysis_mode(self):
        """Interactive mode for critical analysis"""
        print(f"\n🧠 CRITICAL ANALYSIS SYSTEM")
        print("="*50)
        print("This system will critically analyze your queries and store insights.")
        print("Commands:")
        print("  - Type your question or request")
        print("  - Type 'summary' to see analysis patterns")
        print("  - Type 'search <query>' to search knowledge database")
        print("  - Type 'insights' to see generated insights")
        print("  - Type 'quit' to exit")
        print("-"*50)
        
        while True:
            user_input = input("\n🎯 Enter your query: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'summary':
                summary = self.get_analysis_summary()
                self._display_summary(summary)
                continue
            elif user_input.lower().startswith('search '):
                search_query = user_input[7:]
                matches = self.search_knowledge_db(search_query)
                self._display_search_results(matches)
                continue
            elif user_input.lower() == 'insights':
                self._display_insights()
                continue
            elif not user_input:
                print("Please enter a query.")
                continue
            
            # Perform critical analysis
            print(f"\n🔍 Analyzing: '{user_input}'")
            analysis = self.critically_analyze_query(user_input)
            
            # Display results
            self._display_analysis_results(analysis)
            
            # If sentiment-related, offer adaptive learning
            if analysis['sentiment_analysis']:
                self._offer_sentiment_learning(user_input, analysis['sentiment_analysis'])
    
    def _display_analysis_results(self, analysis):
        """Display analysis results in a formatted way"""
        print(f"\n📊 CRITICAL ANALYSIS RESULTS")
        print("="*40)
        
        # Query structure
        qa = analysis['query_analysis']
        print(f"🔍 Query Type: {qa['question_type'].title()}")
        print(f"📝 Word Count: {qa['word_count']}")
        print(f"🎯 Key Entities: {', '.join(qa['entities']) if qa['entities'] else 'None'}")
        
        # Complexity
        ca = analysis['complexity_analysis']
        print(f"\n⚡ Complexity: {ca['complexity_level'].title()} (score: {ca['complexity_score']})")
        if ca['requires_deep_analysis']:
            print("   ⚠️  Requires deep analysis")
        
        # Domain
        da = analysis['domain_analysis']
        print(f"\n🎓 Primary Domain: {da['primary_domain'].title()}")
        if da['is_cross_domain']:
            print("   🔄 Cross-domain query")
        
        # Sentiment (if applicable)
        if analysis['sentiment_analysis']:
            sa = analysis['sentiment_analysis']
            print(f"\n💭 Sentiment: {sa['prediction']} (confidence: {sa['confidence']:.3f})")
            if sa['uncertain']:
                print("   ❓ Uncertain prediction")
        
        # Critical insights
        if analysis['critical_insights']:
            print(f"\n💡 CRITICAL INSIGHTS:")
            for i, insight in enumerate(analysis['critical_insights'], 1):
                print(f"   {i}. {insight['insight']}")
                print(f"      💡 Recommendation: {insight['recommendation']}")
        
        print(f"\n⏱️  Processing time: {analysis['processing_time']:.3f} seconds")
    
    def _display_summary(self, summary):
        """Display analysis summary"""
        if isinstance(summary, str):
            print(f"\n📊 {summary}")
            return
        
        print(f"\n📊 ANALYSIS SUMMARY")
        print("="*30)
        print(f"Total queries analyzed: {summary['total_queries']}")
        print(f"Average processing time: {summary['avg_processing_time']:.3f}s")
        print(f"Total insights generated: {summary['insights_generated']}")
        
        print(f"\n🔍 Question Types:")
        for q_type, count in summary['question_types'].items():
            print(f"  {q_type.title()}: {count}")
        
        print(f"\n⚡ Complexity Levels:")
        for level, count in summary['complexity_levels'].items():
            print(f"  {level.title()}: {count}")
        
        print(f"\n🎓 Primary Domains:")
        for domain, count in summary['primary_domains'].items():
            print(f"  {domain.title()}: {count}")
    
    def _display_search_results(self, matches):
        """Display search results"""
        if not matches:
            print("🔍 No similar queries found in knowledge database.")
            return
        
        print(f"\n🔍 SIMILAR QUERIES FOUND:")
        print("="*30)
        
        for i, match in enumerate(matches, 1):
            print(f"{i}. '{match['query']}'")
            print(f"   Similarity: {match['similarity']:.3f}")
            print(f"   Date: {match['timestamp'][:10]}")
            if match['insights']:
                print(f"   Insights: {len(match['insights'])} generated")
            print()
    
    def _display_insights(self):
        """Display stored insights"""
        if not self.knowledge_db['insights']:
            print("💡 No insights stored yet.")
            return
        
        print(f"\n💡 STORED INSIGHTS")
        print("="*30)
        
        for insight_type, insights in self.knowledge_db['insights'].items():
            print(f"\n🔍 {insight_type.replace('_', ' ').title()}:")
            for insight in insights[-3:]:  # Show last 3
                print(f"  • {insight['insight']}")
                print(f"    Query: '{insight['query'][:50]}...'")
    
    def _offer_sentiment_learning(self, query, sentiment_analysis):
        """Offer sentiment learning opportunity"""
        prediction = sentiment_analysis['prediction']
        confidence = sentiment_analysis['confidence']
        
        print(f"\n🎓 ADAPTIVE LEARNING OPPORTUNITY")
        print(f"Sentiment prediction: {prediction} (confidence: {confidence:.3f})")
        
        feedback = input("Is this sentiment correct? (y/n) or provide correct sentiment: ").strip()
        
        if feedback.lower() in ['n', 'no']:
            correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
            if correct in ['Positive', 'Negative', 'Neutral']:
                self.enhanced_learner.adaptive_learn_from_feedback(query, prediction, correct, confidence)
                print("✅ System learned from your feedback!")
        elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
            if feedback.title() != prediction:
                self.enhanced_learner.adaptive_learn_from_feedback(query, prediction, feedback.title(), confidence)
                print("✅ System learned from your feedback!")
    
    def save_knowledge_db(self):
        """Save knowledge database to file"""
        try:
            with open(self.knowledge_db_file, 'wb') as f:
                pickle.dump(self.knowledge_db, f)
        except Exception as e:
            print(f"⚠️  Warning: Could not save knowledge database: {e}")
    
    def load_knowledge_db(self):
        """Load knowledge database from file"""
        if os.path.exists(self.knowledge_db_file):
            try:
                with open(self.knowledge_db_file, 'rb') as f:
                    loaded_db = pickle.load(f)
                    self.knowledge_db.update(loaded_db)
                print(f"📖 Loaded existing knowledge database")
            except Exception as e:
                print(f"⚠️  Warning: Could not load knowledge database: {e}")

def main():
    """Main interface for critical analysis system"""
    print("🧠 CRITICAL ANALYSIS SYSTEM WITH ADAPTIVE LEARNING")
    print("="*60)
    
    analyzer = CriticalAnalysisSystem()
    analyzer.interactive_analysis_mode()

if __name__ == "__main__":
    main()
