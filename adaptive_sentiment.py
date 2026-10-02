import json
import os
import re
from collections import defaultdict, Counter
import pickle
import pandas as pd
from datetime import datetime

class AdaptiveSentimentAnalyzer:
    def __init__(self, knowledge_file="sentiment_knowledge.pkl"):
        self.knowledge_file = knowledge_file
        
        # Core sentiment dictionaries (will be expanded through learning)
        self.positive_words = {
            'love', 'great', 'awesome', 'amazing', 'excellent', 'fantastic', 'wonderful',
            'good', 'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'marvelous',
            'terrific', 'magnificent', 'splendid', 'fabulous', 'incredible', 'impressive',
            'delightful', 'exceptional', 'remarkable', 'beautiful', 'lovely', 'nice',
            'pleased', 'happy', 'satisfied', 'thrilled', 'excited', 'glad', 'joyful',
            'like', 'liked', 'likes', 'liking', 'enjoy', 'enjoyed', 'enjoying', 'enjoys',
            'adore', 'adored', 'adores', 'adoring', 'appreciate', 'appreciated', 'appreciates'
        }
        
        self.negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'bad',
            'pathetic', 'useless', 'worthless', 'disappointing', 'frustrated', 'angry',
            'furious', 'annoying', 'irritating', 'stupid', 'ridiculous', 'absurd',
            'nightmare', 'disaster', 'catastrophe', 'failure', 'broken', 'damaged',
            'poor', 'inferior', 'subpar', 'unacceptable', 'appalling', 'dreadful',
            'disgusted', 'outraged', 'livid', 'miserable', 'depressed', 'sad'
        }
        
        self.neutral_words = {
            'okay', 'ok', 'fine', 'average', 'normal', 'standard', 'typical',
            'regular', 'ordinary', 'decent', 'acceptable', 'fair', 'moderate',
            'reasonable', 'adequate', 'sufficient', 'satisfactory', 'passable'
        }
        
        # Intensifiers and negations
        self.intensifiers = {
            'very', 'extremely', 'incredibly', 'amazingly', 'absolutely', 'totally',
            'completely', 'utterly', 'really', 'quite', 'so', 'too', 'super',
            'highly', 'exceptionally', 'tremendously', 'enormously', 'immensely',
            'soo', 'sooo', 'much', 'lots', 'tons', 'loads', 'plenty'
        }
        
        self.negations = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'hardly', 'scarcely', 'barely', 'seldom',
            'cannot', 'cant', 'couldnt', 'wouldnt', 'shouldnt', 'dont', 'doesnt'
        }
        
        # Learning data storage
        self.word_sentiment_counts = defaultdict(lambda: defaultdict(int))
        self.pattern_scores = defaultdict(list)
        self.user_corrections = []
        self.training_texts = []
        
        # Load existing knowledge
        self.load_knowledge()
        
        print(f"Known positive words: {len(self.positive_words)}")
        print(f"Known negative words: {len(self.negative_words)}")
        print(f"Learned patterns: {len(self.pattern_scores)}")
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace and split
        words = [word.strip() for word in text.split() if word.strip()]
        return words
    
    def analyze_word_by_word(self, words):
        """Analyze sentiment word by word with context"""
        sentiment_score = 0
        word_contributions = []
        
        i = 0
        while i < len(words):
            word = words[i]
            base_score = 0
            context_info = {}
            
            # Check if word is in our known sentiment words
            if word in self.positive_words:
                base_score = 1
                context_info['type'] = 'positive'
            elif word in self.negative_words:
                base_score = -1
                context_info['type'] = 'negative'
            elif word in self.neutral_words:
                base_score = 0
                context_info['type'] = 'neutral'
            
            # Check learned word patterns
            if word in self.word_sentiment_counts:
                counts = self.word_sentiment_counts[word]
                total = sum(counts.values())
                if total > 2:  # Only use learned patterns if we have enough data
                    learned_score = (counts['positive'] - counts['negative']) / total
                    # Weight the learned score less if it's weak
                    if abs(learned_score) < 0.3:
                        learned_score *= 0.5  # Reduce impact of weak patterns
                    base_score = (base_score + learned_score) / 2
                    context_info['learned'] = True
            
            # Handle negations
            negated = False
            if i > 0 and words[i-1] in self.negations:
                base_score = -base_score
                negated = True
                context_info['negated'] = True
            
            # Handle intensifiers
            intensified = False
            if i > 0 and words[i-1] in self.intensifiers:
                base_score = base_score * 1.5
                intensified = True
                context_info['intensified'] = True
            
            # Handle intensifier words themselves (like "much", "soo")
            if word in self.intensifiers and base_score == 0:
                # Look for nearby sentiment words to intensify
                for j in range(max(0, i-2), min(len(words), i+3)):
                    if j != i:
                        nearby_word = words[j]
                        if nearby_word in self.positive_words:
                            base_score = 0.3  # Boost for intensifier near positive
                            context_info['intensifier_boost'] = True
                            break
                        elif nearby_word in self.negative_words:
                            base_score = -0.3  # Boost for intensifier near negative
                            context_info['intensifier_boost'] = True
                            break
            
            # Handle double negation
            if i > 1 and words[i-2] in self.negations and words[i-1] in self.negations:
                base_score = abs(base_score)  # Double negative = positive
                context_info['double_negation'] = True
            
            sentiment_score += base_score
            word_contributions.append({
                'word': word,
                'score': base_score,
                'context': context_info
            })
            
            i += 1
        
        return sentiment_score, word_contributions
    
    def predict_sentiment(self, text, show_analysis=False):
        """Predict sentiment with detailed analysis"""
        if not text.strip():
            return "Neutral", 0.0, []
        
        words = self.preprocess_text(text)
        sentiment_score, word_contributions = self.analyze_word_by_word(words)
        
        # Normalize score
        if len(words) > 0:
            normalized_score = sentiment_score / len(words)
        else:
            normalized_score = 0
        
        # Determine sentiment class with better thresholds
        if normalized_score > 0.05:  # More sensitive to positive
            prediction = "Positive"
        elif normalized_score < -0.05:  # More sensitive to negative
            prediction = "Negative"
        else:
            prediction = "Neutral"
        
        if show_analysis:
            print(f"\n--- Word-by-Word Analysis ---")
            print(f"Text: '{text}'")
            print(f"Words analyzed: {words}")
            print(f"Raw sentiment score: {sentiment_score}")
            print(f"Normalized score: {normalized_score:.3f}")
            print(f"Prediction: {prediction}")
            print("\nWord contributions:")
            for contrib in word_contributions:
                if contrib['score'] != 0:
                    print(f"  '{contrib['word']}': {contrib['score']:.2f} {contrib['context']}")
        
        return prediction, normalized_score, word_contributions
    
    def learn_from_correction(self, text, predicted_sentiment, correct_sentiment):
        """Learn from user corrections"""
        words = self.preprocess_text(text)
        
        # Record the correction
        correction = {
            'text': text,
            'predicted': predicted_sentiment,
            'correct': correct_sentiment,
            'timestamp': datetime.now().isoformat(),
            'words': words
        }
        self.user_corrections.append(correction)
        
        # Update word sentiment counts
        for word in words:
            self.word_sentiment_counts[word][correct_sentiment.lower()] += 1
        
        # Learn new patterns
        if correct_sentiment == "Positive":
            self.positive_words.update(word for word in words if len(word) > 2)
        elif correct_sentiment == "Negative":
            self.negative_words.update(word for word in words if len(word) > 2)
        elif correct_sentiment == "Neutral":
            self.neutral_words.update(word for word in words if len(word) > 2)
        
        # Save updated knowledge
        self.save_knowledge()
        
        print(f"✅ Learned from correction: '{text}' is {correct_sentiment}")
        print(f"📚 Total corrections learned: {len(self.user_corrections)}")
    
    def add_training_data(self, text, sentiment):
        """Add new training data"""
        self.training_texts.append({
            'text': text,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        })
        
        # Immediately learn from this data
        self.learn_from_correction(text, "Unknown", sentiment)
    
    def save_knowledge(self):
        """Save learned knowledge to file"""
        knowledge = {
            'positive_words': list(self.positive_words),
            'negative_words': list(self.negative_words),
            'neutral_words': list(self.neutral_words),
            'word_sentiment_counts': dict(self.word_sentiment_counts),
            'pattern_scores': dict(self.pattern_scores),
            'user_corrections': self.user_corrections,
            'training_texts': self.training_texts,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.knowledge_file, 'wb') as f:
                pickle.dump(knowledge, f)
            print(f"💾 Knowledge saved to {self.knowledge_file}")
        except Exception as e:
            print(f"❌ Error saving knowledge: {e}")
    
    def load_knowledge(self):
        """Load previously learned knowledge"""
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'rb') as f:
                    knowledge = pickle.load(f)
                
                self.positive_words.update(knowledge.get('positive_words', []))
                self.negative_words.update(knowledge.get('negative_words', []))
                self.neutral_words.update(knowledge.get('neutral_words', []))
                self.word_sentiment_counts.update(knowledge.get('word_sentiment_counts', {}))
                self.pattern_scores.update(knowledge.get('pattern_scores', {}))
                self.user_corrections = knowledge.get('user_corrections', [])
                self.training_texts = knowledge.get('training_texts', [])
                
                print(f"📖 Loaded existing knowledge from {self.knowledge_file}")
                print(f"📊 Previous corrections: {len(self.user_corrections)}")
                
            except Exception as e:
                print(f"⚠️  Error loading knowledge: {e}")
                print("Starting with fresh knowledge base")
    
    def show_learning_stats(self):
        """Show learning statistics"""
        print(f"\n📈 Learning Statistics:")
        print(f"Positive words known: {len(self.positive_words)}")
        print(f"Negative words known: {len(self.negative_words)}")
        print(f"Neutral words known: {len(self.neutral_words)}")
        print(f"User corrections: {len(self.user_corrections)}")
        print(f"Training texts: {len(self.training_texts)}")
        print(f"Learned word patterns: {len(self.word_sentiment_counts)}")
        
        if self.user_corrections:
            print(f"\nRecent corrections:")
            for correction in self.user_corrections[-3:]:
                print(f"  '{correction['text']}' → {correction['correct']}")

def interactive_learning_mode():
    """Interactive mode with learning capability"""
    analyzer = AdaptiveSentimentAnalyzer()
    
    print("\n" + "="*60)
    print("🧠 ADAPTIVE SENTIMENT ANALYZER - LEARNING MODE")
    print("="*60)
    print("This analyzer learns from your feedback!")
    print("Commands:")
    print("  - Type text to analyze")
    print("  - Type 'stats' to see learning statistics")
    print("  - Type 'quit' to exit")
    print("-"*60)
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            print("👋 Goodbye! Your corrections have been saved.")
            break
        elif text.lower() == 'stats':
            analyzer.show_learning_stats()
            continue
        elif not text:
            print("Please enter some text.")
            continue
        
        # Analyze the text
        prediction, score, contributions = analyzer.predict_sentiment(text, show_analysis=True)
        
        # Ask for feedback
        print(f"\n🤖 Prediction: {prediction} (confidence: {abs(score):.2f})")
        feedback = input("Is this correct? (y/n) or provide correct answer (Positive/Negative/Neutral): ").strip()
        
        if feedback.lower() in ['n', 'no']:
            correct = input("What's the correct sentiment? (Positive/Negative/Neutral): ").strip().title()
            if correct in ['Positive', 'Negative', 'Neutral']:
                analyzer.learn_from_correction(text, prediction, correct)
            else:
                print("Invalid sentiment. Please use: Positive, Negative, or Neutral")
        elif feedback.title() in ['Positive', 'Negative', 'Neutral']:
            if feedback.title() != prediction:
                analyzer.learn_from_correction(text, prediction, feedback.title())
            else:
                print("✅ Thanks for confirming!")
        elif feedback.lower() in ['y', 'yes']:
            print("✅ Thanks for confirming!")
        else:
            print("No feedback recorded.")

if __name__ == "__main__":
    interactive_learning_mode()
