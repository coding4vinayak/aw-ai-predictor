import re
import string
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os
import logging

# Simple sentiment lexicon for basic sentiment analysis
POSITIVE_WORDS = [
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
    'brilliant', 'outstanding', 'superb', 'perfect', 'love', 'like', 'happy',
    'pleased', 'satisfied', 'recommend', 'best', 'top', 'quality'
]

NEGATIVE_WORDS = [
    'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
    'worst', 'poor', 'disappointing', 'frustrated', 'angry', 'upset', 'annoyed',
    'dissatisfied', 'problem', 'issue', 'broken', 'defective', 'useless'
]

def preprocess_text(text):
    """
    Preprocess text for NLP tasks
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment_model():
    """Load or create sentiment analysis model"""
    model_path = "ml_models/sentiment_model.pkl"
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info("Loaded existing sentiment model")
        else:
            # Create a simple pipeline model
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            # Create some dummy training data for demonstration
            dummy_texts = [
                "This product is amazing and I love it",
                "Great quality and fast delivery",
                "Excellent service, highly recommend",
                "Terrible product, waste of money",
                "Poor quality and bad customer service",
                "I hate this product, very disappointed"
            ]
            dummy_labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative
            
            model.fit(dummy_texts, dummy_labels)
            
            # Save the model
            os.makedirs('ml_models', exist_ok=True)
            joblib.dump(model, model_path)
            logging.info("Created and saved dummy sentiment model")
        
        return model
    except Exception as e:
        logging.error(f"Error with sentiment model: {str(e)}")
        return None

def analyze_sentiment(text):
    """
    Analyze sentiment of input text
    
    Args:
        text (str): Input text for sentiment analysis
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'score': 0.0,
                'error': 'Invalid or empty text input'
            }
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Try to use trained model first
        model = get_sentiment_model()
        
        if model:
            try:
                # Get prediction from model
                prediction = model.predict([cleaned_text])[0]
                probabilities = model.predict_proba([cleaned_text])[0]
                
                # Calculate confidence and sentiment
                confidence = max(probabilities)
                sentiment = 'positive' if prediction == 1 else 'negative'
                score = probabilities[1] - probabilities[0]  # positive prob - negative prob
                
                return {
                    'sentiment': sentiment,
                    'confidence': round(confidence, 4),
                    'score': round(score, 4),
                    'method': 'ml_model',
                    'text_length': len(text),
                    'cleaned_text_length': len(cleaned_text)
                }
            except Exception as e:
                logging.warning(f"Model prediction failed, falling back to lexicon: {str(e)}")
        
        # Fallback to lexicon-based approach
        words = cleaned_text.split()
        
        positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = 'neutral'
            confidence = 0.5
            score = 0.0
        else:
            if positive_count > negative_count:
                sentiment = 'positive'
                score = (positive_count - negative_count) / len(words)
                confidence = positive_count / total_sentiment_words
            elif negative_count > positive_count:
                sentiment = 'negative'
                score = (positive_count - negative_count) / len(words)
                confidence = negative_count / total_sentiment_words
            else:
                sentiment = 'neutral'
                score = 0.0
                confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'score': round(score, 4),
            'method': 'lexicon',
            'positive_words': positive_count,
            'negative_words': negative_count,
            'text_length': len(text),
            'word_count': len(words)
        }
        
    except Exception as e:
        logging.error(f"Sentiment analysis error: {str(e)}")
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'score': 0.0,
            'error': str(e)
        }

def extract_keywords(text, max_keywords=10):
    """
    Extract keywords from text using TF-IDF
    
    Args:
        text (str): Input text for keyword extraction
        max_keywords (int): Maximum number of keywords to return
        
    Returns:
        dict: Keyword extraction results
    """
    try:
        if not text or not isinstance(text, str):
            return {
                'keywords': [],
                'error': 'Invalid or empty text input'
            }
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        if len(cleaned_text.split()) < 3:
            return {
                'keywords': [],
                'error': 'Text too short for keyword extraction'
            }
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1
        )
        
        try:
            # Fit and transform the text
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, tfidf_scores))
            
            # Sort by score and take top keywords
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            top_keywords = keyword_scores[:max_keywords]
            
            # Format results
            keywords = [
                {
                    'keyword': keyword,
                    'score': round(score, 4),
                    'type': 'bigram' if ' ' in keyword else 'unigram'
                }
                for keyword, score in top_keywords if score > 0
            ]
            
            return {
                'keywords': keywords,
                'total_features': len(feature_names),
                'text_length': len(text),
                'method': 'tfidf'
            }
            
        except ValueError as e:
            # Fallback to simple frequency-based extraction
            logging.warning(f"TF-IDF failed, using frequency-based extraction: {str(e)}")
            return extract_keywords_frequency(cleaned_text, max_keywords)
        
    except Exception as e:
        logging.error(f"Keyword extraction error: {str(e)}")
        return {
            'keywords': [],
            'error': str(e)
        }

def extract_keywords_frequency(text, max_keywords=10):
    """
    Extract keywords using frequency-based approach (fallback)
    
    Args:
        text (str): Input text
        max_keywords (int): Maximum number of keywords
        
    Returns:
        dict: Keyword extraction results
    """
    try:
        # Remove punctuation and split into words
        words = text.translate(str.maketrans('', '', string.punctuation)).split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Filter words
        filtered_words = [
            word.lower() for word in words 
            if len(word) > 2 and word.lower() not in stop_words
        ]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top keywords
        top_words = word_counts.most_common(max_keywords)
        
        # Calculate relative scores
        max_count = max(word_counts.values()) if word_counts else 1
        
        keywords = [
            {
                'keyword': word,
                'score': round(count / max_count, 4),
                'frequency': count,
                'type': 'unigram'
            }
            for word, count in top_words
        ]
        
        return {
            'keywords': keywords,
            'total_words': len(words),
            'unique_words': len(set(filtered_words)),
            'text_length': len(text),
            'method': 'frequency'
        }
        
    except Exception as e:
        logging.error(f"Frequency-based keyword extraction error: {str(e)}")
        return {
            'keywords': [],
            'error': str(e)
        }

def analyze_text_stats(text):
    """
    Analyze basic text statistics
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Text statistics
    """
    try:
        if not text or not isinstance(text, str):
            return {
                'error': 'Invalid or empty text input'
            }
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        paragraph_count = len(text.split('\n\n'))
        
        # Average metrics
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_chars_per_word = char_count / max(word_count, 1)
        
        # Reading level (simple approximation)
        if avg_words_per_sentence > 20:
            reading_level = "Advanced"
        elif avg_words_per_sentence > 15:
            reading_level = "Intermediate"
        else:
            reading_level = "Basic"
        
        return {
            'character_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'avg_chars_per_word': round(avg_chars_per_word, 2),
            'reading_level': reading_level
        }
        
    except Exception as e:
        logging.error(f"Text stats analysis error: {str(e)}")
        return {
            'error': str(e)
        }

def train_sentiment_model(training_data):
    """
    Train a new sentiment model with provided data
    
    Args:
        training_data (list): List of {'text': str, 'sentiment': int} dictionaries
        
    Returns:
        dict: Training results
    """
    try:
        if not training_data or len(training_data) < 10:
            return {
                'error': 'Insufficient training data (minimum 10 samples required)'
            }
        
        # Prepare training data
        texts = [item['text'] for item in training_data]
        labels = [item['sentiment'] for item in training_data]
        
        # Create and train model
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        model.fit(texts, labels)
        
        # Save model
        os.makedirs('ml_models', exist_ok=True)
        joblib.dump(model, 'ml_models/sentiment_model.pkl')
        
        # Calculate training accuracy
        train_predictions = model.predict(texts)
        accuracy = sum(1 for true, pred in zip(labels, train_predictions) if true == pred) / len(labels)
        
        return {
            'message': 'Sentiment model trained successfully',
            'train_accuracy': round(accuracy, 4),
            'training_samples': len(training_data),
            'model_type': 'MultinomialNB with TF-IDF'
        }
        
    except Exception as e:
        logging.error(f"Sentiment model training error: {str(e)}")
        return {
            'error': str(e),
            'message': 'Sentiment model training failed'
        }
