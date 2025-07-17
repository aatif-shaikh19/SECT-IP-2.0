import os
from typing import Dict, List

class Config:
    """Configuration class for the sentiment analysis dashboard"""
    
    # Database configuration
    DATABASE_PATH = "sentiment_data.db"
    
    # Twitter API configuration
    TWITTER_API_VERSION = "2"
    MAX_TWEETS_PER_REQUEST = 100
    DEFAULT_TWEET_COUNT = 100
    
    # Sentiment analysis configuration
    SENTIMENT_THRESHOLDS = {
        'positive': 0.1,
        'negative': -0.1
    }
    
    # Visualization configuration
    CHART_COLORS = {
        'positive': '#10B981',
        'negative': '#EF4444',
        'neutral': '#6B7280'
    }
    
    # Word cloud configuration
    WORDCLOUD_CONFIG = {
        'width': 400,
        'height': 300,
        'background_color': 'white',
        'max_words': 100,
        'colormap': {
            'positive': 'Greens',
            'negative': 'Reds',
            'neutral': 'Blues'
        }
    }
    
    # Streamlit configuration
    STREAMLIT_CONFIG = {
        'page_title': 'Social Media Sentiment Analysis Dashboard',
        'page_icon': '📊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Languages supported
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    # Custom positive and negative words for sentiment analysis
    POSITIVE_WORDS = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'awesome', 'brilliant', 'outstanding', 'superb', 'perfect', 'love',
        'like', 'best', 'happy', 'pleased', 'satisfied', 'delighted',
        'incredible', 'marvelous', 'spectacular', 'phenomenal', 'terrific',
        'fabulous', 'splendid', 'magnificent', 'exceptional', 'remarkable'
    ]
    
    NEGATIVE_WORDS = [
        'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst',
        'hate', 'dislike', 'poor', 'sad', 'angry', 'frustrated', 'upset',
        'disgusted', 'annoyed', 'irritated', 'furious', 'disgusting',
        'dreadful', 'appalling', 'atrocious', 'abysmal', 'pathetic',
        'ridiculous', 'useless', 'worthless', 'hopeless', 'miserable'
    ]
    
    # Stop words for text cleaning
    STOP_WORDS = [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    ]
    
    @classmethod
    def get_database_path(cls) -> str:
        """Get the database path"""
        return cls.DATABASE_PATH
    
    @classmethod
    def get_sentiment_threshold(cls, sentiment_type: str) -> float:
        """Get sentiment threshold for given type"""
        return cls.SENTIMENT_THRESHOLDS.get(sentiment_type, 0.0)
    
    @classmethod
    def get_chart_color(cls, sentiment: str) -> str:
        """Get color for sentiment visualization"""
        return cls.CHART_COLORS.get(sentiment.lower(), '#6B7280')