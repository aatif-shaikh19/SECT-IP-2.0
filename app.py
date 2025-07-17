# Social Media Sentiment Analysis Dashboard
# Compatible with Python 3.13.5

import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import datetime
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import base64
from io import BytesIO
import requests
from PIL import Image
import json
import sqlite3
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    
    .tweet-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1DA1F2;
    }
</style>
""", unsafe_allow_html=True)

class TwitterAPI:
    def __init__(self, bearer_token: str):
        """Initialize Twitter API client"""
        self.bearer_token = bearer_token
        self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    
    def search_tweets(self, query: str, count: int = 100, lang: str = 'en') -> List[Dict]:
        """Search for tweets with given query"""
        try:
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                max_results=min(count, 100),
                limit=max(1, count // 100)
            ).flatten(limit=count)
            
            tweet_data = []
            for tweet in tweets:
                tweet_info = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count']
                }
                tweet_data.append(tweet_info)
            
            return tweet_data
        except Exception as e:
            st.error(f"Error fetching tweets: {str(e)}")
            return []

class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'perfect', 'love',
            'like', 'best', 'happy', 'pleased', 'satisfied', 'delighted'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst',
            'hate', 'dislike', 'poor', 'sad', 'angry', 'frustrated', 'upset',
            'disgusted', 'annoyed', 'irritated', 'furious', 'disgusting'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower().strip()
        return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob and custom rules"""
        cleaned_text = self.clean_text(text)
        
        # TextBlob analysis
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Custom keyword analysis
        words = cleaned_text.split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Combined sentiment score
        keyword_score = (positive_count - negative_count) / max(len(words), 1)
        combined_score = (polarity + keyword_score) / 2
        
        # Determine sentiment label
        if combined_score > 0.1:
            sentiment = 'Positive'
        elif combined_score < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'combined_score': combined_score,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        results = []
        progress_bar = st.progress(0)
        
        for i, text in enumerate(texts):
            result = self.analyze_sentiment(text)
            results.append(result)
            progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()
        return results

class DataStorage:
    def __init__(self, db_path: str = "sentiment_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                text TEXT,
                created_at TEXT,
                author_id TEXT,
                retweet_count INTEGER,
                like_count INTEGER,
                reply_count INTEGER,
                quote_count INTEGER,
                sentiment TEXT,
                polarity REAL,
                subjectivity REAL,
                combined_score REAL,
                query TEXT,
                analyzed_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_tweets(self, tweets_data: List[Dict], query: str):
        """Save tweets to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for tweet in tweets_data:
            cursor.execute('''
                INSERT OR REPLACE INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tweet['id'], tweet['text'], str(tweet['created_at']),
                tweet['author_id'], tweet['retweet_count'], tweet['like_count'],
                tweet['reply_count'], tweet['quote_count'], tweet['sentiment'],
                tweet['polarity'], tweet['subjectivity'], tweet['combined_score'],
                query, str(datetime.datetime.now())
            ))
        
        conn.commit()
        conn.close()
    
    def load_tweets(self, query: str = None) -> pd.DataFrame:
        """Load tweets from database"""
        conn = sqlite3.connect(self.db_path)
        
        if query:
            df = pd.read_sql_query(
                "SELECT * FROM tweets WHERE query = ? ORDER BY created_at DESC",
                conn, params=(query,)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM tweets ORDER BY created_at DESC",
                conn
            )
        
        conn.close()
        return df

class Dashboard:
    def __init__(self):
        self.twitter_api = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_storage = DataStorage()
    
    def setup_api(self, bearer_token: str):
        """Setup Twitter API"""
        self.twitter_api = TwitterAPI(bearer_token)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Social Media Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("### Real-time Twitter sentiment analysis and visualization")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🔧 Configuration")
        
        # API Configuration
        st.sidebar.subheader("Twitter API Settings")
        bearer_token = st.sidebar.text_input(
            "Bearer Token",
            type="password",
            help="Enter your Twitter API Bearer Token"
        )
        
        if bearer_token:
            self.setup_api(bearer_token)
            st.sidebar.success("✅ API Connected")
        
        # Search Parameters
        st.sidebar.subheader("Search Parameters")
        search_query = st.sidebar.text_input(
            "Search Query",
            value="python programming",
            help="Enter keywords to search for"
        )
        
        tweet_count = st.sidebar.slider(
            "Number of Tweets",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
        
        language = st.sidebar.selectbox(
            "Language",
            options=['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko'],
            index=0
        )
        
        # Analysis Options
        st.sidebar.subheader("Analysis Options")
        real_time = st.sidebar.checkbox("Real-time Analysis", value=True)
        save_data = st.sidebar.checkbox("Save to Database", value=True)
        
        return {
            'bearer_token': bearer_token,
            'search_query': search_query,
            'tweet_count': tweet_count,
            'language': language,
            'real_time': real_time,
            'save_data': save_data
        }
    
    def render_metrics(self, df: pd.DataFrame):
        """Render key metrics"""
        if df.empty:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tweets = len(df)
            st.markdown(f'''
                <div class="metric-container">
                    <h3>📝 Total Tweets</h3>
                    <h1>{total_tweets}</h1>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            positive_count = len(df[df['sentiment'] == 'Positive'])
            positive_pct = (positive_count / total_tweets) * 100 if total_tweets > 0 else 0
            st.markdown(f'''
                <div class="metric-container sentiment-positive">
                    <h3>😊 Positive</h3>
                    <h1>{positive_count}</h1>
                    <p>{positive_pct:.1f}%</p>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            negative_count = len(df[df['sentiment'] == 'Negative'])
            negative_pct = (negative_count / total_tweets) * 100 if total_tweets > 0 else 0
            st.markdown(f'''
                <div class="metric-container sentiment-negative">
                    <h3>😞 Negative</h3>
                    <h1>{negative_count}</h1>
                    <p>{negative_pct:.1f}%</p>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            neutral_count = len(df[df['sentiment'] == 'Neutral'])
            neutral_pct = (neutral_count / total_tweets) * 100 if total_tweets > 0 else 0
            st.markdown(f'''
                <div class="metric-container sentiment-neutral">
                    <h3>😐 Neutral</h3>
                    <h1>{neutral_count}</h1>
                    <p>{neutral_pct:.1f}%</p>
                </div>
            ''', unsafe_allow_html=True)
    
    def render_visualizations(self, df: pd.DataFrame):
        """Render data visualizations"""
        if df.empty:
            st.warning("No data available for visualization")
            return
        
        # Sentiment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color_discrete_map={
                    'Positive': '#10B981',
                    'Negative': '#EF4444',
                    'Neutral': '#6B7280'
                },
                title="Sentiment Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Sentiment Over Time")
            df['created_at'] = pd.to_datetime(df['created_at'])
            df_time = df.groupby([df['created_at'].dt.hour, 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(
                df_time,
                x='created_at',
                y='count',
                color='sentiment',
                title="Sentiment Trends by Hour",
                color_discrete_map={
                    'Positive': '#10B981',
                    'Negative': '#EF4444',
                    'Neutral': '#6B7280'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Polarity and Subjectivity Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Polarity Distribution")
            fig = px.histogram(
                df,
                x='polarity',
                nbins=30,
                title="Polarity Distribution",
                labels={'polarity': 'Polarity Score', 'count': 'Frequency'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Subjectivity vs Polarity")
            fig = px.scatter(
                df,
                x='polarity',
                y='subjectivity',
                color='sentiment',
                title="Subjectivity vs Polarity",
                color_discrete_map={
                    'Positive': '#10B981',
                    'Negative': '#EF4444',
                    'Neutral': '#6B7280'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Engagement Analysis
        st.subheader("💬 Engagement vs Sentiment")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Likes vs Sentiment', 'Retweets vs Sentiment', 
                          'Replies vs Sentiment', 'Quotes vs Sentiment')
        )
        
        engagement_metrics = ['like_count', 'retweet_count', 'reply_count', 'quote_count']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(engagement_metrics, positions):
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                sentiment_data = df[df['sentiment'] == sentiment]
                fig.add_trace(
                    go.Box(
                        y=sentiment_data[metric],
                        name=sentiment,
                        legendgroup=sentiment,
                        showlegend=(pos == (1, 1))
                    ),
                    row=pos[0], col=pos[1]
                )
        
        fig.update_layout(height=600, title_text="Engagement Metrics by Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_word_cloud(self, df: pd.DataFrame):
        """Render word cloud"""
        if df.empty:
            return
        
        st.subheader("☁️ Word Cloud")
        
        col1, col2, col3 = st.columns(3)
        
        sentiments = ['Positive', 'Negative', 'Neutral']
        colors = ['Greens', 'Reds', 'Blues']
        
        for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
            with [col1, col2, col3][i]:
                sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['text'].astype(str))
                
                if sentiment_text.strip():
                    # Clean text for word cloud
                    cleaned_text = self.sentiment_analyzer.clean_text(sentiment_text)
                    
                    wordcloud = WordCloud(
                        width=400,
                        height=300,
                        background_color='white',
                        colormap=color,
                        max_words=100
                    ).generate(cleaned_text)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'{sentiment} Sentiment', fontsize=14, fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close()
    
    def render_tweet_samples(self, df: pd.DataFrame):
        """Render sample tweets"""
        if df.empty:
            return
        
        st.subheader("📝 Sample Tweets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 😊 Positive Tweets")
            positive_tweets = df[df['sentiment'] == 'Positive'].head(3)
            for _, tweet in positive_tweets.iterrows():
                st.markdown(f'''
                    <div class="tweet-container">
                        <p>{tweet['text'][:150]}...</p>
                        <small>👍 {tweet['like_count']} | 🔄 {tweet['retweet_count']} | 💬 {tweet['reply_count']}</small>
                    </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 😞 Negative Tweets")
            negative_tweets = df[df['sentiment'] == 'Negative'].head(3)
            for _, tweet in negative_tweets.iterrows():
                st.markdown(f'''
                    <div class="tweet-container">
                        <p>{tweet['text'][:150]}...</p>
                        <small>👍 {tweet['like_count']} | 🔄 {tweet['retweet_count']} | 💬 {tweet['reply_count']}</small>
                    </div>
                ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown("##### 😐 Neutral Tweets")
            neutral_tweets = df[df['sentiment'] == 'Neutral'].head(3)
            for _, tweet in neutral_tweets.iterrows():
                st.markdown(f'''
                    <div class="tweet-container">
                        <p>{tweet['text'][:150]}...</p>
                        <small>👍 {tweet['like_count']} | 🔄 {tweet['retweet_count']} | 💬 {tweet['reply_count']}</small>
                    </div>
                ''', unsafe_allow_html=True)
    
    def render_data_table(self, df: pd.DataFrame):
        """Render data table"""
        if df.empty:
            return
        
        st.subheader("📋 Detailed Data")
        
        # Display controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=['All', 'Positive', 'Negative', 'Neutral']
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=['created_at', 'like_count', 'retweet_count', 'polarity']
            )
        
        with col3:
            sort_order = st.selectbox("Order", options=['Descending', 'Ascending'])
        
        # Apply filters
        filtered_df = df.copy()
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
        
        # Sort data
        ascending = sort_order == 'Ascending'
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
        
        # Display table
        display_columns = ['text', 'sentiment', 'polarity', 'subjectivity', 
                         'like_count', 'retweet_count', 'reply_count', 'created_at']
        
        st.dataframe(
            filtered_df[display_columns].head(100),
            use_container_width=True,
            height=400
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export to JSON"):
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"sentiment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Sidebar configuration
        config = self.render_sidebar()
        
        # Main content
        if not config['bearer_token']:
            st.warning("⚠️ Please enter your Twitter API Bearer Token in the sidebar to get started.")
            st.info("""
            ### How to get Twitter API Bearer Token:
            1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
            2. Create a new app or use an existing one
            3. Navigate to the "Keys and Tokens" section
            4. Generate or copy your Bearer Token
            5. Paste it in the sidebar
            """)
            return
        
        # Search and analyze tweets
        if st.button("🔍 Search & Analyze", type="primary"):
            with st.spinner("Fetching tweets..."):
                tweets = self.twitter_api.search_tweets(
                    query=config['search_query'],
                    count=config['tweet_count'],
                    lang=config['language']
                )
            
            if tweets:
                with st.spinner("Analyzing sentiment..."):
                    # Analyze sentiment
                    tweet_texts = [tweet['text'] for tweet in tweets]
                    sentiment_results = self.sentiment_analyzer.batch_analyze(tweet_texts)
                    
                    # Combine data
                    for i, result in enumerate(sentiment_results):
                        tweets[i].update(result)
                    
                    # Create DataFrame
                    df = pd.DataFrame(tweets)
                    
                    # Save to session state
                    st.session_state.df = df
                    st.session_state.search_query = config['search_query']
                    
                    # Save to database if enabled
                    if config['save_data']:
                        self.data_storage.save_tweets(tweets, config['search_query'])
                    
                    st.success(f"✅ Analyzed {len(tweets)} tweets successfully!")
            else:
                st.error("❌ No tweets found for the given query.")
        
        # Load previous data button
        if st.button("📂 Load Previous Data"):
            df = self.data_storage.load_tweets()
            if not df.empty:
                st.session_state.df = df
                st.session_state.search_query = "Previous Data"
                st.success(f"✅ Loaded {len(df)} previous tweets!")
            else:
                st.info("No previous data found.")
        
        # Display results if available
        if 'df' in st.session_state and not st.session_state.df.empty:
            df = st.session_state.df
            search_query = st.session_state.get('search_query', 'Unknown')
            
            st.markdown(f"### Results for: '{search_query}'")
            
            # Render all sections
            self.render_metrics(df)
            self.render_visualizations(df)
            self.render_word_cloud(df)
            self.render_tweet_samples(df)
            self.render_data_table(df)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Built with ❤️ using Streamlit | Social Media Sentiment Analysis Dashboard</p>
            <p>For internship project purposes | Python 3.13.5 compatible</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()