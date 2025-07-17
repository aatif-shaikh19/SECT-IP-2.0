# SECT-IP 2.0  in collaboration with Civora Nexus
📊 Social Media Sentiment Analysis Dashboard

A comprehensive real-time sentiment analysis dashboard for social media data with advanced visualization and analytics capabilities.

🌟 Overview
This project is a full-featured Social Media Sentiment Analysis Dashboard built for internship and educational purposes. It provides real-time sentiment analysis of Twitter data with beautiful visualizations, comprehensive analytics, and an intuitive user interface.
🎯 Key Features

🔍 Real-time Data Collection: Fetch tweets using Twitter API v2
🧠 Advanced Sentiment Analysis: Hybrid approach combining TextBlob and custom algorithms
📊 Interactive Visualizations: Dynamic charts, graphs, and word clouds
💾 Data Persistence: SQLite database for storing and retrieving analysis results
📱 Responsive Design: Modern, mobile-friendly interface
📥 Export Capabilities: CSV and JSON export functionality
🎨 Professional UI: Custom CSS styling with dark/light themes
⚡ Real-time Updates: Live sentiment tracking and analysis


🛠️ Technology Stack
Core Technologies

Python 3.13.5: Main programming language
Streamlit: Web framework for the dashboard
Tweepy: Twitter API integration
TextBlob: Natural language processing for sentiment analysis
Plotly: Interactive data visualizations
SQLite: Database for data persistence

Libraries & Dependencies
streamlit==1.29.0      # Web framework
tweepy==4.14.0         # Twitter API client
pandas==2.1.4          # Data manipulation
numpy==1.24.3          # Numerical computing
plotly==5.17.0         # Interactive visualizations
textblob==0.17.1       # NLP and sentiment analysis
wordcloud==1.9.2       # Word cloud generation
matplotlib==3.7.2      # Plotting library
seaborn==0.12.2        # Statistical visualization
Pillow==10.1.0         # Image processing
requests==2.31.0       # HTTP requests



📁 Project Structure
sentiment_dashboard/
├── 📄 app.py                    # Main application file
├── 📄 config.py                 # Configuration settings
├── 📄 launcher.py               # Application launcher
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 .gitignore               # Git ignore rules
├── 📄 run.bat                   # Windows launcher script
├── 📄 run.sh                    # Linux/Mac launcher script
├── 📁 data/                     # Data storage directory
├── 📁 exports/                  # Export files directory
├── 📁 sentiment_env/            # Virtual environment
└── 📄 sentiment_data.db         # SQLite database (auto-created)

