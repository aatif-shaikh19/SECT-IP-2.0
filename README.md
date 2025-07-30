# SECT-IP 2.0  in collaboration with Civora Nexus

# 📊 Social Media Sentiment Analysis Dashboard

A powerful and interactive dashboard for real-time **Twitter sentiment analysis**, built with **Streamlit** and **Plotly**. This app fetches tweets using the Twitter API, analyzes their sentiments, and visualizes insights in a clean, customizable interface.

## 🚀 Features

- 🔍 Real-time Twitter data fetching
- 🤖 Sentiment analysis using TextBlob + custom keyword heuristics
- 📈 Beautiful, interactive charts (Pie, Line, Histogram, Box Plots)
- ☁️ Word clouds for positive, negative, and neutral tweets
- 🐦 View and filter real tweet samples with engagement stats
- 💾 Store data in SQLite and reload anytime
- 📤 Export results to CSV and JSON
- 🎨 Stylish and responsive UI (Streamlit + optional HTML React template)

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Plotly, Matplotlib, HTML/CSS
- **Backend:** Python 3.13+, Tweepy (Twitter API v2), TextBlob, SQLite
- **Visualization:** Plotly, Seaborn, WordCloud

## 📁 Project Structure

├── app.py               # Main Streamlit app
├── config.py            # Configurations and constants
├── launcher.py          # Environment checker + app launcher
├── index.html           # Optional UI template (React-based)
├── run.bat              # Windows batch file to launch app
├── requirements.txt     # All Python dependencies

## 🔧 Installation

1. **Clone the repository**:
   
   git clone https://github.com/yourusername/sentiment-dashboard.git
   cd sentiment-dashboard

2. **Install Python packages**:

   
   pip install -r requirements.txt

3. **Ensure Python ≥ 3.8** is installed.

## ▶️ Running the App

### Option 1: Launch via Python script (recommended)

python launcher.py

### Option 2: Directly with Streamlit
streamlit run app.py

The app will open in your default browser on [http://localhost:8501](http://localhost:8501)

## 🔐 Twitter API Setup
To fetch tweets, you’ll need a **Bearer Token** from Twitter:

1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new App (or use an existing one)
3. Go to **Keys and Tokens**
4. Copy your **Bearer Token**
5. Paste it into the sidebar of the dashboard when prompted
## 🧪 How to Use

* Enter a **search query** (e.g., `Python`, `AI`, `Elections`)
* Choose number of tweets and language
* Click **Search & Analyze**
* View sentiment distribution, tweet samples, engagement charts, and more
* Export data as CSV or JSON

## 📌 Notes

* Python 3.13+ supported
* SQLite DB stored as `sentiment_data.db`
* Session data is saved using Streamlit's `st.session_state`
* Customizable sentiment keyword lists and thresholds in `config.py`
* Includes dark/light styling and responsiveness

## ✨ Credits

Developed with ❤️ as part of an **internship project**
Thanks to the open-source community for Streamlit, Plotly, and Twitter API tools.

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.
