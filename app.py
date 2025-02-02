import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Set up NLTK data path for Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download vader_lexicon at runtime
nltk.download('vader_lexicon', download_dir=nltk_data_path)

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title='Sentiment Analysis', layout='wide')
st.sidebar.title('Visualization')
st.sidebar.write('The graph below shows sentiment distribution.')

# Load processed data
data_clean = pd.read_csv('processed_data.csv')
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(x='sentiment', data=data_clean, palette='bright', ax=ax)
ax.set_title('Sentiment Distribution')
st.sidebar.pyplot(fig)

st.title('Sentiment Analysis App')
st.write('Type your review below and the app will analyze its sentiment.')

review = st.text_area('Enter your review:', on_change=None, placeholder='Type here and press Enter')
if review:
    score = sia.polarity_scores(review)['compound']
    if score > 0:
        sentiment = 'Positive'
    elif score < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    st.write(f'The sentiment of the review is: {sentiment}')

    # Visualize the sentiment scores
    labels = ['Positive', 'Negative', 'Neutral']
    scores = [score if sentiment == 'Positive' else 0,
              -score if sentiment == 'Negative' else 0,
              1 - abs(score) if sentiment == 'Neutral' else 0]
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(x=labels, y=scores, palette='bright')
    ax.set_title('Sentiment Score Breakdown')
    st.pyplot(fig)
