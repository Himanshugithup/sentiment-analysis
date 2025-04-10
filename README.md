# sentiment-analysis

Using VADAR AND RoBERTA Model

Amazon Product Review Sentiment Analysis
Overview
This project analyzes sentiment in Amazon product reviews using VADER (Valence Aware Dictionary and sEntiment Reasoner) and the RoBERTA model. The goal is to compare the performance of lexicon-based and transformer-based sentiment analysis methods.

Features
VADER Sentiment Analysis: Uses SentimentIntensityAnalyzer from NLTK to analyze sentiment scores.
RoBERTA Model: Utilizes the pre-trained RoBERTA model for more advanced sentiment classification.
Data Visualization: Generates insights through matplotlib and seaborn.
Preprocessing: Cleans and prepares Amazon product review data for sentiment analysis.
Installation
pip install numpy pandas matplotlib seaborn nltk transformers torch scipy tqdm
Usage
Load the dataset:
import pandas as pd
df = pd.read_csv('Reviews.csv')
Perform sentiment analysis using VADER:
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['vader_score'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
Use RoBERTA for sentiment classification:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
Results
VADER: Fast and works well for short text but may struggle with complex sentiment.
RoBERTA: More accurate for nuanced sentiment but requires more computation.
License
This project is open-source under the MIT License.

Acknowledgments
NLTK VADER
Hugging Face Transformers
