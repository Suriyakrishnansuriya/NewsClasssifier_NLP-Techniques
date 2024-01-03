# Import necessary libraries
import pandas as pd
import re
import numpy as np
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# 1. Web Scraping:
def scrape_news(url):
  response= requests.get(url)
  soup= BeautifulSoup(response.text, 'html.parser')
  title = soup.title.text
  content = ' '.join([p.text for p in soup.find_all('p')])
  return title, content


# news websites
websites = ['https://www.thehindu.com/news/cities/chennai/two-boys-drown-in-lake-in-chennai-selaiyur/article67691160.ece',
            'https://www.thehindu.com/news/cities/chennai/chennai-police-to-deploy-18000-personnel-to-ensure-incident-free-new-year-celebrations/article67690499.ece',
            'https://www.thehindu.com/news/cities/chennai/setc-mtc-buses-to-begin-operations-from-kilambakkam-bus-terminus-from-sunday-says-transport-minister/article67690170.ece',
            'https://www.thehindu.com/news/cities/chennai/all-new-year-celebrations-at-hotels-resorts-must-end-before-1-am-tn-police/article67689954.ece',
            'https://www.thehindu.com/news/cities/chennai/chennai-passport-office-issues-5-lakh-passports-in-2023/article67688042.ece',
            'https://www.thehindu.com/news/cities/chennai/chennai-corporation-dmk-councillors-call-upon-government-agencies-to-coordinate-with-civic-body/article67687990.ece',
            'https://www.thehindu.com/sport/cricket/india-tour-of-south-africa-south-africa-vs-india-second-test-injury-scare-for-indian-team/article67689919.ece',
            'https://www.thehindu.com/sport/cricket/ind-vs-sa-second-test-ravindra-jadeja-likely-to-be-available-for-cape-town-test/article67686556.ece',
            'https://www.dtnext.in/news/cinema/guntur-kaaram-mahesh-babu-sree-leela-bring-electrifying-moves-in-song-kurchi-madatha-petti-757778',
            'https://www.dtnext.in/news/cinema/bigger-and-better-prashanth-neel-about-salaar-2-757806',
            'https://www.dtnext.in/news/city/bangladeshi-woman-with-forged-credentials-detained-at-chennai-airport-handed-over-to-police-757864',
            ]

# scrape news articles
data = []
for website in websites:
    news_url = website
    title, content = scrape_news(news_url)
    data.append({'title': title, 'content': content, 'website': website})

# 2. Data Cleaning and Preprocessing:
df= pd.DataFrame(data)

# Function to perform preprocessing
def preprocess_text(text):
  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  # non-alphanumeric characters
  text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
  # extra whitespaces
  text = re.sub(' +', ' ', text).strip()


  # Tokenize and remove stop words
  stop_words= set(stopwords.words('english'))
  tokens= word_tokenize(text)
  filtered_tokens= [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

  # Lemmatization
  lemmatizer= WordNetLemmatizer()
  lemmatize_tokens= [lemmatizer.lemmatize(word) for word in filtered_tokens]

  # Join the tokens back into a string
  preprocessed_text= ' '.join(lemmatize_tokens)

  return preprocess_text

# Apply preprocessing to the 'title' and 'content' columns
df['title'].apply(lambda x: preprocess_text(x))
df['content'].apply(lambda x: preprocess_text(x))

# Define a function to remove common initial substrings from a text
def preprocess_text(text):
    common_prefix = '\n\t'
    if text.startswith(common_prefix):
      return text[len(common_prefix):]
    return text

# Apply the function to both 'title' and 'content' columns
df['title'] = df['title'].apply(preprocess_text)
df['content'] = df['content'].apply(preprocess_text)

# Define a function to remove common initial substrings from a text
def preprocess_text(text):
    common_prefix2 = 'To enjoy additional benefits CONNECT WITH US'
    if text.startswith(common_prefix2):
      return text[len(common_prefix2):]
    return text

# Apply the function to both 'title' and 'content' columns
df['title'] = df['title'].apply(preprocess_text)
df['content'] = df['content'].apply(preprocess_text)

# 3. Text Representation:
# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# Optionally, convert the sparse matrix to a dense array
dense_tfidf_matrix = tfidf_matrix.toarray()

# Load pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('K:\\New\\GoogleNews-vectors-negative300.bin', binary=True)

# Convert text to Word2Vec embeddings
word2vec_matrix = df['content'].apply(lambda x: np.mean([word2vec_model[w] for w in x.split() if w in word2vec_model] or [np.zeros(word2vec_model.vector_size)], axis=0))


# 4. Topic Clustering:
# Apply K-means clustering
num_clusters = 4  # Adjust based on the topics you want to identify
kmeans = KMeans(n_clusters=num_clusters)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# 5. Topic Labeling:
# Display a sample of articles from each cluster for manual inspection
for cluster_id in range(num_clusters):
    cluster_sample = df[df['cluster'] == cluster_id].sample(5)  # Displaying 5 samples
    print(f"Cluster {cluster_id} - Sample Articles:")
    print(cluster_sample[['title', 'content']])
    print("\n")

# 6. Classification Model:

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['cluster'], test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, predictions))