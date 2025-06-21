In this project, I’ll walk you through a mini-project where I used a pre-trained BERT model to perform sentiment analysis on Flipkart product reviews — extracting them directly from the website and predicting whether each review is positive or negative.

🚀 Project Overview

We’ll:

Scrape product reviews from Flipkart

Use the NLPTown BERT model for multilingual sentiment analysis

Predict sentiment scores (1 to 5) for each review

Display results in a structured DataFrame

 🛠 Tech Stack

Python

Transformers (Hugging Face)

BeautifulSoup (for web scraping)

Regex

PyTorch

Pandas & NumPy

 📦 Installing Dependencies

pip install transformers torch beautifulsoup4 requests pandas numpy
 🧠 Step-by-Step Breakdown

 1. Load Pre-trained BERT Model

 We used Hugging Face’s nlptown/bert-base-multilingual-uncased-sentiment, which can handle text in multiple languages and outputs sentiment from 1 to 5.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
 2. Scrape Flipkart Reviews

Using requests, BeautifulSoup, and some Regex, we fetched review text from a Flipkart product page:

import requests from bs4 import BeautifulSoup import re
url = 'https://www.flipkart.com/marq-flipkart-80cm-32-inch-hd-ready-led-tv/product-reviews/itmewmcbpqhht4df?pid=TVSEWMCBXTP4PJ3J' response = requests.get(url) soup = BeautifulSoup(response.text, 'html.parser')
regex = re.compile('.*ZmyHeo.*') results = soup.findAll('div', {'class': regex}) reviews = [result.text for result in results] 
3. Create a DataFrame of Reviews

import pandas as pd import numpy as np
df = pd.DataFrame(np.array(reviews), columns=['review'])
 4. Define Sentiment Analysis Function

We tokenize the review text, pass it to the BERT model, and extract the highest-scoring sentiment label.

import torch
def sentiment_analyser(review):     tokens = tokenizer.encode(review, return_tensors="pt")     result = model(tokens)     return int(torch.argmax(result.logits)) + 1  # Output: 1 to 5
 5. Run Analysis on All Reviews

We apply the sentiment function to each review (truncated at 512 characters due to BERT’s token limit):

df['sentiment'] = df['review'].apply(lambda x: sentiment_analyser(x[:512]))
 📊 Sample Output

Review                                                                       Sentiment

“Very good product. Picture quality is excellent for the price.”                 5

“Remote stopped working in a week. Poor service by brand.”                       2

“Decent TV for a small room. Don’t expect high-end features.”                    3

💡 Final Thoughts

 This project gave me hands-on experience with:

Web scraping with Python

Using state-of-the-art pre-trained NLP models

Applying sentiment analysis in a real-world context

 It also shows how easy it is to integrate transformer models into simple scripts — without training anything from scratch!

🧠 Learnings

This project strengthened my understanding of:

BERT tokenization and output interpretation

Practical applications of PyTorch models

How to clean and analyze web-scraped data

To test the model using your own custom reviews, run the 'Sentiment score.pynb' code from repository in your environment.




 

 

