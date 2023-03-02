import pandas as pd
import requests
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from bs4 import BeautifulSoup
# from textblob import TextBlob

# Set up API parameters
api_key = "7b13b4dd346749e3b5d59d083d2ccb6e"
query = "Bitcoin"
to_date = datetime.today().date()
from_date = to_date - timedelta(days=28)
url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=popularity&apiKey={api_key}"

newsapi_response = NewsApiClient(api_key=api_key)
data = newsapi_response.get_everything(q=query, language="en", from_param=from_date, sort_by="relevancy")['articles']
# data = newsapi_response.get_top_headlines(q=query, language="en")

# Extract relevant data and preprocess text
articles = []
for article in data:
    # print(article)
    contents = requests.get(article['url'])
    # Use BeautifulSoup to parse the HTML content of the response
    soup = BeautifulSoup(contents.content, 'html.parser')

    articles.append({'title': article['title'],
                     'author': article['author'],
                     'source': article['source'],
                     'date': article['publishedAt'],
                     'url': article['url'],
                     'soup': soup,
                     'soup_p': soup.find_all('p')})

df = pd.DataFrame(articles)

# print(df.keys())
# print(f"df title: {df.iloc[0]['title']}")

for i in df.iloc[0]['soup'].descendants:
    print(f"df soup: {i}")

# print(f"soup title: {soup.title.string}")
# for i in soup.find_all('p'):
#     if i.string is not None:
#         print(f"{i.string}")

# print(f"soup title: {soup.title.string}")
# Extract the title of the article
# title = soup.find('h1', class_='article-title').text.strip()
#
# # Extract the publication date of the article
# date = soup.find('span', class_='article-date').text.strip()
#
# # Extract the author of the article
# author = soup.find('span', class_='article-author').text.strip()
#
# # Extract the body text of the article
# body = soup.find('div', class_='article-body').text.strip()

# Print the extracted information
# print("Title:", title)
# print("Date:", date)
# print("Author:", author)
# print("Body:", body)

# df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# df['text'] = df['text'].str.replace('[^\w\s]', '')
#
# Perform sentiment analysis
# df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
#
# Save data to CSV
# df.to_csv("sentiment_data.csv", index=False)
