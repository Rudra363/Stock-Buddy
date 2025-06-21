import csv
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import yfinance as yf
import pandas as pd
import json  # or use str(list) if you prefer


model = SentenceTransformer('all-MiniLM-L6-v2')

def openFile():
    with open("analyst_ratings_processed.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            news = row[1]
            time = row[2]
            stock_ticker = row[3]

            print(news, time)

            news_embedding = model.encode(news)
            price = price_change(stock_ticker, time[0:10])

            makeFile(news_embedding, price)

def makeFile(embedded_title, price_change):
    with open("embeddedNews.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        embedding_list = [float(x) for x in embedded_title]
        embedding_str = json.dumps(embedding_list)
        writer.writerow([embedding_str, price_change])

def price_change(ticker, date_str):
    date = pd.to_datetime(date_str)
    start_date = "1900-01-01"  # early enough to ensure data is retrieved
    end_date = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # include target date

    # Fetch historical data up to day after target date
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date)

    if hist.empty:
        return None, None

    # Get high of the specified day
    day_data = hist[hist.index.date == date.date()]
    high_of_day = day_data.iloc[0]["High"] if not day_data.empty else None

    # Get previous trading day's close
    prev_day_data = hist[hist.index.date < date.date()]
    prev_close = prev_day_data.iloc[-1]["Close"] if not prev_day_data.empty else None

    return high_of_day - prev_close

# openFile()

