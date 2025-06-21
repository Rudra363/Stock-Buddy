import csv
from sentence_transformers import SentenceTransformer
import yfinance as yf
import pandas as pd
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def openFile():
    data_to_write = []

    with open("analyst_ratings_processed.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)
        count = 0

        for row in reader:
            news = row[1]
            time = row[2]
            stock_ticker = row[3]

            print(news, time)

            news_embedding = model.encode(news)
            price = price_change(stock_ticker, time[0:10])

            if price is not None:
                embedding_list = [float(x) for x in news_embedding]
                embedding_str = json.dumps(embedding_list)
                data_to_write.append([embedding_str, price])

            count += 1
            if count == 5:
                break

    makeFile(data_to_write)

def makeFile(data_rows):
    with open("embeddedNews.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["embedding", "price_change"])  # optional header
        writer.writerows(data_rows)

def price_change(ticker, date_str):
    date = pd.to_datetime(date_str)
    end_date = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    hist = yf.Ticker(ticker).history(start="1900-01-01", end=end_date)

    if hist.empty:
        return None

    day_data = hist[hist.index.date == date.date()]
    high_of_day = day_data.iloc[0]["High"] if not day_data.empty else None

    prev_day_data = hist[hist.index.date < date.date()]
    prev_close = prev_day_data.iloc[-1]["Close"] if not prev_day_data.empty else None

    if high_of_day is None or prev_close is None:
        return None

    return high_of_day - prev_close

# Run it
openFile()
