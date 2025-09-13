import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

stock_results = {}

def download_stock_data(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")
    data.index = pd.to_datetime(data.index)
    return data

def process_stock(data, stock):
    df = data.copy()

    if 'Dividends' in df.columns:
        del df["Dividends"]
    if 'Stock Splits' in df.columns:
        del df["Stock Splits"]

    df["Next_Week_Close"] = df["Close"].shift(-5)
    df["Target"] = (df["Next_Week_Close"] > df["Close"]).astype(int)
    df = df.loc["2003-01-01":].copy()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    predictors = ["Close", "Volume", "Open", "High", "Low", "SMA_50", "MACD"]

    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_avg = df["Close"].rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg
        df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()

    df = df.resample('W-FRI').last()  

    df = df.dropna(subset=df.columns)
    return df, predictors + [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]

def train_and_predict(df, predictors):
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    def backtest(data, model, predictors, start=250, step=50): 
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[:i]
            test = data.iloc[i:i + step]
            model.fit(train[predictors], train["Target"])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index, name="Predictions")
            combined = pd.concat([test["Target"], preds], axis=1)
            all_predictions.append(combined)
        return pd.concat(all_predictions)

    predictions_df = backtest(df, model, predictors)
    return model, predictions_df

def run(stock):
    symbol = stock.getSymbol()
    df, predictors = process_stock(download_stock_data(symbol), stock)

    if symbol in stock_results and stock_results[symbol]['latest_date'] == df.index[-1]:
        return stock_results[symbol]['Prediction'] == 'increase'

    print(f"Processing {symbol}...")
    data = download_stock_data(symbol)
    df, predictors_with_extra = process_stock(data, stock)
    model, predictions = train_and_predict(df, predictors_with_extra)

    latest_data = df.iloc[-1:][predictors_with_extra]

    prediction = model.predict(latest_data)
    predicted_increase = bool(prediction[0])

    stock_results[stock.getSymbol()] = {
        'Prediction': 'increase' if predicted_increase else 'no_increase',
        'latest_date': df.index[-1],
        'latest_price': df['Close'].iloc[-1]
    }

    return predicted_increase

