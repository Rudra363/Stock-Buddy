import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Dictionary to store results
stock_results = {}

def download_stock_data(symbol):
    # Download using yfinance
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")
    data.index = pd.to_datetime(data.index)
    return data

def process_stock(data, stock):
    # Implement the entire pipeline: cleaning, feature engineering, model training, prediction
    df = data.copy()

    # Basic data cleaning
    if 'Dividends' in df.columns:
        del df["Dividends"]
    if 'Stock Splits' in df.columns:
        del df["Stock Splits"]

    df["Next_Week_Close"] = df["Close"].shift(-5)
    df["Target"] = (df["Next_Week_Close"] > df["Close"]).astype(int)
    df = df.loc["2003-01-01":].copy()

    # Add your custom indicators here
    df["SMA_50"] = df["Close"].rolling(window=50).mean() # 50-day Simple Moving Average
    # df["RSI_14"] = getrsi(stock)  # 14-day RSI

    # Moving Average Convergence Divergence (MACD)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Define the base predictors
    predictors = ["Close", "Volume", "Open", "High", "Low", "SMA_50", "MACD"]

    # Generate additional predictors as needed
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_avg = df["Close"].rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg
        df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()

    # Resample the data to weekly frequency (end of week)
    df = df.resample('W-FRI').last()  # Use 'W-FRI' to resample to the end of the week (Friday)

    df = df.dropna(subset=df.columns)
    return df, predictors + [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]

def train_and_predict(df, predictors):
    # Initialize the model
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    # Backtesting for demonstration; for real prediction, consider a different approach
    def backtest(data, model, predictors, start=250, step=50):  # Adjusted for weekly data
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
        return stock_results[symbol]

    print(f"Processing {symbol}...")
    data = download_stock_data(symbol)
    df, predictors_with_extra = process_stock(data, stock)
    model, predictions = train_and_predict(df, predictors_with_extra)

    # Get the latest data for prediction
    latest_data = df.iloc[-1:][predictors_with_extra]

    # Predict for next week
    prediction = model.predict(latest_data)
    predicted_increase = bool(prediction[0])
    print(predictions)
    print(f"Prediction for {symbol}: {'Increase' if predicted_increase else 'No increase'}")
    print(precision_score(predictions["Target"], predictions["Predictions"]))

    # Store result
    stock_results[stock.getSymbol()] = {
        'Prediction': 'increase' if predicted_increase else 'no_increase',
        'latest_date': df.index[-1],
        'latest_price': df['Close'].iloc[-1]
    }

    return predicted_increase

# --------------


# import yfinance as yf
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score
#
# #TODO: Things to work on if or when needed
# #   - Big O is logarithmic making run time long
# #   - line 34 copies data from 1990 onwards, turn it into a ratio instead like maybe : 50% of data if company older than X years and all data if 10 year or less
# #   - Increase precision score - make it more accurate - add more indicators
#
# # Dictionary to store results
# stock_results = {}
#
#
# def download_stock_data(symbol):
#     # Download using yfinance
#     ticker = yf.Ticker(symbol)
#     data = ticker.history(period="max")
#     data.index = pd.to_datetime(data.index)
#     return data
#
# def process_stock(data):
#     # Implement the entire pipeline: cleaning, feature engineering, model training, prediction
#     df = data.copy()
#
#     # Basic data cleaning
#     if 'Dividends' in df.columns:
#         del df["Dividends"]
#     if 'Stock Splits' in df.columns:
#         del df["Stock Splits"]
#
#
#     df["Tomorrow"] = df["Close"].shift(-1)
#     df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
#     df = df.loc["1990-01-01":].copy()
#
#     predictors = ["Close", "Volume", "Open", "High", "Low"]
#
#     # Generate additional predictors as needed
#     horizons = [2, 5, 60, 250, 1000]
#     for horizon in horizons:
#         rolling_avg = df["Close"].rolling(horizon).mean()
#         df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg
#         df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()
#
#     df = df.dropna(subset=df.columns)
#     return df, predictors + [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]
#
#
# def train_and_predict(df, predictors):
#     # Initialize the model
#     model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
#
#     # Backtesting for demonstration; for real prediction, consider a different approach
#     def backtest(data, model, predictors, start=2500, step=250):
#         all_predictions = []
#         for i in range(start, data.shape[0], step):
#             train = data.iloc[:i]
#             test = data.iloc[i:i + step]
#             model.fit(train[predictors], train["Target"])
#             preds = model.predict(test[predictors])
#             preds = pd.Series(preds, index=test.index, name="Predictions")
#             combined = pd.concat([test["Target"], preds], axis=1)
#             all_predictions.append(combined)
#         return pd.concat(all_predictions)
#
#     predictions_df = backtest(df, model, predictors)
#     return model, predictions_df
#
#
# def run(stock):
#     symbol = stock.getSymbol()
#     df, predictors = process_stock(download_stock_data(symbol))
#
#
#     if symbol in stock_results and stock_results[symbol]['latest_date'] == df.index[-1]:
#         return stock_results[symbol]
#
#
#     print(f"Processing {symbol}...")
#     data = download_stock_data(symbol)
#     df, predictors_with_extra = process_stock(data)
#     model, predictions = train_and_predict(df, predictors_with_extra)
#
#         # Get the latest data for prediction
#     latest_data = df.iloc[-1:][predictors_with_extra]
#
#         # Compute additional predictors for latest data
#     for horizon in [2, 5, 60, 250, 1000]:
#         rolling_avg = df["Close"].rolling(horizon).mean().iloc[-1]
#         latest_data[f"Close_Ratio_{horizon}"] = (
#             latest_data["Close"].iloc[0] / rolling_avg if rolling_avg != 0 else 0
#         )
#         trend_value = df["Target"].shift(1).rolling(horizon).sum().iloc[-1]
#         latest_data[f"Trend_{horizon}"] = trend_value
#
#     latest_data = latest_data.reindex(columns=predictors_with_extra, fill_value=0)
#
#         # Predict for next day
#     prediction = model.predict(latest_data)
#     predicted_increase = bool(prediction[0])
#     print(predictions)
#     print(f"Prediction for {symbol}: {'Increase' if predicted_increase else 'No increase'}")
#     print(precision_score(predictions["Target"], predictions["Predictions"]))
#
#
#         # Store result
#     stock_results[stock.getSymbol()] = {
#         'Prediction': 'increase' if predicted_increase else 'no_increase',
#         'latest_date': df.index[-1],
#         'latest_price': df['Close'].iloc[-1]
#     }
#
#     return predicted_increase