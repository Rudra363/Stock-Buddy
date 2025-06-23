from algorithms import *
from StocksClass import Stocks
from machineModel import *
from test import *
from trainModel import *
from stock_forecast import *
from dateutil.parser import parse

def main():
    #Load the trained model
   # csvTickers()

   # stock = Stocks("AAPL", 10)
    # news_items = stock.getNews()
    #
    # for item in news_items:
    #     try:
    #         title = item["content"]["title"]
    #         date = item["content"]["pubDate"]
    #         # Optional: format date to something like 'June 20, 2025'
    #         formatted_date = parse(date).strftime('%B %d, %Y %H:%M')
    #         print(f"{formatted_date} — {title}")
    #     except KeyError:
    #         continue

    # print(SMA_slope(stock, 20))
    # movingAverageConvergenceDivergence(stock)
   # print(run(stock))

    # stock = Stocks("BA", 10)
   # predicted_increase = run(stock)


    model = load_model("stock_label_model.pkl")
   #
    # List of stock symbols you want to test
    #symbols_to_test = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"]
    symbols_to_test = ["BA", "NVDA", "SBUX", "TU"]

    features_list = []
    trend_predictions = []  # New list
    newsPredict = []
    for symbol in symbols_to_test:
        try:
            stock = Stocks(symbol, 10)
            features = extractFeatures(stock)
            if features is None or len(features) == 0:
                print(f"⚠️ No features extracted for {symbol}, skipping...")
                trend_predictions.append(False)
                newsPredict.append((False, 0.0))
                continue
            features_list.append(features)

            predicted_increase = run(stock)
            #print("predicted_increase",predicted_increase)
            trend_predictions.append(predicted_increase)  # Save per stock

            news_predict = newTitle(stock)
            newsPredict.append(news_predict)


        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            trend_predictions.append(False)  # Maintain index alignment'


    if features_list:
        import numpy as np
        X_new = np.array(features_list)
        predictions = model.predict(X_new)


        for sym, pred, trend, news in zip(symbols_to_test, predictions, trend_predictions, newsPredict):
            news_bool, news_percent = news
            print(f"DEBUG: {sym} | Model: {pred} | Trend: {trend}")
            if pred == 1 and trend and news_bool:
                print(f"✅ Stock: {sym} --> Strong signal (Model=1, Trend=True), Current News Outlook:", newsPredict)
            elif pred == 1:
                print(f"➕ Stock: {sym} --> Model says buy, but trend uncertain")
            elif trend:
                print(f"⚠️ Stock: {sym} --> Trend looks good, but model disagrees")
            else:
                print(f"❌ Stock: {sym} --> No signal")

    else:
        print("No valid stock features to predict.")

    X, y = load_data("training_data.csv")
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

#def main():
    # csvTickers()
    #stock = Stocks("AAPL", 10)
    #stock = Stocks("GOOG", 10)

   #  print(stock)
   #  print(stock.getSymbol())
   #  df = simpleMovingAverage(stock)
   #  print(df)
   #
   #  # Access some attributes or methods
   #  print("Symbol:", stock.getSymbol())
   #  print("Name:", stock.getName())
   #  print("Price:", stock.getPrice())
   #  print("Total value:", stock.getValue())
   #  print("Recommendations: ", stock.getRecommendations())
   #  print("RecommendationsSummary: ", stock.getRecommendationsSummary())
   #  print(mostRecommendations(stock))
   #  print(getDividendYield(stock))
   #  print("Earnings:", stock.getEarnings())
   #  print("Growth:", getNextYearGrowth(stock))
   #  #print("Financials:", stock.getFinancials())
   #  print("Income Statement:", stock.getIncome())
   #  print("Earnings This Year:", getCurrentYearEarnings(stock))
   #  print("Earnings Last Year:", getPreviousYearsEarnings(stock))
   #  #print("Insider:", stock.getInsiders())
   #  #print("Net Insider:", getNetInsiderPurchases(stock))
   #  print(getInsiderConfidence(stock))
   #  #print(current_ratio(stock))
   #  print("Price/Earning Ratio:", get_price_earnings_ratio(stock))
   #  print("Return on Investment:", return_on_investments(stock))
   #  #print("Return on Assets:", return_on_assets(stock))
   # # print(stock.ticker.info)
   #  #print(stock.getBalanceSheet())
   #  print("Debt to Equity Ratio:", debtEquityRatio(stock))
   #  #getAssestTurnoverRatio(stock)
   #  #print("other income statement:", stock.getIncomeStatement())
   #  getAssestTurnoverRatio(stock)
   #  print("Volume:", getVolume(stock))
   #  #print(stock.getBalanceSheet())
   #  print("52 week low:", yearLow(stock))
   #  print("52 week high:", yearHigh(stock))
   #  print("npm:", net_profit_margin(stock)
   # print(SMA_slope_Trial(stock))


