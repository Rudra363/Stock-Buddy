from algorithms import *
from StocksClass import Stocks
from machineModel import *
from trainModel import *

def main():
    # Load the trained model
    model = load_model("stock_label_model.pkl")

    # List of stock symbols you want to test
    symbols_to_test = ["AAPL", "MSFT", "TSLA"]

    features_list = []
    for symbol in symbols_to_test:
        try:
            stock = Stocks(symbol, 10)  # adjust args if needed
            features = extractFeatures(stock)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if features_list:
        import numpy as np
        X_new = np.array(features_list)
        predictions = model.predict(X_new)

        for sym, pred in zip(symbols_to_test, predictions):
            print(f"Stock: {sym} --> Predicted label: {pred}")
    else:
        print("No valid stock features to predict.")


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


