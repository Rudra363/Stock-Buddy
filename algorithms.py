from operator import truediv
import numpy as np
from StocksClass import Stocks
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def mostRecommendations(stock):
    rec_summary = stock.getRecommendationsSummary()

    if rec_summary is None or rec_summary.empty:
        return "No recommendation data available."

    rating_totals = {
        "strongBuy": rec_summary["strongBuy"].sum(),
        "buy": rec_summary["buy"].sum(),
        "hold": rec_summary["hold"].sum(),
        "sell": rec_summary["sell"].sum(),
        "strongSell": rec_summary["strongSell"].sum()
    }

    total_recommendations = sum(rating_totals.values())
    if total_recommendations == 0:
        return "No analysts have given recommendations."

    rating_percentages = {
        rating: (count / total_recommendations) * 100
        for rating, count in rating_totals.items()
    }

    most_common_rating = max(rating_percentages, key=rating_percentages.get)
    most_common_percent = rating_percentages[most_common_rating]

    return most_common_rating, most_common_percent

def getDividendYield(stock):
    dividends = stock.getDividends()  
    if dividends.empty:
        return f"{stock.getSymbol()} does not pay dividends."

    recent_dividends = dividends[-4:]
    annual_dividend = recent_dividends.sum()

    current_price = stock.getPrice()
    dividend_yield = (annual_dividend / current_price) * 100

 
    return dividend_yield


def sma(data, window):
    return data["Close"].rolling(window=window).mean()

def smaGraph(stock):
    ticker_symbol = stock.getSymbol()
    data = stock.getHistoricalPrice(period="1y", interval="1d")

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    data["SMA_20"] = sma(data, 20)
    data["SMA_50"] = sma(data, 50)
    data["SMA_100"] = sma(data, 100)
    data["SMA_200"] = sma(data, 200)

   

    plt.figure(figsize=(14, 7))

    plt.plot(data.index, data["Close"], label="Close Price", linewidth=2)
    plt.plot(data["SMA_20"], label="SMA 20", linestyle='--', color="blue")
    plt.plot(data.index, data["SMA_50"], label="SMA 50", linestyle='--', color="red")
    plt.plot(data.index, data["SMA_100"], label="SMA 100", linestyle='--', color="green")
    plt.plot(data.index, data["SMA_200"], label="SMA 200", linestyle='--', color="cyan")
    plt.title('SMA Indicator')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
   

def SMA_current_slope(stock, window):
    ticker_symbol = stock.getSymbol()
    data = stock.getHistoricalPrice(period="1y", interval="1d")

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    data["SMA_Wanted"] = sma(data, window)

    if len(data["SMA_Wanted"].dropna()) < 2:
        raise ValueError("Not enough SMA_20 data to compute slope.")

    current = data["SMA_Wanted"].iloc[-1]
    previous = data["SMA_Wanted"].iloc[-2]

    return (current - previous)

def SMA_previous_slope(stock, window, daysAwayFromToday):
    ticker_symbol = stock.getSymbol()
    data = stock.getHistoricalPrice(period="1y", interval="1d")

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    data["SMA_Wanted"] = sma(data, window)

    if len(data["SMA_Wanted"].dropna()) < 2:
        raise ValueError("Not enough SMA_20 data to compute slope.")

    current = data["SMA_Wanted"].iloc[-(daysAwayFromToday)]
    previous = data["SMA_Wanted"].iloc[-(daysAwayFromToday+1)]

    return (current - previous)

def readSMA(stock):
    data = stock.getHistoricalPrice(period="1y", interval="1d")
    if data.empty:
        raise ValueError(f"No data for {stock.getSymbol()}")

    for window in [20, 50, 100, 200]:
        data[f"SMA_{window}"] = sma(data, window)

    latest = {w: data[f"SMA_{w}"].iloc[-1] for w in [20, 50, 100, 200]}

    slopes = {}
    for window in [20, 50, 100, 200]:
        col = f"SMA_{window}"
        if data[col].dropna().size < 2:
            slopes[window] = None
        else:
            slopes[window] = data[col].iloc[-1] - data[col].iloc[-2]

    signals = []
    if data["SMA_50"].iloc[-1] <= data["Close"].iloc[-1] and SMA_current_slope(stock, 50)>0:
        signals.append("Bullish Behaviour")
        return True
    if data["SMA_50"].iloc[-1] >= data["Close"].iloc[-1] and SMA_current_slope(stock, 50)<0:
        signals.append("Bearish Behaviour")
        return False
    TOLERANCE = 0.5
    if np.isclose(data["SMA_50"].iloc[-1], data["Close"].iloc[-1], atol=TOLERANCE) and SMA_current_slope(stock, 20)>0:
        return False
    if np.isclose(data["SMA_50"].iloc[-1],data["Close"].iloc[-1], atol=TOLERANCE) and SMA_current_slope(stock, 50)<0:
       # print("Bullish Behaviour")
        #print("trigger4")
        return True

    if SMA_current_slope(stock, 50) == 0:
        if SMA_previous_slope(stock, 50, 2) > 0:
            signals.append("Bearish Behaviour")
            #print("trigger5")
            return False
        if SMA_previous_slope(stock, 50,2) < 0:
            signals.append("Bullish Behaviour")
           # print("trigger6")
            return True

    if data["SMA_20"].iloc[-2] <= data["SMA_200"].iloc[-2] and data["SMA_20"].iloc[-1] > data["SMA_200"].iloc[-1]:
        signals.append("Golden Cross (20 over 200)")
        return True
    if data["SMA_50"].iloc[-2] <= data["SMA_200"].iloc[-2] and data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]:
        signals.append("Golden Cross (50 over 200)")
        return True

    if data["SMA_20"].iloc[-2] >= data["SMA_200"].iloc[-2] and latest[20] < latest[200]:
        signals.append("Death Cross (20 below 200)")
        return False
    if data["SMA_50"].iloc[-2] >= data["SMA_200"].iloc[-2] and latest[50] < latest[200]:
        signals.append("Death Cross (50 below 200)")
        return False

    price = data["Close"].iloc[-1]
    price_vs = {w: "above" if price > latest[w] else "below" for w in latest}
    return signals

    # return {
    #     "symbol": stock.getSymbol(),
    #     "latest_sma": latest,
    #     "slopes": slopes,
    #     "price_vs_sma": price_vs,
    #     "signals": signals
    # }


def getrsi(stock):
    data = stock.getHistoricalPrice(period = "2mo", interval = "1d")
    period = 14
    data["Change"] = data["Close"].diff()
    data["Gain"] = data["Change"].apply(lambda x: x if x > 0 else 0)
    data["Loss"] = data["Change"].apply(lambda x: -x if x < 0 else 0)

    avg_gain = data["Gain"].rolling(window=period).mean()
    avg_loss = data["Loss"].rolling(window=period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi

    latest_rsi = data["RSI"].iloc[-1]

    return latest_rsi


def getCurrentYearEarnings(stock):
    
    try:
        income_stmt = stock.getIncome()

        current_year_col = income_stmt.columns[0]

        net_income = income_stmt.loc['Net Income', current_year_col]

        return net_income
    except Exception as e:
        print(f"Error fetching current year earnings: {e}")
        return None


def getPreviousYearsEarnings(stock):
    """
    Returns a dictionary of previous fiscal years' Net Income (earnings),
    excluding the current fiscal year.

    :param stock: yfinance.Ticker object
    :return: dict where keys are fiscal year dates and values are net incomes
    """
    try:
        income_stmt = stock.getIncome()

        years = income_stmt.columns

        if len(years) < 2:
            return None 
            
        prev_year = years[1]

        net_income = income_stmt.loc['Net Income', prev_year]

        return net_income
    except Exception as e:
        return None


def getNextYearGrowth(stock):
    growth_df = stock.getGrowth()

    if growth_df is None or growth_df.empty:
        return "No growth estimate data available."

    try:
        next_year_growth = growth_df.loc['+1y', 'stockTrend']
        return next_year_growth
    except KeyError:
        return "Next year growth estimate not found."


def getNetInsiderPurchases(stock):
   
    insider_df = stock.getInsiders()
    try:
        net_row = insider_df[insider_df['Insider Purchases Last 6m'] == 'Net Shares Purchased (Sold)']

        if not net_row.empty:
            net_shares = net_row['Shares'].values[0]
            return float(net_shares)
        else:
            return 0
    except Exception as e:
        return f"Error retrieving net insider purchases: {e}"



# COME BACK TO FUNDAMENTAL ANALYSIS - not hard jsut dont know what numbers to use ; maybe calculate for each prev year to predict pattern


# def get_total_revenue(stock):
#     return stock.financials.loc['Total Revenue']
#
  # 10 or more % usually good - for this use the most recent year but then also calculate prev year and see if its decline or increase - this tells u if the company is growing or not
def net_profit_margin(stock):
    tot_rev = stock.getInfo().get("totalRevenue")
    #print(tot_rev)
    npm = getCurrentYearEarnings(stock)/tot_rev
    # if npm.round() >= 10:
    #     return True
    # return False
    return npm


 # P/E ratio - below 15 is good but the average is 20 25
def get_price_earnings_ratio(stock):
    priceEarningsRatio = stock.getInfo().get("trailingPE")
    return priceEarningsRatio


def return_on_investments(stock): # 15 or more % good
    roe = stock.getInfo().get("returnOnEquity")
    return roe

def get_return_on_assets(stock): # 5 or more % good
    roa = stock.getInfo().get("returnOnAssets")
    #print(roa)
    return roa

def current_ratio(stock): # 1.2/1.4 - 2 is good - more doesnt mean good necessarily since it would mean they arent investing or selling much or whatver - careful
    current_ratio = stock.getInfo().get("currentRatio")
    return current_ratio

# maybe quick ratio - anything 1 or higher
def quick_ratio(stock):
    quickRatio = stock.getInfo().get("quickRatio")
    return quickRatio

# debt to equity ratio - 0.5 to 1.5 is considered fine
def debtEquityRatio(stock):
    return stock.getInfo().get("debtToEquity")

# asset turnover ratio - 1 or higher is better but depends on industry i think
def getAssestTurnoverRatio(stock):
    income_stmt = stock.getIncomeStatement()
    latest_column = income_stmt.columns[0]
    net_income = income_stmt.loc['Net Income', latest_column]
    #print("Net Income:", net_income)
    roa = get_return_on_assets(stock)
    #print("Return on Assets:", roa)
    averageTotalAssests = net_income/roa
    #print("Average Total Assests:", averageTotalAssests)
    totalRevenue = stock.getInfo().get("totalRevenue")
    #print("Total Revenue:", totalRevenue)
    assestTurnover = totalRevenue/averageTotalAssests
    #print("Assest Turnover:", assestTurnover)
    return assestTurnover

def getVolume(stock):
    return stock.getInfo().get("volume")

def yearLow(stock):
    return stock.getInfo().get("fiftyTwoWeekLow")

def yearHigh(stock):
    return stock.getInfo().get("fiftyTwoWeekHigh")

# MACD Line vs Signal Line:
# 	•	When MACD crosses above the signal line → Bullish signal (price may rise)
# 	•	When MACD crosses below the signal line → Bearish signal (price may fall)
# 	2.	MACD Histogram:
# 	•	The difference between the MACD and signal line.
# 	•	Shows the momentum: larger bars mean stronger trends.
def movingAverageConvergenceDivergence(stock):
    data = stock.getHistoricalPrice(period = "1mo", interval = "1d")
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['Buy_Signal'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift() <= data['Signal'].shift())
    data['Sell_Signal'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift() >= data['Signal'].shift())
    plt.figure(figsize=(14,6))
    plt.plot(data['MACD'], label='MACD', color='blue')
    plt.plot(data['Signal'], label='Signal Line', color='orange')
    plt.bar(data.index, data['MACD'] - data['Signal'], label='Histogram', color='gray')
    plt.legend()
    plt.title('MACD Indicator')
    plt.show()

    if data['Buy_Signal'].any():
       # print("Buy signal")
        return True
    elif data['Sell_Signal'].any():
        #print("Sell signal")
        return False
    else:
        #print("No signal/Hold")
        return None

def newsAlgorithm(stock):
    news_items = stock.getNews()

    for item in news_items:
        try:
            title = item["content"]["title"]
            print(title)
        except KeyError:
            print("Title not found in:", item)

# inventory turnover ratio - 5 to 10 is the common
def inventory_turnover_ratio(stock):
    pass
    ## I need COGS divided by total inventory

# interest coverage ratio - 3 or more is solid
def interestCoverage(stock):
    #EBIT is in the new income statement one
    pass
