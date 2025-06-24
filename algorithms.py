from operator import truediv
import numpy as np
from StocksClass import Stocks
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def mostRecommendations(stock):
    rec_summary = stock.getRecommendationsSummary()

    # Check if the data is valid
    if rec_summary is None or rec_summary.empty:
        return "No recommendation data available."

    # Sum each rating column
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

    # Compute percentage for each rating type
    rating_percentages = {
        rating: (count / total_recommendations) * 100
        for rating, count in rating_totals.items()
    }

    # Find the rating with the highest percentage
    most_common_rating = max(rating_percentages, key=rating_percentages.get)
    most_common_percent = rating_percentages[most_common_rating]

    #return f"Most common rating for {stock.getSymbol()} is '{most_common_rating}' at {most_common_percent:.2f}% of all recommendations."
    return most_common_rating, most_common_percent

def getDividendYield(stock):
    dividends = stock.getDividends()  # Series with datetime index
    if dividends.empty:
        return f"{stock.getSymbol()} does not pay dividends."

    # Get the latest 4 dividends (most stocks pay quarterly)
    recent_dividends = dividends[-4:]
    annual_dividend = recent_dividends.sum()

    current_price = stock.getPrice()
    dividend_yield = (annual_dividend / current_price) * 100

    #return f"Dividend yield for {stock.getSymbol()} is {dividend_yield:.2f}%"
    #dividend_yield = stock.getInfo.get("dividendYield")
    return dividend_yield


def sma(data, window):
    return data["Close"].rolling(window=window).mean()

def smaGraph(stock):
    ticker_symbol = stock.getSymbol()
    data = stock.getHistoricalPrice(period="1y", interval="1d")

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    # Calculate SMAs
    data["SMA_20"] = sma(data, 20)
    data["SMA_50"] = sma(data, 50)
    data["SMA_100"] = sma(data, 100)
    data["SMA_200"] = sma(data, 200)

    # data['Buy_Signal'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift() <= data['Signal'].shift())
    # data['Sell_Signal'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift() >= data['Signal'].shift())
    #     """
    #     Plots stock price along with SMA20, SMA50, SMA100, and SMA200.
    #     """

    plt.figure(figsize=(14, 7))

    # Plot stock closing price
    plt.plot(data.index, data["Close"], label="Close Price", linewidth=2)
    # Plot SMAs if they exist
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
    # plt.figure(figsize=(14,6))
    # plt.plot(data['SMA_20'], label='SMA_20', color='blue')
    # plt.plot(data['SMA_50'], label='SMA_50', color='red')
    # plt.plot(data['SMA_100'], label='SMA_100', color='green')
    # plt.plot(data['SMA_200'], label='SMA_200', color='yellow')
    # plt.plot(stock.getHistoricalPrice(period = "1y", interval = "1d"), label='Signal Line', color='orange')
    # #plt.bar(data.index, data['MACD'] - data['Signal'], label='Line', color='gray')
    # plt.legend()

    # plt.show()

def SMA_current_slope(stock, window):
    ticker_symbol = stock.getSymbol()
    data = stock.getHistoricalPrice(period="1y", interval="1d")

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    #Calculate SMAs
    data["SMA_Wanted"] = sma(data, window)

    # Ensure enough data for SMA_20 slope calculation
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
    #Calculate SMAs
    data["SMA_Wanted"] = sma(data, window)

    # Ensure enough data for SMA_20 slope calculation
    if len(data["SMA_Wanted"].dropna()) < 2:
        raise ValueError("Not enough SMA_20 data to compute slope.")

    current = data["SMA_Wanted"].iloc[-(daysAwayFromToday)]
    previous = data["SMA_Wanted"].iloc[-(daysAwayFromToday+1)]

    return (current - previous)

def readSMA(stock):
    """
    Computes SMA20, 50, 100, and 200, their current values,
    slopes, and basic trend signals (Golden/Death crosses).
    Returns a summary dict.
    """
    # Get historical data
    data = stock.getHistoricalPrice(period="1y", interval="1d")
    if data.empty:
        raise ValueError(f"No data for {stock.getSymbol()}")

    # Compute SMAs
    for window in [20, 50, 100, 200]:
        data[f"SMA_{window}"] = sma(data, window)

    # Get latest SMA values
    latest = {w: data[f"SMA_{w}"].iloc[-1] for w in [20, 50, 100, 200]}

    # Compute slopes (difference between last two values)
    slopes = {}
    for window in [20, 50, 100, 200]:
        col = f"SMA_{window}"
        if data[col].dropna().size < 2:
            slopes[window] = None
        else:
            slopes[window] = data[col].iloc[-1] - data[col].iloc[-2]

    # Check cross signals (using 20 & 50 vs 200)
    signals = []
    #if Sma under price & slope == strong positive slope then bullish
    if data["SMA_50"].iloc[-1] <= data["Close"].iloc[-1] and SMA_current_slope(stock, 50)>0:
        signals.append("Bullish Behaviour")
       # print("trigger1")
        return True
    #if Sma over price & slope == strong negative slope then bearish
    if data["SMA_50"].iloc[-1] >= data["Close"].iloc[-1] and SMA_current_slope(stock, 50)<0:
        signals.append("Bearish Behaviour")
       # print("trigger2")
        return False
    #if sma intersects price from below = bearish
    TOLERANCE = 0.5
    # Treat as intersecting (crossover signal)
    if np.isclose(data["SMA_50"].iloc[-1], data["Close"].iloc[-1], atol=TOLERANCE) and SMA_current_slope(stock, 20)>0:
       # print("Bearish Behaviour")
       # print("trigger3")
        return False
    #if sma intersects price from above = bullish
    if np.isclose(data["SMA_50"].iloc[-1],data["Close"].iloc[-1], atol=TOLERANCE) and SMA_current_slope(stock, 50)<0:
       # print("Bullish Behaviour")
        #print("trigger4")
        return True

    if SMA_current_slope(stock, 50) == 0:
        #if slope approach zero from increasing then bearish
        if SMA_previous_slope(stock, 50, 2) > 0:
            signals.append("Bearish Behaviour")
            #print("trigger5")
            return False
        # if slope approach zero form decreasing then bullish
        if SMA_previous_slope(stock, 50,2) < 0:
            signals.append("Bullish Behaviour")
           # print("trigger6")
            return True

    # Golden cross: short-term crossing above long-term (50 > 200)
    if data["SMA_20"].iloc[-2] <= data["SMA_200"].iloc[-2] and data["SMA_20"].iloc[-1] > data["SMA_200"].iloc[-1]:
        signals.append("Golden Cross (20 over 200)")
        return True
    if data["SMA_50"].iloc[-2] <= data["SMA_200"].iloc[-2] and data["SMA_50"].iloc[-1] > data["SMA_200"].iloc[-1]:
        signals.append("Golden Cross (50 over 200)")
        return True

    # Death cross (200 < 50)
    if data["SMA_20"].iloc[-2] >= data["SMA_200"].iloc[-2] and latest[20] < latest[200]:
        signals.append("Death Cross (20 below 200)")
        return False
    if data["SMA_50"].iloc[-2] >= data["SMA_200"].iloc[-2] and latest[50] < latest[200]:
        signals.append("Death Cross (50 below 200)")
        return False

    # Price relative to SMAs
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

    # Average gains/losses over `period` days
    avg_gain = data["Gain"].rolling(window=period).mean()
    avg_loss = data["Loss"].rolling(window=period).mean()

    # Relative Strength (RS)
    rs = avg_gain / avg_loss

    # RSI calculation
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi

    # Get latest RSI value
    latest_rsi = data["RSI"].iloc[-1]

    return latest_rsi


def getCurrentYearEarnings(stock):
    """
    Returns the Net Income (earnings) for the current fiscal year
    from the stock's income statement.

    :param stock: yfinance.Ticker object
    :return: Net Income as float or None if not available
    """
    try:
        income_stmt = stock.getIncome()

        # The income statement is a DataFrame with metrics as index and fiscal dates as columns.
        # The first column is the most recent fiscal year.
        current_year_col = income_stmt.columns[0]

        # Get 'Net Income' row value for the current fiscal year
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

        # Get all fiscal year columns
        years = income_stmt.columns

        if len(years) < 2:
            return None  # Not enough data for previous year

        # The previous year is the second column (index 1)
        prev_year = years[1]

        # Get net income for previous year
        net_income = income_stmt.loc['Net Income', prev_year]

        return net_income
    except Exception as e:
       # print(f"Error fetching last year earnings: {e}")
        return None


def getNextYearGrowth(stock):
    growth_df = stock.getGrowth()

    if growth_df is None or growth_df.empty:
        return "No growth estimate data available."

    try:
        next_year_growth = growth_df.loc['+1y', 'stockTrend']
       # return f"Estimated earnings growth for next year: {next_year_growth * 100:.2f}%"
        return next_year_growth
    except KeyError:
        return "Next year growth estimate not found."


def getNetInsiderPurchases(stock):
    """
    Returns the net number of insider shares purchased in the last 6 months.

    Parameters:
        insider_df (pd.DataFrame): DataFrame containing insider transaction summary.

    Returns:
        float or str: Net shares purchased, or a message if not available.
    """
    insider_df = stock.getInsiders()
    try:
        # Look for the row labeled 'Net Shares Purchased (Sold)'
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
# Get stock data (e.g., Apple)
    data = stock.getHistoricalPrice(period = "1mo", interval = "1d")
    # Calculate MACD and Signal Line
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Generate Buy/Sell signals
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

# def getInsiderConfidence(stock):
#     net_shares = getNetInsiderPurchases(stock)
#
#     if net_shares is None:
#         return "Net insider purchase data not available."
#
#     if net_shares > 1_000_000:
#         confidence = "Very High insider confidence 🚀"
#     elif net_shares > 500_000:
#         confidence = "High insider confidence 👍"
#     elif net_shares > 100_000:
#         confidence = "Moderate insider confidence 🙂"
#     elif net_shares > 0:
#         confidence = "Slight insider confidence 🧐"
#     else:
#         confidence = "Low or negative insider confidence ⚠️"
#
#     return f"Net insider shares purchased: {int(net_shares):,} — {confidence}"

# def simpleMovingAverage(stock): #Averages the Closing Price based on ___ # of days
#     ticker_symbol = stock.getSymbol()
#     data = yf.Ticker(ticker_symbol).history(period="1y", interval="1d", auto_adjust=True)
#     if data.empty:
#         raise ValueError(f"No data returned for {ticker_symbol}")
#     #data = yf.Ticker(stock.getSymbol()).history(stock, period="6mo", interval = "1d", auto_adjust=True)
#     data["SMA_20"] = data["Close"].rolling(window=20).mean()
#     data["SMA_50"] = data["Close"].rolling(window=50).mean()
#     data["SMA_100"] = data["Close"].rolling(window=100).mean()
#     data["SMA_200"] = data["Close"].rolling(window=200).mean()
#     df = pd.DataFrame({ #Creates a Table with Headers
#         "Current Price": stock.getPrice(),
#         "SMA_": ["20","50","100","200"],
#         "Closing Price Average": [data["SMA_20"].iloc[-1],data["SMA_50"].iloc[-1],data["SMA_100"].iloc[-1],data["SMA_200"].iloc[-1]]
#     })
#     #Golden Cross - Buy Signal ( short-term SMA passes above long-term SMA)
#     #TODO: try to use ML to teach computer how to read the graphs/trends
#     # if data["SMA_20"] > data["SMA_200"]:
#     #     print("Golden Cross: BUY SIGNAL")
#     # if data["SMA_20"] < data["SMA_200"]:
#     #     print("Death Cross: SELL SIGNAL")
#
#     return df, data["SMA_20"], data["SMA_50"], data["SMA_100"], data["SMA_200"]