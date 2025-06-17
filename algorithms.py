from operator import truediv
import pandas as pd
import yfinance as yf
from StocksClass import Stocks

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

def simpleMovingAverage(stock): #Averages the Closing Price based on ___ # of days
    ticker_symbol = stock.getSymbol()
    data = yf.Ticker(ticker_symbol).history(period="1y", interval="1d", auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")
    #data = yf.Ticker(stock.getSymbol()).history(stock, period="6mo", interval = "1d", auto_adjust=True)
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_100"] = data["Close"].rolling(window=100).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    df = pd.DataFrame({ #Creates a Table with Headers
        "Current Price": stock.getPrice(),
        "SMA_": ["20","50","100","200"],
        "Closing Price Average": [data["SMA_20"].iloc[-1],data["SMA_50"].iloc[-1],data["SMA_100"].iloc[-1],data["SMA_200"].iloc[-1]]
    })
    #Golden Cross - Buy Signal ( short-term SMA passes above long-term SMA)
    #TODO: try to use ML to teach computer how to read the graphs/trends
    # if data["SMA_20"] > data["SMA_200"]:
    #     print("Golden Cross: BUY SIGNAL")
    # if data["SMA_20"] < data["SMA_200"]:
    #     print("Death Cross: SELL SIGNAL")

    return df, data["SMA_20"], data["SMA_50"], data["SMA_100"], data["SMA_200"]



def SMA_slope_Trial(stock):
    """
        Calculates SMAs and the slope of the 20-day SMA for a given stock.
        """
    ticker_symbol = stock.getSymbol()
    data = yf.Ticker(ticker_symbol).history(period="1y", interval="1d", auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")

    # Calculate SMAs
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_100"] = data["Close"].rolling(window=100).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    # Ensure at least two SMA_20 values for slope calculation
    if len(data) < 2:
        raise ValueError("Not enough data to compute SMA slope.")

    # Get the last two SMA_20 values
    sma_20_current = data["SMA_20"]
    sma_20_previous = (data["Close"].rolling(window=21) - stock.get).mean()

    # Calculate the slope as difference
    sma_20_slope = sma_20_current - sma_20_previous

    # Prepare output
    # df = pd.DataFrame({
    #     "Current Price": stock.getPrice(),
    #     "SMA_20": sma_20_current,
    #     "SMA_50": data["SMA_50"].iloc[-1],
    #     "SMA_100": data["SMA_100"].iloc[-1],
    #     "SMA_200": data["SMA_200"].iloc[-1],
    #     "SMA_20_Slope": sma_20_slope
    # })

    # return df
    return sma_20_slope


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
        print(f"Error fetching last year earnings: {e}")
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
            return "Net Shares Purchased (Sold) data not available."
    except Exception as e:
        return f"Error retrieving net insider purchases: {e}"

def getInsiderConfidence(stock):
    net_shares = getNetInsiderPurchases(stock)

    if net_shares is None:
        return "Net insider purchase data not available."

    if net_shares > 1_000_000:
        confidence = "Very High insider confidence 🚀"
    elif net_shares > 500_000:
        confidence = "High insider confidence 👍"
    elif net_shares > 100_000:
        confidence = "Moderate insider confidence 🙂"
    elif net_shares > 0:
        confidence = "Slight insider confidence 🧐"
    else:
        confidence = "Low or negative insider confidence ⚠️"

    return f"Net insider shares purchased: {int(net_shares):,} — {confidence}"



# COME BACK TO FUNDAMENTAL ANALYSIS - not hard jsut dont know what numbers to use ; maybe calculate for each prev year to predict pattern


# def get_total_revenue(stock):
#     return stock.financials.loc['Total Revenue']
#
  # 10 or more % usually good - for this use the most recent year but then also calculate prev year and see if its decline or increase - this tells u if the company is growing or not
def net_profit_margin(stock):
    tot_rev = stock.getInfo().get("totalRevenue")
    print(tot_rev)
    npm = getCurrentYearEarnings(stock)/tot_rev
    # if npm.round() >= 10:
    #     return True
    # return False
    return npm


 # P/E ratio - below 15 is good but the average is 20 25
def get_price_earnings_ratio(stock):
    priceEarningsRatio = stock.getInfo().get("trailingPE")
    return priceEarningsRatio
   # stock_price = stock.getPrice()
   # earningPerShare = stock.ticker.info.get("trailingEps")
   # print(stock.ticker.info)
   # print(stock_price, earningPerShare)
   # price_earnings_ratio = stock_price//earningPerShare
   # print(price_earnings_ratio)
   # return price_earnings_ratio


def return_on_investments(stock): # 15 or more % good
    # se = stock.ticker.info.get['Total Stockholders Equity']
    # roe = (stock.getCurrentYearEarnings()/se)*100
    # print(roe)
    roe = stock.getInfo().get("returnOnEquity")
    # if roe > 15:
    #     return False
    # return True
    return roe


def get_return_on_assets(stock): # 5 or more % good
    # total_assets = stock.ticker.balance_sheet.loc["Total Assets"].iloc[0]
    # roa = (stock.getCurrentYearEarnings()/total_assets )*100
    roa = stock.getInfo().get("returnOnAssets")
    print(roa)
    # if roa >= 5:
    #     return True
    # return False
    return roa

def current_ratio(stock): # 1.2/1.4 - 2 is good - more doesnt mean good necessarily since it would mean they arent investing or selling much or whatver - careful
    current_ratio = stock.getInfo().get("currentRatio")
    return current_ratio
    # print(cu)
    # current_assests = balance_sheet.loc['Total Current Assests', latest_period]
    # current_liabilities = balance_sheet.loc['Total Current Liabilities', latest_period]
    # current_ratio = current_assets/current_liabilities



# maybe quick ratio - anything 1 or higher
def quick_ratio(stock):
    quickRatio = stock.getInfo().get("quickRatio")
    return quickRatio


# debt to equity ratio - 0.5 to 1.5 is considered fine
def debtEquityRatio(stock):
    return stock.getInfo().get("debtToEquity")


# interest coverage ratio - 3 or more is solid
def interestCoverage(stock):
    #EBIT is in the new income statement one
    pass


# TODO: create ratios to see if stock is undervalued
# P/E ratio (Done), EV/EBIT,. P/S ratio, insider buyinh,

# asset turnover ratio - 1 or higher is better but depends on industry i think
def getAssestTurnoverRatio(stock):
    income_stmt = stock.getIncomeStatement()
    latest_column = income_stmt.columns[0]
    net_income = income_stmt.loc['Net Income', latest_column]
    print("Net Income:", net_income)
    roa = get_return_on_assets(stock)
    print("Return on Assets:", roa)
    averageTotalAssests = net_income/roa
    print("Average Total Assests:", averageTotalAssests)
    totalRevenue = stock.getInfo().get("totalRevenue")
    print("Total Revenue:", totalRevenue)
    assestTurnover = totalRevenue/averageTotalAssests
    #print("Assest Turnover:", assestTurnover)
    return assestTurnover

def getVolume(stock):
    return stock.getInfo().get("volume")

def yearLow(stock):
    return stock.getInfo().get("fiftyTwoWeekLow")

def yearHigh(stock):
    return stock.getInfo().get("fiftyTwoWeekHigh")

# inventory turnover ratio - 5 to 10 is the common
def inventory_turnover_ratio(stock):
    pass
    ## I need COGS divided by total inventory
