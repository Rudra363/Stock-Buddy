import csv
from algorithms import *
from StocksClass import *

def defineLabel(stock):
    count = 0
    total = 0
    rating, percentage = mostRecommendations(stock)
    if rating in ["strongBuy", "buy"] and percentage is not None and percentage >= 50:
        count += 1
    total += 1

    dividend = getDividendYield(stock)
    if 0.4 < dividend < 0.6  :
        count += 1
    total += 1

    growth = getNextYearGrowth(stock)
    if growth > 0.1 and growth is not None:
        count += 1
    total += 1

    insiderMoney = getNetInsiderPurchases(stock)
    if insiderMoney > 100000 and insiderMoney is not None:
        count += 1
    total += 1

    roe = return_on_investments(stock) # 15 or more % good
    if roe > 0.15 and roe is not None:
        count += 1
    total += 1

    roa = get_return_on_assets(stock)  # 5 or more % good
    if roa > 0.05 and roa is not None:
        count += 1
    total += 1

    currentRatio = current_ratio(stock) # 1.2/1.4 - 2 is good - more doesnt mean good necessarily since it would mean they arent investing or selling much or whatver - careful
    if 1.5 <= currentRatio <= 2.0 and currentRatio is not None:
        count += 1
    total += 1

    quickRatio = quick_ratio(stock) #anything 1 or higher
    if quickRatio > 1 and quickRatio is not None:
        count += 1
    total += 1

    debtEquity = debtEquityRatio(stock)  #0.5 to 1.5 is considered fine
    if 0.5 < debtEquity < 1.5 and debtEquity is not None:
        count += 1
    total += 1

    assetTurnover = getAssestTurnoverRatio(stock) #asset turnover ratio - 1 or higher is better but depends on industry i think
    if assetTurnover > 1 and assetTurnover is not None:
        count += 1
    total += 1

   # df, SMA_20, SMA_50, SMA_100, SMA_200 = SMA_slope(stock)

    npm = net_profit_margin(stock) # 10 or more % usually good
    if npm > 0.1 and npm is not None:
        count += 1
    total += 1

    priceEarningsRatio = get_price_earnings_ratio(stock) # below 15 is good but the average is 20 25
    if priceEarningsRatio < 15 and priceEarningsRatio is not None:
        count += 1
    total += 1

    volume = getVolume(stock)
    if 400000 < volume < 20000000 and volume is not None:
        count += 1
    total += 1

    lowYear = yearLow(stock)
    low, high = percent_range(lowYear)
    if low < lowYear < high and lowYear is not None:
        count += 1
    total += 1

    # rsi = getrsi(stock)
    # if rsi < 30 and rsi is not None:
    #     count += 1
    # total += 1
    #
    # macd = movingAverageConvergenceDivergence(stock)
    # if macd is True and macd is not None:
    #     count += 1
    # total += 1

    #highYear = yearHigh(stock)

    if count/total >= 0.7:
        label = 1
    else:
        label = 0

    return label

def percent_range(value, percent=10):
    factor = percent / 100
    return value * (1 - factor), value * (1 + factor)

def csvTickers():
    dataset = []

    with open("all_stock_data.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        count = 0  # Initialize counter

        for row in reader:
            symbol = row[1]
            try:
                stock = Stocks(symbol, 10)

                features = extractFeatures(stock)
                label = defineLabel(stock)

                dataset.append(features + [label])

                count += 1
                if count >= 9000:
                    break  # Stop after processing 5 stocks

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

    # Write to CSV
    with open("training_data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = [
            "recommendation", "dividend", "growth", "insider", "roe", "roa",
            "current_ratio", "quick_ratio", "debt_equity", "asset_turnover",
            "net_profit_margin", "pe_ratio", "volume", "year_low_in_range", "rsi", "macd", "label"
        ]
        writer.writerow(header)
        writer.writerows(dataset)

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def extractFeatures(stock):
    features = []

    rating, percentage = mostRecommendations(stock)
    features.append(1 if rating in ["strongBuy", "buy"] and percentage is not None and percentage >= 50 else 0)

    dividend = safe_float(getDividendYield(stock))
    features.append(1 if dividend is not None and 0.4 < dividend < 0.6 else 0)

    growth = safe_float(getNextYearGrowth(stock))
    features.append(1 if growth is not None and growth > 0.1 else 0)

    insiderMoney = safe_float(getNetInsiderPurchases(stock))
    features.append(1 if insiderMoney is not None and insiderMoney > 100000 else 0)

    roe = safe_float(return_on_investments(stock))
    features.append(1 if roe is not None and roe > 0.15 else 0)

    roa = safe_float(get_return_on_assets(stock))
    features.append(1 if roa is not None and roa > 0.15 else 0)

    currentRatio = safe_float(current_ratio(stock))
    features.append(1 if currentRatio is not None and 1.5 <= currentRatio <= 2.0 else 0)

    quickRatio = safe_float(quick_ratio(stock))
    features.append(1 if quickRatio is not None and quickRatio > 1 else 0)

    debtEquity = safe_float(debtEquityRatio(stock))
    features.append(1 if debtEquity is not None and 0.5 < debtEquity < 1.5 else 0)

    assetTurnover = safe_float(getAssestTurnoverRatio(stock))
    features.append(1 if assetTurnover is not None and assetTurnover > 1 else 0)

    npm = safe_float(net_profit_margin(stock))
    features.append(1 if npm is not None and npm > 0.1 else 0)

    priceEarningsRatio = safe_float(get_price_earnings_ratio(stock))
    features.append(1 if priceEarningsRatio is not None and priceEarningsRatio < 15 else 0)

    volume = safe_float(getVolume(stock))
    features.append(1 if volume is not None and 400000 < volume < 20000000 else 0)

    lowYear = safe_float(yearLow(stock))
    if lowYear is not None:
        low, high = percent_range(lowYear)
        features.append(1 if low < lowYear < high else 0)
    else:
        features.append(0)

    return features

