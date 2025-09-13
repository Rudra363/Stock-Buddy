import csv
from algorithms import *
from StocksClass import *

def defineLabel(stock):
    weights = {
        "recommendation": 2,
        "dividend": 1,
        "growth": 2,
        "insider": 3,
        "roe": 3,
        "roa": 3,
        "current_ratio": 1,
        "quick_ratio": 1,
        "debt_equity": 1,
        "asset_turnover": 1,
        "net_profit_margin": 2,
        "pe_ratio": 3,
        "volume": 2,
        "year_low_in_range": 1,
        "rsi": 3,
        "macd": 2,
        "sma": 3
    }

    score = 0
    total = 0

    rating, percentage = mostRecommendations(stock)
    if rating in ["strongBuy", "buy"] and percentage is not None and percentage >= 50:
        score += weights["recommendation"]
    total += weights["recommendation"]

    dividend = getDividendYield(stock)
    if dividend is not None and 0.4 < dividend < 0.6:
        score += weights["dividend"]
    total += weights["dividend"]

    growth = getNextYearGrowth(stock)
    if growth is not None and growth > 0.1:
        score += weights["growth"]
    total += weights["growth"]

    insiderMoney = getNetInsiderPurchases(stock)
    if insiderMoney is not None and insiderMoney > 100000:
        score += weights["insider"]
    total += weights["insider"]

    roe = return_on_investments(stock)
    if roe is not None and roe > 0.15:
        score += weights["roe"]
    total += weights["roe"]

    roa = get_return_on_assets(stock)
    if roa is not None and roa > 0.05:
        score += weights["roa"]
    total += weights["roa"]

    currentRatio = current_ratio(stock)
    if currentRatio is not None and 1.5 <= currentRatio <= 2.0:
        score += weights["current_ratio"]
    total += weights["current_ratio"]

    quickRatio = quick_ratio(stock)
    if quickRatio is not None and quickRatio > 1:
        score += weights["quick_ratio"]
    total += weights["quick_ratio"]

    debtEquity = debtEquityRatio(stock)
    if debtEquity is not None and 0.5 < debtEquity < 1.5:
        score += weights["debt_equity"]
    total += weights["debt_equity"]

    assetTurnover = getAssestTurnoverRatio(stock)
    if assetTurnover is not None and assetTurnover > 1:
        score += weights["asset_turnover"]
    total += weights["asset_turnover"]

    npm = net_profit_margin(stock)
    if npm is not None and npm > 0.1:
        score += weights["net_profit_margin"]
    total += weights["net_profit_margin"]

    priceEarningsRatio = get_price_earnings_ratio(stock)
    if priceEarningsRatio is not None and priceEarningsRatio < 15:
        score += weights["pe_ratio"]
    total += weights["pe_ratio"]

    volume = getVolume(stock)
    if volume is not None and 400000 < volume < 20000000:
        score += weights["volume"]
    total += weights["volume"]

    lowYear = yearLow(stock)
    if lowYear is not None:
        low, high = percent_range(lowYear)
        if low < lowYear < high:
            score += weights["year_low_in_range"]
        total += weights["year_low_in_range"]

    rsi = getrsi(stock)
    if rsi is not None and rsi < 30:
        score += weights["rsi"]
    total += weights["rsi"]

    macd = movingAverageConvergenceDivergence(stock)
    if macd is not None and macd is True:
        score += weights["macd"]
    total += weights["macd"]

    sma = readSMA(stock)
    if sma is not None and sma is True:
        score += weights["sma"]
    total += weights["sma"]

    # Final label decision
    return 1 if score / total >= 0.6 else 0


def percent_range(value, percent=10):
    factor = percent / 100
    return value * (1 - factor), value * (1 + factor)

def csvTickers():
    dataset = []

    with open("all_stock_data.csv", newline='') as file:
        reader = csv.reader(file)
        next(reader)

        count = 0 

        for row in reader:
            symbol = row[1]
            try:
                stock = Stocks(symbol, 10)

                features = extractFeatures(stock)
                label = defineLabel(stock)

                dataset.append(features + [label])

                count += 1
                if count >= 9300:
                    break 

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

    with open("training_data.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = [
            "recommendation", "dividend", "growth", "insider", "roe", "roa",
            "current_ratio", "quick_ratio", "debt_equity", "asset_turnover",
            "net_profit_margin", "pe_ratio", "volume", "year_low_in_range", "rsi", "macd", "sma", "label"
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

    rsi = safe_float(getrsi(stock))
    features.append(1 if rsi is not None and rsi < 30 else 0)
    #
    macd = safe_float(movingAverageConvergenceDivergence(stock))
    features.append(1 if macd is not None and macd is True else 0)
    #
    sma = safe_float(readSMA(stock))
    features.append(1 if sma is True and sma is not None else 0)

    return features

