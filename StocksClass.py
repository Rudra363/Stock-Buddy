import yfinance as yf

class Stocks:
    def __init__(self, symbol, quantity):
        self.symbol = symbol
        self.quantity = quantity
        self.ticker = yf.Ticker(symbol)
        self.name = self.ticker.info.get('longName', 'Unknown Company')
        self.price = self.getPrice()  # fetch current price

    def __str__(self):
        return f"{self.name} ({self.symbol}): {self.quantity} shares at ${self.price:.2f} each"

    def setSymbol(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)  # update ticker object
        self.name = self.ticker.info.get('longName', 'Unknown Company')
        self.price = self.getPrice()

    def setName(self, name):
        self.name = name

    def setPrice(self):
        self.price = self.getPrice()

    def setQuantity(self, quantity):
        self.quantity = quantity

    def getName(self):
        return self.name

    def getValue(self):
        return self.price * self.quantity

    def getSymbol(self):
        return self.symbol

    def getPrice(self):
        return self.ticker.history(period="1d")['Close'].iloc[0]

    def getQuantity(self):
        return self.quantity

    def getRecommendations(self):
        return self.ticker.recommendations

    def getRecommendationsSummary(self):
        return self.ticker.recommendations_summary

    def getDividends(self):
        return self.ticker.dividends

    def getEarnings(self):
        return self.ticker.earnings

    def getGrowth(self):
        return self.ticker.growth_estimates

    def getFinancials(self):
        return self.ticker.financials

    def getIncome(self):
        return self.ticker.income_stmt

    def getInsiders(self):
        return self.ticker.insider_purchases

    def getNews(self):
        return self.ticker.news

    def getBalanceSheet(self):
        return self.ticker.balance_sheet

    def getInfo(self):
        return self.ticker.info

    def getIncomeStatement(self):
        return self.ticker.incomestmt
