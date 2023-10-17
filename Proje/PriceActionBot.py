from Data import CoinData
from PriceAction import PriceAction
import pandas as pd
import time


class MarketStructureBot:
    coins = ['BTCUSDT','ETHUSDT','XRPUSDT','BNBUSDT','AVAXUSDT','SOLUSDT','FTMUSDT','MATICUSDT']
    timeInterval = ['15m',"1h","4h","1d"]
    signals = pd.DataFrame(columns=['CoinName','Interval','Date','Price','Type','Signal'])

    def __init__(self,windowSize = 5,limit=250):
        self.limit = limit
        self.windowSize = windowSize
    
    def findSignalForMarketStructure(self):
        coinData = CoinData()
        for interval in self.timeInterval:
            for coinName in self.coins:
                df = coinData.gatInfoInstantlyBinanceByLimit(coinName,self.limit,interval="4h")
                priceAction = PriceAction(df,coinName,self.windowSize)
                signal=priceAction.OnlyMarketStructureBreakSignals()
                self.signals.loc[len(self.signals)] = [coinName,interval,signal['Date'],signal['Price'],signal['Type'],signal['Signal']] 
        
        self.signals.sort_values(by='Date',ascending=False,inplace=True)
        self.signals.to_csv('PriceAction.csv',index=False)

    def findSignalForMarketStructureAutomaticaly(self,sleepInterval = 10):
        coinData = CoinData()
        while True:
            for interval in self.timeInterval:
                for coinName in self.coins:
                    df = coinData.gatInfoInstantlyBinanceByLimit(coinName,self.limit,interval=interval)
                    priceAction = PriceAction(df,coinName,self.windowSize)
                    signal=priceAction.OnlyMarketStructureBreakSignals()
                    self.signals.loc[len(self.signals)] = [coinName,interval,signal['Date'],signal['Price'],signal['Type'],signal['Signal']] 
            
            self.signals.sort_values(by='Date',ascending=False,inplace=True)
            self.signals.to_csv('PriceAction.csv',index=False)
            time.sleep(sleepInterval)



    def PlotAllCoinsMarketStructure(self,interval="15m"):
        coinData = CoinData()
        for coinName in self.coins:
            df = coinData.gatInfoInstantlyBinanceByLimit(coinName,self.limit,interval=interval)
            priceAction = PriceAction(df,coinName,self.windowSize)
            priceAction.PlotAllCoinsMarketStructure()

    def plotOneCoinMarketStructure(self,coinName="BTCUSDT",interval="1h"):
        coinData = CoinData()
        df = coinData.gatInfoInstantlyBinanceByLimit(coinName,self.limit,interval=interval)
        priceAction = PriceAction(df,coinName,self.windowSize)
        priceAction.PlotOnlyOneCoinMarketStructure()
    



