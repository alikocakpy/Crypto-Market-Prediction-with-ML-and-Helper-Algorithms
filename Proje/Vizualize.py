import pandas as pd
from matplotlib import pyplot as plt,dates as mdates
import mplfinance as mpf
import numpy as np

class CoinVizualize:
    coin = pd.DataFrame()
    def __init__(self,coinData):
        self.coin = pd.DataFrame(coinData)
        self.coin.index = pd.to_datetime(self.coin.index)
    
    def plotPriceWithLine(self):
        fig,ax = plt.subplots()
        ax.plot(pd.to_datetime(self.coin.index),self.coin['Close'])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=60))
        plt.show()
    
    def plotPriceWithCandle(self):
        mpf.plot(self.coin,type='candle',volume=True)
        print(self.coin)
        mpf.show()
    def plotsimple(self,windowSize):
        plt.plot(self.coin['Close'])
        plt.plot(self.coin['Close'].rolling(window=windowSize).mean())
        plt.show()

    
    def calculateRSI(self):
        delta = self.coin['Close'].diff()      
        delta = delta[1:]
        gains = delta.where(delta>0,0)
        losses = -delta.where(delta<0,0)
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        rs = avg_gains/avg_losses
        rsi = 100 - (100/(1+rs))
        return rsi
    
    def plotPriceWithRSI(self):
        rsi = self.calculateRSI()
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        fig.suptitle("PRICE & RSI")
        ax1.plot(self.coin['Close'])
        ax2.plot(rsi)
        ax2.axhline(y = 70,color='red',linestyle='--')
        ax2.axhline(y=30,color='green',linestyle='--')
        plt.show()
    
    
    def calculateBollingerBand(self):
        df  = pd.DataFrame()
        df['Close'] = self.coin['Close']
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['SMA']+df['StdDev'].std()*2
        df['Lower Band'] = df['SMA'] - df['StdDev'].std()*2
        return df

    def ploPriceWithBollingerBand(self):
        df = self.calculateBollingerBand()
        plt.title("Bollinger Band")
        plt.plot(df['Close'],label='Price')
        plt.plot(df['SMA'],label='SMA-20')
        plt.plot(df['Upper Band'],label='Upper Band')
        plt.plot(df['Lower Band'],label='Lower Band')
        plt.legend()
        plt.show()
    def plotPriceWithMACD(self):
        df = self.coin
        df['12EMA'] = df['Close'].ewm(span=12).mean()
        df['26EMA'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['12EMA'] - df['26EMA']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Histogram'] = df['MACD'] - df['Signal']

        df['Positive Hist'] = df['Histogram'].map(lambda x : x if x>0 else 0 )
        df['Negative Hist'] = df['Histogram'].map(lambda x : x if x < 0 else 0)

        macd_plot = [
            mpf.make_addplot((df['MACD']), color='#606060',panel=1,ylabel='MACD', secondary_y = False),
            mpf.make_addplot((df['Signal']),color='#1f77b4', panel=1, secondary_y=False),
            mpf.make_addplot((df['Positive Hist']), type='bar', color='#4dc790', panel=1),
            mpf.make_addplot((df['Negative Hist']), type='bar', color='#fd6b6c', panel=1)
        ]
        mpf.plot(df,type='candle',volume=False,addplot = macd_plot,panel_ratios=(4,4),title="PRICE & MACD")

        
    def plotATRStrategy(self):
        df = self.coin
        df['TR'] = pd.DataFrame([df['High'] - df['Low'],
                                    abs(df['Close'].shift(1) - df['High']),
                                    abs(df['Close'].shift(1) - df['Low'])]).T.max(axis=1)
        
        df['ATR'] = df['TR'].rolling(14).mean()

        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(df['Close'])
        ax2.scatter(df.index,df['ATR'],c=df['ATR'],s=3)
        ax2.plot(df['ATR'].rolling(window=14).mean())
        
        plt.show()

    def plotPriceWithEWM(self):
        df = self.coin
        plt.plot(df['Close'],label="Price")
        ewm = pd.DataFrame(df['Close'])

        plt.plot(ewm.ewm(span=14).mean(),label="EWM")
        plt.plot(ewm.rolling(14).mean(),label="EMA")
        plt.legend()
        plt.show()
        