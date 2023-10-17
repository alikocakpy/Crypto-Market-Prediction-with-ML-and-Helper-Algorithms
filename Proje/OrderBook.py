from Data import CoinData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf


#This show us waited order. So it mean support and resistance
class OrderBook:
    def __init__(self):
        pass
    
    def plotOrderBook(self,coinName="BTCUSDT"):
        coin = CoinData()
        orders = coin.getOrderbook(coinName,5000)
        orders = pd.DataFrame(orders)
        price = coin.gatInfoInstantlyBinanceByLimit("BTCUSDT",250,"15m")
        print(price)
        bids = orders['bids']
        asks = orders['asks']
        bids = pd.DataFrame(bids.tolist())
        asks = pd.DataFrame(asks.tolist())

        bids.columns = ["Price","Amount"]
        asks.columns = ["Price","Amount"]
        bids = bids.astype(float)
        asks = asks.astype(float)


        scale = 30
        bottom = (bids['Price'][0] // 30) * 30
        top = bottom - scale
        buyers = []
        sum = 0
        for i in range(len(bids)):
            if bids['Price'][i]>=top:
                sum += bids['Amount'][i]
            if bids['Price'][i] < top :
                buyers.append([top,sum])
                sum = 0
                top -= scale


        bottom = (asks['Price'][0] // 30) * 30
        top = bottom + scale
        sum = 0
        sellers = []
        for i in range(len(asks)):
            if asks['Price'][i]<=top:
                sum += asks['Amount'][i]
            if asks['Price'][i] > top :
                sellers.append([top,sum])
                sum = 0
                top += scale


        orders = buyers + sellers
        orders = pd.DataFrame(orders)
        orders.columns = ["Price","Amount"]
        scale = []
        amount = orders["Amount"]
        min = amount.min()
        max= amount.max()
        for i in range(len(orders)):
            scale.append((amount[i]-min) / (max - min))

        scale = pd.DataFrame(scale)
        orders['Scale'] = scale

        mean = orders['Amount'].mean()

        fig, ax = plt.subplots()

        for i in range(len(price)):
                    date = price.index[i]
                    open_price = price["Open"][i]
                    close_price = price["Close"][i]
                    high = price["High"][i]
                    low = price["Low"][i]
                    if open_price > close_price:
                        color = 'r'
                    else:
                        color = 'g'
                    ax.plot([date, date], [low, high], 'k-', linewidth=1)
                    ax.plot([date, date], [open_price, close_price], color=color, linewidth=2)

        for i in range(len(orders)):
            if(mean >= orders['Amount'][i]):
                ax.hlines(orders["Price"][i],xmin=price.index[-4],xmax=price.index[-1],linestyles="--",linewidth=orders['Scale'][i]*5,color="r")
            else:
                ax.hlines(orders["Price"][i],xmin=price.index[-4],xmax=price.index[-1],linestyles="-",linewidth=orders['Scale'][i]*5,color="r")
        plt.show()