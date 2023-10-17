import pandas as pd
import numpy as np




class Test:
    
    def __init__(self,coin):
        self.coin = coin
    
    def position(self):
        df = self.coin
        
        df['TR'] = pd.DataFrame([df['High'] - df['Low'],
                                    abs(df['Close'].shift(1) - df['High']),
                                    abs(df['Close'].shift(1) - df['Low'])]).T.max(axis=1)
        
        df['ATR'] = df['TR'].rolling(14).mean()
        position = "long"
        money = 500
        entryPrice = df['Open'][0]
        stop = entryPrice - df['Open'][0]/100
        profit = entryPrice + df['Open'][0]/100
        
        for i in range(len(df)):
            price = df['Open'][i]
            if position == "long":
                if df['Open'][i] > profit:
                    profit = profit + df['ATR'][i]*2
                    stop = stop + df['ATR'][i]*2
                elif df['Open'][i] <= stop:
                    print(f"{money:.3f}")      
                    ratio = df['Open'][i] / entryPrice
                    money += ((ratio * money) - money) * 10
                    money  -= money /1000
                    position = "short"
                    entryPrice = df['Open'][i]
                    stop = entryPrice + df['ATR'][i]*2
                    profit = entryPrice - df['ATR'][i]*2
                    
            if position =="short":
                if df['Open'][i] < profit:
                    profit = profit - df['ATR'][i]*2
                    stop = stop - df['ATR'][i]*2
                elif df['Open'][i] >= stop:
                    print(f"{money:.3f}")             
                    ratio = df['Open'][i]  / stop
                    money += ((ratio * money) - money) * 10
                    money  -= money /1000
                    position = "long"
                    entryPrice = df['Open'][i]
                    stop = entryPrice - df['ATR'][i]*2
                    profit = entryPrice + df['ATR'][i]*2
            
            

            
        