import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema

class PriceAction:
    coin = pd.DataFrame()
    priceSignal = pd.DataFrame()
    def __init__(self,coin = pd.DataFrame(),coinName="BTCUSDT",windowSize=10):
        if len(coin) == 0:
            print("This Dataframe is EMPTY!!!")
            return 0
        self.coin = coin
        self.coinName=coinName
        self.windowSize = windowSize
        self.signal = []
    
         
    def OnlyMarketStructureBreakSignals(self):
        prc = self.PriceActionswingMovement()
        signal = prc.loc[prc[prc['Signal'].str.findall("1").astype(bool)].index[-1]] 
        return signal

    def PlotAllCoinsMarketStructure(self):
        prc = self.PriceActionswingMovement()
        self.plotSwingMovement(prc)


    def PlotOnlyOneCoinMarketStructure(self):
        prc = self.PriceActionswingMovement()
        self.plotSwingMovement(prc)

    
    #filter repeat themself movement
    def shrapeninggMinMaxPoints(self,minmaxpoints):
        flag = False
        arr = pd.DataFrame(columns=['Date','Price'])
        minmaxpoints = pd.DataFrame(minmaxpoints)
        prev = minmaxpoints['Price'][0]
        arr.loc[0] = [minmaxpoints.index[0],minmaxpoints['Price'][0]]
        for i in range(1,len(minmaxpoints)):
            if prev < minmaxpoints['Price'][i]:
                if flag == True:
                    arr.loc[len(arr)] = [minmaxpoints.index[i-1],minmaxpoints['Price'][i-1]]
                prev = minmaxpoints['Price'][i]
                flag = False
            else:
                if flag == False:
                    arr.loc[len(arr)] = [minmaxpoints.index[i-1],minmaxpoints['Price'][i-1]]
                prev = minmaxpoints['Price'][i]
                flag = True
        arr.loc[len(arr)] = [self.coin.index[-1],self.coin['High'][-1]]
        return arr

    #Finding low and high points
    def PriceActionswingMovement(self):
        data = self.coin
        max_idx = argrelextrema(data['High'].values, np.greater, order=self.windowSize)[0]
        min_idx = argrelextrema(data['Low'].values, np.less, order=self.windowSize)[0]
        max_points = data['High'][max_idx]
        min_points = data['Low'][min_idx]
        min_max = pd.DataFrame(pd.concat([max_points,min_points])).sort_index()
        min_max.columns = ['Price']
        extremePoints = self.shrapeninggMinMaxPoints(min_max)
        min_max_points = pd.DataFrame(extremePoints)
        min_max_points.set_index(['Date'],inplace=True)
        prc = self.findHighLowPoints(data,min_max_points)
        return pd.DataFrame(prc)

            
    # creating dataframe for HH-LL definition
    def findHighLowPoints(self,data,min_max_points):
        data = pd.DataFrame(data)
        min_max_points = pd.DataFrame(min_max_points)
        
        prc= pd.DataFrame(columns=['Date','Price','Type','Signal'])
        prc.loc[0] = [data.index[0],data['High'][0],'HH',0]
        prc.loc[1] = [data.index[0],data['Low'][0],'LL',0]
        prc.loc[2] = [data.index[0],max(data['Close'][0],data['Open'][1]),'LH',0]
        prc.loc[3] = [data.index[0],min(data['Close'][0],data['Open'][1]),'HL',0]
        prc.loc[4] = [data.index[1],data['High'][1],'HH',0]
        prc.loc[5] = [data.index[1],data['Low'][1],'LL',0]
        prc.loc[6] = [data.index[1],max(data['Close'][1],data['Open'][1]),'LH',0]
        prc.loc[7] = [data.index[1],min(data['Close'][1],data['Open'][1]),'HL',0]


        prev_price = prc['Price'][7]
        for i in range(len(min_max_points)):
            price = min_max_points['Price'][i]
            if price > prev_price:
                if price > prc['Price'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] and (prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] > prc['Date'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]) and\
                    ((prc['Date'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]) or \
                    (prc['Date'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]]) or \
                    (prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]])):
                        prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"HH","1"]
                        prc['Signal'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] = "-1"
                        #after the msb I give the break point 1 and relative point -1.

                elif prc['Type'][len(prc)-1] == "HH" and price >= prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:
                    prc.loc[len(prc)-1] = [min_max_points.index[i],min_max_points['Price'][i],"HH","0"]
                
                elif prc['Type'][len(prc)-1] == "HL" and price >= prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"HH","0"]
                elif prc['Type'][len(prc)-1] == "HL" and price < prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"LH","0"]

                elif prc['Type'][len(prc)-1] == "LH" and price >= prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:   
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"HH","0"]
                elif prc['Type'][len(prc)-1] == "LH" and price < prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:   
                    prc.loc[len(prc)-1] =[min_max_points.index[i],min_max_points['Price'][i],"LH","0"]

                elif prc['Type'][len(prc)-1] == "LL" and price < prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"LH","0"]
                elif prc['Type'][len(prc)-1] == "LL" and price >= prc['Price'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]]: #add lh
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"HH","0"]
            else:
                if price < prc['Price'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] and (prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] > prc['Date'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]) and\
                    (prc['Date'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-1]] or \
                    prc['Date'][prc[prc['Type'].str.findall("HH").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]] or \
                    prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-2]] < prc['Date'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] < prc['Date'][prc[prc['Type'].str.findall("LH").astype(bool)].index[-1]]    ):
                        prc.loc[len(prc)] =[min_max_points.index[i],min_max_points['Price'][i],"LL","1"]
                        prc['Signal'][prc[prc['Type'].str.findall("HL").astype(bool)].index[-1]] = "-1"
                        #as the same of the up trend

                elif prc['Type'][len(prc)-1] == "LL" and price <= prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)-1] =[min_max_points.index[i],min_max_points['Price'][i],"LL","0"]

                elif prc['Type'][len(prc)-1] == "HL" and price <= prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] =[min_max_points.index[i],min_max_points['Price'][i],"LL","0"]
                elif prc['Type'][len(prc)-1] == "HL" and price > prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)-1] =[min_max_points.index[i],min_max_points['Price'][i],"HL","0"]

                elif prc['Type'][len(prc)-1] == "LH" and price > prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"HL","0"]
                elif prc['Type'][len(prc)-1] == "LH" and price <= prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] =[min_max_points.index[i],min_max_points['Price'][i],"LL","0"]                

                elif prc['Type'][len(prc)-1] == "HH" and price <= prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]: #add hl
                    prc.loc[len(prc)] = [min_max_points.index[i],min_max_points['Price'][i],"LL","0"]
                elif prc['Type'][len(prc)-1] == "HH" and price > prc['Price'][prc[prc['Type'].str.findall("LL").astype(bool)].index[-1]]:
                    prc.loc[len(prc)] =[min_max_points.index[i],min_max_points['Price'][i],"HL","0"]

            prev_price = price
        return pd.DataFrame(prc.iloc[8:,:])

   
    def plotSwingMovement(self,prc):
        prc = pd.DataFrame(prc)
        df = self.coin
        fig, ax = plt.subplots(figsize=(16,8))
        candlestick_data = zip(df.index, df['Open'], df['Close'], df['High'], df['Low'])

        for candle in candlestick_data:
            date = candle[0]
            open_price = candle[1]
            close_price = candle[2]
            high = candle[3]
            low = candle[4]
            if open_price > close_price:
                color = 'r'
            else:
                color = 'g'
            ax.plot([date, date], [low, high], 'k-', linewidth=1)
            ax.plot([date, date], [open_price, close_price], color=color, linewidth=2)

        # Set the title and axis labels
        plt.title(self.coinName)
        plt.xlabel('Date')
        plt.ylabel('Price')
        prc.set_index('Date',inplace=True)
        plt.plot(prc.index,prc['Price'],ls="--",linewidth=1,color="black")
        
        for i in range(0,len(prc)):
            plt.text(prc.index[i],prc['Price'][i],(prc['Type'][i],prc['Price'][i]),fontsize = 8,weight="bold")
            if prc['Signal'][i] == "-1":
                plt.hlines(prc['Price'][i],xmin=prc.index[i],xmax=prc.index[i+2],linestyles="--",linewidth=2,color="r")
            
        plt.xticks(rotation=45)
        plt.show()

