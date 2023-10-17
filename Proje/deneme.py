""""import tkinter as tk

def button_clicked():
    print("hello world")

window = tk.Tk()

button = tk.Button(window,text="Click Me",command=button_clicked,width=100,height=100)
button.pack()
window.mainloop()

print(help(tk))"""
import pandas as pd
import matplotlib.pyplot as plt
class asd:
    coin = pd.DataFrame()
    def __init__(self,coin):
        self.coin = coin
    
    def plotTest(self):
        df = self.coin
        x  =[df.index[10],df.index[300]]
        y = [df['Close'][10],df['Close'][300]]
        plt.plot(df['Close'])
        plt.plot(x,y,ls='-')
        plt.show()
    
    def findHillDepth(self):
        df = self.coin
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma50'] = df['Close'].rolling(50).mean()
        df['ma100'] = df['Close'].rolling(100).mean()

        # plot the BTC price data and moving averages
        plt.plot(df['Close'] , label='BTC Price')
        plt.plot(df['ma20'], label='MA 20')
        plt.plot(df['ma50'], label='MA 50')
        plt.plot(df['ma100'], label='MA 100')
        

        # determine the trend based on the position of the moving averages
        if df['ma20'].iloc[-1] > df['ma50'].iloc[-1] > df['ma100'].iloc[-1]:
            trend = 'Uptrend'
        elif df['ma20'].iloc[-1] < df['ma50'].iloc[-1] < df['ma100'].iloc[-1]:
            trend = 'Downtrend'
        else:
            trend = 'No Clear Trend'

        # set chart properties
        plt.title('BTC Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()

        # print the trend
        print('The current trend is:', trend)

        # display the chart
        plt.show()

        