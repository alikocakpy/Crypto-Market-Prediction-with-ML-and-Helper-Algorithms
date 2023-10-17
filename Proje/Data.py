import pandas as pd
from datetime import timedelta 
from binance.client import Client
class CoinData:
    api_key = "Api_Key"
    secret_key = "Secret_Key"
    data = pd.DataFrame()


    #Buraya ilgili coin ile alakalı değişken ekle, bin sayfasından değeri alıp otomatik çekelim
    #self,coinName
    def __init__(self):
        self.readExcel()
    
        
    #Access data information with getter-setter
    
    def getData(self):
        return self.data
    

    #Handle the data information
    #Take Info for writing Excel
    def getInfoBinanceByDate(self,coinName="BTCUSDT",str_date="2023-01-01 03:00:00",interval = "1h"):
        self.data.drop(index=self.data.index,inplace=True)
        client = Client(self.api_key,self.secret_key)
        self.data = client.get_historical_klines(coinName,interval=interval,start_str=str_date)
        client.close_connection()
        self.data = pd.DataFrame(self.data)
        self.dataPreperation()
        self.writeExcel()
    
    def getInfoBinanceByLimit(self,coinName="BTCUSDT",limit="Integer",interval = "1h"):
        self.data.drop(index=self.data.index,inplace=True)
        client = Client(self.api_key,self.secret_key)
        self.data = client.get_historical_klines(coinName,interval=interval ,limit=limit)
        client.close_connection()
        self.data = pd.DataFrame(self.data)
        self.dataPreperation()
        self.writeExcel()

    #Take Info without writing only return info
    def getInfoInstantlyBinanceByDate(self,coinName="BTCUSDT",str_date="2023-01-01 03:00:00",interval = "1h"):
        self.data.drop(index=self.data.index,inplace=True)
        client = Client(self.api_key,self.secret_key)
        self.data = client.get_historical_klines(coinName,interval=interval,start_str=str_date)
        client.close_connection()
        self.data = pd.DataFrame(self.data)
        self.dataPreperation()
        return self.data.iloc[:,:5]
        
    def gatInfoInstantlyBinanceByLimit(self,coinName="BTCUSDT",limit="Integer",interval = "1h"):
        self.data.drop(index=self.data.index,inplace=True)
        client = Client(self.api_key,self.secret_key)
        self.data = client.get_historical_klines(coinName,interval=interval ,limit=limit)
        client.close_connection()
        self.data = pd.DataFrame(self.data)
        self.dataPreperation()
        return self.data.iloc[:,:5]

    
    def getOrderbook(self,coinName="BTCUSDT",limit=5000):
        client = Client(self.api_key, self.secret_key)
        orders = client.get_order_book(symbol = 'BTCUSDT',limit=5000)
        return orders
    
    def writeExcel(self):
        self.data.to_csv('BTCUSDT.csv')


    def readExcel(self):
        self.data = pd.read_csv('BTCUSDT.csv')
        self.data.set_index(['Time'],inplace=True)

    #Setting Data Form
    #Fixing data and prepare useful form
    def dataPreperation(self):
        self.data.columns = ["Time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
        self.data['Time'] = pd.to_datetime(self.data['Time'],unit="ms" )
        self.data['Time'] += timedelta(hours=3) # without that 3 hours before
        self.data.set_index(['Time'],inplace=True)
        self.data = self.data.astype(float)