import numpy as np
import pandas as pd
from matplotlib import pyplot as plt,dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

class TimeSeries:
    coin = pd.DataFrame()
    def __init__(self,coin):
        self.coin = pd.DataFrame(coin)
 
    
    def TimeSeriesAnalize(self):
        df = self.coin
        result = seasonal_decompose(df['Close'],model="multiplicative",period=30)
        df = pd.DataFrame()
        df = pd.concat([pd.Series(result.trend.values),pd.Series(result.seasonal.values),pd.Series(result.observed.values),pd.Series(result.resid.values)],axis=1)
        df.columns = ["Trend","Seasonal","Observed","Residual"]
        df = df.fillna(0)
        return df

        """fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=60))
        ax1.plot(result.trend,label="Trend 1")
        ax1.legend()
        ax2.plot(result.seasonal,label="Seasonal")
        ax2.legend()
        ax3.plot(result.observed,label="Observed")
        ax3.legend()
        ax4.plot(result.resid,label="Residual")
        ax4.legend()
        plt.show()
"""
    def TimeSeriesARIMAModel(self):
        
        df = self.coin
        df.index = pd.to_datetime(df.index)
        model = sm.tsa.statespace.SARIMAX(df['Close'],order=(0,1,0), seasonal_order=(1,1,1,12))
        results = model.fit()
        
        from pandas.tseries.offsets import DateOffset
        future_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,30) ]
        future_dates_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
        future_df = pd.concat([df,future_dates_df])
        future_df['forecast'] = results.predict(start = len(df), end = len(df)+30, dynamic= True)  
        future_df[['Close', 'forecast']].plot(figsize=(12, 8)) 
        plt.show()