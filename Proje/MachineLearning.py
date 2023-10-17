from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import metrics
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import timedelta,datetime
from Data import CoinData
from Vizualize import CoinVizualize
import mplfinance as mpf
import talib

class MachineLearning:
    def __init__(self):
        cd = CoinData()
        self.data = cd.gatInfoInstantlyBinanceByLimit("BTCUSDT",interval = "1h",limit=150)

    def volumeAndPrice(self):
        #Data settings
        data = pd.DataFrame(self.data)
        data = data.astype(float)
        data['Return'] = data['Close'] - data['Close'].shift(1)
        data = data.dropna()



        # We found more effective columns upon the result
        """
        from sklearn.feature_selection import RFE

        column = data.drop(['Return'], axis=1)  # X variables
        result = data['Return'].values  # Y variable
        result = np.reshape(result,(-1,1))


        x_train, x_test, y_train, y_test = train_test_split(
            column, result, test_size=0.33, random_state=0)



        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=5)  # Select top k features
        rfe.fit(x_train, y_train)

        selected_features = pd.Series(rfe.support_, index=column.columns)[rfe.support_]
        print(selected_features)

        """


        #Now we have open high low close columns. This is almost same, all of them show price movements so I want to make dimension reduction
        from sklearn.decomposition import PCA

        prices = data[["Open","Close","High","Low"]]
        mms = MinMaxScaler()
        prices = mms.fit_transform(prices)
        pca = PCA(n_components=1)
        reduced_data = pca.fit_transform(prices)




        df = pd.DataFrame()
        df['Time'] = data.index.values
        df['Volume'] = data['Volume'].values
        df['Return'] = data['Return'].values
        df['Close'] = data['Close'].values
        df['Close'] = df['Close'].shift(1)
        df['Reduced Data'] = reduced_data.flatten()
        df = df.dropna()
        df.set_index('Time',inplace=True)


        y = df['Return']
        x = df.drop('Return',axis=1)


        size = int(0.66 * len(df))
        x_train = x.iloc[0:size]
        x_test = x.iloc[size:]
        y_train = y.iloc[:size]
        y_test = y.iloc[size:]

        from sklearn.preprocessing import PolynomialFeatures

        poly_transform = PolynomialFeatures(degree=2)
        X_poly = poly_transform.fit_transform(x_train)

        poly_regression = LinearRegression()
        poly_regression.fit(X_poly,y_train)


        x_test_poly = poly_transform.transform(x_test)
        y_pred = poly_regression.predict(x_test_poly)
        
        print("\nNext day prediction : ",(self.data['Close'][-1] - y_pred[-1]))

        mse = mean_absolute_error(y_test, y_pred)
        print("Mean Squared Eror: ",mse)

        y_pred = (y_pred > 0.5)
        y_test = (y_test > 0.5)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        print("Confusion Matrix : \n",cm)
    

    def RsiAndBB(self):
        self.cv = CoinVizualize(self.data)
        rsi = self.cv.calculateRSI()
        bb = self.cv.calculateBollingerBand()
        df = pd.concat([bb,rsi],axis=1)
        df = df.dropna()
        df.columns = ["Close","SMA","StdDev","Upper Band","Lower Band","RSI"]
        #df['Return'] = df["Close"]-df['Close'].shift(1)
        df = df.dropna()

        

        mms = MinMaxScaler()
        df = pd.DataFrame(mms.fit_transform(df))
        df.columns = ["Close","SMA","StdDev","Upper Band","Lower Band","RSI"]
        df['Close'] = df['Close'].shift(1)
        df = df.dropna()

        result = df['Close']
        columns = df.drop('Close',axis=1)

        size = int(0.66 * len(df))
        x_train = columns.iloc[0:size]
        x_test = columns.iloc[size:]
        y_train = result.iloc[:size]
        y_test = result.iloc[size:]
        lr = LinearRegression()
        lr.fit(x_train,y_train)
        y_pred = lr.predict(x_test)

        print()
        print("Algorithm Squared and Absolute Error : ")
        print(mean_squared_error(y_test,y_pred))
        print(mean_absolute_error(y_test,y_pred))
        print()

        y_pred = (y_pred > 0.5)
        y_test = (y_test > 0.5)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        print("Confusion Matrix : \n",cm)
    

    
    def predictionPrice(self):
        data = self.data

        # Veri hazırlığı
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values

        # Lineer regresyon modeli oluşturma ve eğitme
        model = LinearRegression()
        model.fit(X, y)

        # Tahmin yapma
        future_X = np.arange(len(data), len(data) + 50).reshape(-1, 1)
        future_y = model.predict(future_X)

        # Tahmin sonuçlarını DataFrame'e dönüştürme
        future_timestamps = pd.date_range(
            start=data.index[-1], periods=51, freq='4H')[1:]
        future_data = pd.DataFrame({'Open': future_y[:-1], 'High': future_y[:-1], 'Low': future_y[:-1], 'Close': future_y[:-1]},
                                index=future_timestamps[:-1])

        # Tüm verileri birleştirme
        all_data = pd.concat([data, future_data])

        # Bollinger Bantlarını hesaplama
        all_data['middle_band'] = talib.SMA(all_data['Close'], timeperiod=20)
        all_data['std_dev'] = talib.STDDEV(all_data['Close'], timeperiod=20)
        all_data['upper_band'] = all_data['middle_band'] + 2 * all_data['std_dev']
        all_data['lower_band'] = all_data['middle_band'] - 2 * all_data['std_dev']

        # RSI hesaplama
        all_data['rsi'] = talib.RSI(all_data['Close'], timeperiod=14)

        # Görselleştirme
        add_plot = mpf.make_addplot(
            all_data[['upper_band', 'middle_band', 'lower_band']])
        add_plot2 = mpf.make_addplot(all_data['rsi'], panel=1, color='purple')
        mpf.plot(all_data, type='candle', volume=False, style='yahoo',
                title='Bitcoin (BTC/USDT) 4 Saatlik Mum Grafiği ve Tahmin', addplot=[add_plot, add_plot2])

        # Fiyat değişim tahminlerini hesaplama ve yazdırma
        last_price = data['Close'][-1]
        price_changes = future_data['Close'] - last_price
        print("Fiyat Değişim Tahminleri:")
        for timestamp, change in zip(future_data.index, price_changes):
            if change > 0:
                direction = "yukarı"
            else:
                direction = "aşağı"
            change = abs(change)
            print(f"{timestamp}: Fiyat {change:.4f} USDT {direction} yönde tahmin ediliyor.")

        # Mum dönüş noktalarını bulma
        turning_points = future_data[(price_changes.shift(
            1) > 0) & (price_changes.shift(-1) < 0)]
        print("Tahmin Edilen Mum Dönüş Noktaları:")
        if not turning_points.empty:
            print(turning_points)
        else:
            print("Mum dönüş noktaları bulunamadı.")
    
    def DeepLearning(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Activation
        data = self.data
        self.cv = CoinVizualize(self.data)
        rsi = self.cv.calculateRSI()
        ema = data['Close'].rolling(14).mean()
        volume = data['Volume']
        close = data['Close']
        df = pd.concat([rsi,volume,ema,close],axis=1)
        df = df.dropna()
        mms = MinMaxScaler()
        df = mms.fit_transform(df)
        df = pd.DataFrame(df)
        df.columns = ["RSI","Volume","EMA","Close"]
        df['Close'] = df['Close'].shift(1)
        df = df.dropna()
        
        result = df['Close']
        columns = df.drop('Close',axis=1)


        size = int(0.66 * len(df))
        x_train = columns.iloc[0:size]
        x_test = columns.iloc[size:]
        y_train = result.iloc[:size]
        y_test = result.iloc[size:]
        
        model = Sequential()
        model.add(Dense(6,activation="relu",input_dim=x_train.shape[1]))
        model.add(Dense(6,activation="relu"))
        model.add(Dense(1,activation="relu"))
        model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
        y_pred = model.predict(x_test)


        mae = mean_absolute_error(y_test, y_pred)
        print('Algorithm Mean Absolute Error:', mae)

    def TimeSeriesAnalyse(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Activation
        from TimeSeries import TimeSeries
        ts = TimeSeries(self.data)
        df = ts.TimeSeriesAnalize()
        a = range(0,len(df))
        df['Index'] = a
        df.set_index("Index",inplace=True)

        mms = MinMaxScaler()
        df = pd.DataFrame(mms.fit_transform(df))
        df.columns = ["Trend","Seasonal","Observed","Residual"]

        result = df['Observed']
        columns = df[['Trend', 'Seasonal', 'Residual']]


        size = int(0.66 * len(df))
        x_train = columns.iloc[0:size]
        x_test = columns.iloc[size:]
        y_train = result.iloc[0:size]
        y_test = result.iloc[size:]

        model = Sequential()
        model.add(Dense(6,activation="sigmoid",input_dim=x_train.shape[1]))
        model.add(Dense(6,activation="sigmoid"))
        model.add(Dense(1,activation="sigmoid"))
        model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
        y_pred = model.predict(x_test)


        mae = mean_absolute_error(y_test, y_pred)
        
        print('Algorithm Mean Absolute Error:', mae)




        