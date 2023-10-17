from Data import CoinData
from Vizualize import CoinVizualize
from TimeSeries import TimeSeries
from PriceAction import PriceAction
from PriceActionBot import MarketStructureBot
from MachineLearning import MachineLearning
from test import Test
from OrderBook import OrderBook
cd =CoinData()
df = cd.gatInfoInstantlyBinanceByLimit("BTCUSDT",1000,"1d")
#cv = CoinVizualize(df)
#cv.plotPriceWithRSI()
#cv.plotPriceWithMACD()


ts = TimeSeries(df)
ts.TimeSeriesARIMAModel()


#ms = MarketStructureBot(5,350)
#ms.PlotAllCoinsMarketStructure()
#ms.findSignalForMarketStructure()
#ms.plotOneCoinMarketStructure("GALAUSDT","1h")

ml = MachineLearning()
#ml.volumeAndPrice()
#ml.RsiAndBB()
#ml.predictionPrice()
#ml.DeepLearning()
ml.TimeSeriesAnalyse()

#test = Test(df)
#test.position()

#ob = OrderBook()
#ob.plotOrderBook("BTCUSDT")

