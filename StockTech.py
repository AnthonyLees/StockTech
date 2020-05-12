#########CITATIONS###########
#Inspiration and calculations/formula for how to determine returns and volatility of each
#stock for stock allocation: https://blog.quantinsti.com/calculating-covariance-matrix-portfolio-variance/
#Inspiration and theory explanation for stock predictions:
#https://towardsdatascience.com/walking-through-support-vector-regression-and-lstms-with-stock-price-prediction-45e11b620650
#Info on which data processing to use and how:
#https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
#https://scikit-learn.org/stable/modules/preprocessing.html
#Formula for MACD:
#https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
#Formula for SMA and EMA:
#https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages
#Formula for Williams%R:
#https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r
#Formula for Ultimate Oscillator:
#https://school.stockcharts.com/doku.php?id=technical_indicators:ultimate_oscillator
#Info on how to use financial ratio API from:
#https://financialmodelingprep.com/developer/docs/
#User interface design taken from:
#https://www.cs.cmu.edu/~112/notes/notes-animations-part2.html
##############################

import pandas as pd
import pandas_datareader.data as web
import datetime
import time
import numpy as np 
import matplotlib.pyplot as plt
import string, copy, random, math
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.dates as mdates
from matplotlib import style
style.use("ggplot")
from datetime import date
from datetime import timedelta
from sklearn.svm import SVR
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests
from urllib.request import urlopen
import bs4
from bs4 import BeautifulSoup
from cmu_112_graphics import *

#StockTech

class MyModalApp(ModalApp):
    def appStarted(app):
        app.splashScreenMode=SplashScreenMode()
        app.hubMode=HubMode()
        app.stockPredictionMode=StockPredictionMode()
        app.stockAllocationMode=StockAllocationMode()
        app.stockRatioAnalysisMode=StockRatioAnalysisMode()
        app.liveStockPrices=LiveStockPrices()
        app.setActiveMode(app.splashScreenMode)
        app.timerDelay=50

class SplashScreenMode(Mode):
    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height,\
        fill="papaya whip")
        canvas.create_text(mode.width/2, mode.height/8,\
        text="StockTech", font="system 30 bold")
        canvas.create_text(mode.width/2, mode.height/2,\
        text="A stock technical analysis tool", font="system 20 bold")
        canvas.create_text(mode.width/2, mode.height*7/8,\
        text="Press any key to begin", font="system 20 bold")

    def keyPressed(mode, event):
        mode.app.setActiveMode(mode.app.hubMode)

class HubMode(Mode):
    def mousePressed(mode, event):
        if mode.width*2/26<=event.x<=mode.width*6/26 and\
        mode.height*5/16<=event.y<=mode.height*13/16:
            mode.app.setActiveMode(mode.app.liveStockPrices)
        elif mode.width*8/26<=event.x<=mode.width*12/26 and\
        mode.height*5/16<=event.y<=mode.height*13/16:
            mode.app.setActiveMode(mode.app.stockRatioAnalysisMode)
        elif mode.width*14/26<=event.x<=mode.width*18/26 and\
        mode.height*5/16<=event.y<=mode.height*13/16:
            mode.app.setActiveMode(mode.app.stockAllocationMode)
        elif mode.width*20/26<=event.x<=mode.width*24/26 and\
        mode.height*5/16<=event.y<=mode.height*13/16:
            mode.app.setActiveMode(mode.app.stockPredictionMode)

    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height, fill="black")
        canvas.create_text(mode.width/2, mode.height*3/16,
        text="Pick a feature", font="verdana 20 bold", fill="indian red")
        canvas.create_rectangle(mode.width*2/26, mode.height*5/16,\
        mode.width*6/26, mode.height*13/16, fill="midnight blue")
        canvas.create_rectangle(mode.width*8/26, mode.height*5/16,\
        mode.width*12/26, mode.height*13/16, fill="midnight blue")
        canvas.create_rectangle(mode.width*14/26, mode.height*5/16,\
        mode.width*18/26, mode.height*13/16, fill="midnight blue")
        canvas.create_rectangle(mode.width*20/26, mode.height*5/16,\
        mode.width*24/26, mode.height*13/16, fill="midnight blue")
        canvas.create_text(mode.width*4/26, mode.height*9/16,\
        text="Live Stock Prices", font="verdana 16 bold", fill="LightSkyBlue1")
        canvas.create_text(mode.width*10/26, mode.height*9/16,\
        text="Ratio Analysis", font="verdana 16 bold", fill="LightSkyBlue1")
        canvas.create_text(mode.width*16/26, mode.height*9/16,\
        text="Stock Allocation", font="verdana 16 bold", fill="LightSkyBlue1")
        canvas.create_text(mode.width*22/26, mode.height*9/16,\
        text="Stock Prediction", font="verdana 16 bold", fill="LightSkyBlue1")

class StockPredictionMode(Mode):
    def appStarted(mode):
        mode.stockTicker="Enter stock here"
        mode.daysSelected=None
        mode.modelSelected=None
    
    def mousePressed(mode, event):
        if mode.width*32/64<=event.x<=mode.width*42/64 and\
        mode.height*23/60<=event.y<=mode.height*25/60:
            enteredMessage=mode.getUserInput("Enter stock ticker")
            if (enteredMessage!= None):
                mode.stockTicker=enteredMessage
            else:
                mode.stockTicker="Enter stock here"
        if mode.width*18/64<=event.x<=mode.width*22/64 and\
        mode.height*35/60<=event.y<=mode.height*37/60:
            if mode.daysSelected==7:
                mode.daysSelected=None
            else:
                mode.daysSelected=7
        if mode.width*25/64<=event.x<=mode.width*29/64 and\
        mode.height*35/60<=event.y<=mode.height*37/60:
            if mode.daysSelected==14:
                mode.daysSelected=None
            else:
                mode.daysSelected=14
        if mode.width*32/64<=event.x<=mode.width*36/64 and\
        mode.height*35/60<=event.y<=mode.height*37/60:
            if mode.daysSelected==21:
                mode.daysSelected=None
            else:
                mode.daysSelected=21
        if mode.width*39/64<=event.x<=mode.width*43/64 and\
        mode.height*35/60<=event.y<=mode.height*37/60:
            if mode.daysSelected==30:
                mode.daysSelected=None
            else:
                mode.daysSelected=30
        if mode.width*17/64<=event.x<=mode.width*30/64 and\
        mode.height*50/64<=event.y<=mode.height*52/64:
            if mode.modelSelected==1:
                mode.modelSelected=None
            else:
                mode.modelSelected=1
        if mode.width*63/128<=event.x<=mode.width*89/128 and\
        mode.height*50/64<=event.y<=mode.height*52/64:
            if mode.modelSelected==2:
                mode.modelSelected=None
            else:
                mode.modelSelected=2
        if mode.width*27/64<=event.x<=mode.width*37/64 and\
        mode.height*113/128<=event.y<=mode.height*58/64:
            if mode.modelSelected==1:
                mode.createSupportVectorMachineModel(mode.stockTicker,\
                "2020-01-02", date.today(), mode.daysSelected)
            elif mode.modelSelected==2:
                mode.createRandomForestRegressorModel(mode.stockTicker,\
                "2020-01-02", date.today(), mode.daysSelected)
        if 0<=event.x<=mode.width*3/24 and mode.height*37/40<=event.y<=\
        mode.height:
            mode.app.setActiveMode(mode.app.hubMode)

    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height,\
        fill="DarkSlategray1")
        canvas.create_text(mode.width/2, mode.height/10,\
        text="Stock Prediction", font="verdana 20 bold")
        canvas.create_text(mode.width/2, mode.height*3/10,\
        text="Pick a stock for future prediction (enter ticker)",\
        font="verdana 16 bold")
        canvas.create_text(mode.width*23/64, mode.height*2/5,\
        text=f"Current Stock:", font="verdana 16 bold")
        canvas.create_text(mode.width*37/64, mode.height*2/5,
        text=f"{mode.stockTicker}", font="verdana 16 bold underline")
        canvas.create_text(mode.width*30/64, mode.height/2,
        text="Pick a timeframe for future stock prices",\
        font="verdana 16 bold")
        if mode.daysSelected==7:
            canvas.create_text(mode.width*20/64, mode.height*3/5,\
            text="1 week", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*20/64, mode.height*3/5,\
            text="1 week", font="verdana 14 bold")
        if mode.daysSelected==14:
            canvas.create_text(mode.width*27/64, mode.height*3/5,\
            text="2 weeks", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*27/64, mode.height*3/5,\
            text="2 weeks", font="verdana 14 bold")
        if mode.daysSelected==21:   
            canvas.create_text(mode.width*34/64, mode.height*3/5,\
            text="3 weeks", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*34/64, mode.height*3/5,\
            text="3 weeks", font="verdana 14 bold")
        if mode.daysSelected==30:
            canvas.create_text(mode.width*41/64, mode.height*3/5,\
            text="1 month", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*41/64, mode.height*3/5,\
            text="1 month", font="verdana 14 bold")
        canvas.create_text(mode.width*27/64, mode.height*7/10,\
        text="Pick a machine learning model", font="verdana 16 bold")
        if mode.modelSelected==1:
            canvas.create_text(mode.width*24/64, mode.height*8/10,\
            text="support vector regression", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*24/64, mode.height*8/10,\
            text="support vector regression", font="verdana 14 bold")
        if mode.modelSelected==2:
            canvas.create_text(mode.width*38/64, mode.height*8/10,\
            text="random forest regression", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*38/64, mode.height*8/10,\
            text="random forest regression", font="verdana 14 bold")
        canvas.create_text(mode.width*1/2, mode.height*9/10,\
        text="Create prediction", font="verdana 16 bold underline")
        canvas.create_text(mode.width*1/16, mode.height*19/20,
        text="Go back", font="verdana 20 bold")

    def adjustedStartDate(mode, startDate, daysBack):
        if type(startDate)==str:
            startDate=datetime.date(int(startDate[0:4]), int(startDate[5:7]),\
            int(startDate[8:]))
        if startDate.weekday()==6:
            startDate=startDate-datetime.timedelta(days=1)
        elif startDate.weekday()==5:
            startDate=startDate-datetime.timedelta(days=2)
        if startDate.weekday()+1<daysBack and daysBack<=5:
            weekendsPassed=1
        elif daysBack%5==0 and startDate.weekday()==0:
            weekendsPassed=daysBack//5-1
        else:
            weekendsPassed=daysBack//5
        return startDate-datetime.timedelta(days=daysBack+3*weekendsPassed)

    def movingAverageConvergenceDivergence(mode, df):
        MACD=pd.DataFrame(mode.exponentialMovingAverage(df, "Close", 12)-\
        mode.exponentialMovingAverage(df, "Close", 26))
        MACD.columns=["MACD"]
        signalLine=pd.DataFrame(mode.exponentialMovingAverage(MACD, "MACD", 9))
        signalLine.columns=["signalLine"]
        difference=pd.Series(MACD["MACD"]-signalLine["signalLine"], name="MACD")
        df=df.join(difference)
        return df
    
    def williamsR(mode, df, days):
        williamsR=pd.Series(((df["High"].rolling(days).max()-df["Close"])/(df\
        ["High"].rolling(days).max()-df["Low"].rolling(days).min()))*-100,\
        name="WR")
        df=df.join(williamsR)
        return df

    def ultimateOscillator(mode, df):
        buyingPressure=pd.DataFrame(df["Close"]-\
        np.minimum(df["Close"].shift(1), df["Low"]))
        buyingPressure.columns=["buyingPressure"]
        trueRange=pd.DataFrame(np.maximum(df["Close"].shift(1), df["High"])-\
        np.minimum(df["Close"].shift(1), df["Low"]))
        trueRange.columns=["trueRange"]
        average7=pd.DataFrame(buyingPressure["buyingPressure"].rolling(7).sum(\
        )/trueRange["trueRange"].rolling(7).sum())
        average7.columns=["average7"]
        average14=pd.DataFrame(buyingPressure\
        ["buyingPressure"].rolling(14).sum()/trueRange\
        ["trueRange"].rolling(14).sum())
        average14.columns=["average14"]
        average28=pd.DataFrame(buyingPressure\
        ["buyingPressure"].rolling(28).sum()/trueRange\
        ["trueRange"].rolling(28).sum())
        average28.columns=["average28"]
        ultimateOscillator=pd.Series(100*((4*average7["average7"]+2*\
        average14["average14"]+average28["average28"])/7),\
        name="UO")
        df=df.join(ultimateOscillator)
        return df

    def simpleMovingAverage(mode, df, days): 
        simpleMovingAverage=pd.Series(df["Close"].rolling(days).mean(),\
        name="SMA") 
        df=df.join(simpleMovingAverage) 
        return df
    
    def exponentialMovingAverage(mode, df, item, days):
        exponentialMovingAverage=df[item].ewm(days).mean()
        return exponentialMovingAverage

    def rateOfChangePrice(mode, df, days):
        difference=df["Close"].diff(days)
        original=df["Close"].shift(days)
        roc=pd.Series(difference/original, name="ROCP")
        df=df.join(roc)
        return df
    
    def rateOfChangeVolume(mode, df, days):
        difference=df["Volume"].diff(days)
        original=df["Volume"].shift(days)
        roc=pd.Series(difference/original, name="ROCV")
        df=df.join(roc)
        return df
    
    def createInitialDataFrame(mode, ticker, startDate, endDate):
        tempStartDate=datetime.date(int(startDate[0:4]), int(startDate[5:7]),\
        int(startDate[8:]))
        startDate=mode.adjustedStartDate(tempStartDate, 30)
        data=pdr.DataReader(ticker, "yahoo", start=startDate, end=endDate)
        df=pd.DataFrame(data=data)
        df=mode.movingAverageConvergenceDivergence(df)
        df=mode.williamsR(df, 14)
        df=mode.ultimateOscillator(df)
        df=mode.simpleMovingAverage(df, 14)
        df=mode.rateOfChangePrice(df, 1)
        df=mode.rateOfChangeVolume(df, 1)
        df=df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)
        if tempStartDate.weekday()==6:
            tempStartDate=tempStartDate+datetime.timedelta(days=1)
        elif tempStartDate.weekday()==5:
            tempStartDate=tempStartDate-datetime.timedelta(days=1)
        currentStartDate=df.index[0].to_pydatetime()
        currentStartDate=currentStartDate.date()
        while currentStartDate!=tempStartDate:
            df=df[1:]
            currentStartDate=df.index[0].to_pydatetime()
            currentStartDate=currentStartDate.date()
        return df

    def updateSVMModel(mode, df):
        df, xTrain, yTrain=mode.trainingData(df, "Close", 1)
        svrClose=SVR(kernel="rbf", C=1e3, gamma=0.1)
        svrClose.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "MACD", 1)
        svrMACD=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrMACD.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "WR", 1)
        svrWR=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrWR.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "UO", 1)
        svrUO=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrUO.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "SMA", 1)
        svrSMA=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrSMA.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCP", 1)
        svrROCP=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrROCP.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCV", 1)
        svrROCV=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrROCV.fit(xTrain, yTrain)
        if "result" in df.columns:
            df=df.drop(["result"], 1)
        predictionData=np.array(df.iloc[-1])
        predictionData=predictionData.reshape(1, len(predictionData))
        return df, predictionData, svrClose, svrMACD, svrWR, svrUO, svrSMA,\
        svrROCP, svrROCV

    def createSupportVectorMachineModel(mode, ticker, startDate, endDate, days):
        df=mode.createInitialDataFrame(ticker, startDate, endDate)
        df.fillna(value=-99999, inplace=True)
        df, xTrain, yTrain=mode.trainingData(df, "Close", 1)
        svrClose=SVR(kernel="rbf", C=1e3, gamma=0.1)
        svrClose.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "MACD", 1)
        svrMACD=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrMACD.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "WR", 1)
        svrWR=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrWR.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "UO", 1)
        svrUO=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrUO.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "SMA", 1)
        svrSMA=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrSMA.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCP", 1)
        svrROCP=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrROCP.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCV", 1)
        svrROCV=SVR(kernel="rbf", C=1e3, gamma=0.1) 
        svrROCV.fit(xTrain, yTrain)
        df=mode.createInitialDataFrame(ticker, startDate, endDate)
        lastDate=df.iloc[-1].name
        predictionData=np.array(df.iloc[-1])
        predictionData=predictionData.reshape(1, len(predictionData))
        for i in range(days):
            lastClosePrediction=svrClose.predict(predictionData)
            lastMACDPrediction=svrMACD.predict(predictionData)
            lastWRPrediction=svrWR.predict(predictionData)
            lastUOPrediction=svrUO.predict(predictionData)
            lastSMAPrediction=svrSMA.predict(predictionData)
            lastROCPPrediction=svrROCP.predict(predictionData)
            lastROCVPrediction=svrROCV.predict(predictionData)
            newDate=lastDate+datetime.timedelta(days=1)
            newRow=pd.Series(data={"Close": float(lastClosePrediction),\
            "MACD": float(lastMACDPrediction), "WR": float(lastMACDPrediction),\
            "UO": float(lastUOPrediction), "SMA": float(lastSMAPrediction),\
            "ROCP": float(lastROCPPrediction),\
            "ROCV": float(lastROCVPrediction)}, name=newDate)
            df=df.append(newRow, ignore_index=False)
            df, predictionData, svrClose, svrMACD, svrWR, svrUO, svrSMA,\
            svrROCP, svrROCV=mode.updateSVMModel(df)
            lastDate=newDate
        plt.plot(df["Close"][:-days])
        plt.plot(df["Close"][-days-1:])
        axi=plt.axes()
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        for tick in axi.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        plt.xticks(rotation=30)
        plt.show()

    def updateRFRModel(mode, df):
        df, xTrain, yTrain=mode.trainingData(df, "Close", 1)
        svrClose=RandomForestRegressor()
        rfrClose.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "MACD", 1)
        rfrMACD=RandomForestRegressor()
        rfrMACD.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "WR", 1)
        rfrWR=RandomForestRegressor() 
        rfrWR.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "UO", 1)
        rfrUO=RandomForestRegressor() 
        rfrUO.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "SMA", 1)
        rfrSMA=RandomForestRegressor() 
        rfrSMA.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCP", 1)
        rfrROCP=RandomForestRegressor() 
        rfrROCP.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "M", 1)
        rfrM=RandomForestRegressor() 
        rfrM.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCV", 1)
        rfrROCV=RandomForestRegressor() 
        rfrROCV.fit(xTrain, yTrain)
        if "result" in df.columns:
            df=df.drop(["result"], 1)
        predictionData=np.array(df.iloc[-1])
        predictionData=predictionData.reshape(1, len(predictionData))
        return df, predictionData, rfrClose, rfrMACD, rfrWR, rfrUO, rfrSMA,\
        rfrROCP, rfrM, rfrROCV

    def trainingData(mode, df, forecastCol, days):
        tempdf=df.copy(deep=True)
        df.fillna(value=-99999, inplace=True)
        df["result"]=df[forecastCol].shift(-days)
        X=np.array(df.drop(["result"], 1))
        X=preprocessing.scale(X)
        XLast=X[-days:]
        X=X[:-days]
        df.dropna(inplace=True)
        y=np.array(df["result"])
        xTrain, xTest, yTrain, yTest=train_test_split(X, y, test_size=0.2)
        return tempdf, xTrain, yTrain

    def createRandomForestRegressorModel(mode, ticker, startDate, endDate,\
        days):
        df=mode.createInitialDataFrame(ticker, startDate, endDate)
        df.fillna(value=-99999, inplace=True)
        df, xTrain, yTrain=mode.trainingData(df, "Close", 1)
        rfrClose=RandomForestRegressor()
        rfrClose.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "MACD", 1)
        rfrMACD=RandomForestRegressor() 
        rfrMACD.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "WR", 1)
        rfrWR=RandomForestRegressor() 
        rfrWR.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "UO", 1)
        rfrUO=RandomForestRegressor() 
        rfrUO.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "SMA", 1)
        rfrSMA=RandomForestRegressor() 
        rfrSMA.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCP", 1)
        rfrROCP=RandomForestRegressor() 
        rfrROCP.fit(xTrain, yTrain)
        df, xTrain, yTrain=mode.trainingData(df, "ROCV", 1)
        rfrROCV=RandomForestRegressor() 
        rfrROCV.fit(xTrain, yTrain)
        df=mode.createInitialDataFrame(ticker, startDate, endDate)
        lastDate=df.iloc[-1].name
        predictionData=np.array(df.iloc[-1])
        predictionData=predictionData.reshape(1, len(predictionData))
        for i in range(days):
            lastClosePrediction=rfrClose.predict(predictionData)
            lastMACDPrediction=rfrMACD.predict(predictionData)
            lastWRPrediction=rfrWR.predict(predictionData)
            lastUOPrediction=rfrUO.predict(predictionData)
            lastSMAPrediction=rfrSMA.predict(predictionData)
            lastROCPPrediction=rfrROCP.predict(predictionData)
            lastROCVPrediction=rfrROCV.predict(predictionData)
            newDate=lastDate+datetime.timedelta(days=1)
            newRow=pd.Series(data={"Close": float(lastClosePrediction),\
            "MACD": float(lastMACDPrediction), "WR": float(lastMACDPrediction),\
            "UO": float(lastUOPrediction), "SMA": float(lastSMAPrediction),\
            "ROCP": float(lastROCPPrediction),\
            "ROCV": float(lastROCVPrediction)}, name=newDate)
            df=df.append(newRow, ignore_index=False)
            df, predictionData, rfrClose, rfrMACD, rfrWR, rfrUO, rfrSMA,\
            rfrROCP, rfrROCV=mode.updateSVMModel(df)
            lastDate=newDate
        plt.plot(df["Close"][:-days])
        plt.plot(df["Close"][-days-1:])
        axi=plt.axes()
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        for tick in axi.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        plt.xticks(rotation=30)
        plt.show()

class StockAllocationMode(Mode):
    def appStarted(mode):
        mode.riskLevel=0
        mode.stocks=[]
    
    def mousePressed(mode, event):
        if mode.width*23/80<=event.x<=mode.width*13/40 and\
        mode.height*17/60<=event.y<=mode.height*19/60:
            if mode.riskLevel=="Low":
                mode.riskLevel=None
            else:
                mode.riskLevel="Low"
        if mode.width*43/160<=event.x<=mode.width*53/160 and\
        mode.height*23/60<=event.y<=mode.height*25/60:
            if mode.riskLevel=="Medium":
                mode.riskLevel=None
            else:
                mode.riskLevel="Medium"
        if mode.width*45/160<=event.x<=mode.width*13/40 and\
        mode.height*29/60<=event.y<=mode.height*31/60:
            if mode.riskLevel=="High":
                mode.riskLevel=None
            else:
                mode.riskLevel="High"
        for i in range(len(mode.stocks)):
            if mode.width*51/80<=event.x<=mode.width*53/80 and\
            mode.height*(20+4*i-1)/80<=event.y<=mode.height*(20+4*i+1)/80:
                mode.stocks.pop(i)
        if mode.width*15/20<=event.x<=mode.width*19/20 and\
        mode.height*31/80<=event.y<=mode.height*33/80:
            enteredMessage=mode.getUserInput("Enter stock ticker")
            if (enteredMessage!= None):
                mode.stocks.append(enteredMessage)
        if mode.width*15/40<=event.x<=mode.width*25/40 and\
        mode.height*59/80<=event.y<=mode.height*61/80:
            mode.createStockPieChart("2019-04-20", date.today(), mode.stocks,\
            mode.riskLevel)
        if 0<=event.x<=mode.width*3/24 and mode.height*37/40<=event.y<=\
        mode.height:
            mode.app.setActiveMode(mode.app.hubMode)

    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height, fill=\
        "SeaGreen1")
        canvas.create_text(mode.width/2, mode.height/10,\
        text="Stock Allocation", font="verdana 20 bold")
        canvas.create_text(mode.width*3/10, mode.height*2/10,\
        text="Select Risk Level", font="verdana 16 bold")
        canvas.create_text(mode.width*6/10, mode.height*2/10,\
        text="Stocks", font="verdana 16 bold")
        canvas.create_text(mode.width*17/20, mode.height*4/10,\
        text="Click to add more stocks", font="verdana 14 bold underline")
        canvas.create_text(mode.width/2, mode.height*3/4,\
        text="Create stock allocation chart", font="verdana 16 bold underline")
        for i in range(len(mode.stocks)):
            stock=mode.stocks[i]
            canvas.create_text(mode.width*6/10, mode.height*(5+i)/20,\
            text=stock, font="verdana 12 bold")
            canvas.create_text(mode.width*13/20, mode.height*(5+i)/20,\
            text="delete", font="verdana 8 bold")
        if mode.riskLevel=="Low":
            canvas.create_text(mode.width*3/10, mode.height*3/10,\
            text="Low", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*3/10, mode.height*3/10,\
            text="Low", font="verdana 14 bold")
        if mode.riskLevel=="Medium":
            canvas.create_text(mode.width*3/10, mode.height*4/10,\
            text="Medium", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*3/10, mode.height*4/10,\
            text="Medium", font="verdana 14 bold")
        if mode.riskLevel=="High":
            canvas.create_text(mode.width*3/10, mode.height*5/10,\
            text="High", font="verdana 14 bold underline")
        else:
            canvas.create_text(mode.width*3/10, mode.height*5/10,\
            text="High", font="verdana 14 bold")
        canvas.create_text(mode.width*1/16, mode.height*19/20,
        text="Go back", font="verdana 20 bold")
    
    def dailyReturns(mode, startDate, endDate, stockList):
        data=pdr.DataReader(stockList, "yahoo", start=startDate, end=endDate)
        df=pd.DataFrame(data=data)
        df=df.drop(["High", "Low", "Open", "Volume", "Adj Close"], axis=1)
        df=df[-252:]
        priceDict={}
        for i in range(len(stockList)):
            stock=stockList[i]
            priceDict[stock]=df["Close"][stock]
        pricedf=pd.DataFrame(priceDict)
        logOfReturns=np.log(pricedf/pricedf.shift(1))
        return logOfReturns

    def calculateOptimalWeights(mode, startDate, endDate, stockList, sims,\
    riskLevel):
        returns=mode.dailyReturns(startDate, endDate, stockList)
        weights=[]
        expectedReturns=[]
        volatilityRates=[]
        sharpeRatios=[]
        for i in range(sims):
            currentWeights=np.random.random(len(stockList))
            currentWeights=currentWeights/sum(currentWeights)
            weightedCovMatrix=np.dot(returns.cov()*252, currentWeights)
            volatilityRate=math.sqrt(np.dot(weightedCovMatrix,\
            currentWeights.transpose()))
            volatilityRates.append(volatilityRate)
            weights.append(currentWeights.tolist())
            expectedReturn=sum(returns.mean()*currentWeights*252)
            expectedReturns.append(expectedReturn)
            sharpeRatio=expectedReturn/volatilityRate
            sharpeRatios.append(sharpeRatio)
        weights=sorted(weights)
        volatilityRates=sorted(volatilityRates)
        expectedReturns=sorted(expectedReturns)
        sharpeRatios=sorted(sharpeRatios)
        if riskLevel=="Low":
            weights=weights[:len(weights)//4]
            volatilityRates=volatilityRates[:len(volatilityRates)//4]
            expectedReturns=expectedReturns[:len(expectedReturns)//4]
            sharpeRatios=sharpeRatios[:len(sharpeRatios)//4]
        elif riskLevel=="Medium":
            weights=weights[len(weights)//4:3*len(weights)//4]
            volatilityRates=\
            volatilityRates[len(volatilityRates)//4:3*len(volatilityRates)//4]
            expectedReturns=\
            expectedReturns[len(expectedReturns)//4:3*len(expectedReturns)//4]
            sharpeRatios=sharpeRatios\
            [len(sharpeRatios)//4:3*len(sharpeRatios)//4]
        elif riskLevel=="High":
            weights=weights[3*len(weights)//4:]
            volatilityRates=volatilityRates[3*len(volatilityRates)//4:]
            expectedReturns=expectedReturns[3*len(expectedReturns)//4:]
            sharpeRatios=sharpeRatios[3*len(sharpeRatios)//4:]
        bestIndex=sharpeRatios.index(max(sharpeRatios))
        optimalWeights=weights[bestIndex]
        return optimalWeights, expectedReturns[bestIndex],\
        volatilityRates[bestIndex]

    def createStockPieChart(mode, startDate, endDate, stockList, riskLevel):
        optimalWeights, expectedReturn, volatilityRate=\
        mode.calculateOptimalWeights(startDate, endDate,\
        stockList, 5000, riskLevel)
        for i in range(len(stockList)):
            optimalWeights[i]=round(optimalWeights[i]*100, 2)
        fig1, ax1=plt.subplots()
        ax1.pie(optimalWeights, labels=stockList, autopct="%1.1f%%",
            shadow=True, startangle=90)
        ax1.axis("equal")
        plt.show()

class StockRatioAnalysisMode(Mode):
    def appStarted(mode):
        mode.ratioType=0
        mode.stocks=[]
    
    def mousePressed(mode, event):
        if mode.width*27/40<=event.x<=mode.width*31/40 and\
        mode.height*3/10<=event.y<=mode.height*4/10:
            if mode.ratioType=="Debt to Capital":
                mode.ratioType=0
            else:
                mode.ratioType="Debt to Capital"
        if mode.width*33/40<=event.x<=mode.width*37/40 and\
        mode.height*3/10<=event.y<=mode.height*4/10:
            if mode.ratioType=="Cash Ratio":
                mode.ratioType=0
            else:
                mode.ratioType="Cash Ratio"
        if mode.width*27/40<=event.x<=mode.width*31/40 and\
        mode.height*5/10<=event.y<=mode.height*6/10:
            if mode.ratioType=="Return on Equity":
                mode.ratioType=0
            else:
                mode.ratioType="Return on Equity"
        if mode.width*33/40<=event.x<=mode.width*37/40 and\
        mode.height*5/10<=event.y<=mode.height*6/10:
            if mode.ratioType=="D/E Ratio":
                mode.ratioType=0
            else:
                mode.ratioType="D/E Ratio"
        if mode.width*27/40<=event.x<=mode.width*31/40 and\
        mode.height*7/10<=event.y<=mode.height*8/10:
            if mode.ratioType=="Current Ratio":
                mode.ratioType=0
            else:
                mode.ratioType="Current Ratio"
        if mode.width*33/40<=event.x<=mode.width*37/40 and\
        mode.height*7/10<=event.y<=mode.height*8/10:
            if mode.ratioType=="Quick Ratio":
                mode.ratioType=0
            else:
                mode.ratioType="Quick Ratio"
        if mode.width*8/20<=event.x<=mode.width*12/20 and\
        mode.height*31/80<=event.y<=mode.height*33/80:
            enteredMessage=mode.getUserInput("Enter stock ticker")
            if (enteredMessage!= None):
                mode.stocks.append(enteredMessage)
        for i in range(len(mode.stocks)):
            if mode.width*19/80<=event.x<=mode.width*21/80 and\
            mode.height*(20+4*i-1)/80<=event.y<=mode.height*(20+4*i+1)/80:
                mode.stocks.pop(i)
        if mode.width*31/80<=event.x<=mode.width*49/80 and\
        mode.height*63/80<=event.y<=mode.height*65/80:
            mode.createRatioGraph(mode.ratioType, mode.stocks)
        if 0<=event.x<=mode.width*3/24 and mode.height*37/40<=event.y<=\
        mode.height:
            mode.app.setActiveMode(mode.app.hubMode)
        
    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height, fill="salmon1")
        canvas.create_text(mode.width/2, mode.height/10,\
        text="Stock Ratio Analysis", font="verdana 20 bold")
        canvas.create_text(mode.width*2/10, mode.height*2/10,\
        text="Stocks", font="verdana 16 bold")
        for i in range(len(mode.stocks)):
            stock=mode.stocks[i]
            canvas.create_text(mode.width*2/10, mode.height*(5+i)/20,\
            text=stock, font="verdana 12 bold")
            canvas.create_text(mode.width*5/20, mode.height*(5+i)/20,\
            text="delete", font="verdana 8 bold")
        canvas.create_text(mode.width/2, mode.height*4/10,\
        text="Click to add more stocks", font="verdana 14 bold underline")
        canvas.create_text(mode.width/2, mode.height*8/10,
        text="Click to create ratio graph", font="verdana 16 bold underline")
        canvas.create_text(mode.width*8/10, mode.height*2/10,\
        text="Ratio Analysis Type", font="verdana 16 bold")
        canvas.create_rectangle(mode.width*27/40, mode.height*3/10,\
        mode.width*31/40, mode.height*4/10)
        if mode.ratioType=="Debt to Capital":
            canvas.create_text(mode.width*29/40, mode.height*7/20,\
            text="Debt to Capital", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*29/40, mode.height*7/20,\
            text="Debt to Capital", font="verdana 8 bold")
        canvas.create_rectangle(mode.width*33/40, mode.height*3/10,\
        mode.width*37/40, mode.height*4/10)
        if mode.ratioType=="Cash Ratio":
            canvas.create_text(mode.width*35/40, mode.height*7/20,\
            text="Cash Ratio", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*35/40, mode.height*7/20,\
            text="Cash Ratio", font="verdana 8 bold")
        canvas.create_rectangle(mode.width*27/40, mode.height*5/10,
        mode.width*31/40, mode.height*6/10)
        if mode.ratioType=="Return on Equity":
            canvas.create_text(mode.width*29/40, mode.height*11/20,\
            text="Return on Equity", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*29/40, mode.height*11/20,\
            text="Return on Equity", font="verdana 8 bold")
        canvas.create_rectangle(mode.width*33/40, mode.height*5/10,
        mode.width*37/40, mode.height*6/10)
        if mode.ratioType=="D/E Ratio":
            canvas.create_text(mode.width*35/40, mode.height*11/20,\
            text="D/E Ratio", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*35/40, mode.height*11/20,\
            text="D/E Ratio", font="verdana 8 bold")
        canvas.create_rectangle(mode.width*27/40, mode.height*7/10,
        mode.width*31/40, mode.height*8/10)
        if mode.ratioType=="Current Ratio":
            canvas.create_text(mode.width*29/40, mode.height*15/20,\
            text="Current Ratio", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*29/40, mode.height*15/20,\
            text="Current Ratio", font="verdana 8 bold")
        canvas.create_rectangle(mode.width*33/40, mode.height*7/10,\
        mode.width*37/40, mode.height*8/10)
        if mode.ratioType=="Quick Ratio":
            canvas.create_text(mode.width*35/40, mode.height*15/20,\
            text="Quick Ratio", font="verdana 8 bold underline")
        else:
            canvas.create_text(mode.width*35/40, mode.height*15/20,\
            text="Quick Ratio", font="verdana 8 bold")
        canvas.create_text(mode.width*1/16, mode.height*19/20,
        text="Go back", font="verdana 20 bold")
    
    def createRatioGraph(mode, ratioType, stocks):
        for stock in stocks:
            df=mode.createRatioDataFrame(stock)
            plt.plot(df[ratioType], label=stock)
        axi=plt.axes()
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))
        for tick in axi.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        plt.xticks(rotation=30)
        plt.legend()
        plt.show()

    def createRatioDataFrame(mode, stock):
        data=requests.get\
        ("https://financialmodelingprep.com/api/v3/financial-ratios/"+stock)
        data=data.json()
        ratioDict={"Date": [], "Debt to Capital": [], "Cash Ratio": [],\
        "Return on Equity": [], "D/E Ratio": [], "Current Ratio": [],\
        "Quick Ratio": []}
        for i in range(len(data["ratios"])):
            ratioDict["Date"].append(data["ratios"][i]["date"])
            if data["ratios"][i]["debtRatios"]\
            ["longtermDebtToCapitalization"]!="":
                ratioDict["Debt to Capital"].append(float(\
                data["ratios"][i]["debtRatios"]\
                ["longtermDebtToCapitalization"]))
            else:
                ratioDict["Debt to Capital"].append(0)
            if data["ratios"][i]["liquidityMeasurementRatios"]["cashRatio"]!="":
                ratioDict["Cash Ratio"].append(float(\
                data["ratios"][i]["liquidityMeasurementRatios"]["cashRatio"]))
            else:
                ratioDict["Cash Ratio"].append(0)
            if data["ratios"][i]["profitabilityIndicatorRatios"]\
            ["returnOnEquity"]!="":
                ratioDict["Return on Equity"].append(float(\
                data["ratios"][i]["profitabilityIndicatorRatios"]\
                ["returnOnEquity"]))
            else:
                ratioDict["Return on Equity"].append(0)
            if data["ratios"][i]["debtRatios"]["debtEquityRatio"]!="":
                ratioDict["D/E Ratio"].append(float(\
                data["ratios"][i]["debtRatios"]["debtEquityRatio"]))
            else:
                ratioDict["D/E Ratio"].append(0)
            if data["ratios"][i]\
            ["liquidityMeasurementRatios"]["currentRatio"]!="":
                ratioDict["Current Ratio"].append(float(\
                data["ratios"][i]["liquidityMeasurementRatios"]\
                ["currentRatio"]))
            else:
                ratioDict["Current Ratio"].append(0)
            if data["ratios"][i]\
            ["liquidityMeasurementRatios"]["quickRatio"]!="":
                ratioDict["Quick Ratio"].append(float(\
                data["ratios"][i]["liquidityMeasurementRatios"]["quickRatio"]))
            else:
                ratioDict["Quick Ratio"].append(0)
        df=pd.DataFrame(data=ratioDict)
        for i in range(len(df.index)):
            date=df["Date"][i]
            month=int(date[5:7])
            day=int(date[8:])
            year=int(date[:4])
            newDate=pd.Timestamp(year, month, day)
            df["Date"][i]=newDate
        df=df.sort_values("Date")
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df=df.drop(["index"], axis=1)
        return df

class LiveStockPrices(Mode):
    def appStarted(mode):
        mode.stocks=[]
        mode.prices=[]
        mode.priceChanges=[]
        mode.opens=[]
        mode.previousCloses=[]
    
    def livePrice(mode, ticker):
        url="https://finance.yahoo.com/quote/"+ticker+"?p="+\
        ticker+"&.tsrc=fin-srch"
        stockhtml=urlopen(url)
        parser=BeautifulSoup(stockhtml, "html.parser")
        priceFound=False
        for item in parser.find_all("span"):
            if priceFound==True:
                return item.contents[0]
            if "Currency in USD" in item.contents[0]:
                priceFound=True
    
    def livePreviousClose(mode, ticker):
        url="https://finance.yahoo.com/quote/"+ticker\
        +"?p="+ticker+"&.tsrc=fin-srch"
        stockhtml=urlopen(url)
        parser=BeautifulSoup(stockhtml, "html.parser")
        previousCloseFound=False
        for item in parser.find_all("span"):
            if previousCloseFound==True:
                return item.contents[0]
            if "Previous Close" in item.contents[0]:
                previousCloseFound=True

    def liveOpen(mode, ticker):
        url="https://finance.yahoo.com/quote/"+ticker\
        +"?p="+ticker+"&.tsrc=fin-srch"
        stockhtml=urlopen(url)
        parser=BeautifulSoup(stockhtml, "html.parser")
        openFound=False
        for item in parser.find_all("span"):
            if openFound==True:
                return item.contents[0]
            if "Open" in item.contents[0]:
                openFound=True

    def livePriceChange(mode, ticker):
        url="https://finance.yahoo.com/quote/"+ticker\
        +"?p="+ticker+"&.tsrc=fin-srch"
        stockhtml=urlopen(url)
        parser=BeautifulSoup(stockhtml, "html.parser")
        priceFound=False
        count=0
        for item in parser.find_all("span"):
            if priceFound==True:
                count+=1
            if count==2:
                return item.contents[0]
            if "Currency in USD" in item.contents[0]:
                priceFound=True
    
    def mousePressed(mode, event):
        if mode.width*1/40<=event.x<=mode.width*35/160 and\
        mode.height*39/80<=event.y<=mode.height*41/80:
            enteredMessage=mode.getUserInput("Enter stock ticker")
            if (enteredMessage!= None):
                mode.stocks.append(enteredMessage)
                mode.prices.append(mode.livePrice(enteredMessage))
                mode.priceChanges.append(mode.livePriceChange(enteredMessage))
                mode.opens.append(mode.liveOpen(enteredMessage))
                mode.previousCloses.append\
                (mode.livePreviousClose(enteredMessage))
        for i in range(len(mode.stocks)):
            if mode.width*27/80<=event.x<=mode.width*29/80 and\
            mode.height*(20+4*i-1)/80<=event.y<=mode.height*(20+4*i+1)/80:
                mode.stocks.pop(i)
                mode.prices.pop(i)
                mode.priceChanges.pop(i)
                mode.opens.pop(i)
                mode.previousCloses.pop(i)
        if 0<=event.x<=mode.width*3/24 and mode.height*37/40<=event.y<=\
        mode.height:
            mode.app.setActiveMode(mode.app.hubMode)

    def timerFired(mode):
        for i in range(len(mode.stocks)):
            mode.prices[i]=mode.livePrice(mode.stocks[i])
            mode.priceChanges[i]=mode.livePriceChange(mode.stocks[i])
            mode.opens[i]=mode.liveOpen(mode.stocks[i])
            mode.previousCloses[i]=mode.livePreviousClose(mode.stocks[i])

    def redrawAll(mode, canvas):
        canvas.create_rectangle(0, 0, mode.width, mode.height,\
        fill="light goldenrod")
        canvas.create_text(mode.width/2, mode.height/10,\
        text="Live Stock Prices", font="verdana 20 bold")
        canvas.create_text(mode.width*5/40, mode.height*5/10,\
        text="Click to add more stocks", font="verdana 14 bold underline")
        canvas.create_text(mode.width*3/10, mode.height*2/10,\
        text="Stocks", font="verdana 16 bold")
        canvas.create_text(mode.width*5/10, mode.height*2/10,\
        text="Price", font="verdana 16 bold")
        canvas.create_text(mode.width*7/10, mode.height*2/10,\
        text="Open", font="verdana 16 bold")
        canvas.create_text(mode.width*9/10, mode.height*2/10,\
        text="Previous Close", font="verdana 16 bold")
        for i in range(len(mode.stocks)):
            stock=mode.stocks[i]
            price=mode.prices[i]
            priceChange=mode.priceChanges[i]
            priceOpen=mode.opens[i]
            previousClose=mode.previousCloses[i]
            canvas.create_text(mode.width*3/10, mode.height*(5+i)/20,\
            text=stock, font="verdana 12 bold")
            canvas.create_text(mode.width*7/20, mode.height*(5+i)/20,\
            text="delete", font="verdana 8 bold")
            canvas.create_text(mode.width*9/20, mode.height*(5+i)/20,\
            text=price, font="verdana 12 bold")
            canvas.create_text(mode.width*11/20, mode.height*(5+i)/20,\
            text=priceChange, font="verdana 12 bold")
            canvas.create_text(mode.width*7/10, mode.height*(5+i)/20,\
            text=priceOpen, font="verdana 12 bold")
            canvas.create_text(mode.width*9/10, mode.height*(5+i)/20,\
            text=previousClose, font="verdana 12 bold")
        canvas.create_text(mode.width*1/16, mode.height*19/20,
        text="Go back", font="verdana 20 bold")

app=MyModalApp(width=1366, height=705)