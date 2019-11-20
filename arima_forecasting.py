import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from scipy.stats import sem, t
from scipy import mean



df = pd.read_csv(r'C:\Users\Hamza\Documents\PycharmProjects\Research\clean_n2ex_2016_hourly.csv')


df['month'] = pd.DatetimeIndex(df['date']).month

df['hour_start'] = pd.to_datetime(df['hour_start'], format='%H').dt.time

df['date_time'] = df['date'].astype(str) + ' ' +df['hour_start'].astype(str)


df['season'] = ['winter' if i < 3 or i == 12 else 'Autumn' if i >= 9 and i <= 11 else 'Spring' if i >= 3 and i <= 5 else 'Summer' for i in df['month']]


def scatterchart():
    xticklabels = df['date_time']
    l = sns.scatterplot(data=df, x='date_time', y='price_sterling')
    l.set(xlim=(0,24),ylim=(0,100))
    l.set_xticklabels(xticklabels, rotation=45, ha='right')
    plt.tight_layout
    plt.setp(l.get_xticklabels(), visible=True)
    plt.show()

#start prepping data for linear regression analysis or for ARIMA comparison

#plot autocorrelation plot, finding the AR componenet
def auto_corr():
    df1 = df['price_sterling']
    autocorrelation_plot(df1)
    plt.show()


#shows the MA component
def pacf():
    df1 = df['price_sterling']
    plot_acf(df1)
    plt.show()

#fit model
#order, lag value of 5 for autoregression, difference order of 1 to make time series stationary, moving avergae of 0
def run_arim():
    df1 = df['price_sterling']
    model = ARIMA(df1, order=(5,1,4))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
#plot residual errors

def ARIM_pre():
    df1 = df['price_sterling']
    df2 = df1.loc[:300]
    X = df2.values
    size = int(len(X) * 0.66)
    train, test = X[0:size] , X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for k in range(len(test)):
        model = ARIMA(history, order=(5,2,2))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[k]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('TEST MSE: %.3f' % error)
    d = dict(data=X, forecast=predictions)
    results_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items()]))
    results_table.to_csv(r'arima_forecasting.csv')


def chart_creation():
    df1 = pd.read_csv(r'arima_forecasting.csv')
    df1['predictions'] = df1['forecast'].str.strip('[]').astype(float)
    df1['predictions'] = df1['predictions'].shift(200)
    df1.fillna({x:0 for x in ['data', 'predictions']}, inplace=True)
    test = df1['data']
    print(df1)
    std_err = df1['predictions'].sem()
    predictions = df1['predictions'][200:].values
    confidence = 0.95
    n = len(df1['predictions'])
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    g = np.arange(len(predictions))
    plt.plot(df1['predictions'][200:], color='red')
    plt.plot(test, color='blue')
    plt.fill_between(g+200 ,predictions + h, predictions - h, alpha=0.25, interpolate=True, color='red')
    plt.show()
    
    
    




chart_creation()

   

    

