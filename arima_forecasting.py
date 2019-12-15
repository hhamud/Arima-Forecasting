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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import arma_generate_sample
import itertools
import warnings


df = pd.read_csv(r'C:\Users\Hamza\Documents\PycharmProjects\Research\clean_n2ex_2016_hourly.csv')


df['month'] = pd.DatetimeIndex(df['date']).month

df['hour_start'] = pd.to_datetime(df['hour_start'], format='%H').dt.time

df['date_time'] = df['date'].astype(str) + ' ' +df['hour_start'].astype(str)


df['season'] = ['winter' if i < 3 or i == 12 else 'Autumn' if i >= 9 and i <= 11 else 'Spring' if i >= 3 and i <= 5 else 'Summer' for i in df['month']]


def scatterchart():
    xticklabels = df['date_time']
    l = sns.scatterplot(data=df, x='date_time', y='price_sterling')
    l.set(xlim=(0,300),ylim=(0,200))
    l.set_xticklabels(xticklabels, rotation=45, ha='right')
    plt.tight_layout
    plt.setp(l.get_xticklabels(), visible=True)
    plt.show()




def stat_test():
    df1 = df.loc[:302, 'price_sterling']
    X = df1.values
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))




def auto_corr():
    corr_data = df.loc[:302, 'price_sterling'].diff()
    corr_data = corr_data.replace(np.nan, 0)
    print(corr_data)
    autocorrelation_plot(corr_data)
    plt.show()



def pacf():
    pacf_data = df.loc[:302, 'price_sterling'].diff()
    pacf_data = pacf_data.replace(np.nan, 0)
    plot_acf(pacf_data)
    plt.show()




def run_arim():
    df1 = df.loc[ :300, 'price_sterling']
    model = ARIMA(df1, order=(2,1,2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())




def compute_arima_para():
    import pmdarima as pm
    df1 = df['price_sterling']
    df2 = df1.loc[:300]
    X = df2.values
    arima = pm.auto_arima(X, start_P=1, max_P=6, start_q=1, max_q=6,  information_criterion='aic')
    print(arima)
    


def ARIM_pre():
    df1 = df['price_sterling']
    df2 = df1.loc[:300]
    X = df2.values
    size = int(len(X) * 0.66)
    train, test = X[0:size] , X[size:len(X)]
    history = [x for x in train]
    predictions = []
    predictionslower = []
    predictionsupper = []
    for k in range(len(test)):
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit(disp=0)
        forecast, stderr, conf_int = model_fit.forecast()
        yhat = forecast[0]
        yhatlower = conf_int[0][0]
        yhatupper = conf_int[0][1]
        predictions.append(yhat)
        predictionslower.append(yhatlower)
        predictionsupper.append(yhatupper)
        obs = test[k]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        print('95 prediction interval: %f to %f' % (yhatlower, yhatupper))
    error = mean_squared_error(test, predictions)
    print('TEST MSE: %.3f' % error)
    d = dict(data=X, forecast=predictions, lower=predictionslower, upper=predictionsupper)
    results_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items()]))
    results_table.to_csv(r'arima_forecasting.csv')


def chart_creation():
    df1 = pd.read_csv(r'arima_forecasting.csv')
    df1['forecast'] = df1['forecast'].shift(200)
    df1['lower'] = df1['lower'].shift(200)
    df1['upper'] = df1['upper'].shift(200)
    df1['date_time'] = df['date_time']
    df1.fillna({x:0 for x in ['data', 'forecast', 'lower', 'upper']}, inplace=True)
    test = df1['data']
    std_err = df1['forecast'].sem()
    predictions = df1['forecast'][200:].values
    lower = df1['lower'][200:].values
    upper = df1['upper'][200:].values
    g = np.arange(len(predictions))
    plt.plot(df1['forecast'][200:], color='red', label='Arima Forecast')
    plt.plot(test, color='blue', label='Real Data')
    plt.fill_between(g+200 ,lower, upper, alpha=0.25, interpolate=True, color='red', label='95% confidence interval')
    #plt.xticks(df1.index.values, df1['date_time'], rotation=45, ha='right')
    plt.xlabel('Time (h)')
    plt.ylabel('£ / Mwh')
    plt.title('Day Ahead Electricity Pricing')
    plt.legend(loc='upper left')
    plt.show()
    




def AIC_iteration():
    warnings.filterwarnings("ignore")
    df1 = df['price_sterling']
    df2 = df1.loc[:300]
    X = df2.values
    size = int(len(X) * 0.66)
    train, test = X[0:size] , X[size:len(X)]
    history = [x for x in train]
    p = d = q = range(0,8)
    pdq = list(itertools.product(p,d,q))
    aic_results = []
    for param in pdq:
        try:
            model = ARIMA(history, order=param)
            results = model.fit(disp=0)
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            aic_results.append(results.aic)
        except:
            continue
    d = dict(ARIMA=pdq, AIC=aic_results)
    results_table = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in d.items()]))
    results_table.to_csv(r'AIC.csv')

ARIM_pre()
chart_creation()