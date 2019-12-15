# Arima-Forecasting

Using ARIMA Forecasting to predict the day ahead electricity market. The data is from Nordpool for the year 2016 and due to time constraints only 300 values out of the potential 8800 values were used for this process


The Box Jenkins method created by George Box and Gwilym Jenkins in their landmark book on time series analysis. This method is an iterative process of data preparation, model selection, parameter estimation, model checking and finally forecasting using ARIMA. ARIMA stands for AutoRegressive Integated Moving Average and in this model there are three main parameters used represented as (p,d,q). This model combines the Autoregressive model and the moving average model and applies this mixed forecasting model to a differenced time series.

The P parameter represents the order of the autoregressive model of the ARIMA model that uses the dependent relationship between an observation and a number of lagged observations, meaning that this is a tool used to test for randomness within the time series data by testing to see how much of X is affected by the previous values of X. 

![Autocorrelation Chart of the differenced time series](https://github.com/hhamud/Arima-Forecasting/blob/master/autocorrchart.png "Autocorrelation Chart of the differenced time series")

The d parameter represents the Integrated order of differencing required of the model such that the time series data is becomes stationary meaning that the statistical properties of the time series data does not vary across time (mean, variance becomes constant over time). The time series is made stationary by subtracting the observation from an observation from a previous time step. 

The q parameter represents the order of the moving average and shows the size of the window function as it moves. This parameter is calculated from the partial autocorrelation chart.

![Partial Autocorrelation chart of the differenced time series](https://github.com/hhamud/Arima-Forecasting/blob/master/partialcorrchart.png "Partial Autocorrelation chart of the differenced time series")

The main method used to test the time series data for stationarity is the Augmented Dickney-Fuller test which is also called the unit root test. It tests the null hypothesis that a unit root is present in a time series sample, that it is not stationary (has some time-dependent structure) and the alternative hypothesis is that it is stationary in that its statistical properties do vary over time. 

![Day Ahead Electricity Arima Forecasting Chart](https://github.com/hhamud/Arima-Forecasting/blob/master/chart.png "Day Ahead Electricity Arima Forecasting Chart")








