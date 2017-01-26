import pandas as pd
import matplotlib.pyplot as plt

def moving_average(ts, window):

    moving_avg = pd.rolling_mean(ts,window)
    plt.plot(moving_avg, color='red')
    plt.plot(ts)

    ts_moving_avg_diff = ts - moving_avg
    ts_moving_avg_diff.dropna(inplace=True)

    return ts_moving_avg_diff
    
def exponential_moving_average(ts, window):
    plt.plot(ts)
    expwighted_avg = pd.ewma(ts,halflife=window)
    plt.plot(expwighted_avg, color='red')

    ts_expwighted_avg_diff = ts - expwighted_avg
    ts_expwighted_avg_diff.dropna(inplace=True)

    return ts_expwighted_avg_diff
    
def differencing(ts):
    ts_diff = ts - ts.shift()
    plt.plot(ts_diff)
    ts_diff.dropna(inplace=True)
    
    return ts_diff

    