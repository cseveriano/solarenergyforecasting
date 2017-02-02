#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import glob
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Panel
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
#from mpl_toolkits.mplot3d import Axes3D
import importlib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from pyFTS.partitioners import Grid
from pyFTS.common import FLR,FuzzySet,Membership
from pyFTS.timeseries import TSAnalysis, Detrending

#from pyFTS import fts
from pyFTS import hofts
#from pyFTS import ifts
from pyFTS import pfts
#from pyFTS import tree
from pyFTS.benchmarks import benchmarks as bchmk

os.chdir("C:\\Users\\cseve\\Google Drive\\Doutorado\\Projetos de Pesquisa\\Base de Dados\\INPE-SONDA")
path = "C:\\Users\\cseve\\Google Drive\\Doutorado\\Projetos de Pesquisa\\Base de Dados\\INPE-SONDA"

tseries = pd.DataFrame()

#read the header
header = pd.read_csv("ED_header_new.csv", sep=";")


#read the whole directory
all_files = glob.glob(os.path.join(path, "*ED.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

np_array_list = []
for file_ in all_files:
    df = pd.read_csv(file_, header=None, sep=";")
    print (file_,  " " , df.shape)
    np_array_list.append(df.as_matrix())

comb_np_array = np.vstack(np_array_list)
big_frame = pd.DataFrame(comb_np_array)

big_frame.columns = header.columns

#####

# set time series range
dates = pd.date_range("2013-01-01 00:00", "2015-12-01 00:00", freq="1min")
dates = dates[:-1]

ts = pd.DataFrame(index = dates)
ts["glo_avg"] = big_frame["glo_avg"].values.astype(float)
ts30 = ts.glo_avg.resample("30min").mean()

## Analyzing time series


plt.plot(ts30)
plt.plot(ts30["2013-01-01":"2014-01-01"])

## DETRENDING PROCESS

## moving average
ts30_moving_avg_diff = Detrending.moving_average(ts30["2013-01-01":"2014-01-01"], 48)
TSAnalysis.test_stationarity(ts30_moving_avg_diff)

## exponential moving average
ts30_expwighted_avg_diff = Detrending.exponential_moving_average(ts30["2013-01-01":"2014-01-01"], 48)
TSAnalysis.test_stationarity(ts30_expwighted_avg_diff)
##

## Differencing
ts30_diff = Detrending.differencing(ts30)
test_stationarity(ts30_diff)
## Test Statistic = -40
##

## Decomposing
TSAnalisys.decomposing(ts30, 48)

## Test Statistic = -47
###


##ACF and PACF plots:
lag_acf = acf(ts30_diff, nlags=60)
lag_pacf = pacf(ts30_diff, nlags=60, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts30_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts30_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts30_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts30_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

####

## ARMA model

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts30, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts30[1:])
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts30[1:])**2))

ts30_train = ts30["2013-01-01":"2014-01-01"]
ts30_test = ts30["2014-01-02":"2015-12-01"]

order = 12
partitions = 100
sets = Grid.GridPartitionerTrimf(ts30_train,partitions)
fts = hofts.HighOrderFTS("k = " + str(partitions) + " w = " + str(order))
fts.train(ts30_train, sets, order)
forecasted = fts.forecast(ts30_test)
error = Measures.rmse(np.array(forecasted), np.array(ts30_test[order:]))
