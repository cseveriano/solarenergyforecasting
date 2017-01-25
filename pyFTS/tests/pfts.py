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
from pyFTS.timeseries import TSAnalysis

#from pyFTS import fts
#from pyFTS import hofts
#from pyFTS import ifts
from pyFTS import pfts
#from pyFTS import tree
#from pyFTS.benchmarks import benchmarks as bchmk

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
moving_avg = pd.rolling_mean(ts30["2013-01-01":"2014-01-01"],48)
plt.plot(moving_avg, color='red')

ts30_moving_avg_diff = ts30 - moving_avg
ts30_moving_avg_diff.dropna(inplace=True)
TSAnalysis.test_stationarity(ts30_moving_avg_diff)

## exponential moving average
plt.plot(ts30["2013-01-01":"2014-01-01"])
expwighted_avg = pd.ewma(ts30["2013-01-01":"2014-01-01"],halflife=48)
plt.plot(expwighted_avg, color='red')

ts30_expwighted_avg_diff = ts30 - expwighted_avg
ts30_expwighted_avg_diff.dropna(inplace=True)
TSAnalysis.test_stationarity(ts30_expwighted_avg_diff)
##

## Differencing
ts30_diff = ts30 - ts30.shift()
plt.plot(ts30_diff)
ts30_diff.dropna(inplace=True)
test_stationarity(ts30_diff)

## Test Statistic = -40
##

## Decomposing
ts30.dropna(inplace=True)
decomposition = seasonal_decompose(ts30, freq=48)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts30, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts30_decompose = residual
ts30_decompose.dropna(inplace=True)
test_stationarity(ts30_decompose)
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
#enrollments_fs1 = Grid.GridPartitionerTrimf(enrollments,6)

#pfts1_enrollments = pfts.ProbabilisticFTS("1")
#pfts1_enrollments.train(enrollments,enrollments_fs1,1)
#pfts1_enrollments.shortname = "1st Order"
#pfts2_enrollments = pfts.ProbabilisticFTS("2")
#pfts2_enrollments.dump = False
#pfts2_enrollments.shortname = "2nd Order"
#pfts2_enrollments.train(enrollments,enrollments_fs1,2)


#pfts1_enrollments.forecastAheadDistribution2(enrollments[:15],5,100)
