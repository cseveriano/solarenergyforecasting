#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import glob
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, Panel
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from pyFTS.partitioners import Grid
from pyFTS.common import FLR,FuzzySet,Membership
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
ts = pd.Series(big_frame["glo_avg"].values, index=dates)
ts30 = ts.resample("30min")

#enrollments_fs1 = Grid.GridPartitionerTrimf(enrollments,6)

#pfts1_enrollments = pfts.ProbabilisticFTS("1")
#pfts1_enrollments.train(enrollments,enrollments_fs1,1)
#pfts1_enrollments.shortname = "1st Order"
#pfts2_enrollments = pfts.ProbabilisticFTS("2")
#pfts2_enrollments.dump = False
#pfts2_enrollments.shortname = "2nd Order"
#pfts2_enrollments.train(enrollments,enrollments_fs1,2)


#pfts1_enrollments.forecastAheadDistribution2(enrollments[:15],5,100)
