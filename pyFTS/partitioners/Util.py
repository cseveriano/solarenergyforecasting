import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyFTS.common import Membership


def plotSets(data, sets, titles):
    num = len(sets)
    fig = plt.figure(figsize=[12, 10])
    maxx = max(data)
    minx = min(data)
    h = 1/num
    for k in range(num):
        ax0 = fig.add_axes([0, (k+1)*h, 0.65, h*0.7])  # left, bottom, width, height
        ax0.set_title(titles[k])
        ax0.set_ylim([0, 1])
        ax0.set_xlim([minx, maxx])
        for s in sets[k]:
            if s.mf == Membership.trimf:
                ax0.plot([s.parameters[0],s.parameters[1],s.parameters[2]],[0,1,0])
            elif s.mf == Membership.gaussmf:
                tmpx = [ kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax0.plot(tmpx, tmpy)