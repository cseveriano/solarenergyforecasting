import numpy as np
from pyFTS import *


class FTS:
    def __init__(self, order, name):
        self.sets = {}
        self.flrgs = {}
        self.order = order
        self.shortname = name
        self.name = name
        self.detail = name
        self.hasSeasonality = False
        self.hasPointForecasting = True
        self.hasIntervalForecasting = False
        self.hasDistributionForecasting = False
        self.dump = False

    def fuzzy(self, data):
        best = {"fuzzyset": "", "membership": 0.0}

        for f in self.sets:
            fset = self.sets[f]
            if best["membership"] <= fset.membership(data):
                best["fuzzyset"] = fset.name
                best["membership"] = fset.membership(data)

        return best

    def forecast(self, data):
        pass

    def forecastInterval(self, data):
        pass

    def forecastDistribution(self, data):
        pass

    def forecastAhead(self, data, steps):
        pass

    def forecastAheadInterval(self, data, steps):
        pass

    def forecastAheadDistribution(self, data, steps):
        pass

    def train(self, data, sets):
        pass

    def getMidpoints(self, flrg):
        ret = np.array([s.centroid for s in flrg.RHS])
        return ret

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            tmp = tmp + str(self.flrgs[r]) + "\n"
        return tmp
