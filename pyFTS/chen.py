import numpy as np
from pyFTS.common import FuzzySet, FLR
import fts


class ConventionalFLRG:
    def __init__(self, LHS):
        self.LHS = LHS
        self.RHS = set()

    def append(self, c):
        self.RHS.add(c)

    def __str__(self):
        tmp = self.LHS.name + " -> "
        tmp2 = ""
        for c in sorted(self.RHS, key=lambda s: s.name):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c.name
        return tmp + tmp2


class ConventionalFTS(fts.FTS):
    def __init__(self, name):
        super(ConventionalFTS, self).__init__(1, "CFTS")
        self.name = "Conventional FTS"
        self.detail = "Chen"
        self.flrgs = {}

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = ConventionalFLRG(flr.LHS);
                flrgs[flr.LHS.name].append(flr.RHS)
        return (flrgs)

    def train(self, data, sets):
        self.sets = sets
        tmpdata = FuzzySet.fuzzySeries(data, sets)
        flrs = FLR.generateNonRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data):

        ndata = np.array(data)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            mv = FuzzySet.fuzzyInstance(ndata[k], self.sets)

            actual = self.sets[np.argwhere(mv == max(mv))[0, 0]]

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                flrg = self.flrgs[actual.name]
                mp = self.getMidpoints(flrg)

                ret.append(sum(mp) / len(mp))

        return ret
