import numpy as np
import math
from pyFTS import *


def trimf(x, parameters):
    xx = round(x, 3)
    if (xx < parameters[0]):
        return 0
    elif (xx >= parameters[0] and xx < parameters[1]):
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif (xx >= parameters[1] and xx <= parameters[2]):
        return (parameters[2] - xx) / (parameters[2] - parameters[1])
    else:
        return 0


def trapmf(x, parameters):
    if (x < parameters[0]):
        return 0
    elif (x >= parameters[0] and x < parameters[1]):
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif (x >= parameters[1] and x <= parameters[2]):
        return 1
    elif (x >= parameters[2] and x <= parameters[3]):
        return (parameters[3] - x) / (parameters[3] - parameters[2])
    else:
        return 0


def gaussmf(x, parameters):
    return math.exp((-(x - parameters[0])**2)/(2 * parameters[1]**2))
    #return math.exp(-0.5 * ((x - parameters[0]) / parameters[1]) ** 2)


def bellmf(x, parameters):
    return 1 / (1 + abs((x - parameters[2]) / parameters[0]) ** (2 * parameters[1]))


def sigmf(x, parameters):
    return 1 / (1 + math.exp(-parameters[0] * (x - parameters[1])))
