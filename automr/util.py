import numpy as np

def check_uno(noon, thresh=1.98):
    ndb = np.count_nonzero(noon > thresh)
    nex = np.count_nonzero(noon < (2.0-thresh))
    nacto = len(noon) - ndb - nex
    return nacto, ndb, nex

