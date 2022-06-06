import numpy as np

def check_uno(noon, thresh=1.98):
    ndb = np.count_nonzero(noon > thresh)
    nex = np.count_nonzero(noon < (2.0-thresh))
    nacto = len(noon) - ndb - nex
    return nacto, ndb, nex

chemcore_atm = [
    0,                                                                  0,
    0,  0,                                          1,  1,  1,  1,  1,  1,
    1,  1,                                          5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  9,  9,  9,  9,  9,  9,
    9,  9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 18, 18, 18, 18, 18, 18, 
   18, 18, 
           18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 23,
               23, 23, 23, 23, 23, 23, 23, 23, 23, 34, 34, 34, 34, 34, 34, 
   34, 34, 
           34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 
               50, 50, 50, 50, 50, 50, 50, 50, 50]
def chemcore(mol):
    core = 0
    for a in mol.atom_charges():
        core += chemcore_atm[a]
    return core
