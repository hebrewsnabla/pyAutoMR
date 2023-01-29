#!/usr/bin/env python
"""
Parse SUPDFT output;
Interpolate dissociation curve
"""


import subprocess, sys
import numpy as np

class suData():
    def __init__(self, data):
        self.suhf = float(data[0])
        self.j = float(data[3])
        self.k = float(data[4])
        self.c = float(data[5])
        self.ddxc = float(data[6])
        self.otx = float(data[13])
        self.otc = float(data[14])
        self.otxc = float(data[15])
    
    def sudd(self, hyb):
        return self.suhf + (self.ddxc - self.k - self.c)*(1.0-hyb)
    def supd(self, hyb):
        return self.suhf + (self.otxc - self.k - self.c)*(1.0-hyb)
    def supd_k(self, hyb, k):
        return self.suhf + (self.otx - self.k - self.c)*(1.0-hyb) + (1.0-hyb**k)*self.otc

def runcmd(cmd):
    p = subprocess.Popen(cmd, 
         shell=True,
         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
         encoding='utf-8').communicate()[0]
    return p

h = float(sys.argv[2])
k = float(sys.argv[3])
task = sys.argv[1]
rscale = float(sys.argv[4])
sub = bool(sys.argv[5])
print("     SUDD    SUPD    SUHF   SUPDk")

filelist = runcmd("ls %s*out" % task).strip().split('\n')
#print(filelist)

#exit()

x = []
e_dd = []
e_pd = []
e_su = []
e_supdk = []
for s in filelist:
    r = s.replace(task, '').replace('.out', '')
    #print(s, r)
    runcmd("cp %s tmp" % s)
    runcmd("sed -i 's/://' tmp")
    p = runcmd("grep E_ tmp | awk '{print $2}'")
    data = p.split('\n')
    #print(data)
    su = suData(data)
    print('%s  %6.3f %6.3f %6.3f %6.3f'%(r, su.sudd(h), su.supd(h), su.suhf, su.supd_k(h,k)))
    if r[0].isdigit():
        x.append(float(r)/rscale)
    else:
        x.append(-1)
    e_dd.append(su.sudd(h))
    e_pd.append(su.supd(h))
    e_su.append(su.suhf)
    e_supdk.append(su.supd_k(h,k))

#exit()
def sub_atom(lst):
    array = np.array(lst)
    return array[:-2] - array[-1] - array[-2]

Ha2ev = 27.21138602
if sub:
    x_sub = np.array(x)[:-2]
    e_dd_sub = sub_atom(e_dd)*Ha2ev
    e_pd_sub = sub_atom(e_pd)*Ha2ev
    e_su_sub = sub_atom(e_su)*Ha2ev
    e_supdk_sub = sub_atom(e_supdk)*Ha2ev
    for i in range(len(x_sub)):
        print('%s  %6.3f %6.3f %6.3f %6.3f'%(x_sub[i], 
              e_dd_sub[i], e_pd_sub[i], e_su_sub[i], e_supdk_sub[i]))

#exit()
from scipy.interpolate import make_interp_spline as spl
from scipy.optimize import root
def spline_findmin(x, y):
    f = spl(x, y, k=3)
    deriv = f.derivative()
    res = root(deriv, x[2])
    print('root:    %.6f' %res.x)
    print('y(root): %.6f' %f(res.x))
    return res.x
for y in [e_dd_sub, e_pd_sub, e_su_sub, e_supdk_sub]:
    spline_findmin(x_sub, y)
