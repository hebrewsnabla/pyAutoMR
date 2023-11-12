#!/usr/bin/env python
"""
Parse SUPDFT output;
Interpolate dissociation curve
"""

import argparse

def argument_parse():
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                               description='')
    parser.add_argument("-p","--parse",dest='type',metavar='type',type=str,default='dump',
                        required=False, #choices=['dh','scf'],
                        help='')
    parser.add_argument("-f","--fun",dest='fun',metavar='fun',type=str,
                        default='suhf',
                        required=True,
                        )
    parser.add_argument("-t","--task",dest='task',metavar='task',type=str,
                        default='dump',
                        required=True,)
    parser.add_argument("-s","--rscale",dest='rscale',metavar='rscale',type=int,
                        default=100,
                        required=False,)
    parser.add_argument("-S","--sub",dest='sub',metavar='sub',type=int,
                        default=1,
                        required=False,)
    parser.add_argument("-m","--mode",dest='mode',metavar='mode',type=str,
                        default='supd',
                        required=False,)
    args=parser.parse_args()
    return parser, args

parser, args = argument_parse()

import subprocess, sys
import numpy as np

mode = args.mode
shift = 0
if 'dd' in mode:
    shift += 7
#if 'mc' in mode:
#    shift += 9

class suData():
    def __init__(self, data):
        self.suhf = float(data[0])
        self.j = float(data[3])
        self.k = float(data[4])
        self.c = float(data[5])
        if 'dd' in mode:
            self.ddxc = float(data[6])
        self.otx = float(data[6+shift])
        self.otc = float(data[7+shift])
        self.otxc = float(data[8+shift])
    
    def sudd(self, hyb):
        if hasattr(self, 'ddxc'):
            return self.suhf + (self.ddxc - self.k - self.c)*(1.0-hyb)
        else:
            return 0.0
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

FUN_param = {
    'suhf': [1.0, 1.0],
    'pbe': [0.0, 1.0],
    'pbe0': [0.25, 1.0],
    'pbe02': [0.25, 2.0]
}
def get_param(fun):
    return FUN_param[fun.lower()]

h, k = get_param(args.fun) #float(sys.argv[2])
#float(sys.argv[3])
task = args.task
rscale = args.rscale #float(sys.argv[4])
sub = bool(args.sub) #bool(sys.argv[5])
print("     SUDD    SUPD    SUHF   SUPDk")

filelist = runcmd("ls %s" % task).strip().split('\n')
print(filelist)

#exit()

x = []
e_dd = []
e_pd = []
e_su = []
e_supdk = []
for s in filelist:
    #r = s.replace(task, '').replace('.out', '')
    r = s.split('_')[1].split('.')[0]
    #print(s, r)
    runcmd("cp %s tmp" % s)
    runcmd("sed -i 's/://' tmp")
    p = runcmd("grep ^E_ tmp | awk '{print $2}'")
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

exit()
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
