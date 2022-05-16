from pyscf import lib
#from pyphf import util, guess
from automr import guess, autocas, cidump

lib.num_threads(4)

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.0' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf = guess.gen(xyz, bas, 0, 0)
#guess.check_stab(mf)

#mf2 = autocas.cas(mf, lmo='pm', gvb=True)
# This will invoke CAS(4,4)
mf2 = autocas.cas(mf, lmo='pm', gvb=True, act_user=(6,(3,3)))
# User can enlarge the active space
