from pyscf import lib
#from pyphf import util, guess
from automr import guess, autocas, cidump

lib.num_threads(4)

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.9' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
mf = guess.check_stab(mf)

mf2 = autocas.cas(mf)
#cidump.dump(mf2)

# UNO -> CASSCF, no localization in this case.
