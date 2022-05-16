from pyscf import lib
#from pyphf import util, guess
from automr import guess, autocas, cidump

lib.num_threads(4)

xyz = 'H 0.0 0.0 0.0; H  0.0 0.0 1.9' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf = guess.mix(xyz, bas, conv='tight')
#mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
#mf = guess.check_stab(mf)

mf2 = autocas.cas(mf, lmo=False)
# UNO -> CASSCF, no localization in this case.


#cidump.dump(mf2)

