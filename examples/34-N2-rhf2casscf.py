from pyscf import lib
#from pyphf import util, guess
import guess, autocas, cidump

lib.num_threads(4)

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.0' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf = guess.gen(xyz, bas, 0, 0)
#guess.check_stab(mf)

mf2 = autocas.cas(mf)
cidump.dump(mf2)
