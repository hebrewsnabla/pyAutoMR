from pyscf import lib
#from pyphf import util, guess
from automr import guess

lib.num_threads(4)

xyz = 'F 0.0 0.0 0.0; F 0.0 0.0 1.9' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [1,-1], cycle=50, 
                     rmdegen=True, bgchg = [-0.1, -0.1])
guess.check_stab(mf)

