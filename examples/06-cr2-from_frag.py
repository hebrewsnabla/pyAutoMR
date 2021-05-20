from pyscf import lib
import guess

lib.num_threads(4)

xyz = 'Cr 0.0 0.0 0.0; Cr  0.0 0.0 2.00' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'def2-svp'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [6,-6], cycle=100)
guess.check_stab(mf)

#mf2 = util.SUHF(mf)
#mf2.cut_no = False
#mf2.verbose = 4
#mf2.diis_on = True
#mf2.diis_start_cyc = 5
#mf2.level_shift = 0.5
#mf2.max_cycle = 50
#mf2.kernel()
# E = -2086.32909103
