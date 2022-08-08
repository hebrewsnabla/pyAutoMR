from pyscf import lib
#from pyphf import util, guess
from automr import guess

lib.num_threads(4)


xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.9' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = 'cc-pvdz'

mf0 = guess.mix(xyz, bas, conv='tight')
mf1 = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
mf1.stability()
#guess.check_stab(mf)
#mf2 = guess.flipspin(xyz, bas, 6, 'lmo', [3,4,5], cycle=0)
mf3 = guess.flipspin(xyz, bas, 6, 'site', site=[1])
mf3.stability()


xyz = 'C 0.0 0.0 0.0; C 0.0 0.0 2.0'
bas = 'cc-pvdz'
mf1 = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [4,-4], cycle=20)
if not mf1.converged: mf1.newton().run()
mf1.stability()
mf3 = guess.flipspin(xyz, bas, 8, 'site', site=[1], cycle=20)
if not mf3.converged: mf3.newton().run()
mf3.stability()


xyz = 'Li 0.0 0.0 0.0; Li 0.0 0.0 4.0'
bas = 'cc-pvdz'
mf0 = guess.mix(xyz, bas, conv='tight')
mf1 = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [1,-1], cycle=20)
if not mf1.converged: mf1.newton().run()
mf1.stability()
mf3 = guess.flipspin(xyz, bas, 2, 'site', site=[1], cycle=20)
if not mf3.converged: mf3.newton().run()
mf3.stability()
