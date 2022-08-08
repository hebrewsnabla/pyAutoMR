from automr import guess, dump_mat
#from automr.autocas import get_uno
from pyscf import lib
#from pyscf.lo import PM

lib.num_threads(4)

xyz = '''Cr .0 .0 .0; V .0 .0 2.0'''
bas = 'def2-svp'

mf0 = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [6,-5])
mf0.stability()
mf1 = guess.flipspin(xyz, bas, 11, 'site', site=[1])
mf1.stability()
