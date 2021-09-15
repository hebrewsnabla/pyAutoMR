from pyscf import dft
from automr import guess

mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()
mf2 = guess.check_stab(mf, newton=True)

