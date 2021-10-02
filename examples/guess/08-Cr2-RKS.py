from pyscf import gto, dft
from automr import guess

#mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()

mol = gto.Mole(atom='''Cr 0.0 0.0 0.0; Cr 0.0 0.0 1.6''', basis='def2-tzvp', verbose=5).build()
mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()

mf2 = guess.check_stab(mf, newton=True, res=True)

