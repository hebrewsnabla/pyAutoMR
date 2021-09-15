from pyscf import gto, dft
from automr import guess

#mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()

mol = gto.Mole(atom='''V 0.0 0.0 0.0; V 0.0 0.0 1.77''', basis='def2-tzvp', spin=2)
mol.verbose = 4
mol.build()
mf = dft.ROKS(mol)
mf.xc = 'pbe0'
mf.kernel()

mf2 = guess.check_stab(mf, newton=True, res=True)

