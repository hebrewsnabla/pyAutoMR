from pyscf import dft
from automr import guess

mf=guess.from_fch_simp("v2.fchk", cycle=0)

mf2 = dft.UKS(mf.mol)    
mf2.xc='pbe0'
mf2.max_cycle=1
mf2.kernel()

mf2.mo_coeff = mf.mo_coeff   
dm = mf2.make_rdm1()
mf2.max_cycle=50
mf2.kernel(dm)

#mf2.verbose=9
#mf2.stability()
guess.check_stab(mf2, newton=True)

