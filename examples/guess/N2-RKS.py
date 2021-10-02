from pyscf import gto, dft
from automr import guess, dump_mat
import numpy as np
#mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()
for r in np.arange(3.0, 9.0, 0.25):
    mol = gto.Mole(atom='''N 0.0 0.0 0.0; N 0.0 0.0 %f'''%r, basis='def2-svp', verbose=5).build()
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    dump_mat.dump_mo(mol, mf.mo_coeff[:,:12])
    print(mf.mo_energy[:12])
    
    mf2 = guess.check_stab(mf, newton=True, res=True)
    dump_mat.dump_mo(mol, mf2.mo_coeff[:,:12])
    print(mf2.mo_energy[:12])
    print('scan %.2f, %.6f' % (r, mf2.e_tot))
