from pyscf import gto, dft, mcscf
from automr import guess, dump_mat, cidump
import numpy as np
#mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()
for r in np.arange(4.5, 4.55, 0.25):
    mol = gto.Mole(atom='''N 0.0 0.0 0.0; N 0.0 0.0 %f'''%r, basis='ccpvdz', verbose=5 #, symmetry='D2h'
).build()
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    dump_mat.dump_mo(mol, mf.mo_coeff[:,:12])
    print(mf.mo_energy[:12])
    
    #mf2 = guess.check_stab(mf, newton=True, res=True)
    #dump_mat.dump_mo(mol, mf2.mo_coeff[:,:12])
    #print(mf2.mo_energy[:12])
    #print('scan %.2f, %.6f' % (r, mf2.e_tot))
    mc = mcscf.CASSCF(mf, 6, 6)
    mc = mc.state_average_([0.25, 0.25, 0.25, 0.25])
    mc.fix_spin_(ss=0)
    mc.kernel()
    dump_mat.dump_mo(mol, mc.mo_coeff[:,:12])
    cidump.dump(mc)
    
