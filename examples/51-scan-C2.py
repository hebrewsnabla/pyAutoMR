from pyscf import lib, gto, scf, mcscf, mrpt
from automr import dump_mat, cidump
import os, contextlib
import numpy as np

lib.num_threads(16)

for r in np.arange(1.4, 1.45, 0.1):
    output = 'c2_%.1f.out' % r
    os.system("echo '\n' > %s" % output)
    print('scan %.1f' % r)
    with open(output, 'a', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            xyz = '''C 0.0 0.0 0.0; C 0.0 0.0 %f'''%r
            mol = gto.M(atom=xyz, basis='cc-pvtz')
            mf = scf.RHF(mol)
            mf.kernel()
            
            mc = mcscf.CASSCF(mf, 8, (4,4))
            mc.fix_spin_(ss=0)
            mc.fcisolver.nroots = 3
            mc = mc.state_specific_(0)
            mc.run()
            
            dump_mat.dump_mo(mol, mc.mo_coeff[:,:12])
            cidump.dump(mc)

            #mc2 = mcscf.CASCI(mf, 8, (4,4))
            #mc2.fix_spin_(ss=0)
            #mc2.fcisolver.nroots = 3
            #mc2.kernel(mc.mo_coeff)
            #cidump.dump(mc2)

            e_corr = mrpt.NEVPT(mc).kernel()
            e_tot = mc.e_tot + e_corr
            print('Total energy of ground state', e_tot)
