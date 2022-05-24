from automr import guess, dump_mat
from automr.autocas import get_uno
from pyscf import lib
from pyscf.lo import PM

lib.num_threads(4)

xyz = '''Cr .0 .0 .0; V .0 .0 2.0'''
bas = 'def2-svp'
mf = guess.gen(xyz, bas, 0, 9, xc='pbe')

mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)

act_idx = range(ncore, ncore+nacto)
loc_orb = PM(mf.mol, mf.mo_coeff[:,act_idx], mf).kernel()
dump_mat.dump_mo(mf.mol, loc_orb, ncol=10)
#mf.mo_coeff[:, act_idx] = loc_orb.copy()

import numpy as np
def mulliken(mol, mo):
    atm_loc = {}
    S = mol.intor_symmetric('int1e_ovlp')
    theta = lib.einsum('ai, ab, bi -> ia', mo, S, mo)
    for i in range(theta.shape[0]):
        print('LMO %d' %i, end='')
        chg = np.zeros(mol.natm)
        for a, s in enumerate(mol.ao_labels(fmt=None)):
            chg[s[0]] += theta[i,a]
        if chg.max() > 0.65:
            max_idx = chg.argmax()
            if max_idx in atm_loc:
                atm_loc[max_idx].append(i)
            else:
                atm_loc[max_idx] = [i]
        for s in range(mol.natm):
            if chg[s] > 0.05:
                print('  %d%2s  %.2f  ' % (s, mol.atom_symbol(s), chg[s]), end='')
        print('')
    return atm_loc

def flip(mf, act_idx, loc_orb, site):
    mo = mf.mo_coeff
    mo_core = mo[:, :act_idx[0]]
    #mo_ext = mo[:, act_idx[-1]+1:]
    for 

atm_loc = mulliken(mf.mol, loc_orb)
print(atm_loc)


