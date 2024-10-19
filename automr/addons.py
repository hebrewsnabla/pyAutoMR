from pyscf import gto
import numpy as np
import scipy
from mokit.lib.assoc_rot import assoc_loc
from pyscf.lo.boys import dipole_integral

def addvir_by_aochar(mf, aolabels, nadd, nocc, mo_coeff=None):
    mol = mf.mol
    pmol = mol.copy()
    #pmol.atom = mol._atom
    #pmol.unit = 'B'
    #pmol.symmetry = False
    pmol.basis = 'minao'
    pmol.build(False, False)
    baslst = pmol.search_ao_label(aolabels)
    print('reference AO indices for %s %s:\n %s'%(
             'minao', aolabels, baslst))
    
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    mo_vir = mo_coeff[:, nocc:]
    s2 = pmol.intor_symmetric('int1e_ovlp')[baslst][:,baslst]
    s21 = gto.intor_cross('int1e_ovlp', pmol, mol)[baslst]
    s21 = np.dot(s21, mo_vir)
    sa = s21.T.dot(scipy.linalg.solve(s2, s21, assume_a='pos'))

    w, u = np.linalg.eigh(sa)
    print(w)
    mo_kept = mo_coeff[:,nocc:].dot(u[:,:-nadd])
    mo_add = mo_coeff[:,nocc:].dot(u[:,-nadd:])
    mo_coeff_new = np.hstack((mo_coeff[:,:nocc], mo_add, mo_kept))
    return mo_coeff_new

def addvir(mf, orb_ref, nocc, mo_coeff=None):
    '''
    Find antibonding orbitals of given orb_ref from the space spanned by all virtual orbitals.
    Then place them after the active orbitals.
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    orb_rot = (nocc, mo_coeff.shape[1])
    return addvir_generic(mf, orb_ref, orb_rot, mo_coeff)

def addvir_generic(mf, orb_ref, orb_rot, mo_coeff=None):
    '''
    Find antibonding orbitals of given orb_ref from the space spanned by orb_rot.
    Then place them at the beginning of orb_rot.
    '''
    mol = mf.mol
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nbf, nif = mo_coeff.shape
    mo_dipole = dipole_integral(mol, mo_coeff)
    bonding_start, bonding_end = orb_ref
    antibonding_start, antibonding_end = orb_rot
    new_mo = assoc_loc(nbf, nif, bonding_start, bonding_end, 
                       antibonding_start, antibonding_end, mo_coeff, mo_dipole)
    return new_mo