from uno import uno
from construct_vir import construct_vir
from pyscf import mcscf
import numpy as np

def cas(mf, crazywfn=False, max_memory=2000):
    mol = mf.mol 
    nbf = mf.mo_coeff[0].shape[0]
    nif = mf.mo_coeff[0].shape[1]
    S = mol.intor_symmetric('int1e_ovlp')
    # transform UHF canonical orbitals to UNO
    na = np.sum(mf.mo_occ[0]==1)
    nb = np.sum(mf.mo_occ[1]==1)
    idx, noon, alpha_coeff = uno(nbf,nif,na,nb, mf.mo_coeff[0], mf.mo_coeff[1], S, 0.98)
    alpha_coeff = construct_vir(nbf, nif, idx[1], alpha_coeff, S)
    #print(alpha_coeff.shape)
    mf = mf.to_rhf()
    mf.mo_coeff = alpha_coeff
    # done transform

    ndb, nocc, nopen = idx
    nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2

    mc = mcscf.CASSCF(mf,nacto,(nacta,nactb))
    mc.fcisolver.max_memory = max_memory // 2
    mc.max_memory = max_memory // 2
    mc.max_cycle = 200
    mc.fcisolver.spin = nopen
    if crazywfn:
        mc.fix_spin_(ss=nopen)
        mc.fcisolver.level_shift = 0.2
        mc.fcisolver.pspace_size = 1200
        mc.fcisolver.max_space = 100
        mc.fcisolver.max_cycle = 300
    else:
        mc.fcisolver.max_cycle = 100
    mc.natorb = True
    mc.verbose = 5
    mc.kernel()
    return mc
