#from uno import uno
#from construct_vir import construct_vir
from pyscf import mcscf, mrpt
import numpy as np
import scipy
from functools import partial, reduce

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)
np.set_printoptions(precision=6, linewidth=160, suppress=True)

def get_uno(mf, st='st2'):
    mol = mf.mol 
    mo = mf.mo_coeff
    S = mol.intor_symmetric('int1e_ovlp')
    na, nb = mf.nelec
    nopen = na - nb
    if st=='st1':
        # deprecated
        unos, noon, nacto = get_uno_st1(mo, mo_occ, S)
    elif st=='st2':
        dm = mf.make_rdm1()
        unos, noon = get_uno_st2(dm[0] + dm[1], S)
        nacto = check_uno(noon)
    print('UNO ON:', noon)
    #ndb, nocc, nopen = idx
    #nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2
    print('nacto, nacta, nactb: %d %d %d' % (nacto, nacta, nactb))
    mf = mf.to_rhf()
    mf.mo_coeff = unos
    return mf, unos, noon, nacto, (nacta, nactb)

def check_uno(noon, thresh=1.98):
    n1 = np.count_nonzero(noon < thresh)
    n2 = np.count_nonzero(noon > (2.0-thresh))
    nacto = n1 + n2 - len(noon)
    return nacto

def get_uno_st2(dm, S):
    A = reduce(np.dot, (S, dm, S))
    w, v = scipy.linalg.eigh(A, b = S)

    # Flip NOONs (and NOs) since they're in increasing order
    noons = np.flip(w)
    natorbs = np.flip(v, axis=1)
    return natorbs, noons

def get_uno_st1(mo, mo_occ, S, thresh=0.98):
    moa, mob = mo
    #nbf, nif = mf.mo_coeff[0].shape
    # transform UHF canonical orbitals to UNO
    na = np.sum(mo_occ[0]==1)
    nb = np.sum(mo_occ[1]==1)
    print('na,nb', na, nb)
    moab_ovlp = einsum('ji,jk,kl->il', moa[:,:na], S, mob[:,:nb])
    u, sv_occ, v = scipy.linalg.svd(moab_ovlp)
    print('sv_occ', sv_occ)
    ca = np.dot(moa[:,:na], u)
    cb = np.dot(mob[:,:nb], v.T)
    print(ca, '\n', cb)
    nact = np.count_nonzero(sv_occ < thresh)
    nopen = na - nb
    ndb = na - nact
    nact0 = nact - nopen
    nacto = nact0*2 + nopen
    noon = np.zeros(ndb + nacto)
    noon[:na] += 1.0 + sv_occ
    noon[na:] += 2.0 - np.flip(noon[ndb:ndb+nact0])
    print(noon)
    unos = np.zeros_like(moa)
    unos[:,:ndb] += ca[:,:ndb]
    if nopen > 0:
        unos[:,nb:na] += ca[:,nb:na]
    unos[:,ndb:nb] += (ca[:,ndb:nb] + cb[:,ndb:nb]) / np.sqrt(2.0 * noon[ndb:nb])
    unos[:,na:ndb+nacto] += np.flip(ca[:,ndb:nb] - cb[:,ndb:nb], axis=-1) / np.sqrt(2.0 * noon[na:na+nacto])
    #unos[:,ndb+nacto:] 
    print(unos)
    #alpha_coeff = construct_vir(nbf, nif, idx[1], alpha_coeff, S)
    #print(alpha_coeff.shape)
    # done transform

    return unos, noon, nacto

def cas(mf, crazywfn=False, max_memory=2000):
    mf, unos, unoon, nacto, (nacta, nactb) = get_uno(mf)
    nopen = nacta - nactb
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

def nevpt2(mc):
    nev = mrpt.NEVPT(mc)
    nev.kernel()