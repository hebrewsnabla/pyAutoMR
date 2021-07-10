#from uno import uno
#from construct_vir import construct_vir
from pyscf import mcscf, mrpt, gto, scf
import numpy as np
import scipy
from functools import partial, reduce
from lo import pm
from pyscf.lo.boys import dipole_integral
from auto_pair import pair_by_tdm
import dump_mat
import sys

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
        mo_occ = mf.mo_occ
        unos, noon, nacto = get_uno_st1(mo, mo_occ, S)
    elif st=='st2':
        dm = mf.make_rdm1()
        unos, noon = get_uno_st2(dm[0] + dm[1], S)
        nacto, ndb, nex = check_uno(noon)
    print('UNO ON:', noon)
    #ndb, nocc, nopen = idx
    #nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2
    print('nacto, nacta, nactb: %d %d %d' % (nacto, nacta, nactb))
    mf = mf.to_rhf()
    mf.mo_coeff = unos
    print('UNO in active space')
    dump_mat.dump_mo(mol, unos[:,ndb:ndb+nacto], ncol=10)
    return mf, unos, noon, nacto, (nacta, nactb), ndb, nex

def check_uno(noon, thresh=1.98):
    ndb = np.count_nonzero(noon > thresh)
    nex = np.count_nonzero(noon < (2.0-thresh))
    nacto = len(noon) - ndb - nex
    return nacto, ndb, nex

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

def get_locorb(mf, localize='pm1', pair=True):
    mol = mf.mol
    mo = mf.mo_coeff
    nbf = mf.mo_coeff.shape[0]
    nif = mf.mo_coeff.shape[1]
    mol2 = mf.mol.copy()
    mol2.basis = 'sto-6g'
    mol2.build()
    mf2 = scf.RHF(mol2)
    mf2.max_cycle = 150
    #dm = mf2.from_chk('loc_rhf_proj.chk')
    mf2.kernel()
    mo2 = mf2.mo_coeff
    #nbf2 = mf2.mo_coeff.shape[0]
    #nif2 = mf2.mo_coeff.shape[1]
    idx = np.count_nonzero(mf.mo_occ)
    cross_S = gto.intor_cross('int1e_ovlp', mol, mol2)
    print(idx, mo.shape, mo2.shape ,cross_S.shape)
    vir_cross = einsum('ji,jk,kl->il', mo[:,idx:], cross_S, mo2[:,idx:])

    u,s,v = scipy.linalg.svd(vir_cross)
    print('SVD', s)
    projmo = np.dot(mo[:,idx:], u)
    mf.mo_coeff[:, idx:] = projmo

    npair = np.sum(mf2.mo_occ==0)
    if localize=='pm1':
        idx2 = np.count_nonzero(mf.mo_occ)
        idx1 = idx2 - npair
        idx3 = idx2 + npair
        print('MOs after projection')
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,idx1:idx3], ncol=10)
        occ_idx = range(idx1,idx2)
        vir_idx = range(idx2,idx3)
        S = mol.intor_symmetric('int1e_ovlp')
        occ_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[:,occ_idx],S,'mulliken')
        vir_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[:,vir_idx],S,'mulliken')
        mf.mo_coeff[:,occ_idx] = occ_loc_orb.copy()
        mf.mo_coeff[:,vir_idx] = vir_loc_orb.copy()
        print('MOs after PM localization')
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,idx1:idx3], ncol=10)
    
    if pair:
        mo_dipole = dipole_integral(mol, mf.mo_coeff)
        ncore = idx1
        nopen = np.sum(mf.mo_occ==1)
        nalpha = idx2
        #nvir_lmo = npair
        alpha_coeff = pair_by_tdm(ncore, npair, nopen, nalpha, npair, nbf, nif, mf.mo_coeff, mo_dipole)
        mf.mo_coeff = alpha_coeff.copy()
        print('MOs after pairing')
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,idx1:idx3], ncol=10)
    return mf, alpha_coeff, npair, ncore

def check_uhf(mf):
    dm = mf.make_rdm1()
    ndim = np.ndim(dm)
    if ndim==2:
        return False, mf
    elif ndim==3:
        diff = abs(dm[0] - dm[1])
        #print(diff)
        #print(np.max(diff))
        if diff.max() > 1e-4:
            return True, mf
        else:
            mf = mf.to_rhf()
            return False, mf


def cas(mf, act_user=None, crazywfn=False, max_memory=2000, natorb=True):
    is_uhf, mf = check_uhf(mf)
    if is_uhf:
        mf, unos, unoon, nacto, (nacta, nactb), ndb, nex = get_uno(mf)
    else:
        mf, lmos, npair, ndb = get_locorb(mf)
        nacto = npair*2
        nacta = nactb = npair
    nopen = nacta - nactb
    if act_user is not None:
        print('Warning: using user defined active space')
        nacto = act_user[0]
        nacta, nactb = act_user[1]
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
    mc.natorb = natorb
    mc.verbose = 4
    mc.kernel()
    #mc.analyze(with_meta_lowdin=False)
    if natorb:
        print('Natrual Orbs')
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,ndb:ndb+nacto], ncol=10)
    return mc

def nevpt2(mc):
    nev = mrpt.NEVPT(mc)
    nev.kernel()
