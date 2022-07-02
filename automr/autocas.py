#from uno import uno
#from construct_vir import construct_vir
from pyscf import mcscf, mrpt, gto, scf
import numpy as np
import scipy
from functools import partial, reduce
try:
    #from lo import pm
    from auto_pair import pair_by_tdm
    #from assoc_rot import assoc_rot
except:
    print('Warning: auto_pair not found. Install MOKIT if you need that.')
from pyscf.lo import PM
from pyscf.lo.boys import dipole_integral
from automr import dump_mat, bridge, cidump
from automr.util import check_uno, chemcore
import sys, os
#try:
#    from pyphf import suscf
#except:
#    print('Warning: pyphf not found. SUHF is disabled. Install ExSCF if you need that.')
#try:
#    import nof
#except:
#    print('Warning: pyNOF not found. GVB is disabled. Install pyNOF if you need that.')

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)
np.set_printoptions(precision=6, linewidth=160, suppress=True)

def do_suhf(mf):
    from pyphf import suscf
    mol = mf.mol 
    #mo = mf.mo_coeff
    #S = mol.intor_symmetric('int1e_ovlp')
    na, nb = mf.nelec
    nopen = na - nb
    mf2 = suscf.SUHF(mf)
    mf2.kernel()
    no = mf2.natorb[2]
    noon = mf2.natocc[2]
    nacto, ndb, nex = check_uno(noon)
    #ndb, nocc, nopen = idx
    #nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2
    print('nacto, nacta, nactb: %d %d %d' % (nacto, nacta, nactb))
    mf = mf.to_rhf()
    mf.mo_coeff = no
    print('UNO in active space')
    dump_mat.dump_mo(mol, no[:,ndb:ndb+nacto], ncol=10)
    return mf, no, noon, nacto, (nacta, nactb), ndb, nex

def get_gvbno(thenof, mf, thresh=1.98):
    mol = mf.mol 
    mo = thenof.mo_coeff
    S = mol.intor_symmetric('int1e_ovlp')
    na, nb = thenof.nelecas
    nopen = na - nb
    dm = thenof.make_rdm1()
    no, noon = get_uno_st2(dm, S)
    nacto, ndb, nex = check_uno(noon, thresh)
    print('GVBNO ON:', noon)
    #ndb, nocc, nopen = idx
    #nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2
    print('nacto, nacta, nactb: %d %d %d' % (nacto, nacta, nactb))
    mf.mo_coeff = no
    #mf.mo_occ = noon
    print('GVBNO in active space')
    dump_mat.dump_mo(mol, no[:,ndb:ndb+nacto], ncol=10)
    return mf, no, noon, nacto, (nacta, nactb), ndb, nex

def get_uno(mf, st='st2', uks=False, thresh=1.98):
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
        nacto, ndb, nex = check_uno(noon, thresh)
    print('UNO ON:', noon)
    #ndb, nocc, nopen = idx
    #nacto = nocc - ndb
    nacta = (nacto + nopen)//2
    nactb = (nacto - nopen)//2
    print('nacto, nacta, nactb: %d %d %d' % (nacto, nacta, nactb))
    #print(mf.mo_occ[0], mf.mo_occ[1])
    if uks:
        mf = mf.to_rks()
    else:
        mf = mf.to_rhf()
    mf.mo_coeff = unos
    #if mf.mo_occ[0] == mf.mo_occ[1]:
    if mf.mo_occ is None:
        mf.mo_occ = mf.mo_occ[0] + mf.mo_occ[1]
    #else:
    #    mf.mo_occ = np.zeros(unos.shape[1])
    #    mf.mo_occ[
    #mf.mo_occ = noon
    print(mf.mo_occ)
    print('UNO in active space')
    dump_mat.dump_mo(mol, unos[:,ndb:ndb+nacto], ncol=10)
    return mf, unos, noon, nacto, (nacta, nactb), ndb, nex

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

def get_locorb(mf, localize='pm', pair=True):
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
    print('npair', npair)
    idx2 = np.count_nonzero(mf.mo_occ)
    ncore = chemcore(mol)
    idx1 = min(idx2 - npair, ncore)
    idx3 = idx2 + npair
    print('MOs after projection')
    dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,idx1:idx3], ncol=10)
    occ_idx = slice(idx1,idx2)
    vir_idx = slice(idx2,idx3)
    if localize:
        mf = loc(mf, occ_idx, vir_idx, localize)
    if pair:
        mo_dipole = dipole_integral(mol, mf.mo_coeff)
        #ncore = idx1
        nopen = np.sum(mf.mo_occ==1)
        nalpha = idx2
        #nvir_lmo = npair
        alpha_coeff = pair_by_tdm(ncore, npair, nopen, nalpha, npair, nbf, nif, mf.mo_coeff, mo_dipole)
        mf.mo_coeff = alpha_coeff.copy()
        print('MOs after pairing')
        ncore = idx2 - npair
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,idx1:idx3], ncol=10)
    return mf, mf.mo_coeff, npair, ncore

def loc(mf, occ_idx, vir_idx, localize='pm'):
    mol = mf.mol
    if localize=='pm1':
        from lo import pm
        S = mol.intor_symmetric('int1e_ovlp')
        nbf = mf.mo_coeff.shape[0]
        nif = mf.mo_coeff.shape[1]
        npair = len(occ_idx)
        occ_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[:,occ_idx],S,'mulliken')
        vir_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[:,vir_idx],S,'mulliken')
    elif localize=='pm':
        occ_loc_orb = PM(mol, mf.mo_coeff[:,occ_idx], mf).kernel()
        vir_loc_orb = PM(mol, mf.mo_coeff[:,vir_idx], mf).kernel()
    mf.mo_coeff[:,occ_idx] = occ_loc_orb.copy()
    mf.mo_coeff[:,vir_idx] = vir_loc_orb.copy()
    print('MOs after PM localization')
    dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,slice(occ_idx.start, vir_idx.stop)], ncol=10)
    return mf

def loc_asrot(mf, nacto, nelecact, ncore, localize='pm'):
    #idx2 = idx[0] + npair - 1
    #idx3 = idx2 + idx[2]
    #npair = min(npair,3)
    #idx1 = idx2 - npair
    #idx4 = idx3 + npair
    mol = mf.mol
    na, nb = nelecact
    nopen = na-nb
    npair = (nacto - nopen)//2
    occ_idx = slice(ncore,ncore+npair)
    vir_idx = slice(ncore+nacto-npair,ncore+nacto)
    """nbf = mf.mo_coeff.shape[0]
    S = mf.get_ovlp()
    occ_loc_orb = pm(mol.nbas,mol._bas[:,0],mol._bas[:,1],mol._bas[:,3],mol.cart,nbf,npair,mf.mo_coeff[:,occ_idx],S,'mulliken')
    vir_loc_orb = assoc_rot_py(mf.mo_coeff[:,occ_idx], occ_loc_orb, mf.mo_coeff[:,vir_idx])
    mf.mo_coeff[:,occ_idx] = occ_loc_orb.copy()
    mf.mo_coeff[:,vir_idx] = vir_loc_orb.copy()"""
    mf = loc(mf, occ_idx, vir_idx, localize=localize)
    print('MOs after assoc_rot')
    dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,ncore:ncore+nacto], ncol=10)
    #vir_loc_orb2 = assoc_rot(nbf, npair, mf.mo_coeff[:,occ_idx], occ_loc_orb, mf.mo_coeff[:,vir_idx])
    #dump_mat.dump_mo(mf.mol,vir_loc_orb2, ncol=10)
    return mf, npair

def assoc_rot_py(mo_g, mo_g_loc, mo_u):
    #p,l,u = scipy.linalg.lu(mo_g)
    #pl = p @ l
    #y = scipy.linalg.solve_triangular(pl, mo_g_loc)
    #v = scipy.linalg.solve_triangular(u, y)
    #print(mo_g.shape, np.linalg.matrix_rank(mo_g))
    #v0 = np.linalg.pinv(mo_g)
    #print(np.linalg.norm(mo_g @ v0 @ mo_g_loc - mo_g_loc))
    
    v, res, r, s = np.linalg.lstsq(mo_g, mo_g_loc, rcond=-1)
    #print(np.linalg.norm(mo_g @ v - mo_g_loc), res, r, s)
    
    #lu, piv, _ = scipy.linalg.lapack.dgetrf(mo_g)
    #v, _ = scipy.linalg.lapack.dgetrs(lu, piv, mo_g_loc)
    tmp = np.flip(mo_u, axis=1)
    tmp = tmp @ v
    new_mo_u = np.flip(tmp, axis=1)
    return new_mo_u 

def do_gvb_qchem(mf, npair):
    mo = mf.mo_coeff
    basename = str(os.getpid())
    bridge.py2qchem(mf, basename)
    #os.system('mkdir -p /tmp/qchem/' + basename)
    #os.system('cp test.q53 /tmp/qchem/test_qc/53.0
    os.system('qchem %s.in %s.qout %s' % (basename, basename, basename))
    gvbno = bridge.qchem2py(basename)[0]
    mf.mo_coeff = gvbno
    dump_mat.dump_mo(mf.mol, gvbno, ncol=10)
    return mf, gvbno, npair

def do_gvb(mf, npair, ndb):
    import nof
    thenof = nof.SOPNOF(mf, npair*2, npair*2)
    thenof.verbose = 5
    #thenof.mo_occ = noon / 2
    thenof.fcisolver = nof.fakeFCISolver()
    thenof.fcisolver.ncore = ndb
    thenof.fcisolver.npair = npair
    thenof.internal_rotation = True
    thenof.mc2step()

    
    mf, gvbno, noon, nacto, (nacta, nactb), ndb, nex = get_gvbno(thenof, mf, thresh=1.98)
    nopen = nacta - nactb
    npair = (nacto - nopen)//2
    return mf, gvbno, noon, npair


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

def sort_mo(mf, sort, ncore, base=1):
    caslst = sort
    mo_coeff = mf.mo_coeff
    def ext_list(nmo, caslst):
        mask = np.ones(nmo, dtype=bool)
        mask[caslst] = False
        idx = np.where(mask)[0]
        #if len(idx) + casscf.ncas != nmo:
        #    raise ValueError('Active space size is incompatible with caslist. '
        #                     'ncas = %d.  caslist %s' % (casscf.ncas, caslst))
        return idx

    #if isinstance(ncore, (int, numpy.integer)):
    nmo = mo_coeff.shape[1]
    if base != 0:
        caslst = [i-base for i in caslst]
    idx = ext_list(nmo, caslst)
    mo = np.hstack((mo_coeff[:,idx[:ncore]],
                           mo_coeff[:,caslst],
                           mo_coeff[:,idx[ncore:]]))
    mf.mo_coeff = mo
    return mf

def cas(mf, act_user=None, crazywfn=False, max_memory=2000, natorb=True, 
            gvb=False, suhf=False, unothresh=1.98, lmo=False, no=(None,None), sort=None, 
            dry=False, symmetry=None):
    '''
        mf: RHF/UHF object
        lmo: False, 
             'pm',  # PM from pyscf
             'pm1'  # PM from mokit
    '''
    is_uhf, mf = check_uhf(mf)
    if no[0] is not None:
        mo_coeff, noon = no
        nacto, ndb, nex = check_uno(noon)
        nopen = mf.nelec[0] - mf.nelec[1]
        nacta = (nacto + nopen)//2
        nactb = (nacto - nopen)//2
        mf = mf.to_rhf()
        mf.mo_coeff = mo_coeff
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,ndb:ndb+nacto], ncol=10)
    elif is_uhf:
        if suhf:
            mf, unos, unoon, nacto, (nacta, nactb), ndb, nex = do_suhf(mf)
        else:
            mf, unos, unoon, nacto, (nacta, nactb), ndb, nex = get_uno(mf, thresh=unothresh)
        nacto0 = nacto
        if lmo:
            mf, npair = loc_asrot(mf, nacto, (nacta, nactb), ndb, localize=lmo)
            if gvb:
                if nacta - nactb != 0:
                    raise NotImplementedError('GVB for nopen > 0 not implemented')
                mf, gvbno, noon, npair = do_gvb(mf, npair, ndb)
                nacto = npair*2
                nacta = nactb = npair
    else:
        if lmo:
            mf, lmos, npair, ndb = get_locorb(mf, localize=lmo)
            if gvb:
                #nacto0 = npair*2
                mf, gvbno, noon, npair = do_gvb(mf, npair, ndb)
            else:
                print('''Warning: LMO obtained but no GVB performed. 
                                  That's not reliable usually.''')
            nacto = npair*2
            nacta = nactb = npair
        else:
            print('Warning: no lmo step so no special actions on RHF orbitals')
        #    ndb = np.count_nonzero(mf.mo_occ) - nacto//2
        
    if act_user is not None:
        print('Warning: using user defined active space')
        if is_uhf:
            if act_user[0] > nacto0:
                print('Warning: User defined active space larger than UNO-determined space.')
            elif gvb and act_user[0] > nacto:
                print('Warning: User defined active space larger than GVB-determined space.')
        else:
            if gvb and act_user[0] > nacto:
                print('Warning: User defined active space larger than GVB-determined space.')
        nacto = act_user[0]
        nacta, nactb = act_user[1]
        print('user-defined guess orbs')
        if not is_uhf and not lmo:
            ndb = np.count_nonzero(mf.mo_occ) - nacto//2
            if sort is not None:
                mf = sort_mo(mf, sort, ndb)
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,ndb:ndb+nacto], ncol=10)
    nopen = nacta - nactb
    #print(mf.mo_coeff.shape)

    if symmetry is not None:
        mf.mol.build(symmetry=symmetry)
    mc = mcscf.CASSCF(mf,nacto,(nacta,nactb))
    mc.fcisolver.max_memory = max_memory // 2
    mc.max_memory = max_memory // 2
    mc.max_cycle = 200
    mc.fcisolver.spin = nopen
    mc.fix_spin_(ss=nopen)
    if crazywfn:
        mc.fcisolver.level_shift = 0.2
        mc.fcisolver.pspace_size = 1200
        mc.fcisolver.max_space = 100
        mc.fcisolver.max_cycle = 300
    else:
        mc.fcisolver.max_cycle = 100
    mc.natorb = natorb
    mc.verbose = 4
    #if not is_uhf and sort is not None:
    #    mo = mc.sort_mo(sort)
    #    mf.mo_coeff = mo
    if dry:
        return mc
    #print(mc.mo_coeff.shape)
    #print(mc.mo_coeff)
    mc.kernel()
    #mc.analyze(with_meta_lowdin=False)
    if natorb:
        print('Natrual Orbs')
        dump_mat.dump_mo(mf.mol,mf.mo_coeff[:,ndb:ndb+nacto], ncol=10)
    cidump.dump(mc)
    return mc

def nevpt2(mc, root=0):
    nev = mrpt.NEVPT(mc, root=root)
    nev.kernel()
