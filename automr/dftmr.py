from automr import autocas, mcpdft
from pyscf import mcscf, lib
from pyscf.dft import rks, uks
import numpy as np
from functools import partial

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, np.ndarray):
        dm = np.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = np.asarray((dm*.5,dm*.5))
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    #t0 = (logger.process_clock(), logger.perf_counter())

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.grids)
        #t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.nlcgrids = rks.prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.nlcgrids)
            #t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        if ks.nlc:
            assert 'VV10' in ks.nlc.upper()
            _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm[0]+dm[1],
                                      max_memory=max_memory)
            exc += enlc
            vxc += vnlc
        #logger.debug(ks, 'nelec by numeric integration = %s', n)
        #t0 = logger.timer(ks, 'vxc', *t0)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj = ks.get_j(mol, ddm[0]+ddm[1], hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm[0]+dm[1], hermi)
        vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, ddm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj = vj[0] + vj[1] + vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vj = vj[0] + vj[1]
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk
        
        if ground_state:
            exc0 = exc
            ek = -(np.einsum('ij,ji', dm[0], vk[0]).real +
                   np.einsum('ij,ji', dm[1], vk[1]).real) * .5
    if ground_state:
        ecoul = np.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None

    #vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc, exc0, ecoul, ek

def ks_decomp(ks):
    e_nn = ks.energy_nuc()
    dm1 = ks.make_rdm1()
    dm1t = dm1[0] + dm1[1]
    e_core = einsum('ij,ji->', ks.get_hcore(), dm1t)
    vhf, e_xcdft, e_coul, e_xhf = get_veff(ks)
    #e_coul = vhf.ecoul
    #print('e_mc   : %15.8f' % e_mcscf)
    print('e_nn   : %15.8f' % e_nn)
    print('e_core : %15.8f' % e_core)
    print('e_coul : %15.8f' % e_coul)
    print('e_xhf    : %15.8f' % e_xhf)
    print('e_xcdft  : %15.8f' % e_xcdft)
    #    print('e_otc  : %15.8f' % e_otc)
    #    print('e_c    : %15.8f' % e_c)


def dftcasci(ks, act_user):
    #mo = ks.mo_coeff
    ks_decomp(ks)
    nacto = act_user[0]
    nacta, nactb = act_user[1]
    nopen = nacta - nactb
    mf, unos, unoon, _, _, ndb, nex = autocas.get_uno(ks, uks=True)

    mc = mcscf.CASCI(ks,nacto,(nacta,nactb))
    #mc.fcisolver.max_memory = max_memory // 2
    #mc.max_memory = max_memory // 2
    #mc.max_cycle = 200
    mc.fcisolver.spin = nopen
    mc.fcisolver.max_cycle = 100
    mc.natorb = True
    mc.verbose = 4
    mc.kernel()

    e_nn, e_core, e_coul, e_x, _, _, e_c = mcpdft.get_energy_decomposition(mc)