import time
import numpy as np
from scipy import linalg
from pyscf import mcscf, fci, ao2mo
from pyscf.lib import logger
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft.mcpdft import get_E_ot, _PDFT
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density

from functools import partial
print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

# mc is mcscf.CASSCF obj
def get_energy_decomposition (mc, ot, mo_coeff=None, ci=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear potential, core, Coulomb, exchange,
    and correlation terms. The exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already hybrids. Most return arguments
    are lists if mc is a state average instance. '''
    #e_tot, e_ot, e_mcscf, e_cas, ci, mo_coeff = mc.kernel (mo=mo_coeff, ci=ci)[:6]
    e_mcscf = mc.e_tot
    ci = mc.ci
    mo_coeff = mc.mo_coeff
    #if isinstance (mc, StateAverageMCSCFSolver):
    #    e_tot = mc.e_states
    #e_nuc = mc._scf.energy_nuc ()
    #h = mc.get_hcore ()
    xfnal, cfnal = ot.split_x_c ()
    #if isinstance (mc, StateAverageMCSCFSolver):
    #    e_core = []
    #    e_coul = []
    #    e_otx = []
    #    e_otc = []
    #    e_wfnxc = []
    #    for ci_i, ei_mcscf in zip (ci, e_mcscf):
    #        row = _get_e_decomp (mc, ot, mo_coeff, ci_i, ei_mcscf, xfnal, cfnal)
    #        e_core.append  (row[0])
    #        e_coul.append  (row[1])
    #        e_otx.append   (row[2])
    #        e_otc.append   (row[3])
    #        e_wfnxc.append (row[4])
    
    #else:
    if True:
        e_nn, e_core, e_coul, e_x, e_otx, e_otc, e_c = _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, xfnal, cfnal)
        print('e_mc   : %15.8f' % e_mcscf)
        print('e_nn   : %15.8f' % e_nn)
        print('e_core : %15.8f' % e_core)
        print('e_coul : %15.8f' % e_coul)
        print('e_x    : %15.8f' % e_x)
        print('e_otx  : %15.8f' % e_otx)
        print('e_otc  : %15.8f' % e_otc)
        print('e_c    : %15.8f' % e_c)
    return e_nn, e_core, e_coul, e_x, e_otx, e_otc, e_c

def _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, xfnal, cfnal):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    _rdms = mcscf.CASCI (mc._scf, ncas, nelecas)
    _rdms.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    _rdms.mo_coeff = mo_coeff
    _rdms.ci = ci
    _casdms = _rdms.fcisolver
    _scf = _rdms._scf.to_uhf()
    dm1s = np.stack (_rdms.make_rdm1s (), axis=0)
    dm1 = dm1s[0] + dm1s[1]
    h = _scf.get_hcore()
    j,k = _scf.get_jk (dm=dm1s)
    e_nn = _scf.energy_nuc ()
    e_core = np.tensordot (h, dm1, axes=2)
    #print(j.shape, k.shape, dm1.shape)
    e_coul = np.tensordot (j[0] + j[1], dm1, axes=2) / 2
    e_x = -(np.tensordot (k[0], dm1s[0]) + np.tensordot (k[1], dm1s[1])) / 2
    adm1s = np.stack (_casdms.make_rdm1s (ci, ncas, nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (_casdms.make_rdm12 (_rdms.ci, ncas, nelecas)[1], adm1s)
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    e_otx = get_E_ot (xfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_otc = get_E_ot (cfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_c = e_mcscf - e_nn - e_core - e_coul - e_x
    return e_nn, e_core, e_coul, e_x, e_otx, e_otc, e_c

def get_E_ot (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=2000, hermi=1):
    ''' E_MCPDFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[rho,Pi] 
        or, in other terms, 
        E_MCPDFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[rho, Pi]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[rho, Pi] 
        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 2000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2amo.shape[0]

    E_ot = 0.0

    t0 = (time.process_time (), time.time ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        if ot.verbose > logger.DEBUG and dens_deriv > 0:
            for ideriv in range (1,4):
                rho_test  = einsum ('ijk,aj,ak->ia', oneCDMs, ao[ideriv], ao[0])
                rho_test += einsum ('ijk,ak,aj->ia', oneCDMs, ao[ideriv], ao[0])
                logger.debug (ot, "Spin-density derivatives, |PySCF-einsum| = %s", linalg.norm (rho[:,ideriv,:]-rho_test))
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv, mask)
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top exchange-correlation energy calculation', *t0)

    return E_ot

class PDFT(_PDFT):
    def __init__(self, mc, my_ot):
        self.mc = mc
        self.mol = mc.mol
        self.verbose = 5
        self.stdout = mc.stdout
        self._init_ot_grids(my_ot, grids_level=4)
    def kernel(self):
        ot = self.otfnal
        mc = self.mc
        get_energy_decomposition (mc, ot)
