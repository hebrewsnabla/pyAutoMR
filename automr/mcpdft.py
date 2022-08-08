import time
import numpy as np
from scipy import linalg
from pyscf import mcscf, fci, ao2mo
from pyscf.dft import gen_grid
from pyscf.lib import logger
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
from mrh.my_pyscf.mcpdft.mcpdft import _PDFT
from mrh.my_pyscf.mcpdft.otfnal import energy_ot as get_E_ot
from mrh.my_pyscf.mcpdft.otfnal import get_transfnal
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density

from functools import partial
print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

# mc is mcscf.CASSCF obj
def get_energy_decomposition (mc, ot=None, mo_coeff=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear potential, core, Coulomb, exchange,
    and correlation terms. The exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already hybrids. Most return arguments
    are lists if mc is a state average instance. '''
    e_mcscf = mc.e_tot
    #ci = mc.ci
    mo_coeff = mc.mo_coeff
    
    print('energy decomposition of MCPDFT')
    e_nn, e_core, e_coul, e_x,  e_c = _get_e_decomp (mc, mo_coeff)
    e_otx, e_otc = get_pd(mc, ot, mo_coeff)
    print('e_mc   : %15.8f' % e_mcscf)
    print('e_nn   : %15.8f' % e_nn)
    print('e_core : %15.8f' % e_core)
    print('e_coul : %15.8f' % e_coul)
    print('e_x    : %15.8f' % e_x)
    print('e_c    : %15.8f' % e_c)
    print('e_otx  : %15.8f' % e_otx)
    print('e_otc  : %15.8f' % e_otc)
    return e_nn, e_core, e_coul, e_x, e_otx, e_otc, e_c

def _get_e_decomp (mc, mo_coeff):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    if isinstance(mc, mcscf.casci.CASCI):
        _rdms = mc
        _casdms = mc.fcisolver
    else:
        _rdms = mcscf.CASCI (mc._scf, ncas, nelecas)
        _rdms.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
        _rdms.mo_coeff = mo_coeff
        _rdms.ci = mc.ci
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
    e_c = mc.e_tot - e_nn - e_core - e_coul - e_x
    
    return e_nn, e_core, e_coul, e_x, e_c

def get_pd(mc, ot, mo_coeff):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    if isinstance(mc, mcscf.casci.CASCI):
        _rdms = mc
        _casdms = mc.fcisolver
    else:
        _rdms = mcscf.CASCI (mc._scf, ncas, nelecas)
        _rdms.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
        _rdms.mo_coeff = mo_coeff
        _rdms.ci = mc.ci
        _casdms = _rdms.fcisolver
    e_otx = 0.0; e_otc = 0.0
    
    xfnal, cfnal = ot.split_x_c ()
    adm1s = np.stack (_casdms.make_rdm1s (_rdms.ci, ncas, nelecas), axis=0)
    adm2s = _casdms.make_rdm12s (_rdms.ci, ncas, nelecas)[1]
    adm2 = sum_adm2(adm2s)
    #mo_cas = mo_coeff[:,ncore:ncore+ncas]
    e_otx = get_E_ot (xfnal, adm1s, adm2, mo_coeff, _rdms.ncore, max_memory=mc.max_memory)
    e_otc = get_E_ot (cfnal, adm1s, adm2, mo_coeff, _rdms.ncore, max_memory=mc.max_memory)
    return e_otx, e_otc

def sum_adm2(adm2s):
    #print(adm2s)
    #adm2s = get_2CDMs_from_2RDMs (adm2s, adm1s)
    adm2_ss = adm2s[0] + adm2s[2]
    adm2_os = adm2s[1]
    adm2 = adm2_ss + adm2_os + adm2_os.transpose (2,3,0,1)
    return adm2 

def get_E_ot2 (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=2000, hermi=1):
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

def kernel(pdft, mc, ot):
    print('\n******** %s ********' % pdft.__class__)
    return get_energy_decomposition (mc, ot)

class PDFT():
    def __init__(self, mc, my_ot, grids_level=4):
        self.mc = mc
        self.mol = mc.mol
        self.verbose = 5
        self.stdout = mc.stdout
        self.grids_level = grids_level
        grids_attr = {}
        if self.grids_level is not None:
            grids_attr['level'] = self.grids_level
        self._init_ot(my_ot)
        self._init_grids(grids_attr)
    
    def kernel(self):
        return kernel(self, self.mc, self.otfnal)

    def _init_grids(self, grids_attr=None):
        if grids_attr is None: grids_attr = {}
        #old_grids = getattr (self, 'grids', None)
        if hasattr(self, 'otfnal'):
            self.otfnal.grids = gen_grid.Grids(self.mol)
            self.otfnal.grids.__dict__.update (grids_attr)
            self.otfnal.grids.build()
            self.grids = self.otfnal.grids
        else:
            self.grids = gen_grid.Grids(self.mol)
            self.grids.__dict__.update (grids_attr)
            self.grids.build()
        #for key in grids_attr:
        #    assert (getattr (self.grids, key, None) == getattr (
        #        self.otfnal.grids, key, None))

    def _init_ot (self, my_ot):
        if isinstance (my_ot, (str, np.string_)):
            self.otfnal = get_transfnal (self.mol, my_ot)
        else:
            self.otfnal = my_ot
        # Make sure verbose and stdout don't accidentally change 
        # (i.e., in scanner mode)
        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout
