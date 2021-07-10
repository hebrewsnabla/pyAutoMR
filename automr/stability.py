import numpy
#import scipy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
#from pyscf.scf import hf, hf_symm, uhf_symm
#from pyscf.scf import _response_functions  # noqa
from pyscf.soscf import newton_ah
from pyscf.scf.stability import _rotate_mo

def uhf_internal(mf, with_symmetry=True, verbose=None):
    log = logger.new_logger(mf, verbose)
    log.note('**** checking UHF/UKS internal stability ...')
    g, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ,
                                            with_symmetry=with_symmetry)
    hdiag *= 2
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    def hessian_x(x): # See comments in function rhf_internal
        return hop(x).real * 2

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[numpy.argmin(hdiag)] = 1
    e, v = lib.davidson(hessian_x, x0, precond, tol=1e-4, verbose=log)
    if e < -1e-5:
        log.note('UHF/UKS wavefunction has an internal instability.')
        nocca = numpy.count_nonzero(mf.mo_occ[0]> 0)
        nvira = numpy.count_nonzero(mf.mo_occ[0]==0)
        mo = (_rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[:nocca*nvira]),
              _rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[nocca*nvira:]))
        stable = False
    else:
        log.note('UHF/UKS wavefunction is stable in the internal stability analysis')
        mo = mf.mo_coeff
        stable = True
    return mo, stable