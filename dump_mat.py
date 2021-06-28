import sys

def dump_mo(mol, c, label2=None,
             ncol=6, digits=3, start=0):
    ''' Print an array in rectangular format
    Args:
        mol: `gto.Mole` object
        c : numpy.ndarray
            coefficients
    Kwargs:
        label2 : list of strings
            Col labels (default is 1,2,3,4,...)
        ncol : int
            Number of columns in the format output (default 5)
        digits : int
            Number of digits of precision for floating point output (default 5)
        start : int
            The number to start to count the index (default 0)
    '''
    stdout = sys.stdout
    label = mol.ao_labels()
    nc = c.shape[1]
    if label2 is None:
        fmt = '#%%-%dd' % (digits+3)
        label2 = [fmt%i for i in range(start,nc+start)]
    else:
        fmt = '%%-%ds' % (digits+4)
        label2 = [fmt%i for i in label2]
    for ic in range(0, nc, ncol):
        dc = c[:,ic:ic+ncol]
        m = dc.shape[1]
        fmt = (' %%%d.%df'%(digits+4,digits))
        #if label is None:
        #    stdout.write(((' '*(digits+3))+'%s\n') % ' '.join(label2[ic:ic+m]))
        #    for k, v in enumerate(dc):
        #        stdout.write(('%-5d' % (k+start)) + (fmt % tuple(v)))
        #else:
        stdout.write(((' '*(digits+10))+'%s\n') % ' '.join(label2[ic:ic+m]))
        for k, v in enumerate(dc):
            coefs = ''
            for j in v:
                if abs(j) < 0.1:
                    coefs += '        '
                else:
                    coefs += fmt % j
            if len(coefs.strip()) == 0:
                continue
            stdout.write(('%12s' % label[k]) + coefs + '\n')
    stdout.flush()

'''
def analyze(casscf, mo_coeff=None, ci=None, verbose=None,
            large_ci_tol=LARGE_CI_TOL, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    from pyscf.mcscf import addons
    log = logger.new_logger(casscf, verbose)

    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ci is None: ci = casscf.ci
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncore + ncas
    mocore = mo_coeff[:,:ncore]
    mocas = mo_coeff[:,ncore:nocc]

    label = casscf.mol.ao_labels()
    if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
        not isinstance(casscf.fcisolver, addons.StateAverageFCISolver)):
        log.warn('Mulitple states found in CASCI/CASSCF solver. Density '
                 'matrix of the first state is generated in .analyze() function.')
        civec = ci[0]
    else:
        civec = ci
    if getattr(casscf.fcisolver, 'make_rdm1s', None):
        casdm1a, casdm1b = casscf.fcisolver.make_rdm1s(civec, ncas, nelecas)
        casdm1 = casdm1a + casdm1b
        dm1b = numpy.dot(mocore, mocore.conj().T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.conj().T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.conj().T))
        dm1 = dm1a + dm1b
        if log.verbose >= logger.DEBUG2:
            log.info('alpha density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1a, label, **kwargs)
            log.info('beta density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1b, label, **kwargs)
    else:
        casdm1 = casscf.fcisolver.make_rdm1(civec, ncas, nelecas)
        dm1a = (numpy.dot(mocore, mocore.conj().T) * 2 +
                reduce(numpy.dot, (mocas, casdm1, mocas.conj().T)))
        dm1b = None
        dm1 = dm1a

    if log.verbose >= logger.INFO:
        ovlp_ao = casscf._scf.get_ovlp()
        # note the last two args of ._eig for mc1step_symm
        occ, ucas = casscf._eig(-casdm1, ncore, nocc)
        log.info('Natural occ %s', str(-occ))
        mocas = numpy.dot(mocas, ucas)
        if with_meta_lowdin:
            log.info('Natural orbital (expansion on meta-Lowdin AOs) in CAS space')
            orth_coeff = orth.orth_ao(casscf.mol, 'meta_lowdin', s=ovlp_ao)
            mocas = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mocas))
        else:
            log.info('Natural orbital (expansion on AOs) in CAS space')
        dump_mat.dump_rec(log.stdout, mocas, label, start=1, **kwargs)
        if log.verbose >= logger.DEBUG2:
            if not casscf.natorb:
                log.debug2('NOTE: mc.mo_coeff in active space is different to '
                           'the natural orbital coefficients printed in above.')
            if with_meta_lowdin:
                c = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff))
                log.debug2('MCSCF orbital (expansion on meta-Lowdin AOs)')
            else:
                c = mo_coeff
                log.debug2('MCSCF orbital (expansion on AOs)')
            dump_mat.dump_rec(log.stdout, c, label, start=1, **kwargs)

        if casscf._scf.mo_coeff is not None:
            addons.map2hf(casscf, casscf._scf.mo_coeff)

        if (ci is not None and
            (getattr(casscf.fcisolver, 'large_ci', None) or
             getattr(casscf.fcisolver, 'states_large_ci', None))):
            log.info('** Largest CI components **')
            if isinstance(ci, (list, tuple, RANGE_TYPE)):
                if hasattr(casscf.fcisolver, 'states_large_ci'):
                    # defined in state_average_mix_ mcscf object
                    res = casscf.fcisolver.states_large_ci(ci, casscf.ncas, casscf.nelecas,
                                                           large_ci_tol, return_strs=False)
                else:
                    res = [casscf.fcisolver.large_ci(civec, casscf.ncas, casscf.nelecas,
                                                     large_ci_tol, return_strs=False)
                           for civec in ci]
                for i, civec in enumerate(ci):
                    log.info('  [alpha occ-orbitals] [beta occ-orbitals]  state %-3d CI coefficient', i)
                    for c,ia,ib in res[i]:
                        log.info('  %-20s %-30s %.12f', ia, ib, c)
            else:
                log.info('  [alpha occ-orbitals] [beta occ-orbitals]            CI coefficient')
                res = casscf.fcisolver.large_ci(ci, casscf.ncas, casscf.nelecas,
                                                large_ci_tol, return_strs=False)
                for c,ia,ib in res:
                    log.info('  %-20s %-30s %.12f', ia, ib, c)

        if with_meta_lowdin:
            casscf._scf.mulliken_meta(casscf.mol, dm1, s=ovlp_ao, verbose=log)
        else:
            casscf._scf.mulliken_pop(casscf.mol, dm1, s=ovlp_ao, verbose=log)
    return dm1a, dm1b
    '''