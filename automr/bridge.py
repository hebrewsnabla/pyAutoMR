import numpy as np
import os
from automr import dump_mat
from functools import partial, reduce
print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

def print_mol(mol):
    print(mol._basis)
    print(mol.atom)
    print(mol._atom)
    print(mol.aoslice_by_atom())
    print(mol.ao_labels())
#if mol.verbose >= logger.DEBUG:
    mol.stdout.write('[INPUT] ---------------- BASIS SET ---------------- \n')
    mol.stdout.write('[INPUT] l, kappa, [nprim/nctr], '
                      'expnt,             c_1 c_2 ...\n')
    for atom, basis_set in mol._basis.items():
        mol.stdout.write('[INPUT] %s\n' % atom)
        for b in basis_set:
            if isinstance(b[1], int):
                kappa = b[1]
                b_coeff = b[2:]
            else:
                kappa = 0
                b_coeff = b[1:]
            nprim = len(b_coeff)
            nctr = len(b_coeff[0])-1
            if nprim < nctr:
                logger.warn(mol, 'num. primitives smaller than num. contracted basis')
            mol.stdout.write('[INPUT] %d   %2d    [%-5d/%-4d]  '
                              % (b[0], kappa, nprim, nctr))
            for k, x in enumerate(b_coeff):
                if k == 0:
                    mol.stdout.write('%-15.12g  ' % x[0])
                else:
                    mol.stdout.write(' '*32+'%-15.12g  ' % x[0])
                for c in x[1:]:
                    mol.stdout.write(' %4.12g' % c)
                mol.stdout.write('\n')

def py2qchem(mf, basename, is_uhf=False):
    if is_uhf:
        mo_coeffa = mf.mo_coeff[0]
        mo_coeffb = mf.mo_coeff[1]
        #mo_enea = mf.mo_energy[0]
        #mo_eneb = mf.mo_energy[1]
    else:
        mo_coeffa = mf.mo_coeff
        mo_coeffb = mf.mo_coeff
        #mo_enea = mf.mo_energy
        #mo_eneb = mf.mo_energy
    mo_enea = np.zeros(len(mo_coeffa))
    mo_eneb = np.zeros(len(mo_coeffa))
    Sdiag = mf.get_ovlp().diagonal()**(0.5)
    mo_coeffa = einsum('ij,i->ij', mo_coeffa, Sdiag).T
    mo_coeffb = einsum('ij,i->ij', mo_coeffb, Sdiag).T
    #dump_mat.dump_mo(mf.mol, mo_coeffa, ncol=10)

    guess_file = np.vstack([mo_coeffa, mo_coeffb, mo_enea, mo_eneb]).flatten()
    tmpbasename = '/tmp/qchem/' + basename
    os.system('mkdir -p ' + tmpbasename)
    with open(tmpbasename + '/53.0', 'w') as f:
        guess_file.tofile(f, sep='')
    create_qchem_in(mf, basename)

def create_qchem_in(mf, basename, uhf=False, sph=True):
    atom = mf.mol.format_atom(mf.mol.atom, unit=1)
    with open(basename + '.in', 'w') as f:
        f.write('$molecule\n')
        f.write(' %d %d\n' % (mf.mol.charge, mf.mol.spin+1))
        for a in atom:
            f.write(' %s %12.6f %12.6f %12.6f\n' % (a[0], a[1][0], a[1][1], a[1][2]))
        f.write('$end\n\n')
        '''f.write('$rem\n')
        f.write(' method = hf\n')
        if uhf:
            f.write(' unrestricted = true\n')
        f.write(' basis = cc-pvdz\n')
        f.write(' print_orbitals = true\n')
        f.write(' sym_ignore = true\n')
        if sph:
            f.write(' purecart = 1111\n')
        else:
            f.write(' purecart = 2222\n')
        f.write(' scf_guess_print = 2\n')
        f.write(' scf_guess = read\n')
        f.write(' scf_convergence = 0\n')
        f.write(' thresh = 12\n')
        f.write('$end\n\n')
        f.write('@@@\n\n')
        f.write('$molecule\n')
        f.write('read\n')
        f.write('$end\n\n')'''
        f.write('$rem\n')
        #f.write(' method = hf\n')
        f.write(' correlation = pp\n')
        f.write(' gvb_local = 0\n')
        f.write(' gvb_n_pairs = 2\n')
        f.write(' gvb_print = 1\n')
        if uhf:
            f.write(' unrestricted = true\n')
        f.write(' basis = cc-pvdz\n')
        f.write(' print_orbitals = true\n')
        f.write(' sym_ignore = true\n')
        if sph:
            f.write(' purecart = 1111\n')
        else:
            f.write(' purecart = 2222\n')
        f.write(' scf_guess_print = 2\n')
        f.write(' scf_guess = read\n')
        f.write(' thresh = 12\n')
        f.write('$end\n\n')



def qchem2py(basename):
    with open('/tmp/qchem/' + basename + '/53.0', 'r') as f:
        data = np.fromfile(f)
    print(data.shape)
    n = data.shape[0]
    #x = sympy.Symbol('x')
    #nmo = sympy.solve(2*x*(x+1) -n, x)
    nmo = int(np.sqrt(n/2.0+0.25)-0.5)
    moa = data[:nmo*nmo].reshape(nmo,nmo).T
    mob = data[nmo*nmo:2*nmo*nmo].reshape(nmo,nmo).T
    mo = (moa, mob)
    return mo