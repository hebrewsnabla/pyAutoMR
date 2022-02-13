from pyscf import gto, scf, dft
import numpy as np
try:
    from fch2py import fch2py
    import gaussian
    from rwwfn import read_eigenvalues_from_fch as readeig
except:
    print('fch2py, rwwfn not found. Interface with fch is disabled. Install MOKIT if you need that.')
from automr import stability, dump_mat, autocas
import time
import copy

def gen(xyz, bas, charge, spin, conv='tight', level_shift=0, xc=None):
    '''for states other than singlets'''
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = bas
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 4
    mol.build()
    
    if xc is None:
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc
    if conv == 'loose':
        mf.conv_tol = 1e-6
        mf.max_cycle = 10
    mf.level_shift = level_shift
    mf.kernel()

    return mf

def from_fch_noiter(fch, no=True):
    mol = gaussian.load_mol_from_fch(fch)
    mf = _from_fchk(mol, fch, no=no)
    return mf 


def from_fch_simp(fch, cycle=None, xc=None, no=False):
    mol = gaussian.load_mol_from_fch(fch)
    mf = _from_fchk(mol, fch, xc)    
    dm = mf.make_rdm1()
    if cycle is None:
        if xc is None:
            cycle = 2
        else:
            cycle = 6
    mf.max_cycle = cycle
    mf.kernel(dm)
    return mf

def from_fchk(xyz, bas, fch, cycle=None, xc=None):
    mol = gto.Mole()
    mol.atom = xyz
    #with open(xyz, 'r') as f:
    #    mol.atom = f.read()
    #print(mol.atom)
    mol.basis = bas
    #mol.output = 'test.pylog'
    mol.verbose = 4
    mol.build()
    mf = _from_fchk(mol, fch, xc)
    dm = mf.make_rdm1()
    if cycle is None:
        if xc is None:
            cycle = 2
        else:
            cycle = 6
    mf.max_cycle = cycle
    mf.kernel(dm)
    return mf

def _from_fchk(mol, fch, xc=None, no=False):    
    if xc is None:
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc
    if no:
        mf = scf.RHF(mol)
    #mf.init_guess = '1e'
    #mf.init_guess_breaksym = True
    mf.max_cycle = 1
    mf.kernel()
    
    # read MOs from .fch(k) file
    nbf = mf.mo_coeff[0].shape[0]
    nif = mf.mo_coeff[0].shape[1]
    #S = mol.intor_symmetric('int1e_ovlp')
    #Sdiag = S.diagonal()
    alpha_coeff = fch2py(fch, nbf, nif, 'a')
    beta_coeff  = fch2py(fch, nbf, nif, 'b')
    if no:
        mf.mo_coeff = alpha
        noon = readeig(fch, nif, 'a')
        mf.mo_occ = noon
    else:
        mf.mo_coeff = (alpha_coeff, beta_coeff)
    # read done
    
    return mf

def mix_tight(xyz, bas, charge=0, xc=None):
    return mix(xyz, bas, charge, 'tight', xc=xc)

def mix(xyz, bas, charge=0, conv='loose', cycle=5, skipstb=False, xc=None, newton=False):
    mol = gto.Mole()
    mol.atom = xyz
    #with open(xyz, 'r') as f:
    #    mol.atom = f.read()
    #print(mol.atom)
    mol.basis = bas
    mol.charge = charge
    #mol.output = 'test.pylog'
    mol.verbose = 4
    mol.build()

    t1 = time.time()
    if xc is None: 
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    mf.conv_tol = 1e-5
    #mf.verbose = 4
    mf.kernel() # Guess by 1e is poor,
    #dm, mo_coeff, mo_energy, mo_occ = init_guess_by_1e(mf)
    print('**** generating mix guess ****')
    dm_mix = init_guess_mixed(mf.mo_coeff, mf.mo_occ)
    if xc is None:
        mf_mix = scf.UHF(mol)
    else:
        mf_mix = dft.UKS(mol)
        mf_mix.xc = xc
    #mf_mix.verbose = 4
    if conv == 'loose':
        mf_mix.conv_tol = 1e-3
        mf_mix.max_cycle = cycle
    elif conv == 'tight':
        mf_mix.max_cycle = 100
    mf_mix.kernel(dm0=dm_mix)
    if not mf_mix.converged and conv == 'tight':
        raise RuntimeError('UHF not converged')
    ss, s = mf_mix.spin_square()
    if s < 0.1:
        print('Warning: S too small, symmetry breaking may be failed')
    
    if conv == 'tight' and not skipstb:
        mf_mix = check_stab(mf_mix, newton)

    t2 = time.time()
    print('time for guess: %.3f' % (t2-t1))
    #dm = mf.make_rdm1()
    #mf.max_cycle = 0
    #mf_mix.kernel(dm)
    return mf_mix

def check_uhf2(mf):
    ss, s = mf.spin_square()
    spin = mf.mol.spin
    if abs(s - 1 - spin) > 0.01:
        is_uhf = True
    else:
        is_uhf = False
        #mf = mf.to_rhf()
    return is_uhf, mf

def check_stab(mf_mix, newton=False, res=False):
    if res:
        stab = stability.rhf_internal
        if mf_mix.mol.spin is not 0:
            stab = stability.rohf_internal
        is_uhf, mf_mix = check_uhf2(mf_mix)
        if is_uhf:
            raise ValueError('UHF/UKS wavefunction detected. RHF/RKS is required for res=True')
    else:
        stab = stability.uhf_internal
    mf_mix.verbose = 9
    mo, stable = stab(mf_mix)
    cyc = 0
    while(not stable and cyc < 10):
        mf_mix.verbose = 4
        dm_new = mf_mix.make_rdm1(mo, mf_mix.mo_occ)
        if newton:
            mf_mix=mf_mix.newton()
        mf_mix.kernel(dm0=dm_new)
        mf_mix.verbose = 9
        mo, stable = stab(mf_mix)
        cyc += 1
    if not stable:
        raise RuntimeError('Stablility Opt failed after %d attempts.' % cyc)
    if not mf_mix.converged:
        print('Warning: SCF finally not converged')
    mf_mix.mo_coeff = mo
    return mf_mix

def from_frag_tight(xyz, bas, frags, chgs, spins):
    if not isinstance(spins[0], list):
        spins = [spins]
    lowest_mf = None
    lowest_e = 0.0
    lowest_spin = None
    for s in spins:
        print('Attempt spins ', s)
        mf = from_frag(xyz, bas, frags, chgs, s, cycle=70)
        mf = check_stab(mf)
        if mf.e_tot < lowest_e:
            lowest_e = mf.e_tot
            lowest_mf = mf
            lowest_spin = s
    print('Lowest UHF energy %.6f from spin guess ' % lowest_e, lowest_spin)
    return lowest_mf
    



def from_frag(xyz, bas, frags, chgs, spins, cycle=2, xc=None, verbose=4):
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = bas
    mol.verbose = 4
    mol.charge = sum(chgs)
    mol.build()
    
    t1 = time.time() 
    dm, mo, occ = guess_frag(mol, frags, chgs, spins)
    print('Frag guess orb alpha')
    dump_mat.dump_mo(mol, mo[0])
    print('Frag guess orb beta')
    dump_mat.dump_mo(mol, mo[1])
    if xc is None:
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc
    mf.verbose = verbose
    #mf.conv_tol = 1e-2
    mf.max_cycle = cycle
    mf.kernel(dm0 = dm)
    ss, s = mf.spin_square()
    if s < 0.1:
        print('Warning: S too small, symmetry breaking may be failed')
    t2 = time.time()
    print('time for guess: %.3f' % (t2-t1))
    return mf


def guess_frag(mol, frags, chgs, spins):
    '''
    frags: e.g. [[0], [1]] for N2
    '''
    #mol.build()
    print('**** generating fragment guess ****')
    atom = mol.format_atom(mol.atom, unit=1)
    #print(atom)
    fraga, fragb = frags
    chga, chgb = chgs
    spina, spinb = spins
    atoma = [atom[i] for i in fraga]
    atomb = [atom[i] for i in fragb]
    print('fragments:', atoma, atomb)
    ca_a, cb_a, na_a, nb_a = do_uhf(atoma, mol.basis, chga, spina)
    ca_b, cb_b, na_b, nb_b = do_uhf(atomb, mol.basis, chgb, spinb)
    print('       na   nb')
    print('atom1  %2d   %2d' % (na_a, nb_a))
    print('atom2  %2d   %2d' % (na_b, nb_b))
    #print(mo_a)
    #print(mo_b)
    nbasa = ca_a.shape[0]
    nbasb = ca_b.shape[0]
    ca = np.vstack((
                    np.hstack((ca_a[:,:na_a], np.zeros((nbasa,na_b)), ca_a[:,na_a:], np.zeros((nbasa, ca_b.shape[1]-na_b)) )),
                    np.hstack((np.zeros((nbasb, na_a)), ca_b[:,:na_b], np.zeros((nbasb, ca_a.shape[1]-na_a)), ca_b[:,na_b:]))
                  ))
    cb = np.vstack((
                    np.hstack((cb_a[:,:nb_a], np.zeros((nbasa,nb_b)), cb_a[:,nb_a:], np.zeros((nbasa, cb_b.shape[1]-nb_b)) )),
                    np.hstack((np.zeros((nbasb, nb_a)), cb_b[:,:nb_b], np.zeros((nbasb, cb_a.shape[1]-nb_a)), cb_b[:,nb_b:]))
                  ))
    mo = np.array([ca, cb])
    na = na_a + na_b
    nb = nb_a + nb_b
    #print(ca.shape, cb.shape)
    occa = np.hstack((np.ones(na), np.zeros(ca.shape[1]-na))) 
    occb = np.hstack((np.ones(nb), np.zeros(cb.shape[1]-nb)))
    occ = np.array([occa, occb]) 
    #print(occ)
    dm = scf.uhf.make_rdm1(mo, occ)
    #print(dm.shape)
    return dm, mo, occ
    
def do_uhf(atoma, basisa, chga, spina):
    mola = gto.Mole()
    mola.atom = atoma
    mola.basis = basisa
    mola.charge = chga
    mola.spin = spina
    mola.build()
    mfa = scf.UHF(mola)
    mfa.kernel()
    #print(mfa.nelec)
    ca, cb = mfa.mo_coeff
    na, nb = mfa.nelec
    return ca, cb, na, nb

'''
modified from pyscf/examples/scf/56-h2_symm_breaking.py, by James D Whitfield
The initial guess is obtained by mixing the HOMO and LUMO and is implemented
as a function that can be used in other applications.
See also 16-h2_scan.py, 30-scan_pes.py, 32-break_spin_symm.py
'''

def init_guess_by_1e(rhf, mol=None):
    h1e = rhf.get_hcore(mol)
    s1e = rhf.get_ovlp(mol)
    mo_energy, mo_coeff = scf.hf.eig(h1e, s1e)
    mo_occ = rhf.get_occ(mo_energy, mo_coeff)
    return rhf.make_rdm1(mo_coeff, mo_occ), mo_coeff, mo_energy, mo_occ

def init_guess_mixed(mo_coeff, mo_occ, mixing_parameter=np.pi/4):
    ''' Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.
    
    psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
    psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
        
    psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    Returns: 
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi

    homo_idx=0
    lumo_idx=1

    for i in range(len(mo_occ)-1):
        if mo_occ[i]>0 and mo_occ[i+1]<0.1:
            homo_idx=i
            lumo_idx=i+1

    psi_homo=mo_coeff[:, homo_idx]
    psi_lumo=mo_coeff[:, lumo_idx]
    
    Ca=copy.deepcopy(mo_coeff)
    Cb=copy.deepcopy(mo_coeff)

    #mix homo and lumo of alpha and beta coefficients
    q=mixing_parameter
    angle = q / np.pi
    print('rotating angle: %.2f pi' % angle)

    Ca[:,homo_idx] = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
    Cb[:,homo_idx] = np.cos(q)*psi_homo - np.sin(q)*psi_lumo

    Ca[:,lumo_idx] = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
    Cb[:,lumo_idx] =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo

    dm = scf.uhf.make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm
