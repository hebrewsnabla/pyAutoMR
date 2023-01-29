from pyscf import gto, scf, dft, lib, qmmm
from pyscf.scf import stability
from pyscf.lib import logger
from pyscf.lo import PM, Boys
import numpy as np
try:
    from fch2py import fch2py
    import gaussian
    from rwwfn import read_eigenvalues_from_fch as readeig
except:
    print('fch2py, rwwfn not found. Interface with fch is disabled. Install MOKIT if you need that.')
from automr import dump_mat, autocas
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
    nbf = mf.mo_coeff.shape[-2]
    nif = mf.mo_coeff.shape[-1]
    #S = mol.intor_symmetric('int1e_ovlp')
    #Sdiag = S.diagonal()
    alpha_coeff = fch2py(fch, nbf, nif, 'a')
    if not no:
        beta_coeff  = fch2py(fch, nbf, nif, 'b')
    if no:
        mf.mo_coeff = alpha_coeff
        noon = readeig(fch, nif, 'a')
        mf.mo_occ = noon
    else:
        mf.mo_coeff = (alpha_coeff, beta_coeff)
    # read done
    
    return mf

def mix_tight(xyz, bas, charge=0, *args, **kwargs):
    return mix(xyz, bas, charge=charge, conv='tight', **kwargs)

def mix(xyz, bas, charge=0, *args, **kwargs
):
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
    return _mix(mol, 
#conv=conv, cycle=cycle, level_shift=level_shift,
#                     skipstb=skipstb, xc=xc, newton=newton, mix_param=mix_param, hl=hl, field=field
*args, **kwargs)

def _mix(mol, 
        conv='loose', cycle=5, level_shift=0.0, 
        skipstb=False, xc=None, newton=False, mix_param=np.pi/4, hl=[[0,0]], field=(0.0, 0.0, 0.0)
):
    t1 = time.time()
    if xc is None: 
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    mf.conv_tol = 1e-5
    if field[0] != 0.0 or field[1] != 0.0 or field[2] != 0.0:
        mf = apply_field(mf, field)
    #mf.verbose = 4
    mf.kernel() # Guess by 1e is poor,
    #dm, mo_coeff, mo_energy, mo_occ = init_guess_by_1e(mf)
    print('**** generating mix guess ****')
    mo_mix = mf.mo_coeff
    nocc = mol.nelectron//2
    print('RHF MO energy and HOMO-3~LUMO+3 before mixing')
    print('mo_e', mf.mo_energy[nocc-4:nocc+4])
    dump_mat.dump_mo(mol, mo_mix[:,nocc-4:nocc+4], ncol=8)
    if not isinstance(hl[0], list):
        hl = [hl]
    for hlitem in hl:
        dm_mix, mo_mix = init_guess_mixed(mo_mix, mf.mo_occ, ho=hlitem[0], lu=hlitem[1], mix_param=mix_param)
    print('After mixing')
    dump_mat.dump_mo(mol, mo_mix[0][:,nocc-4:nocc+4], ncol=8)
    dump_mat.dump_mo(mol, mo_mix[1][:,nocc-4:nocc+4], ncol=8)
    if xc is None:
        mf_mix = scf.UHF(mol)
    else:
        mf_mix = dft.UKS(mol)
        mf_mix.xc = xc
    if cycle == 0:
        mf_mix.mo_coeff = mo_mix
        nelec = mf_mix.nelec
        mf_mix.mo_occ = np.zeros((2, mo_mix[0].shape[1]))
        mf_mix.mo_occ[0,:nelec[0]] = 1.0
        mf_mix.mo_occ[1,:nelec[1]] = 1.0
        mf_mix.mo_energy = np.zeros((2, mo_mix[0].shape[1]))
    #    return mf_mix
    #mf_mix.verbose = 4
    if conv == 'loose':
        mf_mix.conv_tol = 1e-3
        mf_mix.max_cycle = cycle
    elif conv == 'tight':
        mf_mix.level_shift = level_shift
        mf_mix.max_cycle = 100
    mf_mix.kernel(dm0=dm_mix)
    mf_mix = postscf_check(mf_mix, conv, skipstb, newton)

    t2 = time.time()
    print('time for guess: %.3f' % (t2-t1))
    return mf_mix

def postscf_check(mf, conv, skipstb, newton):
    if not mf.converged and conv == 'tight':
        raise RuntimeError('UHF not converged')
    ss, s = mf.spin_square()
    if s < 0.1:
        print('Warning: S too small, symmetry breaking may be failed')
    
    if conv == 'tight' and not skipstb:
        mf = check_stab(mf, newton)
    return mf

def apply_field(mf, f):
    h2 = mf.get_hcore() + np.einsum('x,xij->ij', f, mf.mol.intor('int1e_r', comp=3))
    get_h2 = lambda *args : h2
    mf.get_hcore = get_h2
    return mf

def check_uhf2(mf):
    ss, s = mf.spin_square()
    spin = mf.mol.spin
    if abs(s - 1 - spin) > 0.01:
        is_uhf = True
    else:
        is_uhf = False
        #mf = mf.to_rhf()
    return is_uhf, mf

def check_stab(mf_mix, newton=False, goal='uhf', debug=False):
    if goal=='rhf':
        stab = stability.rhf_internal
        if mf_mix.mol.spin != 0:
            stab = stability.rohf_internal
        is_uhf, mf_mix = check_uhf2(mf_mix)
        if is_uhf:
            raise ValueError('UHF/UKS wavefunction detected. RHF/RKS is required for res=True')
    elif goal=='uhf':
        stab = stability.uhf_internal
    elif goal=='ghf':
        stab = stability.ghf_stability
    elif goal=='cghf':
        if mf_mix.mo_coeff.dtype is not np.complex:
            dm_c = mf_mix.make_rdm1() + 0j
            dm_c = dm_c*np.sqrt(2)/2 + dm_c*1.0j*np.sqrt(2)/2
            #dm_c[0,:] += 0.1j
            #dm_c[:,0] -= 0.1j
            mf_mix.kernel(dm0=dm_c)
        stab = stability.ghf_stability

    if debug: mf_mix.verbose = 9
    mo, stable = stab(mf_mix, return_status=True)
    if newton:
        mf_mix=mf_mix.newton()
    cyc = 0
    while(not stable and cyc < 10):
        print('Stability Opt Attempt %d' %cyc)
        if debug: mf_mix.verbose = 4
        mf_mix.mo_coeff = mo
        dm_new = mf_mix.make_rdm1(mo, mf_mix.mo_occ)
        mf_mix.kernel(dm0=dm_new)
        if debug: mf_mix.verbose = 9
        mo, stable = stab(mf_mix, return_status=True)
        cyc += 1
    if not stable:
        raise RuntimeError('Stability Opt failed after %d attempts.' % cyc)
    if not mf_mix.converged:
        print('Warning: SCF finally not converged')
    mf_mix.mo_coeff = mo
    return mf_mix

def from_frag_tight(xyz, bas, frags, chgs, spins, newton=False, **kwargs):
    if not isinstance(spins[0], list):
        spins = [spins]
    lowest_mf = None
    lowest_e = 0.0
    lowest_spin = None
    for s in spins:
        print('Attempt spins ', s)
        mf = from_frag(xyz, bas, frags, chgs, s, cycle=70, **kwargs)
        mf = check_stab(mf, newton=newton)
        if mf.e_tot < lowest_e:
            lowest_e = mf.e_tot
            lowest_mf = mf
            lowest_spin = s
    print('Lowest UHF energy %.6f from spin guess ' % lowest_e, lowest_spin)
    return lowest_mf
    
def from_frag(xyz, bas, frags, chgs, spins, **kwargs):
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = bas
    mol.verbose = 4
    mol.charge = sum(chgs)
    mol.spin = sum(spins)
    mol.build()
    return _from_frag(mol, frags, chgs, spins, **kwargs) 


def _from_frag(mol_or_mf, frags, chgs, spins, 
               conv='loose', cycle=2, level_shift=0.0, 
               skipstb=False, xc=None, newton=False, verbose=4, rmdegen=False, bgchg=None):
    
    t1 = time.time() 
    dm, mo, occ = guess_frag(mol_or_mf, frags, chgs, spins, rmdegen=rmdegen, bgchg=bgchg)
    if isinstance(mol_or_mf, gto.Mole):
        mol = mol_or_mf
    else:
        mol1 = mol_or_mf[0].mol
        mol2 = mol_or_mf[1].mol
        mol = gto.conc_mol(mol1, mol2)
        mol.spin = mol1.spin + mol2.spin
    if verbose > 4:
        print('Frag guess orb alpha')
        dump_mat.dump_mo(mol, mo[0])
        print('Frag guess orb beta')
        dump_mat.dump_mo(mol, mo[1])
    if xc is None:
        mf = scf.UHF(mol)
    else:
        mf = dft.UKS(mol)
        mf.xc = xc
    if cycle == 0:
        mf.mo_coeff = mo
        mf.mo_occ = occ
        mf.mo_energy = np.zeros((2, mo[0].shape[1]))
    mf.verbose = verbose
    if conv == 'loose':
        mf.conv_tol = 1e-3
        mf.max_cycle = cycle
    elif conv == 'tight':
        mf.level_shift = level_shift
        mf.max_cycle = 100
    mf.kernel(dm0 = dm)
    mf = postscf_check(mf, conv, skipstb, newton)
    t2 = time.time()
    print('time for guess: %.3f' % (t2-t1))
    return mf

#def make_bgchg(atom, chgs):
#    return 

def guess_frag(mol_or_mf, frags, chgs, spins, rmdegen=False, bgchg=None):
    '''
    mol_or_mf: 
        if mol inputted, it will be fragmented by frags, chgs, spins
        otherwise, it should be a list of mf
    frags: e.g. [[0], [1]] for N2
    '''
    #mol.build()
    if rmdegen:
         mol = mol_or_mf
         if bgchg is not None:
             mul_chg = bgchg
         else:
             mol2 = mol.copy()
             mol2.set(basis = 'def2-svp', verbose = 0).build()
             mf2 = scf.HF(mol2).set(conv_tol = 1e-4).run()
             pop, mul_chg = mf2.mulliken_pop()
    
    print('**** generating fragment guess ****')
    if isinstance(mol_or_mf, gto.Mole):
        mol = mol_or_mf
        atom = mol.format_atom(mol.atom, unit=1)
        #print(atom)
        fraga, fragb = frags
        chga, chgb = chgs
        spina, spinb = spins
        atoma = [atom[i] for i in fraga]
        atomb = [atom[i] for i in fragb]
        print('fragments:', atoma, atomb)
        if rmdegen:
            bgcoord_a = atomb
            bgcoord_b = atoma
            bgchg_a = np.array([mul_chg[i] for i in fragb])
            bgchg_b = np.array([mul_chg[i] for i in fraga])
        else:
            bgcoord_a = bgcoord_b = bgchg_a = bgchg_b = None
        ca_a, cb_a, na_a, nb_a = do_uhf(atoma, mol.basis, chga, spina, bgcoord_a, bgchg_a)
        ca_b, cb_b, na_b, nb_b = do_uhf(atomb, mol.basis, chgb, spinb, bgcoord_b, bgchg_b)
    else:
        mflist = mol_or_mf
        ca_a, cb_a = mflist[0].mo_coeff
        ca_b, cb_b = mflist[1].mo_coeff
        na_a, nb_a = mflist[0].nelec
        na_b, nb_b = mflist[1].nelec
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
    
def do_uhf(atoma, basisa, chga, spina, bg_coord=None, bg_chg = None):
    mola = gto.Mole()
    mola.atom = atoma
    mola.basis = basisa
    mola.charge = chga
    mola.spin = spina
    #mola.verbose = 4
    mola.build()
    mfa = scf.UHF(mola)
    if bg_coord is not None:
        mfa = qmmm.mm_charge(mfa, bg_coord, bg_chg)
        #mfa.dump_flags(verbose=5)
        logger.note(mfa, 'Charge      Location')
        coords = mfa.mm_mol.atom_coords()
        charges = mfa.mm_mol.atom_charges()
        for i, z in enumerate(charges):
            logger.note(mfa, '%.9g    %s', z, coords[i])
    mfa.kernel()
    mfa.mulliken_pop()
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

def init_guess_mixed(mo_coeff, mo_occ, mix_param=np.pi/4, ho=0, lu=0):
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

    homo_idx += ho
    lumo_idx += lu
    if isinstance(mo_coeff, tuple):
        psi_homo=mo_coeff[0][:, homo_idx]
        psi_lumo=mo_coeff[0][:, lumo_idx]
        Ca=copy.deepcopy(mo_coeff[0])
        Cb=copy.deepcopy(mo_coeff[1])
    else:
        psi_homo=mo_coeff[:, homo_idx]
        psi_lumo=mo_coeff[:, lumo_idx]
        Ca=copy.deepcopy(mo_coeff)
        Cb=copy.deepcopy(mo_coeff)

    #mix homo and lumo of alpha and beta coefficients
    q=mix_param
    angle = q / np.pi
    print('rotating angle: %.2f pi' % angle)

    Ca[:,homo_idx] = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
    Cb[:,homo_idx] = np.cos(q)*psi_homo - np.sin(q)*psi_lumo

    Ca[:,lumo_idx] = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
    Cb[:,lumo_idx] =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo

    dm = scf.uhf.make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm, (Ca, Cb)

def flipspin(xyz, bas, highspin, flipstyle='lmo', loc='pm', fliporb=[-1], site=None, cycle=50):
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = bas
    mol.verbose = 4
    #mol.build()
    return _flipspin(mol, highspin, flipstyle, 
                     loc=loc, fliporb=fliporb, site=site, cycle=cycle)

def _flipspin(mol, highspin, flipstyle='lmo', loc='pm', fliporb=[-1], site=None, cycle=50):
    mol.spin = highspin
    mf = scf.UHF(mol.build())
    mf.conv_tol = 1e-6
    mf.run()
    
    mf, unos, noon, _nacto, _, _ncore, _ = autocas.get_uno(mf, thresh=1.98)
    nacto = min(_nacto, highspin)
    ncore = _ncore + nacto - _nacto
    act_idx = slice(ncore, ncore+nacto)
    if loc=='boys':
        locl = Boys(mf.mol, mf.mo_coeff[:,act_idx])
    elif loc=='pm':
        locl = PM(mf.mol, mf.mo_coeff[:,act_idx], mf)
        #loc.pop_method = 'meta-lowdin'
    loc_orb = locl.kernel()
    dump_mat.dump_mo(mf.mol, loc_orb, ncol=10)
    """pm.pop_method = 'mulliken'
    loc_orb = pm.kernel()
    dump_mat.dump_mo(mf.mol, loc_orb, ncol=10)
    pm.pop_method = 'iao'
    loc_orb = pm.kernel()
    dump_mat.dump_mo(mf.mol, loc_orb, ncol=10)
    pm.pop_method = 'becke'
    loc_orb = pm.kernel()
    dump_mat.dump_mo(mf.mol, loc_orb, ncol=10)
    #mf.mo_coeff[:, act_idx] = loc_orb.copy()
    #print(mf.mo_occ)    
    exit() """           
    
    atm_loc = mulliken(mf.mol, loc_orb)
    print('atm_loc', atm_loc)
    if flipstyle=='lmo':
        mf_bs = flip_bylmo(mf, act_idx, loc_orb, fliporb)
    elif flipstyle=='site':
        mf_bs = flip_bysite(mf, act_idx, loc_orb, atm_loc, site)
    else:
        raise ValueError('flipstyle can only be lmo or site')
    dm0 = mf_bs.make_rdm1()
    #print(dm0[0].trace(), dm0[1].trace())
    #print(np.linalg.norm(dm0[0]-dm0[1]))
    #mf_bs.level_shift = 0.3
    mf_bs.max_cycle = cycle
    #print(mf_bs.mo_coeff[0].shape)
    print(mf_bs.mo_occ)
    mf_bs.kernel(dm0=dm0)
    return mf_bs


def mulliken(mol, mo):
    atm_loc = {}
    S = mol.intor_symmetric('int1e_ovlp')
    theta = lib.einsum('ai, ab, bi -> ia', mo, S, mo)
    for i in range(theta.shape[0]):
        print('LMO %d' %i, end='')
        chg = np.zeros(mol.natm)
        for a, s in enumerate(mol.ao_labels(fmt=None)):
            chg[s[0]] += theta[i,a]
        if chg.max() > 0.65:
            max_idx = chg.argmax()
            if max_idx in atm_loc:
                atm_loc[max_idx].append(i)
            else:
                atm_loc[max_idx] = [i]
        else:
            if -1 in atm_loc:
                atm_loc[-1].append(i)
            else:
                atm_loc[-1] = [i]

        for s in range(mol.natm):
            if chg[s] > 0.05:
                print('  %d%2s  %.2f  ' % (s, mol.atom_symbol(s), chg[s]), end='')
        print('')
    return atm_loc

def flip_bylmo(mf, act_idx, loc_orb, fliporb):
    mo = mf.mo_coeff
    mo_core = mo[:, :act_idx.start]
    mo_ext = mo[:, act_idx.stop:]
    a = []
    b = []
    for i in range(loc_orb.shape[1]):
        if i in fliporb:
            b.append(i)
        else:
            a.append(i)
    mo_a = np.hstack((loc_orb[:,a], loc_orb[:,b]))
    mo_b = np.hstack((loc_orb[:,b], loc_orb[:,a]))
    #mo_a = np.hstack(tuple(moa)+tuple(mob))
    #mo_b = np.hstack(tuple(mob)+tuple(moa))
    #mol_bs = mf.mol
    #mol_bs.spin = 0
    mf_bs = mf.to_uhf()
    nelec = mo_core.shape[1] + len(a), mo_core.shape[1] + len(b)
    mf_bs.mol.spin = nelec[0] - nelec[1]
    mf_bs.mol.build()
    #mf_bs.mo_coeff = mo
    mf_bs.mo_occ = np.zeros_like(mf_bs.mo_energy)
    mf_bs.mo_occ[0,:nelec[0]] = 1
    mf_bs.mo_occ[1,:nelec[1]] = 1
    mf_bs.mo_coeff = ( np.hstack((mo_core, mo_a, mo_ext)),
                       np.hstack((mo_core, mo_b, mo_ext)))
    dump_mat.dump_mo(mf_bs.mol, mo_a, ncol=10)
    return mf_bs

def flip_bysite(mf, act_idx, loc_orb, atm_loc, site):
    mo = mf.mo_coeff
    mo_core = mo[:, :act_idx.start]
    mo_ext = mo[:, act_idx.stop:]
    moa = []
    mob = []
    print('flip lmo on site', site)
    act_a = 0; act_b = 0
    for atm in atm_loc:
        mo_atm = loc_orb[:,atm_loc[atm]]
        if atm in site:
            mob.append(mo_atm)
            act_b += mo_atm.shape[1]
        else:
            moa.append(mo_atm)
            act_a += mo_atm.shape[1]
    mo_a = np.hstack(tuple(moa)+tuple(mob))
    mo_b = np.hstack(tuple(mob)+tuple(moa))
    #mol_bs = mf.mol
    #mol_bs.spin = 0
    mf_bs = mf.to_uhf() #scf.UHF(mol_bs)
    #mf_bs.mol.spin = 0
    nelec = mo_core.shape[1] + act_a, mo_core.shape[1] + act_b
    mf_bs.mol.spin = nelec[0] - nelec[1]
    mf_bs.mol.build()
    #mf_bs.mo_coeff = mo
    mf_bs.mo_occ = np.zeros_like(mf_bs.mo_energy)
    mf_bs.mo_occ[0,:nelec[0]] = 1
    mf_bs.mo_occ[1,:nelec[1]] = 1
    mf_bs.mo_coeff = ( np.hstack((mo_core, mo_a, mo_ext)),
                       np.hstack((mo_core, mo_b, mo_ext)))
    #print(mf_bs.mo_occ)
    dump_mat.dump_mo(mf_bs.mol, mf_bs.mo_coeff[0][:, act_idx], ncol=10)
    dump_mat.dump_mo(mf_bs.mol, mf_bs.mo_coeff[1][:, act_idx], ncol=10)
    #dm0 = mf_bs.make_rdm1()
    #print(dm0[0].trace(), dm0[1].trace())
    #print(np.linalg.norm(dm0[0]-dm0[1]))
    return mf_bs
