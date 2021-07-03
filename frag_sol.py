from pyscf import gto
import radii

def from_frag(xyz, frags, chgs, spins, gjfhead='', scrfhead='', gjfname='', basis=None, wfnpath=None):
    mol = gto.Mole()
    mol.atom = xyz
#    mol.basis = bas
    mol.verbose = 1
    mol.build()

    guess_frag(mol, frags, chgs, spins, gjfhead.lstrip('\n'), scrfhead, gjfname, basis, wfnpath)

def spin_p2g(spin):
    if spin>0:
        spin = spin + 1
    elif spin < 0:
        spin = spin - 1
    return spin



def guess_frag(mol, frags, chgs, spins, gjfhead, scrfhead, gjfname, basis, wfnpath):
    '''
    frags: e.g. [[1], [2]] for N2
    chgs:  e.g. [0, 0] for N2
    spins: e.g. [3, -3] for N2
    '''
    #mol.build()
    print('**** generating fragments ****')
    atom = mol.format_atom(mol.atom, unit=1)
    #print(atom)
    #fraga, fragb = frags
    #chga, chgb = chgs
    #spina, spinb = spins
    allatom = range(1,len(atom)+1)
    for k in range(len(frags)):
        frag = frags[k]
        chg = chgs[k]
        spin = spins[k]
        g_spin = spin_p2g(spin)
        atomk = [atom[i-1] for i in frag]
        atomother = [atom[i-1] for i in allatom if i not in frag]
        print('fragment %d, chg %d, spin %d' % (k, chg, spin))
        #print(atomk)
        with open(gjfname+'%d.gjf'%k, 'w') as f:
            f.write(gjfhead)
            f.write('%d %d\n' % (chg, g_spin))
            for a in atomk:
                f.write('%s  %10.5f %10.5f %10.5f\n' % (a[0], a[1][0], a[1][1], a[1][2]))
            #f.write('\n')
            if basis is not None:
                f.write(basis)
                #f.write('\n')
            f.write(scrfhead)
            f.write('ExtraSph=%d\n\n' % len(atomother))
            for b in atomother:
                rad = radii.uff_radii[b[0]] / 2.0
                f.write(' %10.5f %10.5f %10.5f  %10.5f\n' % (b[1][0], b[1][1], b[1][2], rad))
            f.write('\n')
            if wfnpath is not None:
                f.write(wfnpath + '%d.wfn'%k + '\n')
                f.write('\n')
       
            
        
