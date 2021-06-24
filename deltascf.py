import copy
from pyscf import scf

def mom(mf, type='U', aexci=[[],[]], bexci=[[],[]]):
    ''' aexci: [[3,4],[5,6]]
    '''
    if not mf.converged:
        mf.kernel()
    e0 = mf.e_tot
    mo0 = mf.mo_coeff
    occ0 = mf.mo_occ
    occ = copy.copy(occ0)
    if type=='U':
        for i in aexci[0]:
            occ[0][i] -= 1
        for j in aexci[1]:
            occ[0][j] += 1
        for i in bexci[0]:
            occ[1][i] -= 1
        for j in bexci[1]:
            occ[1][j] += 1
    print('Excitation: Alpha ',aexci[0], ' -> ', aexci[1])
    print('             Beta ',bexci[0], ' -> ', bexci[1])
    print('Former occ pattern: Alpha', occ0[0])
    print('                     Beta', occ0[1])
    print('New occ pattern:    Alpha', occ[0])
    print('                     Beta', occ[1])
    #mf2 = copy.deepcopy(mf)
    dm = mf.make_rdm1(mo0, occ)
    mf.verbose = 8
    mf2 = scf.addons.mom_occ(mf, mo0, occ)
    mf2.kernel(dm)
    e2 = mf2.e_tot
    print(e0,e2)
