from pyscf import gto,scf, dft, lib
from pyphf import guess

lib.num_threads(16)

mol = gto.Mole()
mol.atom = '''O 0.0 0.0 0.0; H 0.0 0.0 0.9697'''
mol.basis = 'aug-pc-4'
mol.spin = 1
mol.max_memory = 12000
mol.build()

def get_hcore(mol=mol):
    return scf.hf.get_hcore(mol) + 1e-6*mol.intor('int1e_r')[0]
mf = dft.UKS(mol)
mf.get_hcore = get_hcore
mf.xc = 'b3lypg'
mf.kernel()
mf = guess.check_stab(mf)


