from pyscf import gto, scf, lib
from pyscf.geomopt.geometric_solver import optimize
from automr import guess, autocas
lib.num_threads(4)

mol = gto.M(atom='c7.xyz', basis='6-31gs', spin=1).build()
mf = guess._mix(mol)
mf2 = scf.UKS(mol)
mf2.xc = 'b3lyp'
mf2.kernel(dm0=mf.make_rdm1())

mol_eq = optimize(mf2)

mf3 = guess._mix(mol_eq, conv='tight')
mf4 = autocas.cas(mf3)



