from pyscf import lib, gto, dmrgscf, mcscf
from automr import guess, autocas, cidump

lib.num_threads(4)

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.2' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = '6-31g'

mol = gto.M(atom = xyz, basis = bas, verbose = 4).build()
mf = mol.RHF().run()

nacto = mf.mo_coeff.shape[1]

print(nacto)
mc = mcscf.CASCI(mf, nacto-2, 10)
mc.fcisolver = dmrgscf.DMRGCI(mol, tol=1e-8)
mc.fcisolver.threads = 4
mc.fcisolver.memory = 6

autocas.auto_schedule(mc, 1000, 2000)
