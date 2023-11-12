from pyscf import lib, gto, dmrgscf, mcscf
from automr import guess, autocas, cidump

lib.num_threads(24)
dmrgscf.settings.MPIPREFIX = ''

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.2' #sys.argv[1]
#fch = 'n2.fchk' #sys.argv[2]
bas = '6-31gs'

mol = gto.M(atom = xyz, basis = bas, verbose = 4).build()
mf = mol.RHF().run()

nacto = mf.mo_coeff.shape[1]

print(nacto)
mc = mcscf.CASCI(mf, nacto-2, 10)
mc.fcisolver = dmrgscf.DMRGCI(mol, tol=1e-8)
mc.fcisolver.threads = 24
mc.fcisolver.memory = 50

autocas.auto_schedule(mc, 1000, 2000)

autocas.auto_schedule(mc, 1000, 1000)
autocas.auto_schedule(mc, 1500, 1500)
autocas.auto_schedule(mc, 2000, 2000)
