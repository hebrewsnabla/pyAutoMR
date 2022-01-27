from pyscf import lib, mcscf
from automr import guess, autocas

lib.num_threads(4)

mf = guess.gen(xyz='''Co 0.0 0.0 0.0''', bas='def2-tzvp', charge=2, spin=3)

mc = autocas.cas(mf, (5,(5,2)), dry=True)

mc = mc.state_average_([1e0/7e0,1e0/7e0,1e0/7e0,1e0/7e0,1e0/7e0,1e0/7e0,1e0/7e0])
mc.kernel()
ave_mo = mc.mo_coeff.copy()

mc = mcscf.CASCI(mf, 5, (5,2))

mc.fcisolver.spin = 1
mc.fcisolver.nroots = 20
mc.kernel(ave_mo)
