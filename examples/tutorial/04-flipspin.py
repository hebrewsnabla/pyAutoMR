from automr import guess
from pyscf import scf, fci, lib
import numpy

lib.num_threads(4)

bl = 2.0
theta = 104.52 * numpy.pi / 180.0 / 2.0
z_coord = bl * numpy.cos(theta)
y_coord = bl * numpy.sin(theta)

atoms = f"O 0.00000000    0.00000000    0.00000000\n"
atoms += f"H 0.00000000  {y_coord: 12.8f}  {z_coord: 12.8f}\n"
atoms += f"H 0.00000000  {-y_coord: 12.8f}  {z_coord: 12.8f}\n"

bas = 'sto-3g'

#mf = guess.mix(atoms, bas, conv='tight', level_shift=0.2)
mf = guess.flipspin(atoms, bas, 4, 'site', site=[0]) 
mf.analyze()

rhf = scf.RHF(mf.mol).run()
full = fci.FCI(rhf).run()
print(full.e_tot)
