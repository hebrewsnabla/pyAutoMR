from pyscf import scf,dft, gto, lib
from automr import guess, autocas

lib.num_threads(8)
#mf=guess.from_fch_simp("v2.fchk", xc='pbe0')

#mf2.verbose=9
#mf2.stability()
mol = gto.Mole(atom='''V 0.0 0.0 0.0; V 0.0 0.0 1.77''', basis='def2-tzvp', spin=2)
mol.verbose = 4
mol.build()
mf = scf.UHF(mol)
mf.kernel()
mf = guess.check_stab(mf, newton=True)
mf2 = autocas.cas(mf, (8, (5,3)) )
mf4 = autocas.nevpt2(mf2)

mol2 = gto.Mole(atom='''V 0.0 0.0 0.0''', basis='def2-tzvp', spin=3)
mol2.verbose = 4
mol2.build()
mfb = scf.UHF(mol2)
mfb.kernel()
mfb = guess.check_stab(mfb, newton=True)
mfb2 = autocas.cas(mfb)
mfb4 = autocas.nevpt2(mfb2)

mol3 = gto.Mole(atom='''V 0.0 0.0 0.0''', basis='def2-tzvp', spin=1)
mol3.verbose = 4
mol3.build()
mfc = scf.UHF(mol3)
mfc.kernel()
mfc = guess.check_stab(mfc, newton=True)
mfc2 = autocas.cas(mfc)
mfc4 = autocas.nevpt2(mfc2)
