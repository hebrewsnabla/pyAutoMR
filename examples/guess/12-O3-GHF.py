from pyscf import gto, scf
from automr import guess

mol = gto.M(atom='O 0.0 0.0 0.0; O 0.0 0.8 1.0; O 0.0 0.8 -1.0', basis='def2-svp', 
        spin=2, verbose=4).build()

mf = scf.UHF(mol).run()
mf = guess.check_stab(mf)
mf.stability(external=True)

mf2 = mf.to_ghf()
mf2.kernel(dm0=mf2.make_rdm1())
#mf2.run()
#mo = scf.stability.ghf_stability(mf2)
#dm = mf2.make_rdm1(mo_coeff=mo)
#mf2.kernel(dm0=dm)
#mf2.stability()
mf2 = guess.check_stab(mf2, goal='ghf')
print(mf2.mo_coeff)
mf2 = guess.check_stab(mf2, goal='cghf')

print(mf2.mo_coeff.imag.max())

print(mf.e_tot, mf2.e_tot)
