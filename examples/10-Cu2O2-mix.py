from pyscf import lib
import guess

lib.num_threads(8)

# structure from dx.doi.org/10.1021/ct300689e | J. Chem. Theory Comput. 2012, 8, 4944−4949
xyz = '''
 Cu                 1.4    0.0     0.0
 Cu                -1.4    0.0     0.0
 O                  0.0    1.15    0.0
 O                  0.0   -1.15    0.0
'''
mf = guess.mix(xyz, 'def2-tzvp', charge=2)
# stability check of UHF will be performed

#mf2 = util.SUHF(mf)
#mf2.level_shift = 0.3
#mf2.max_cycle = 100
#mf2.conv_tol = 1e-6
#mf2.kernel()
# E = -3426.9194811

