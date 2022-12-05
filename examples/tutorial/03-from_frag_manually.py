from automr import guess, dump_mat

xyz = '''H 0.0 0.0 0.0; F 0.0 0.0 2.0'''
bas = 'def2-svp'
mf = guess.from_frag_tight(xyz, bas, [[0],[1]], [0,0], [1,-1])
mf.mulliken_pop() 
dump_mat.dump_mo(mf.mol, mf.mo_coeff[0], ncol=10)

mf0 = guess.gen('''H 0.0 0.0 0.0''', bas, 0, 1)
mf1 = guess.gen('''F 0.0 0.0 2.0''', bas, 0, -1).set(max_cycle=5)
#mf1 = guess.apply_field(mf1, (0.0,0.01,0.02)).run() # add field in z direction
mf1.set(symmetry = True)
mf1.irrep_nelec = {'s+0':(2,2),'p-1':(1,1),'p+0':(0,1),'p+1':(1,1)}
mf1.kernel()
mf1.mulliken_pop() # check odd electron on pz
mf = guess._from_frag([mf0, mf1], None, None, None, conv='tight')
mf.mulliken_pop() 
dump_mat.dump_mo(mf.mol, mf.mo_coeff[0], ncol=10)
