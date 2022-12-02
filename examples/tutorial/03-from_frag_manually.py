from automr import guess

xyz = '''H 0.0 0.0 0.0; F 0.0 0.0 2.0'''
bas = 'def2-svp'
mf = guess.from_frag_tight(xyz, bas, [[0],[1]], [0,0], [1,-1])

mf0 = guess.gen('''H 0.0 0.0 0.0''', bas, 0, 1)
#mf0 = guess.apply_field(mf0, (0.0,0.0,0.01)).run() # add field in z direction
#mf0.mulliken_pop() # check odd electron on pz
mf1 = guess.gen('''F 0.0 0.0 2.0''', bas, 0, -1).set(max_cycle=5)
mf1 = guess.apply_field(mf1, (0.0,0.01,0.02)).run() # add field in z direction
mf1.mulliken_pop() # check odd electron on pz
mf = guess._from_frag([mf0, mf1], None, None, None, conv='tight')
