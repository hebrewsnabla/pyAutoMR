from automr import guess

xyz = '''F 0.0 0.0 0.0; F 0.0 0.0 4.0'''
bas = 'cc-pvtz'
mf = guess.from_frag_tight(xyz, bas, [[0],[1]], [0,0], [1,-1])

mf0 = guess.gen('''F 0.0 0.0 0.0''', bas, 0, 1)
mf0 = guess.apply_field(mf0, (0.01,0.01,0.0).run()
#mf0.analyze()
#mf = guess._from_frag(xyz, bas, [[0],[1]], [0,0], [1,-1])
#
