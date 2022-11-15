from automr import guess

xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 4.0'''
bas = 'cc-pvtz'
mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3])
mf = guess.from_frag_tight(xyz, bas, [[0],[1]], [0,0], [3,-3])



