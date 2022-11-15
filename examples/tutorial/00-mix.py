from automr import guess

xyz = '''H 0.0 0.0 0.0; H 0.0 0.0 2.0'''
bas = 'def2-svp'
mf = guess.mix(xyz, bas) # only do 5 cycle SCF
mf = guess.mix_tight(xyz, bas) # normal SCF and stable=opt
