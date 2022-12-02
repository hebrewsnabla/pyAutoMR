from automr import guess

xyz = '''F 0.0 0.0 0.0; F 0.0 0.0 1.5'''
bas = 'def2-svp'

mf = guess.mix_tight(xyz, bas) # end up no polarized spin
mf = guess.mix_tight(xyz, bas, hl=[-2,0]) # correct polarized spin
