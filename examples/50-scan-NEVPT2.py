from pyscf import lib
from automr import guess, autocas, mcpdft
import os, contextlib
import numpy as np

lib.num_threads(4)

os.system('mkdir -p scan')
for r in np.arange(0.8, 2.55, 0.1):
    output = 'scan/n2_%.1f.out' % r
    os.system("echo '\n' > %s" % output)
    print('scan %.1f' % r)
    with open(output, 'a', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f):
            xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 %f'''%r
            bas = 'cc-pvtz'

            mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
            mf = guess.check_stab(mf)

            mf2 = autocas.cas(mf, (6,(3,3)), natorb=False)
            
            #mf3 = mcpdft.PDFT(mf2, 'tpbe')
            #mf3.kernel()
            
            mf4 = autocas.nevpt2(mf2)
