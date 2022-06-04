from pyscf import fci
import numpy as np

def merge_vec(va, vb, n):
    #print(str(bin(va)))
    va = str(bin(va))[2:].rjust(n,'0')[::-1]
    vb = str(bin(vb))[2:].rjust(n,'0')[::-1]
    v = ['0']*n
    vo = ['0']*n
    for i in range(n):
        a = int(va[i])
        b = int(vb[i])
        if a and b :
            v[i] = '1'
            vo[i]  = '2'
        elif a:
            v[i] = 'a'
            vo[i] = '1'
        elif b:
            v[i] = 'b'
            vo[i] = '1'
 
    return ''.join(v), ''.join(vo), int(va), int(vb)

def dump(mc, thresh=1e-2):
    '''
    input:
        mc: PySCF CASCI/CASSCF object
    return:
        dump_g, dump_o: dict for occupation vector and c**2, 
                        in Gaussian and ORCA format, respectively
    Gaussian format: 11a00
    ORCA format    : 22100
    '''
    if isinstance(mc.ci, (list, tuple)):
        civecs = mc.ci
    else:
        civecs = [mc.ci]
    for ici, ci in enumerate(civecs):
        print('***** CI components ROOT %d ******' % ici)
        ncas = mc.ncas
        #na, nb = mc.nelecas
        #print(mc.fcisolver.nelec)
        spin = mc.fcisolver.spin
        na,nb =  fci.addons._unpack_nelec(mc.nelecas, spin)
        #print(mc.ci)
        lena, lenb = ci.shape
        addra, addrb = np.where(abs(ci) > np.sqrt(thresh))
        dump_g = {}
        dump_o = {}
        #print(addra)
        #print(addrb)
        for k in range(len(addra)):
            #for j in addrb:
            i = addra[k]
            j = addrb[k]
            #print('ncas %d na %d i %d'%(ncas, na,i))
            #print(fci.cistring.num_strings(ncas, na) )
            veca = fci.cistring.addr2str(ncas, na, i)
            vecb = fci.cistring.addr2str(ncas, nb, j)
            coeff = ci[i,j]
            #    if coeff**2 > thresh:
            #print(veca)
            vec, veco, va, vb = merge_vec(veca, vecb, ncas)
            dump_g[vec] = coeff**2
            if veco in dump_o:
                dump_o[veco] += coeff**2
            else:
                dump_o[veco] = coeff**2
        dump_g = sorted(dump_g.items(),  key=lambda d: d[1], reverse=True)
        dump_o = sorted(dump_o.items(),  key=lambda d: d[1], reverse=True)
        #print(dump_g)
        #print(dump_o)
        print(" c**2   Gaussian-type vector")
        for k,v in dump_g:
            print("{: .6f}  {:7s}".format(v,k))
        print(" c**2     ORCA-type vector")
        for k,v in dump_o:
            print("{: .6f}  {:7s}".format(v,k))
        
    return dump_g, dump_o

