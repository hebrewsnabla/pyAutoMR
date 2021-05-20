from pyscf import fci

def merge_vec(va, vb, n):
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

def dump(mc):
    '''
    input:
        mc: PySCF CASCI/CASSCF object
    return:
        dump_g, dump_o: dict for occupation vector and c**2, 
                        in Gaussian and ORCA format, respectively
    Gaussian format: 11a00
    ORCA format    : 22100
    '''
    ci = mc.ci
    ncas = mc.ncas
    na, nb = mc.nelecas
    #print(mc.ci)
    lena, lenb = mc.ci.shape
    dump_g = {}
    dump_o = {}
    for i in range(lena):
        for j in range(lenb):
            veca = fci.cistring.addr2str(ncas, na, i)
            vecb = fci.cistring.addr2str(ncas, nb, j)
            coeff = ci[i,j]
            if coeff**2 > 1e-3:
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

