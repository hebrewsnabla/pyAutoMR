from pyscf import gto

def rigid(internal, var, vrange, unit=1):
    for v in vrange:
        newint = internal.replace(var, '%f'%v)
        newcart = gto.format_atom(newint, unit=unit)
        newcart = atm2str(newcart)
        print(newcart)

def atm2str(atm):
    s = ''
    for a in atm:
        s += a[0]
        s += ' %f %f %f\n' % tuple(a[1])
    return s
