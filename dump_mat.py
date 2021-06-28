import sys

def dump_mo(mol, c, label2=None,
             ncol=6, digits=5, start=0):
    ''' Print an array in rectangular format
    Args:
        mol: `gto.Mole` object
        c : numpy.ndarray
            coefficients
    Kwargs:
        label2 : list of strings
            Col labels (default is 1,2,3,4,...)
        ncol : int
            Number of columns in the format output (default 5)
        digits : int
            Number of digits of precision for floating point output (default 5)
        start : int
            The number to start to count the index (default 0)
    '''
    stdout = sys.stdout
    label = mol.ao_labels()
    nc = c.shape[1]
    if label2 is None:
        fmt = '#%%-%dd' % (digits+3)
        label2 = [fmt%i for i in range(start,nc+start)]
    else:
        fmt = '%%-%ds' % (digits+4)
        label2 = [fmt%i for i in label2]
    for ic in range(0, nc, ncol):
        dc = c[:,ic:ic+ncol]
        m = dc.shape[1]
        fmt = (' %%%d.%df'%(digits+4,digits))
        #if label is None:
        #    stdout.write(((' '*(digits+3))+'%s\n') % ' '.join(label2[ic:ic+m]))
        #    for k, v in enumerate(dc):
        #        stdout.write(('%-5d' % (k+start)) + (fmt % tuple(v)))
        #else:
        stdout.write(((' '*(digits+10))+'%s\n') % ' '.join(label2[ic:ic+m]))
        for k, v in enumerate(dc):
            coefs = ''
            for j in v:
                if abs(j) < 0.1:
                    coefs += '          '
                else:
                    coefs += fmt % j
            stdout.write(('%12s' % label[k]) + coefs + '\n')
    stdout.flush()