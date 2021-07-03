import frag_sol

xyz = '''H 0.0 0.0 0.0
O 0.0 0.0 1.0
O 0.0 0.0 2.0
H 0.0 0.0 3.0
'''
#bas = 'cc-pvdz'
gjfhead = '''%nproc=1
%mem=1gb
#p b3lyp/cc-pvdz scrf=read out=wfn

title

'''
scrfhead = '''eps=1.0
epsinf=1.0
'''
wfnpath = 'D:\\xxx\\yyy\\frag'
## the optput wfn will be frag0.wfn, frag1.wfn, ...

frag_sol.from_frag(xyz, [[0,1],[2,3]], [0,0],[1,-1], gjfhead, scrfhead, 'frag', wfnpath)
