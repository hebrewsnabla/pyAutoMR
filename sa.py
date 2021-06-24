from pyscf import mcscf, tdscf

def td_density():

def get_no():


def seek_states(mf, nstates):
    td = tdscf.TDHF(mf, nstates*2)
    td.kernel()
    n = int(nstates*1.5)
    xy = td.xy[:n]
    td_dm = td_density(td, xy[0])
    for i in range(1,n):
        dm = td_density(td, xy[i])
        td_dm += dm
    td_dm = td_dm * (1.0 / n)
    no = get_no(td_dm)
    return mf, no, nacto, (nacta, nactb)

def sa_cas(mf, nroots, crazywfn=False, max_memory=2000):
    mf, no, nacto, (nacta, nactb) = seek_states(mf, nroots)
    mc0 = mcscf.CASCI(mf,nacto,(nacta,nactb))
    mc0.fcisolver.nroots = nroots
    mc0.kernel()
    mc = mcscf.CASSCF(mf,nacto,(nacta,nactb)).state_average_(weights)
    mc.fcisolver.max_memory = max_memory // 2
    mc.max_memory = max_memory // 2
    mc.max_cycle = 200
    mc.fcisolver.spin = nopen
    if crazywfn:
        mc.fix_spin_(ss=nopen)
        mc.fcisolver.level_shift = 0.2
        mc.fcisolver.pspace_size = 1200
        mc.fcisolver.max_space = 100
        mc.fcisolver.max_cycle = 300
    else:
        mc.fcisolver.max_cycle = 100
    mc.natorb = True
    mc.verbose = 5
    mc.kernel()
    return mc