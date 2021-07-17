# pyAutoMR

The method used by this program is quite similar to [MOKIT](https://gitlab.com/jxzou/mokit). However, we try to do everything with PySCF and without Gaussian.

This program aims to do:
* HF guess strategy
* automatic guess for CASSCF/SUHF 
* interface for post-MR

## Installation
Pre-requisites
* MOKIT (no need to fully compile, only lo, autopair are needed)
* [PySCF](https://github.com/pyscf/pyscf)
* [mrh](https://github.com/MatthewRHermes/mrh) (optional, for MC-PDFT)
* [ExSCF](https://github.com/hebrewsnabla/ExSCF) (optional, for SUHF)


Install
* git clone and add `/path/to/pyAutoMR` to your `PYTHONPATH`

## Features
* UHF -> UNO -> CASSCF
* UHF -> SUHF -> CASSCF
* RHF -> vir MO projection -> PM LMO -> CASSCF
* CASSCF -> NEVPT2
* CASSCF -> MC-PDFT

UHF, RHF can be auto-detected.

## Utilities
* guess for UHF
  + mix
  + fragment
  + from_fch
* UHF stability
* dump CASCI coefficients
* dump (active) orbital compositions

## Quick Start
```
from automr import guess, autocas, cidump

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.9' 
bas = 'cc-pvdz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
guess.check_stab(mf)

mf2 = autocas.cas(mf)
```

## Tutorials
* [UHF case](https://blog-quoi.readthedocs.io/en/latest/mr_tutor.html#uhf-case)

## TODO
* UNO -> GVB(Q-Chem) -> CASSCF
* TDDFT NO -> CASSCF
* SA-CASSCF
