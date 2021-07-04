# pyAutoMR

The method used by this program is quite similar to [MOKIT](https://gitlab.com/jxzou/mokit). However, we try to do everything with PySCF and without Gaussian.

This program aims to do:
* HF guess strategy
* automatic guess for CASSCF/SUHF 

## Pre-requisites
* MOKIT (no need to fully compile, only lo, autopair are needed)
* PySCF

## Features
* UHF -> UNO -> CASSCF
* RHF -> vir MO projection -> PM LMO -> CASSCF

UHF, RHF can be auto-detected.

## Utilities
* guess for UHF
  + mix
  + fragment
  + from_fch
* UHF stability
* dump CASCI coefficients
* dump active orbital compositions

## Quick Start
```
import guess, autocas, cidump

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.9' 
bas = 'cc-pvdz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
guess.check_stab(mf)

mf2 = autocas.cas(mf)
cidump.dump(mf2)
```

## Tutorials
* [UHF case](https://blog-quoi.readthedocs.io/en/latest/mr_tutor.html#uhf-case)

## TODO
* UNO -> GVB(Q-Chem) -> CASSCF
* TDDFT NO -> CASSCF
* SA-CASSCF
