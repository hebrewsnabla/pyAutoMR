# pyAutoMR

The method used by this program is quite similar to [MOKIT](https://gitlab.com/jxzou/mokit). However, we try to do everything with PySCF and without Gaussian.

This program aims to do:
* HF guess strategy
* automatic guess for CASSCF/GVB/SUHF 
* interface for post-MR

## Installation
Pre-requisites
* MOKIT (no need to fully compile, only `autopair` is needed)
* [PySCF](https://github.com/pyscf/pyscf)
* [mrh](https://github.com/MatthewRHermes/mrh) (optional, for MC-PDFT)
* [ExSCF](https://github.com/hebrewsnabla/ExSCF) (optional, for SUHF)
* [pyNOF](https://github.com/hebrewsnabla/pyNOF) (optional, for GVB)


Install
* git clone and add `/path/to/pyAutoMR` to your `PYTHONPATH`

## Features
* UHF -> UNO (-> PM LMO -> assoc rot) (-> GVB) -> CASSCF
* UHF -> SUHF -> CASSCF
* RHF (-> vir MO projection -> PM LMO -> pairing) (-> GVB ) -> CASSCF
* CASSCF -> NEVPT2
* CASSCF -> MC-PDFT
* CASSCF(dry run) -> SA-CASSCF

UHF, RHF can be auto-detected.

## Utilities
* guess for UHF/UKS
  + mix
  + fragment
  + from_fch
  + flipspin (by lmo or by site)
* internal stability for RHF/RKS, UHF/UKS, ROHF/ROKS
  + optimize wavefunction until stable
* dump CASCI coefficients
* dump (active) orbital compositions

## Quick Start
```
from automr import guess, autocas

xyz = 'N 0.0 0.0 0.0; N  0.0 0.0 1.9' 
bas = 'cc-pvdz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3], cycle=50)
mf = guess.check_stab(mf)

mf2 = autocas.cas(mf)
```

## Tutorials
* [UHF case](https://blog-quoi.readthedocs.io/en/latest/mr_tutor.html#uhf-case)

## TODO
* TDDFT NO -> CASSCF

## Citation
Please cite pyAutoMR as
> Shirong Wang, pyAutoMR, https://github.com/hebrewsnabla/pyAutoMR (accessed month day, year)

and cite every program called by pyAutoMR, such as PySCF, MOKIT, mrh, etc.

If you peform calculations involving GVB, please also cite the following two papers
> DOI: 10.1021/acs.jctc.8b00854; DOI: 10.1021/acs.jpca.0c05216.
