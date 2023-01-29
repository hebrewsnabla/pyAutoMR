# pyAutoMR

[![Latest Version](https://img.shields.io/github/v/release/hebrewsnabla/pyAutoMR)](https://github.com/hebrewsnabla/pyAutoMR/releases/latest)
[![pypi version](https://img.shields.io/pypi/v/pyAutoMR.svg)](https://pypi.python.org/pypi/pyAutoMR)
[![Downloads](https://pepy.tech/badge/pyAutoMR/month)](https://pepy.tech/project/pyAutoMR)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pyAutoMR.svg?label=PyPI%20downloads)](https://pypi.org/project/pyAutoMR/)
![GitHub Downloads][gh-downloads]

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
* [Tutorial: Symmetry-broken wavefunction](https://blog.shi-rong.wang/pyautomr_1.html)
* [Auto CASSCF: UHF case](https://blog.shi-rong.wang/mr_practice/mr_tutor.html#uhf-case)

## TODO
* TDDFT NO -> CASSCF

## Citation
Please cite pyAutoMR as
> Shirong Wang, pyAutoMR, https://github.com/hebrewsnabla/pyAutoMR (accessed month day, year)

and cite every program called by pyAutoMR, such as PySCF, MOKIT, mrh, etc.

If you peform calculations involving GVB, please also cite the following two papers
> DOI: 10.1021/acs.jctc.8b00854; DOI: 10.1021/acs.jpca.0c05216.


[![Ko-Fi][kofi-badge]][kofi]


[gh-downloads]: https://img.shields.io/github/downloads/hebrewsnabla/pyAutoMR/total?color=pink&label=GitHub%20Downloads
[kofi-badge]: https://img.shields.io/badge/Ko--fi-Buy%20me%20a%20coffee!-%2346b798.svg
[kofi]: https://ko-fi.com/srwang
