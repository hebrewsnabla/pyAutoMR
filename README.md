# pyAutoMR

The method used by this program is quite similar to [MOKIT](https://gitlab.com/jxzou/mokit). However, we try to do everything with PySCF and without Gaussian.

This program aims to do:
* HF guess strategy
* automatic guess for CASSCF/SUHF 

## Pre-requisites
* [MOKIT](https://gitlab.com/jxzou/mokit) (no need to fully compile, only pyuno, construct_vir are needed)
* PySCF

## Features
* UHF -> UNO -> CASSCF

## Utilities
* guess for UHF
  + mix
  + fragment
  + from_fch
* UHF stability
* dump CASCI coefficients

## TODO
* remove pyuno dependence
* CASSCF -> NEVPT2
* UNO -> GVB(Q-Chem) -> CASSCF
* TDDFT NO -> CASSCF
