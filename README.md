# pyAutoMR

The method used by this program is quite similar to [MOKIT](https://gitlab.com/jxzou/mokit). However, we try to do everything with PySCF and without Gaussian.

This program aims to do:
* HF guess strategy
* automatic guess for CASSCF/SUHF 

## Pre-requisites
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
* UNO -> GVB(Q-Chem) -> CASSCF
* TDDFT NO -> CASSCF
* SA-CASSCF
