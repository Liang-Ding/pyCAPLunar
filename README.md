# pyCAPLunar
[![DOI](https://zenodo.org/badge/310115978.svg)](https://zenodo.org/badge/latestdoi/310115978)

pyCAPLunar is a python package to create the receiver-side 3D strain Greens' function (SGT) and Greens' function (the displacement, DGF). 
It also uses waveform fitting to determine the moment tensor of earthquake sources and 
uses the Hamiltonian Monte Carlo method to estimate the posterior probability of the inversion result as its uncertainty. 

## Citation
```text
Liang Ding, 2022, pyCAPLunar Ver 2.1. Source code. https://www.github.com/Liang-Ding/pyCAPLunar. doi:10.5281/zenodo.7003977
```
```text
@misc{Ding_pyCAPLunar,
title = {pyCAPLunar, Ver 2.1},
author = {Ding, Liang},
abstractNote = {pyCAPLunar is a python package to determine the moment tensor of earthquake sources and their uncertainty by using the 3D receiver-side SGT database.},
url = {https://www.github.com/Liang-Ding/pyCAPLunar},
doi = {10.5281/zenodo.7003977}, 
year = {2022},
month = {Aug},
}
```

## Modules
![pyCAPLunar Modules](https://github.com/Liang-Ding/pyCAPLunar/blob/master/Documentation/pyCAPLunar_modules.jpg)


## Folder: SPECFEM_2_StrainField/
SPECFEM3D_Cartesian to write out the strain field
![instruction](https://github.com/Liang-Ding/pyCAPLunar/blob/master/SPECFEM_2_StrainField/SPECFEM3D_Cartesian_2_strain_field.png)
