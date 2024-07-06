Inference of simple selection models from the distribution of genetic associations.

[Genetic association data are broadly consistent with stabilizing selection shaping human common diseases and traits](https://doi.org/10.1101/2024.06.19.599789)
## Project components
### gwas_processing
Code for downloading and processing GWAS summary statistics.
### results
Processed GWAS summary statistics, model fits, simulation output, and scripts for generating figures.
### smilenfer
Main code used for analyses, simulations, and plotting. This is organized as a python package and which can be installed by first creating a conda environment with the necessary dependencies `conda env create -f smilenfer.yml`. After activating the environment run `install.sh`. 
### snakemake
Snakemake piplines used for model fits and simulations.
