Snakemake files used to run simulations and analyses of GWAS summary statistics.

`Snakefile_trait_fits_simple` is the primary pipeline file for fitting the different selection models.

`simple_all_traits_nofilter.yml` is the config file used for primary analyses. File paths would need to be changed.

`test_run` test run on a single trait (breast cancer). Results should match those in `results/data/ML/SIMPLE_ALL_TRAITS_NOFILTER_GENOMEWIDE/ML_all_flat_5e-08_new.csv`

`Snakefile_trait_ascertainment_sim` pipeline for trait-based ascertainment simulations.

`Snakefile_dfe_sims` pipeline for DFE-based simuations.

`Snakefile_WF_sfs` pipeline ot generate the SFS over a grid.
