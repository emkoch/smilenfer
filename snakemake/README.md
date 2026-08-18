Legacy Snakemake workflows used for the previous submission and retained for reproducibility. Current production model-fitting workflows are under `results/all_opt_fits`.

`Snakefile_trait_fits_simple`, `simple_all_traits_nofilter.yml`, and `test_run` implement the earlier grid-based model-fitting workflow. They are not used for the current production fits.

The current original-trait production workflow is `results/all_opt_fits/original_traits/orignal_opt_fits.smk`. The current workflows for BBJ, UK Biobank fine-mapping, and MVP analyses are in the corresponding directories under `results/all_opt_fits`.

`Snakefile_trait_ascertainment_sim` is the legacy pipeline for trait-based ascertainment simulations.

`Snakefile_dfe_sims` is the legacy pipeline for DFE-based simulations.

`Snakefile_WF_sfs` is the legacy pipeline used to generate SFS grids.
