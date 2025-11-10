import os
import numpy as np
import pandas as pd
import smilenfer.plotting as splot

# Original traits
fit_dir = "../all_opt_fits/original_traits/"
fits_hat = pd.read_csv(os.path.join(fit_dir, "opt_results_original_traits_eur_raw.csv"))
fits_post = pd.read_csv(os.path.join(fit_dir, "opt_results_original_traits_eur_post.csv"))

trait_groups = {"Quantitative":["height", "bmi", "ldl", "hdl", "dbp", "sbp", "triglycerides", "urate", "rbc", "wbc", "grip_strength", "fvc", "pulse_rate"],
                 "Disease": ["bc", "cad", "ibd", "scz", "t2d", "arthrosis", "asthma", "diverticulitis", "gallstones", "glaucoma", "hypothyroidism",
                             "malignant_neoplasms", "uterine_fibroids", "varicose_veins"]}
trait_group_labels = ["Quantitative", "Disease"]
trait_names = {
    # Quantitative traits
    "height": "Height",
    "bmi": "Body Mass Index",
    "ldl": "LDL Cholesterol",
    "hdl": "HDL Cholesterol",
    "dbp": "Diastolic Blood Pressure",
    "sbp": "Systolic Blood Pressure",
    "triglycerides": "Triglycerides",
    "urate": "Serum Urate",
    "rbc": "Red Blood Cell Count",
    "wbc": "White Blood Cell Count",
    "grip_strength": "Grip Strength",
    "fvc": "Forced Vital Capacity",
    "pulse_rate": "Pulse Rate",

    # Disease traits
    "bc": "Breast Cancer",
    "cad": "Coronary Artery Disease",
    "ibd": "Inflammatory Bowel Disease",
    "scz": "Schizophrenia",
    "t2d": "Type 2 Diabetes",
    "arthrosis": "Arthrosis",
    "asthma": "Asthma",
    "diverticulitis": "Diverticulitis",
    "gallstones": "Gallstones",
    "glaucoma": "Glaucoma",
    "hypothyroidism": "Hypothyroidism",
    "malignant_neoplasms": "Malignant Neoplasms",
    "uterine_fibroids": "Uterine Fibroids",
    "varicose_veins": "Varicose Veins"
}

fig, axes = splot.plot_ML_table_3(fits_post, trait_groups, trait_group_labels, trait_names, 
                                   ML_table_2=fits_hat, pval=False, ML_table_samples=None, main_title="Original Traits")

# Save the figure as a PDF
fig.savefig("original_trait_pval.pdf", bbox_inches='tight')


# BBJ traits
fit_dir = "../bbj/"
fits_bbj_pval = pd.read_csv(os.path.join(fit_dir, "opt_results_pval_bbj.csv"))
fits_bbj_high = pd.read_csv(os.path.join(fit_dir, "opt_results_high_bbj.csv"))
fits_bbj_random = pd.read_csv(os.path.join(fit_dir, "opt_results_random_bbj.csv"))

trait_groups = {"Quantitative": ["bmi", "dbp", "hdl", "height", "ldl", "rbc", "sbp", "triglycerides"],
                "Disease": ["bc", "cad", "gallstones", "t2d", "uterine_fibroids"]}
trait_group_labels = ["Quantitative", "Disease"]
trait_names = {
    # Quantitative traits
    "bmi": "Body Mass Index",
    "dbp": "Diastolic Blood Pressure",
    "hdl": "HDL Cholesterol",
    "height": "Height",
    "ldl": "LDL Cholesterol",
    "rbc": "Red Blood Cell Count",
    "sbp": "Systolic Blood Pressure",
    "triglycerides": "Triglycerides",
    # Disease traits
    "bc": "Breast Cancer",
    "cad": "Coronary Artery Disease",
    "gallstones": "Gallstones",
    "t2d": "Type 2 Diabetes",
    "uterine_fibroids": "Uterine Fibroids"
}

fig, axes = splot.plot_ML_table_3(fits_bbj_high, trait_groups, trait_group_labels, trait_names,
                                      ML_table_2=fits_bbj_pval, pval=True, ML_table_samples=fits_bbj_random, main_title="BBJ Traits")

# Save the figure as a PDF
fig.savefig("bbj_trait_pval.pdf", bbox_inches='tight')


# MVP finemapping traits
fit_dir = "../mvp/"
fits_mvp = pd.read_csv(os.path.join(fit_dir, "opt_results_mvp_finemapping_eur.csv"))
traits_groups = {"Disease": ["Atrial fibrillation", "Basal cell carcinoma", "Cancer of prostate", "Coronary atherosclerosis",
                             "Diverticulosis and diverticulitis", "Glaucoma", "Gout", "Hyperlipidemia", 
                             "Hypertension", "Hypothyroidism", "Type 2 diabetes"]}
trait_group_labels = ["Disease"]
trait_names = {
    "Atrial fibrillation": "Atrial Fibrillation",
    "Basal cell carcinoma": "Basal Cell Carcinoma",
    "Cancer of prostate": "Prostate Cancer",
    "Coronary atherosclerosis": "Coronary Atherosclerosis",
    "Diverticulosis and diverticulitis": "Diverticulosis and Diverticulitis",
    "Glaucoma": "Glaucoma",
    "Gout": "Gout",
    "Hyperlipidemia": "Hyperlipidemia",
    "Hypertension": "Hypertension",
    "Hypothyroidism": "Hypothyroidism",
    "Type 2 diabetes": "Type 2 Diabetes"
}

fig, ax = splot.plot_ML_table_3(fits_mvp, traits_groups, trait_group_labels, trait_names,
                                  pval=True, ML_table_2=None, ML_table_samples=None, main_title="MVP Finemapping Traits")

# Save the figure as a PDF
fig.savefig("mvp_trait_pval.pdf", bbox_inches='tight')

# Original traits with MVP matched effects
fit_dir = "../mvp/matching/"
fits_mvp_matched_eur = pd.read_csv(os.path.join(fit_dir, "opt_results_mvp_matching_eur.csv"))
fits_mvp_matched_afr = pd.read_csv(os.path.join(fit_dir, "opt_results_mvp_matching_afr.csv"))
fits_mvp_matched_amr = pd.read_csv(os.path.join(fit_dir, "opt_results_mvp_matching_amr.csv"))
fits_mvp_matched_meta = pd.read_csv(os.path.join(fit_dir, "opt_results_mvp_matching_meta.csv"))

traits_groups = {"Quantitative": ["bmi", "dbp", "hdl", "height", "ldl", "sbp", "triglycerides", "wbc"],
                 "Disease": ["arthrosis", "asthma", "bc", "cad", "diverticulitis", "gallstones", "glaucoma",
                             "hypothyroidism", "ibd", "t2d", "varicose"]}

trait_group_labels = ["Quantitative", "Disease"]
trait_names = {
    # Quantitative traits
    "bmi": "Body Mass Index",
    "dbp": "Diastolic Blood Pressure",
    "hdl": "HDL Cholesterol",
    "height": "Height",
    "ldl": "LDL Cholesterol",
    "sbp": "Systolic Blood Pressure",
    "triglycerides": "Triglycerides",
    "wbc": "White Blood Cell Count",
    # Disease traits
    "arthrosis": "Arthrosis",
    "asthma": "Asthma",
    "bc": "Breast Cancer",
    "cad": "Coronary Artery Disease",
    "diverticulitis": "Diverticulitis",
    "gallstones": "Gallstones",
    "glaucoma": "Glaucoma",
    "hypothyroidism": "Hypothyroidism",
    "ibd": "Inflammatory Bowel Disease",
    "t2d": "Type 2 Diabetes",
    "varicose": "Varicose Veins"
}

# Plot and save each table
fig, axes = splot.plot_ML_table_3(fits_mvp_matched_eur, traits_groups, trait_group_labels, trait_names,
                                   ML_table_2=None, pval=True, ML_table_samples=None, main_title="MVP Matched EUR")
fig.savefig("mvp_matched_eur_trait_pval.pdf", bbox_inches='tight')
fig, axes = splot.plot_ML_table_3(fits_mvp_matched_afr, traits_groups, trait_group_labels, trait_names,
                                   ML_table_2=None, pval=True, ML_table_samples=None, main_title="MVP Matched AFR")
fig.savefig("mvp_matched_afr_trait_pval.pdf", bbox_inches='tight')
fig, axes = splot.plot_ML_table_3(fits_mvp_matched_amr, traits_groups, trait_group_labels, trait_names,
                                   ML_table_2=None, pval=True, ML_table_samples=None, main_title="MVP Matched AMR")
fig.savefig("mvp_matched_amr_trait_pval.pdf", bbox_inches='tight')
fig, axes = splot.plot_ML_table_3(fits_mvp_matched_meta, traits_groups, trait_group_labels, trait_names,
                                   ML_table_2=None, pval=True, ML_table_samples=None, main_title="MVP Matched Meta")
fig.savefig("mvp_matched_meta_trait_pval.pdf", bbox_inches='tight')

# UKBB susiex
fit_dir = "../ukbb_fine_mapping/"
fits_susiex = pd.read_csv(os.path.join(fit_dir, "opt_results_ukbb_susiex.csv"))

traits_groups = {"Quantitative": ["bmi", "dbp", "hdl", "height", "ldl", "sbp", "triglycerides", "wbc"]}
trait_group_labels = ["Quantitative"]
trait_names = {
    # Quantitative traits
    "bmi": "Body Mass Index",
    "dbp": "Diastolic Blood Pressure",
    "hdl": "HDL Cholesterol",
    "height": "Height",
    "ldl": "LDL Cholesterol",
    "sbp": "Systolic Blood Pressure",
    "triglycerides": "Triglycerides",
    "wbc": "White Blood Cell Count"
}

fig, axes = splot.plot_ML_table_3(fits_susiex, traits_groups, trait_group_labels, trait_names,
                                   ML_table_2=None, pval=True, ML_table_samples=None, main_title = "UKBB SusieX Traits")
# Save the figure as a PDF
fig.savefig("ukbb_susiex_trait_pval.pdf", bbox_inches='tight')