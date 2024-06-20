import os
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import smilenfer.posterior as post
import smilenfer.plotting as splot
splot._plot_params()
matplotlib.rcParams.update({'font.size': 18})

data_dir = "../data"
min_x = 0.01
p_thresh = 5e-08
p_cutoff = 5e-08

# traits_update = ["BMI", "BC", "HDL", "GRIP_STRENGTH", "FVC", "DBP", "CAD", 
#           "SBP", "RBC", "PULSE_RATE", "LDL", "IBD", "HEIGHT", "WBC", "URATE", 
#           "TRIGLYCERIDES", "T2D", "SCZ"]
# diseases_update =  ["ARTHROSIS", "ASTHMA", "DIVERTICULITIS", "GALLSTONES", "GLAUCOMA", "HYPOTHYROIDISM", 
#                     "MALIGNANT_NEOPLASMS", "UTERINE_FIBROIDS", "VARICOSE_VEINS"]
# # make all these names lowercase
# traits_update = [trait.lower() for trait in traits_update]
# diseases_update = [disease.lower() for disease in diseases_update]

# traits_update_labels = ["BMI", "Breast cancer", "HDL levels", "Grip strength", 
#                         "FVC", "Diastolic BP", "CAD", 
#                         "Systolic BP", "RBC", "Pulse rate", "LDL levels", "IBD", 
#                         "Standing height", "WBC", "Urate", 
#                         "Triglycerides", "Type 2 Diabetes", "SCZ"]

# diseases_update_labels = ["Arthrosis", "Asthma", "Diverticulitis", "Gallstones",
#                             "Glaucoma", "Hypothyroidism", "Malignant neoplasms", "Uterine fibroids",
#                             "Varicose veins"]

# # Sort the labels 
# trait_update_order = np.argsort(traits_update_labels)
# disease_update_order = np.argsort(diseases_update_labels)

# # Reorder the traits/diseases and then the labels
# traits_update = np.array(traits_update)[trait_update_order]
# traits_update_labels = np.array(traits_update_labels)[trait_update_order]
# diseases_update = np.array(diseases_update)[disease_update_order]
# diseases_update_labels = np.array(diseases_update_labels)[disease_update_order]

# # Create a merged list of traits and diseases and sort that as well
# all_traits = np.concatenate([traits_update, diseases_update])
# all_labels = np.concatenate([traits_update_labels, diseases_update_labels])
# all_order = np.argsort(all_labels)
# all_traits = all_traits[all_order]
# all_labels = all_labels[all_order]

# fname_trait = "clumped.{trait}.maf.5e-05.tsv.gz"
# fname_disease = "ash.{trait}.normal.block_mhc.finngen.tsv.gz"

# # Read in data for each trait
# data_traits_update = {trait: post.read_and_process_trait_data(os.path.join(data_dir, "clumped_ash", fname_trait.format(trait=trait))) 
#                       for trait in traits_update}
# data_diseases_update = {disease: post.read_and_process_trait_data(os.path.join(data_dir, "clumped_ash", 
#                                                                                fname_disease.format(trait=disease))) 
#                         for disease in diseases_update}

_, _, _, _, _, _, all_traits, all_labels, data_traits_all = splot.read_trait_files(os.path.join(data_dir, "clumped_ash"))


n_traits = len(all_traits)
n_rows = math.ceil(n_traits/4)
fig, ax = plt.subplots(n_rows, 4, figsize=(30, 4.5*n_rows))
ax = ax.flatten()
for i, trait in enumerate(all_traits):
    data = data_traits_all[trait]
    splot.plot_local_neff(data.mlogp, data.n_eff, ax[i], trait_name = all_labels[i])

# remove empty axes
for i in range(n_traits, len(ax)):
    fig.delaxes(ax[i])

fig.tight_layout()
fig.savefig("all_traits_neff.pdf", bbox_inches='tight')