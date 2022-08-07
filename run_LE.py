import os

datasets = ["Ar", "SJAFFE", "Yeast_spoem", "Yeast_spo5",
           "Yeast_dtt", "Yeast_cold", "Yeast_heat",
           "Yeast_spo", "Yeast_diau", "Yeast_elu",
           "Yeast_cdc", "Yeast_alpha", "SBU_3DFE",
           "Movie"]
# datasets = ['Ar', 'SJAFFE']
task_format = "python -u label_enhancement.py -ds {} -ep 100 -lr 1e-3 -wd 1e-5" \
              " -alpha1 1 -alpha2 1 -beta 0.01 -gamma 0.01 -theta 1 -gpu 0"
for dataset in datasets:
    os.system(task_format.format(dataset))

