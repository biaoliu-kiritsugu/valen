import os

datasets = ["Ar", "SJAFFE", "Yeast_spoem", "Yeast_spo5",
           "Yeast_dtt", "Yeast_cold", "Yeast_heat",
           "Yeast_spo", "Yeast_diau", "Yeast_elu",
           "Yeast_cdc", "Yeast_alpha", "SBU_3DFE",
           "Movie"]
lr = [1e-1, 1e-2, 1e-3]
wd = [1e-1, 1e-2, 1e-3, 1e-4]
alpha = [1, 1e-1, 1e-2]
beta_gamma = [1, 1e-1, 1e-2]
theta = [1, 1e-1]
# # datasets = ['Ar', 'SJAFFE']
# task_format = "python -u label_enhancement.py -ds {} -ep 50 -lr 1e-3 -wd 1e-5" \
#               " -alpha1 1 -alpha2 1 -beta 0.01 -gamma 0.01 -theta 1 -gpu 0"
# for dataset in datasets:
#     os.system(task_format.format(dataset))
# task_format = "python -u label_enhancement.py -ds {} -ep 50 -lr 1e-2 -wd 1e-5" \
#               " -alpha1 1 -alpha2 1 -beta 0.01 -gamma 0.01 -theta 1 -gpu 0"
# for dataset in datasets:
#     os.system(task_format.format(dataset))
# task_format = "python -u label_enhancement.py -ds {} -ep 50 -lr 1e-2 -wd 1e-5" \
#               " -alpha1 1 -alpha2 1 -beta 0.01 -gamma 0.01 -theta 1 -gpu 0"
# for dataset in datasets:
#     os.system(task_format.format(dataset))
task_format = "python -u -W ignore label_enhancement.py -ds {} -ep 100 -lr {} -wd {}" \
              " -alpha1 {} -alpha2 {} -beta {} -gamma {} -theta {} -gpu 0"
for ds in datasets:
    print(ds)
    for l in lr:
        for w in wd:
            for a in alpha:
                for bg in beta_gamma:
                    for t in theta:
                        os.system(task_format.format(ds, l, w, a, a, bg, bg, t))

# python -u -W ignore label_enhancement.py -ds Yeast_spoem -ep 10 -lr 1e-3 -wd 1e-4 -alpha1 1 -alpha2 1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0.5 -gpu 0"