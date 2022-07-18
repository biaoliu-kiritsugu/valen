z_dims = [8]
alpha_1s = [1, 1e-1, 1e-2]
alpha_2s = [1, 1e-1, 1e-2]
alpha_3s = [1, 1e-1, 1e-2]
betas = [1e-2, 1e-3]
gammas = [1e-2, 1e-3]
thetas = [1]
sigmas = [5, 10]
corrects = [0]

task_fmt = 'nohup python -u main.py -bs 100 -dt realworld -ds lost -lr 1e-2 -wd 1e-3 ' \
           '-z_dim {} -alpha1 {} -alpha2 {} -alpha3 {} -beta {} -gamma {} -theta {} -sigma {} -correct {} -warm_up 10 -gpu {} &'
task_list = []
for z_dim in z_dims:
    for alpha_1 in alpha_1s:
        for alpha_2 in alpha_2s:
            for alpha_3 in alpha_3s:
                for beta in betas:
                    for gamma in gammas:
                        for theta in thetas:
                            for sigma in sigmas:
                                for correct in corrects:
                                    if not (alpha_1 < beta or alpha_1 < gamma
                                    or alpha_2 < beta or alpha_2 < gamma
                                    or alpha_3 < beta or alpha_3 < gamma
                                    or beta != gamma):
                                        task_list.append(task_fmt.format(z_dim, alpha_1, alpha_2,
                                                                         alpha_3, beta, gamma, theta, sigma, correct, 0))
print(len(task_list))
print(task_list)