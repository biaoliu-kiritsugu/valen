import os
import sys
import time
import argparse

import numpy as np

gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')
print(gpu_status)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CMD code')
    parser.add_argument('scripts', help='...')
    args = parser.parse_args()
    return args


def gpu_info():
    gpu_mem_list = []
    gpu_pow_list = []
    gpu_status_list = os.popen('nvidia-smi | grep %').read().split('\n')
    gpu_status_list.pop()
    for gpu_status in gpu_status_list:
        gpu_status = gpu_status.split('|')
        gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
        gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
        gpu_mem_list.append(gpu_memory)
        gpu_pow_list.append(gpu_power)
    gpu_mem_list = np.array(gpu_mem_list)
    min_idx = np.argmin(gpu_mem_list)
    return min_idx, gpu_pow_list[min_idx], gpu_mem_list[min_idx]


def narrow_setup(interval=60, task_list=None):
    for task_id in range(len(task_list)):
        gpu_id, gpu_power, gpu_memory = gpu_info()
        while gpu_memory > 2000:  # set waiting condition
            gpu_id, gpu_power, gpu_memory = gpu_info()
            time.sleep(interval)
        print('\n' + str(task_id) + ': ' + task_list[task_id].format(gpu_id))
        os.system(task_list[task_id].format(gpu_id))
        # print(task_list[task_id].format(gpu_id))
        time.sleep(interval)


if __name__ == '__main__':
    # task_list = [
    #     'nohup python -u main.py -bs 256 -partial_type random -dt realworld -ds lost -lr 1e-1 -wd 1e-3 -alpha1 0.1 -alpha2 0.1 -alpha3 0 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0.5 -warm_up 10 -gpu {} &',
    # ]
    task_list = [
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u main.py -bs 32 -partial_type random -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu 0 &',
        # 'nohup python -u main.py -bs 32 -partial_type random -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu 0 &',
        # 'nohup python -u main.py -bs 32 -partial_type random -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu 0 &',
        # 'nohup python -u main.py -bs 32 -partial_type random -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu 0 &',
        # 'nohup python -u main.py -bs 32 -partial_type random -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu 0 &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_proden.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss proden -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss rc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_rccc.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss cc -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_lws.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss lws -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        # 'nohup python -u train_d2cnn.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -ep 150 -lr 1e-2 -wd 1e-4 -loss d2cnn -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 32 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
        'nohup python -u main.py -bs 32 -partial_type feature -dt benchmark -ds cub200 -lr 1e-2 -wd 1e-4 -z_dim 64 -alpha1 0.1 -alpha2 0.1 -alpha3 0.1 -beta 0.01 -gamma 0.01 -theta 1 -sigma 1 -correct 0 -warm_up 5 -gpu {} &',
    ]

    narrow_setup(interval=120, task_list=task_list)
