import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from argparse import ArgumentParser
from plot_utils import compute_mean_ci

parser = ArgumentParser()
parser.add_argument("-T", "--timestamp", nargs='+', default=[''])
parser.add_argument("--nreps", type=int, default=10)

args = parser.parse_args()
timestamp = args.timestamp[0]
n_reps = args.nreps

# Selected betas for SafeOPT
betas = [90, 15, 10, 3.5, 2] 

rmean = 10
lw_val = 4
file_format = 'pdf'
n_bins = 20
save = 1
ls_val = '-'
alpha = 0.30

low_percentile = 15
high_percentile = 85

figsize_vals = (8,6)
fontsize_label=28
fontsize_ticks=20
fontsize_legend=20

benchmarks = ['NCB', 'SC_DNCB', 'MC_NCB']
benchmarks_b = ['safeOPT']

# var_ = [0.01, 0.05, 0.1, 0.2, 0.3]
# var_ = [0.01]

#Fixed std deviation for the first plots 
# var = 0.2
var = '[1e-4,1e-4]'

filenames_train = []
config_files = []
algo_list = []
timestamps = []

# labels = [r'RANCB, $\alpha = 0.995$', r'RANCB $\alpha = 0.5$', 'NCB', 'SC-DNCB', 'MC-NCB', 'SafeOPT']
# labels = [r'RANCB, $\alpha = 0.995$', r'RANCB $\alpha = 0.5$']
# labels = [r'RANCB $\alpha = 0.5$', r'RANCB $\alpha = 0.8$', r'RANCB $\alpha = 0.9$',\
#         r'RANCB $\alpha = 0.99$', r'RANCB, $\alpha = 0.995$', r'RANCB $\alpha = 0.999$']
# labels = [r'RANCB $\alpha = 0.999$']
labels = [r'MC_DNCB']
err_bar_kwargs = {'barsabove':True, 'capsize':4, 'lw':lw_val, 'elinewidth':2.5}


# list_configs = [1,9]
# list_configs = [9, 6, 5, 4, 1, 3]
list_configs = [3]

for i in list_configs:
    aux = []
    aux_conf = []
    for rep in np.arange(1, n_reps+1):
        f = f'{timestamp}_MC_DNCB_train_syn_var{var}-rep{rep}_log.csv'
        # f = '{timestamp}_RANCB_train_syn_var{var}-rep{rep}-conf{i}_log.csv'.\
        #     format(timaestamp, )
        aux.append(f)
        aux_conf.append(f'{timestamp}_MC_DNCB_syn_var{var}-rep{rep}_config.json')
    filenames_train.append(aux)
    config_files.append(aux_conf)
    algo_list.append(f'MC_DNCB')
    timestamps.append(timestamp)


# for algo in benchmarks:
#     aux = []
#     aux_conf = []
#     for rep in np.arange(1, n_reps+1):
#         f = f'{timestamp}_{algo}_train_syn_var{var}-rep{rep}_log.csv'
#         aux.append(f)
#         aux_conf.append(f'{timestamp}_{algo}_syn_var{var}-rep{rep}_config.json')
#     filenames_train.append(aux)
#     config_files.append(aux_conf)
#     algo_list.append(algo)
#     timestamps.append(timestamp)
    

conf_file = os.path.join(os.getcwd(), '..', 'results',  timestamp, config_files[0][0])
with open(conf_file) as f:
    conf_data = json.load(f)
constraint_val = conf_data['constr_val']




for fn_list, algo, lab, ts in zip(filenames_train, algo_list, labels, timestamps):
    
    v_len = len(pd.read_csv(os.path.join(os.path.join(os.getcwd(), '..', 'results',  ts), fn_list[0])))
    # reward_all = np.zeros((n_reps, v_len))
    # constraint_1_all = np.zeros((n_reps, v_len))
    # constraint_2_all = np.zeros((n_reps, v_len))
    
    # for i, fn in enumerate(fn_list):
    #     path = os.path.join(os.getcwd(), '..', 'results',  ts)
    #     df = pd.read_csv(os.path.join(path, fn))
    
    path = os.path.join(os.getcwd(), '..', 'results',  ts)
    df = pd.read_csv(os.path.join(path, fn_list[0]))
        
    #     reward1_all[i, :] = df['reward1'].rolling(rmean).mean()
    #     reward2_all[i, :] = df['reward2'].rolling(rmean).mean()
    #     # constraint_1_all[i, :] = np.maximum(0, df['constraint_1'] - constraint_val[0])
    #     # constraint_2_all[i, :] = np.maximum(0, df['constraint_2'] - constraint_val[1])
    #     # constraint_all = constraint_1_all + constraint_2_all
    #     # constraint_all = constraint_1_all

        
    # av_reward1 = np.mean(reward1_all, axis=0)
    # low_reward1 = np.percentile(reward1_all, low_percentile, axis=0)
    # high_reward1 = np.percentile(reward1_all, high_percentile, axis=0)
    
    # # constraint_all = constraint_all.cumsum(axis=1)
    # # av_constraint = np.mean(constraint_all, axis=0)
    # # low_constraint = np.percentile(constraint_all, low_percentile, axis=0)
    # # high_constraint = np.percentile(constraint_all, high_percentile, axis=0)
    
    # av_reward = np.mean(reward_all, axis=0)
    # low_reward = np.percentile(reward2_all, low_percentile, axis=0)
    # high_reward = np.percentile(reward2_all, high_percentile, axis=0)
    

    plt.figure(0, figsize=figsize_vals)
    # plt.plot(df['step'], -av_reward, label=lab, ls=ls_val)
    # plt.fill_between(df['step'], -low_reward, -high_reward, alpha=alpha)
    plt.plot(df['step'], df['reward1'], label=lab, ls=ls_val)
    # plt.fill_between(df['step'], low_reward, high_reward, alpha=alpha)
    plt.ylabel(r'Reward1 ($r_t$)', fontsize=fontsize_label)
    plt.xlabel(r'$t$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)
    # plt.legend(loc='best', fontsize=fontsize_legend)
    # plt.title('Reward training', fontsize=28)
    if save:
        plt.savefig('./syn_training_reward1.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
    
    plt.figure(1, figsize=figsize_vals)
    plt.plot(df['step'], df['reward2'], label=lab, ls=ls_val)
    # plt.fill_between(df['step'], low_constraint, high_constraint, alpha=alpha)
    plt.ylabel(r'Reward2 ($r_t$)', fontsize=fontsize_label)
    plt.xlabel(r'$t$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid(True)
    plt.legend(loc='best', fontsize=fontsize_legend)
    # plt.title('Constraint training', fontsize=28)
    if save:
        plt.savefig('./syn_training_reward2.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
        
    # # ### zoom in ###
    # plt.figure(2, figsize=figsize_vals)
    # # plt.plot(df['step'], -av_reward, label=algo, ls=ls_val)
    # # plt.fill_between(df['step'], -low_reward, -high_reward, alpha=alpha)
    # plt.plot(df['step'], av_reward, label=algo, ls=ls_val)
    # plt.fill_between(df['step'], low_reward, high_reward, alpha=alpha)
    # # plt.ylabel('reward', fontsize=fontsize_label)
    # # plt.xlabel('step', fontsize=fontsize_label)
    # plt.xticks(fontsize=fontsize_ticks)
    # plt.yticks(fontsize=fontsize_ticks)
    # plt.grid(True)
    # # plt.legend(loc='best', fontsize=fontsize_legend)
    # # plt.title('Reward training (zoom in)', fontsize=28)
    # # plt.xlim([2750, 3000])
    # plt.xlim([30000, 100000])
    # # plt.ylim([-.7, 0.1])
    # plt.ylim([0, 300])
    # if save:
    #     plt.savefig('./syn_training_reward_zoom.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
        





# ######### Evaluation statistics figure

# filenames_eval = []
# algo_list = []
# timestamps = []
# v_nreps = []

# for i in list_configs:
#     aux1 = []
#     for v in var_:
#         aux = []
#         for rep in np.arange(1, n_reps+1):
#             f = f'{timestamp}_RANCB_syn_var{v}-rep{rep}-conf{i}_log.csv'
#             aux.append(f)
#         aux1.append(aux)
#     filenames_eval.append(aux1)
#     algo_list.append(f'DNCB_conf{i}')
#     timestamps.append(ts)
#     v_nreps.append(n_reps)
      

# # for algo in benchmarks:
# #     aux1 = []
# #     for v in var_:
# #         aux = []
# #         for rep in np.arange(1, n_reps+1):
# #             f = f'{timestamp}_{algo}_syn_var{v}-rep{rep}_log.csv'
# #             aux.append(f)
# #         aux1.append(aux)
# #     filenames_eval.append(aux1)
# #     algo_list.append(algo)
# #     timestamps.append(timestamp)
# #     v_nreps.append(n_reps)


# # for algo in benchmarks_b:
# #     aux1 = []
# #     for v, beta in zip(var_, betas):
# #         aux = []
# #         for rep in np.arange(1, n_reps+1):
# #             f = f'{timestamp}_{algo}_syn_var{v}-rep{rep}-beta_{beta}_log.csv'
# #             aux.append(f)
# #         aux1.append(aux)
# #     filenames_eval.append(aux1)
# #     algo_list.append(algo)
# #     timestamps.append(timestamp)
# #     v_nreps.append(n_reps)



# n_vars = len(var_)


# for fn_list, algo, lab, ts, n_reps_i in zip(filenames_eval, algo_list, labels, timestamps, v_nreps):
    
#     reward = np.zeros((n_vars, n_reps_i))
#     constraint = np.zeros((n_vars, n_reps_i))
#     constraint_prob = np.zeros((n_vars, n_reps_i))
    
#     reward_av = np.zeros(n_vars)
#     constraint_av = np.zeros(n_vars)
#     constraint_prob_av = np.zeros(n_vars)
#     reward_ci = np.zeros(n_vars)
#     constraint_ci = np.zeros(n_vars)
#     constraint_prob_ci = np.zeros(n_vars)
        
#     for i, fn_reps in enumerate(fn_list): # var
#         for j, fn in enumerate(fn_reps): # reps
#             path = os.path.join(os.getcwd(), '..', 'results',  ts)
#             df = pd.read_csv(os.path.join(path, fn))
            
#             reward[i,j] = -np.mean(df['reward'])
#             constr_violation_1 = np.maximum(0, df['constraint_1'] - constraint_val[0])
#             # constr_violation_2 = np.maximum(0, df['constraint_2'] - constraint_val[1])
#             # constraint[i,j] = np.mean(constr_violation_1 + constr_violation_2)
#             # constraint_prob[i,j] = np.mean((constr_violation_1 + constr_violation_2)>0)
#             constraint[i,j] = np.mean(constr_violation_1)
#             constraint_prob[i,j] = np.mean(constr_violation_1 > 0)
        
#     for i in range(n_vars):
#         reward_av[i], reward_ci[i] = compute_mean_ci(reward[i,:])
#         constraint_av[i], constraint_ci[i] = compute_mean_ci(constraint[i,:])
#         constraint_prob_av[i], constraint_prob_ci[i] = compute_mean_ci(constraint_prob[i,:])

#     plt.figure(20, figsize=figsize_vals)
#     plt.errorbar(var_, reward_av, yerr=reward_ci, label=lab, **err_bar_kwargs)
#     plt.ylabel('Average reward', fontsize=fontsize_label)
#     plt.xlabel(r'$\sigma_{env}$', fontsize=fontsize_label)
#     plt.xticks(fontsize=fontsize_ticks)
#     plt.yticks(fontsize=fontsize_ticks)
#     plt.grid(True)
#     # plt.legend(loc='best', fontsize=fontsize_legend)
#     # plt.title('Reward eval', fontsize=28)
#     if save:
#         plt.savefig('./syn_eval_reward.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
    
#     plt.figure(21, figsize=figsize_vals)
#     plt.errorbar(var_, constraint_av, yerr=constraint_ci, label=lab, **err_bar_kwargs)
#     plt.ylabel('Average constraint violation', fontsize=fontsize_label)
#     plt.xlabel(r'$\sigma_{env}$', fontsize=fontsize_label)
#     plt.xticks(fontsize=fontsize_ticks)
#     plt.yticks(fontsize=fontsize_ticks)
#     plt.grid(True)
#     plt.legend(loc='upper left', fontsize=fontsize_legend)
#     # plt.title('Constraint eval', fontsize=28)
#     if save:
#         plt.savefig('./syn_eval_constraint.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')
            






# ######### Evaluation alpha statistics  figure
# filenames_eval = []

# # confs = [6, 5, 4, 1, 3]
# # alphas = [0.8, 0.9, 0.99, 0.995, 0.999]
# confs = [9, 6, 5, 4, 1, 3]
# alphas = [0.5, 0.8, 0.9, 0.99, 0.995, 0.999]



# for i in confs:
#     aux1 = []
#     for v in var_:
#         aux = []
#         for rep in np.arange(1, n_reps+1):
#             f = f'{timestamp}_RANCB_syn_var{v}-rep{rep}-conf{i}_log.csv'
#             aux.append(f)
#         aux1.append(aux)
#     filenames_eval.append(aux1)



# n_vars = len(var_)
# n_alphas = len(alphas)

# reward = np.zeros((n_vars, n_alphas, n_reps))
# constraint = np.zeros((n_vars, n_alphas, n_reps))

# reward_av = np.zeros((n_vars, n_alphas))
# constraint_av = np.zeros((n_vars, n_alphas))
# reward_ci = np.zeros((n_vars, n_alphas))
# constraint_ci = np.zeros((n_vars, n_alphas))

# for i, fn_list in enumerate(filenames_eval): #alpha
#     for j, fn_reps in enumerate(fn_list): # var
#         for k, fn in enumerate(fn_reps): # reps
#             path = os.path.join(os.getcwd(), '..', 'results',  timestamp)
#             df = pd.read_csv(os.path.join(path, fn))
            
#             reward[j, i, k] = -np.mean(df['reward'])
#             constr_violation_1 = np.maximum(0, df['constraint_1'] - constraint_val[0])
#             # constr_violation_2 = np.maximum(0, df['constraint_2'] - constraint_val[1])
#             # constraint[j, i, k] = np.mean(constr_violation_1 + constr_violation_2)
#             constraint[j, i, k] = np.mean(constr_violation_1)
   
# for i, fn_list in enumerate(filenames_eval): #alpha
#     for j, fn_reps in enumerate(fn_list): # var
#         reward_av[j,i], reward_ci[j,i] = compute_mean_ci(reward[j,i,:])
#         constraint_av[j,i], constraint_ci[j,i] = compute_mean_ci(constraint[j,i,:])
    

        
# plt.figure(30, figsize=figsize_vals)
# for i, v in enumerate(var_):
#     plt.errorbar(alphas, reward_av[i,:], yerr=reward_ci[i,:], label=r'$\sigma_{env}$ = '+str(v), **err_bar_kwargs)
# plt.ylabel('Average reward', fontsize=fontsize_label)
# plt.xlabel(r'$\alpha$', fontsize=fontsize_label)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)
# plt.grid(True)
# # plt.legend(loc='best', fontsize=fontsize_legend)
# plt.xscale('logit')
# # plt.title('Reward eval', fontsize=28)
# if save:
#     plt.savefig('./syn_eval_alpha_reward.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')

# plt.figure(31, figsize=figsize_vals)
# for i, v in enumerate(var_):
#     plt.errorbar(alphas, constraint_av[i,:], yerr=constraint_ci[i,:], label=r'$\sigma_{env}$ = '+str(v), **err_bar_kwargs)
# plt.ylabel('Average constraint violation', fontsize=fontsize_label)
# plt.xlabel(r'$\alpha$', fontsize=fontsize_label)
# plt.xticks(fontsize=fontsize_ticks)
# plt.yticks(fontsize=fontsize_ticks)
# plt.grid(True)
# plt.legend(loc='upper right', fontsize=fontsize_legend)
# plt.xscale('logit')
# # plt.title('Constraint eval', fontsize=28)
# if save:
#     plt.savefig('./syn_eval_alpha_constraint.{}'.format(file_format), format=file_format, dpi=300, bbox_inches='tight')


