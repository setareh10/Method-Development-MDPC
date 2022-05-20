#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 19:07:27 2021

@author: sr05
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler

colorUV = ['darkred', 'indianred', 'salmon']
colorMD = ['darkslategray', 'teal', 'lightseagreen']

##############################################################################
normalize = False


def my_scaler(x):

    trial, v_x = x.shape
    x_vector = x.reshape(trial*v_x, 1)
    scaler = StandardScaler()
    x_vec_scaled = scaler.fit_transform(x_vector)
    x_scaled = x_vec_scaled.reshape(trial, v_x)
    return x_scaled


def l_estimation(x, y, normalize, kf):

    regr_cv = RidgeCV(
        alphas=np.logspace(-3, 3, 5), scoring="explained_variance", normalize=normalize
    )
    scores = cross_validate(
        regr_cv,
        x,
        y,
        scoring="explained_variance",
        cv=kf,
        n_jobs=1,
    )

    gof_ev = np.mean(scores["test_score"])

    return gof_ev


#############################
def noise_connectivity(v_x, v_y, trial, normalize, repeat):

    gof_md = np.zeros([repeat])
    gof_uv = np.zeros([repeat])
    l_output = []

    for r in range(repeat):

        print(r)
        std = np.random.uniform(0.5, 1.5)
        x0 = np.random.normal(0, std, (trial, v_x))
        y0 = np.random.normal(0, std, (trial, v_y))

        x = my_scaler(x0)
        y = my_scaler(y0)
        x_uv = (x.copy().mean(1)).reshape(x.shape[0], 1)
        y_uv = (y.copy().mean(1)).reshape(x.shape[0], 1)

        n_splits = 5
        if (trial / 5) > 10:
            n_splits = 10
        kf = KFold(n_splits=n_splits)
        yx_md = max(l_estimation(x, y, normalize, kf), 0)
        xy_md = max(l_estimation(y, x, normalize, kf), 0)

        yx_uv = max(l_estimation(x_uv, y_uv, normalize, kf), 0)
        xy_uv = max(l_estimation(y_uv, x_uv, normalize, kf), 0)

        gof_md[r] = np.mean([yx_md, xy_md])
        gof_uv[r] = np.mean([yx_uv, xy_uv])
        # gof_md[r] = l_estimation(x, y, normalize, kf)
        # gof_uv[r] = l_estimation(x_uv, y_uv, normalize, kf)

    l_output.append([trial, v_x, v_y, gof_md.mean(),
                      gof_md.std(), gof_uv.mean(), gof_uv.std()])
    return l_output


trials = [30, 50, 100, 200, 300]
v_x = [5, 15]
v_y = [5, 15]
combinations = []
for x in range(len(v_x)):
    for y in range(x, len(v_y)):
        for trial in trials:
            print(v_x[x], v_y[y], trial)
            combinations.append([v_x[x], v_y[y], trial])
n_jobs = 20
repeat = 1000
start = time.time()
normalize = False

op = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(noise_connectivity)(v_x, v_y, trial, normalize, repeat)
    for v_x, v_y, trial in combinations)

end = time.time()
print(end-start)


gof_noise_md = np.zeros([len(trials), 3, 2])
gof_noise_uv = np.zeros([len(trials), 3, 2])
p = 0
for t, trial in enumerate(trials):
    for x in range(len(v_x)):
        for y in range(x, len(v_y)):
            # if (trial == op[p][0][2] and vx == op[p][0][0] and vy == op[p][0][1]):
            print(trial, x, y, p)
            gof_noise_md[t, x+y, 0] = op[p][0][3]
            gof_noise_md[t, x+y, 1] = op[p][0][4]

            gof_noise_uv[t, x+y, 0] = op[p][0][5]
            gof_noise_uv[t, x+y, 1] = op[p][0][6]

            p += 1


# colors = ['salmon', 'teal', 'mediumblue', 'goldenrod']
colors = ['royalblue', 'salmon', 'teal']
linestyles = ['dashed', 'solid', 'dotted']
fmts = [':', '-', ':']
linewidths = [2, 2, 4]
c = 0
fig = plt.figure(figsize=(12, 6))
for n_x in range(len(v_x)):
    for n_y in range(n_x, len(v_y)):
        plt.rcParams['font.size'] = '18'

        plt.errorbar(trials, np.round(
            gof_noise_uv[:, n_x+n_y, 0], 4),
            yerr=np.round(gof_noise_uv[:, n_x+n_y, 1], 4),
            label=f'({v_x[n_x]},{v_y[n_y]})', color=colors[c],
            linestyle=linestyles[c], linewidth=linewidths[c], fmt=fmts[c])

        plt.legend(fontsize=16, loc ='lower right')
        plt.xlabel('Trials', fontsize=18)
        plt.ylabel('Explained Variance', fontsize=18)
        plt.title('UV approach', fontsize=18)
        # plt.tick_params(axis='both', colors='black',width=2, length= 4,labelsize=18)

        c += 1
# plt.savefig(f'/home/sr05/Method_dev/method_fig/UV_noise_connectivity')


# c = 0
# fig = plt.figure(figsize=(12, 6))
# for n_x in range(len(v_x)):
#     for n_y in range(n_x, len(v_y)):
#         plt.rcParams['font.size'] = '18'

#         plt.errorbar(trials, np.round(
#             gof_noise_md[:, n_x+n_y, 0], 4),
#             yerr=np.round(gof_noise_md[:, n_x+n_y, 1], 4),
#             label=f'({v_x[n_x]},{v_y[n_y]})', color=colors[c],
#             linestyle=linestyles[c], linewidth=linewidths[c], fmt=fmts[c])

#         plt.legend(fontsize=16, loc ='lower right')
#         plt.xlabel('Trials', fontsize=18)
#         plt.ylabel('Explained Variance', fontsize=18)
#         plt.title('MD approach', fontsize=18)
#         # plt.tick_params(axis='both', colors='black',width=2, length= 4,labelsize=18)

#         c += 1


# plt.savefig(f'/home/sr05/Method_dev/method_fig/MD_noise_connectivity')
# # # plt.close('all')
############################################################################
# ## MD connectivity
# std_pow = [1.5, 1, .5, 0, -.5, -1, -1.5]
# trials = [30, 50, 100]
# v_x = [5, 15]
# v_y = [5, 15]
# combinations = []
# for trial in trials:
#     for std in std_pow:
#         for x in range(len(v_x)):
#             for y in range(x, len(v_y)):
#                 combinations.append([trial, std, v_x[x], v_y[y]])


# def l_connectivity(trial, std, v_x, v_y, repeat):

#     gof_md = np.zeros([repeat])
#     gof_uv = np.zeros([repeat])
#     snr = np.zeros([repeat])
#     l_output = []

#     for r in range(repeat):

#         print(r)

#         xx = np.random.normal(0, 1, (trial, v_x))
#         T = np.random.normal(0, 1, (v_x, v_y))

#         noise = np.random.normal(0, 10**std, (trial, v_y))
#         y0 = np.matmul(xx, T)
#         yy = y0 + noise

#         x = my_scaler(xx)
#         y = my_scaler(yy)

#         x_uv = (x.copy().mean(1)).reshape(x.shape[0], 1)
#         y_uv = (y.copy().mean(1)).reshape(x.shape[0], 1)

#         n_splits = 5
#         if (trial / 5) > 10:
#             n_splits = 10

#         kf = KFold(n_splits=n_splits)

#         yx_md = max(l_estimation(x, y, normalize, kf), 0)
#         xy_md = max(l_estimation(y, x, normalize, kf), 0)

#         yx_uv = max(l_estimation(x_uv, y_uv, normalize, kf), 0)
#         xy_uv = max(l_estimation(y_uv, x_uv, normalize, kf), 0)

#         gof_md[r] = np.mean([yx_md, xy_md])
#         gof_uv[r] = np.mean([yx_uv, xy_uv])
#         snr[r] = np.var(y0)/np.var(noise)
#     l_output.append([trial, std, v_x, v_y, gof_md.mean(),
#                      gof_md.std(), gof_uv.mean(), gof_uv.std(), snr.mean()])
#     return l_output


# n_jobs = 20
# repeat = 1000
# start = time.time()

# op = Parallel(n_jobs=n_jobs, backend="loky")(
#     delayed(l_connectivity)(trial, std, v_x, v_y, repeat)
#     for trial, std, v_x, v_y in combinations)

# end = time.time()
# print(end-start)

# gof_md = np.zeros([len(std_pow), len(trials), 3, 2])
# gof_uv = np.zeros([len(std_pow), len(trials), 3, 2])
# snr = np.zeros([len(std_pow), len(trials), 3])
# p = 0
# for t, trial in enumerate(trials):
#     for s, std in enumerate(std_pow):
#         for x in range(len(v_x)):
#             for y in range(x, len(v_y)):
#                 # if (trial == op[p][0][2] and vx == op[p][0][0] and vy == op[p][0][1]):
#                 print(trial, s, x, y, p)
#                 gof_md[s, t, x+y, 0] = op[p][0][4]
#                 gof_md[s, t, x+y, 1] = op[p][0][5]

#                 gof_uv[s, t, x+y, 0] = op[p][0][6]
#                 gof_uv[s, t, x+y, 1] = op[p][0][7]
#                 snr[s, t, x+y] = op[p][0][8]
#                 p += 1


# # colors = ['salmon', 'teal', 'mediumblue', 'goldenrod']
# colors = ['royalblue', 'salmon', 'teal']
# linestyles = ['dashed', 'solid', 'dotted']
# fmts = [':', '-', ':']
# linewidths = [2, 2, 4]

# for n_x in range(len(v_x)):
#     for n_y in range(n_x, len(v_y)):
#         fig = plt.figure(figsize=(11, 6))
#         plt.title(f'Vertices in (x,y):({v_x[n_x]},{v_y[n_y]})')
#         for n_t in np.arange(len(trials)):
#             snr_db = 10 * np.log10(snr[:, n_t, n_x+n_y])

#             plt.errorbar(snr_db, np.round(
#                 gof_uv[:, n_t, n_x+n_y, 0], 4),
#                 yerr=np.round(gof_uv[:, n_t, n_x+n_y, 1], 4),
#                 label=f'UV:{trials[n_t]}', color=colorUV[n_t],
#                 linestyle=linestyles[n_t], linewidth=linewidths[n_t], fmt=fmts[n_t])
#             plt.xlabel('SNR(db)', fontsize=16)
#             plt.ylabel('Explained Variance', fontsize=16)
            
            
#             plt.errorbar(snr_db, np.round(
#                 gof_md[:, n_t, n_x+n_y, 0], 4),
#                 yerr=np.round(gof_md[:, n_t, n_x+n_y, 1], 4),
#                 label=f'MD:{trials[n_t]}', color=colorMD[n_t],
#                 linestyle=linestyles[n_t], linewidth=linewidths[n_t], fmt=fmts[n_t])
#             plt.xlabel('SNR(db)', fontsize=16)
#             plt.ylabel('Explained Variance', fontsize=16)
            
#             plt.legend(loc='upper left')
        # plt.savefig(f'/home/sr05/Method_dev/method_fig/MD_effect_x{v_x[n_x]}_y{v_y[n_y]}')


# # plt.close('all')
############################################################################
# ## UV connectivity
std_pow = [1.5, 1, .5, 0, -.5, -1, -1.5]
trials = [30, 50, 100]
v_x = [5, 15]
v_y = [5, 15]
combinations = []
for trial in trials:
    for std in std_pow:
        for x in range(len(v_x)):
            for y in range(x, len(v_y)):
                combinations.append([trial, std, v_x[x], v_y[y]])


def l_connectivity(trial, std, v_x, v_y, repeat):

    gof_md = np.zeros([repeat])
    gof_uv = np.zeros([repeat])
    snr = np.zeros([repeat])
    l_output = []
    alpha = 2
    for r in range(repeat):

        print(r)

        xx = np.random.normal(0, 1, (trial, 1))
        noise_y = alpha*np.random.normal(0, 10**std, (trial, v_y))
        x_md = np.repeat(xx, v_x, axis=1) 
        y0 = alpha*np.repeat(xx, v_y, axis=1)
        y_md = y0 + noise_y


        x = my_scaler(x_md)
        y = my_scaler(y_md)

        x_uv = (x.copy().mean(1)).reshape(x.shape[0], 1)
        y_uv = (y.copy().mean(1)).reshape(x.shape[0], 1)

        n_splits = 5
        if (trial / 5) > 10:
            n_splits = 10

        kf = KFold(n_splits=n_splits)

        yx_md = max(l_estimation(x, y, normalize, kf), 0)
        xy_md = max(l_estimation(y, x, normalize, kf), 0)

        yx_uv = max(l_estimation(x_uv, y_uv, normalize, kf), 0)
        xy_uv = max(l_estimation(y_uv, x_uv, normalize, kf), 0)

        gof_md[r] = np.mean([yx_md, xy_md])
        gof_uv[r] = np.mean([yx_uv, xy_uv])
        snr[r] = np.var(y0)/np.var(noise_y)
    l_output.append([trial, std, v_x, v_y, gof_md.mean(),
                     gof_md.std(), gof_uv.mean(), gof_uv.std(), snr.mean()])
    return l_output


n_jobs = 20
repeat = 100
start = time.time()

op = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(l_connectivity)(trial, std, v_x, v_y, repeat)
    for trial, std, v_x, v_y in combinations)

end = time.time()
print(end-start)

gof_md = np.zeros([len(std_pow), len(trials), 3, 2])
gof_uv = np.zeros([len(std_pow), len(trials), 3, 2])
snr = np.zeros([len(std_pow), len(trials), 3])
p = 0
for t, trial in enumerate(trials):
    for s, std in enumerate(std_pow):
        for x in range(len(v_x)):
            for y in range(x, len(v_y)):
                # if (trial == op[p][0][2] and vx == op[p][0][0] and vy == op[p][0][1]):
                print(trial, s, x, y, p)
                gof_md[s, t, x+y, 0] = op[p][0][4]
                gof_md[s, t, x+y, 1] = op[p][0][5]

                gof_uv[s, t, x+y, 0] = op[p][0][6]
                gof_uv[s, t, x+y, 1] = op[p][0][7]
                snr[s, t, x+y] = op[p][0][8]
                p += 1


# colors = ['salmon', 'teal', 'mediumblue', 'goldenrod']
colors = ['royalblue', 'salmon', 'teal']
linestyles = ['dashed', 'solid', 'dotted']
fmts = [':', '-', ':']
linewidths = [2, 2, 4]

for n_x in range(len(v_x)):
    for n_y in range(n_x, len(v_y)):
        fig = plt.figure(figsize=(11, 6))
        plt.title(f'Vertices in (x,y):({v_x[n_x]},{v_y[n_y]})')
        for n_t in np.arange(len(trials)):
            snr_db = 10 * np.log10(snr[:, n_t, n_x+n_y])

            plt.errorbar(snr_db, np.round(
                gof_uv[:, n_t, n_x+n_y, 0], 4),
                yerr=np.round(gof_uv[:, n_t, n_x+n_y, 1], 4),
                label=f'UV:{trials[n_t]}', color=colorUV[n_t],
                linestyle=linestyles[n_t], linewidth=linewidths[n_t], fmt=fmts[n_t])
            plt.xlabel('SNR(db)', fontsize=16)
            plt.ylabel('Explained Variance', fontsize=16)
            
            
            plt.errorbar(snr_db, np.round(
                gof_md[:, n_t, n_x+n_y, 0], 4),
                yerr=np.round(gof_md[:, n_t, n_x+n_y, 1], 4),
                label=f'MD:{trials[n_t]}', color=colorMD[n_t],
                linestyle=linestyles[n_t], linewidth=linewidths[n_t], fmt=fmts[n_t])
            plt.xlabel('SNR(db)', fontsize=16)
            plt.ylabel('Explained Variance', fontsize=16)
            
            plt.legend(loc='upper left')
        plt.savefig(f'/home/sr05/Method_dev/method_fig/UV_effect_x{v_x[n_x]}_y{v_y[n_y]}')


# # plt.close('all')

# gof_noise_l=np.zeros([len(std_pow), len(trials)])
# gof_noise_n=np.zeros([len(std_pow), len(trials)])
# snr=np.zeros([len(std_pow), len(trials)])
# p = 0
# for t, trial  in enumerate(trials):
#     for s, std in enumerate(std_pow):
#         if (trial==op[p][0][0] and std==op[p][0][1]):
#             gof_noise_l[s, t]= op[p][0][2]
#             gof_noise_n[s, t]= op[p][0][3]
#             snr[s, t]= op[p][0][4]
#             p +=1


# fig=plt.figure(figsize=(11, 6))
# plt.title(f'L - v[x,y]: [{v_x},{v_y}]')
# for n_t in np.arange(len(trials)):
#     snr_db=10 * np.log10(snr[:, n_t])

#     plt.plot(snr_db, gof_noise_l[:, n_t], label=f'L:{trials[n_t]}', color=colorUV[n_t])
#     plt.legend(fontsize=16)
#     plt.plot(snr_db, gof_noise_n[:, n_t], label=f'NL:{trials[n_t]}', color=colorMD[n_t])
#     plt.legend()
#     plt.xlabel('SNR(db)', fontsize=16)
#     plt.ylabel('Explained Variance', fontsize=16)
#     plt.tick_params(axis='both', colors='black',
#                     width=2, length=4, labelsize=18)
#     plt.savefig(f'/home/sr05/Method_dev/method_fig/L_x{v_x}_y{v_y}')


# # # UV effect
# alpha = 2
# std_pow = [2, 1.5, 1, .5, 0, -.5, -1, -1.5]
# vertices_x = [5, 15]
# vertices_y = [5, 15]
# trials = [30, 50, 100]
# repeat = 100

# gof_noise_uv = np.zeros(
#     [len(std_pow), len(trials), len(vertices_x), len(vertices_x), repeat])
# gof_noise_md = np.zeros(
#     [len(std_pow), len(trials), len(vertices_x), len(vertices_x), repeat])
# snr_x = np.zeros([len(std_pow), len(trials), len(
#     vertices_x), len(vertices_x), repeat])
# snr_y = np.zeros([len(std_pow), len(trials), len(
#     vertices_x), len(vertices_x), repeat])

# for n_s, std in enumerate(std_pow):
#     for n_t, trial in enumerate(trials):
#         for n_x, v_x in enumerate(vertices_x):
#             for n_y, v_y in enumerate(vertices_y):
#                 # noise_x = np.random.normal(0, 10**std, (trial, v_x))
#                 # noise_y = alpha*np.random.normal(0, 10**std, (trial, v_y))
#                 for r in np.arange(repeat):
#                     print(std, trial, v_x, v_y, r)
#                     x = np.random.normal(0, 1, (trial, 1))
#                     noise_x = np.random.normal(0, 10**std, (trial, v_x))
#                     noise_y = alpha*np.random.normal(0, 10**std, (trial, v_y))
#                     x_md = np.repeat(x, v_x, axis=1) #+ noise_x
#                     y_md = alpha*np.repeat(x, v_y, axis=1) + noise_y

#                     x_uv = x_md.copy().mean(1).reshape(trial, 1)
#                     y_uv = y_md.copy().mean(1).reshape(trial, 1)

#                     n_splits = 5
#                     if (trial / 5) > 10:
#                         n_splits = 10
#                     kf = KFold(n_splits=n_splits)
#                     regr_md = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                       normalize=True)
#                     scores_md = cross_validate(regr_md, x_md, y_md,
#                                                 scoring='explained_variance',
#                                                 cv=kf, n_jobs=-1)

#                     regr_uv = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                       normalize=True)
#                     scores_uv = cross_validate(regr_uv, x_uv, y_uv,
#                                                 scoring='explained_variance',
#                                                 cv=kf, n_jobs=-1)

#                     gof_noise_uv[n_s, n_t, n_x, n_y, r] = np.mean(
#                         scores_uv['test_score'])
#                     gof_noise_md[n_s, n_t, n_x, n_y, r] = np.mean(
#                         scores_md['test_score'])
#                     # snr_x[n_s, n_t, n_x, n_y, r] = np.var(
#                     #     np.repeat(x, v_x, axis=1))/np.var(noise_x)
#                     snr_y[n_s, n_t, n_x, n_y, r] = np.var(
#                         np.repeat(x, v_y, axis=1))/np.var(noise_y)

# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         # plt.rcParams['font.size'] = '14'
#         # plt.figure(figsize=(11, 5))
#         fig = plt.figure(figsize=(11, 6))
#         plt.title(f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})')
#         for n_t in np.arange(len(trials)):
#             snr_db = 10 * np.log10(snr_y[:, n_t, n_x, n_y, :].copy().mean(-1))

#             plt.plot(snr_db, gof_noise_uv[:, n_t, n_x, n_y, :].copy(
#             ).mean(-1), label=f'UV:{trials[n_t]}', color=colorUV[n_t],
#                 linestyle='solid')
#             plt.legend()
#             plt.plot(snr_db, gof_noise_md[:, n_t, n_x, n_y, :].copy(
#             ).mean(-1), label=f'MD:{trials[n_t]}', color=colorMD[n_t])
#             plt.legend(fontsize=16)
#             plt.xlabel('SNR(db)', fontsize=16)
#             plt.ylabel('Explained Variance', fontsize=16)
#             plt.tick_params(axis='both', colors='black',width=2, length= 4,labelsize=18)

#             # plt.savefig(f'/home/sr05/Method_dev/method_fig'
#             # f'/UV_effect_connectivity_x{vertices_x[n_x]}_y{vertices_x[n_y]}')
# # # plt.close('all')
# ##########################################################################
# # MD effect
# std_pow = [2, 1.5, 1, .5, 0, -.5, -1, -1.5]
# vertices_x = [5, 15]
# vertices_y = [5, 15]
# trials = [30, 50, 100]
# repeat = 1
# std_pow = [-1.5]
# vertices_x = [100]
# vertices_y = [100]
# trials = [30]
# repeat = 1000
# gof_noise_uv = np.zeros(
#     [len(std_pow), len(trials), len(vertices_x), len(vertices_x), repeat])
# gof_noise_md = np.zeros(
#     [len(std_pow), len(trials), len(vertices_x), len(vertices_x), repeat])
# snr = np.zeros([len(std_pow), len(trials), len(
#     vertices_x), len(vertices_x), repeat])
# snr_u = np.zeros([len(std_pow), len(trials), len(
#     vertices_x), len(vertices_x), repeat])

# for n_s, std in enumerate(std_pow):
#     for n_t, trial in enumerate(trials):
#         for n_x, v_x in enumerate(vertices_x):
#             for n_y, v_y in enumerate(vertices_y):
#                 # T = np.random.normal(0, 1, (v_x, v_y))
#                 # noise = np.random.normal(0, 10**std, (trial, v_y))
#                 for r in np.arange(repeat):
#                     print(std, trial, v_x, v_y, r)
#                     x_md = np.zeros([trial, v_x])
#                     # for v in range(v_x):
#                     #     x_md[:,v] = np.random.normal(0, 1, (trial, 1)).reshape(trial)
#                     xx = np.random.normal(0, 1, (trial, v_x))
#                     T = np.random.normal(0, 1, (v_x, v_y))
#                     noise = np.random.normal(0, 10**std, (trial, v_y))
#                     yy = np.matmul(xx, T) + noise
#                     scaler = Normalizer()
#                     x_md = (scaler.fit(xx.transpose()).transform(xx.transpose())).transpose()

#                     scaler = Normalizer()
#                     y_md = (scaler.fit(yy.transpose()).transform(yy.transpose())).transpose()

#                     x_uv = x_md.copy().mean(1).reshape(trial, 1)
#                     y_uv = y_md.copy().mean(1).reshape(trial, 1)

#                     y_u = np.matmul(x_md, T).copy().mean(1).reshape(trial, 1)
#                     n_u = noise.copy().mean(1).reshape(trial, 1)

#                     n_splits = 5
#                     if (trial / 5) > 10:
#                         n_splits = 10

#                     kf = KFold(n_splits=n_splits)

#                     regr_md = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                       normalize=True)
#                     scores_md = cross_validate(regr_md, x_md, y_md,
#                                                 scoring='explained_variance',
#                                                 cv=kf, n_jobs=-1)

#                     regr_uv = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                       normalize=True)
#                     scores_uv = cross_validate(regr_uv, x_uv, y_uv,
#                                                 scoring='explained_variance',
#                                                 cv=kf, n_jobs=-1)

#                     gof_noise_uv[n_s, n_t, n_x, n_y, r] = np.mean(
#                         scores_uv['test_score'])
#                     gof_noise_md[n_s, n_t, n_x, n_y, r] = np.mean(
#                         scores_md['test_score'])
#                     snr[n_s, n_t, n_x, n_y, r] = np.var(
#                         np.matmul(x_md, T))/np.var(noise)

#                     snr_u[n_s, n_t, n_x, n_y, r] = np.var(
#                         y_u)/np.var(n_u)

# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         fig = plt.figure(figsize=(11, 6))
#         plt.title(f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})')
#         for n_t in np.arange(len(trials)):
#             snr_db = 10 * np.log10(snr[:, n_t, n_x, n_y, :].copy().mean(-1))
#             snr_db_u = 10 * \
#                 np.log10(snr_u[:, n_t, n_x, n_y, :].copy().mean(-1))

#             plt.plot(snr_db, gof_noise_uv[:, n_t, n_x, n_y, :].copy(
#             ).mean(-1), label=f'UV:{trials[n_t]}', color=colorUV[n_t])
#             plt.legend(fontsize=16)
#             plt.plot(snr_db, gof_noise_md[:, n_t, n_x, n_y, :].copy(
#             ).mean(-1), label=f'MD:{trials[n_t]}', color=colorMD[n_t])
#             plt.legend()
#             plt.xlabel('SNR(db)', fontsize=16)
#             plt.ylabel('Explained Variance', fontsize=16)
#             plt.tick_params(axis='both', colors='black',
#                             width=2, length=4, labelsize=18)


# # plt.close('all')


# ##########################################################################
# ## noise connectivity: MD patterns
# vertices_x = [5, 15]
# vertices_y = [5, 15]
# trials = [30, 50, 100, 150, 300]
# repeat = 1000
# colors = ['salmon', 'teal', 'mediumblue', 'goldenrod']
# gof_noise_uv = np.zeros(
#     [len(trials), len(vertices_x), len(vertices_x), repeat])
# gof_noise_md = np.zeros(
#     [len(trials), len(vertices_x), len(vertices_x), repeat])

# for n_t, trial in enumerate(trials):
#     for n_x, v_x in enumerate(vertices_x):
#         for n_y, v_y in enumerate(vertices_y):
#             for r in np.arange(repeat):
#                 print(trial, v_x, v_y, r)

#                 x_md = np.random.normal(0, 1, (trial, v_x))
#                 y_md = np.random.normal(0, 1, (trial, v_y))

#                 x_uv = x_md.copy().mean(1).reshape(trial, 1)
#                 y_uv = y_md.copy().mean(1).reshape(trial, 1)

#                 n_splits = 5
#                 if (trial / 5) > 10:
#                     n_splits = 10
#                 kf = KFold(n_splits=n_splits)
#                 regr_md = RidgeCV(alphas=np.logspace(-5, 5, 100), normalize=True)
#                 scores_md = cross_validate(regr_md, x_md, y_md,
#                                             scoring='explained_variance',
#                                             cv=kf, n_jobs=-1)

#                 regr_uv = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                   normalize=True)
#                 scores_uv = cross_validate(regr_uv, x_uv, y_uv,
#                                             scoring='explained_variance',
#                                             cv=kf, n_jobs=-1)

#                 gof_noise_uv[n_t, n_x, n_y, r] = np.mean(
#                     scores_uv['test_score'])
#                 gof_noise_md[n_t, n_x, n_y, r] = np.mean(
#                     scores_md['test_score'])
# c = 0
# fig = plt.figure(figsize=(12, 6))
# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         plt.rcParams['font.size'] = '18'
#         plt.plot(trials, np.round(gof_noise_uv[:, n_x, n_y, :].copy(
#         ).mean(-1),4), label=f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})', color=colors[c])
#         plt.legend( fontsize=16)
#         plt.xlabel('Trials', fontsize=18)
#         plt.ylabel('Explained Variance', fontsize=18)
#         plt.title('UV approach', fontsize=18)
#         # plt.tick_params(axis='both', colors='black',width=2, length= 4,labelsize=18)

#         c += 1
# plt.savefig(f'/home/sr05/Method_dev/method_fig/UV_noise_connectivity')
# c = 0
# fig = plt.figure(figsize=(12, 6))
# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         plt.rcParams['font.size'] = '18'
#         plt.plot(trials, np.round(gof_noise_md[:, n_x, n_y, :].copy(
#         ).mean(-1),4), label=f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})', color=colors[c])
#         plt.legend(fontsize=16)
#         plt.xlabel('Trials', fontsize=18)
#         plt.ylabel('Explained Variance', fontsize=18)
#         plt.title('MD approach')
#         plt.tick_params(axis='both', colors='black',width=2, length= 4,labelsize=18)
#         c += 1
# plt.savefig(f'/home/sr05/Method_dev/method_fig/MD_noise_connectivity')
# # # # plt.close('all')
##########################################################################
# ## noise connectivity: UV patterns
# vertices_x = [5, 15]
# vertices_y = [5, 15]
# trials = [30, 50, 100, 150, 300]
# repeat = 100

# gof_noise_uv = np.zeros(
#     [len(trials), len(vertices_x), len(vertices_x), repeat])
# gof_noise_md = np.zeros(
#     [len(trials), len(vertices_x), len(vertices_x), repeat])

# for n_t, trial in enumerate(trials):
#     for n_x, v_x in enumerate(vertices_x):
#         for n_y, v_y in enumerate(vertices_y):
#             for r in np.arange(repeat):
#                 print(trial, v_x, v_y, r)

#                 x_md = np.repeat(np.random.normal(0, 1, (trial, 1)),v_x,1)
#                 y_md = np.repeat(np.random.normal(0, 1, (trial, 1)),v_y,1)

#                 x_uv = x_md.copy().mean(1).reshape(trial, 1)
#                 y_uv = y_md.copy().mean(1).reshape(trial, 1)

# n_splits = 5
# if (trial / 5) > 10:
#     n_splits = 10
# kf = KFold(n_splits=n_splits)
#                 regr_md = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                   normalize=True)
#                 scores_md = cross_validate(regr_md, x_md, y_md,
#                                            scoring='explained_variance',
#                                            cv=kf, n_jobs=-1)

#                 regr_uv = RidgeCV(alphas=np.logspace(-5, 5, 100),
#                                   normalize=True)
#                 scores_uv = cross_validate(regr_uv, x_uv, y_uv,
#                                            scoring='explained_variance',
#                                            cv=kf, n_jobs=-1)

#                 gof_noise_uv[n_t, n_x, n_y, r] = np.mean(
#                     scores_uv['test_score'])
#                 gof_noise_md[n_t, n_x, n_y, r] = np.mean(
#                     scores_md['test_score'])

# fig = plt.figure()
# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         plt.plot(trials, gof_noise_uv[:, n_x, n_y, :].copy(
#         ).mean(-1), label=f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})')
#         plt.legend()

# fig = plt.figure()
# for n_x in np.arange(len(vertices_x)):
#     for n_y in np.arange(len(vertices_y)):
#         plt.plot(trials, gof_noise_md[:, n_x, n_y, :].copy(
#         ).mean(-1), label=f'v_x,v_y: ({vertices_x[n_x]},{vertices_y[n_y]})')
#         plt.legend()
##############################################################################
# ## UV effect imshow
# trial = 10
# v_x = 6
# v_y = 4
# fontsize = 22
# noise_x = np.random.normal(0, .1, (trial, v_x))
# noise_y = np.random.normal(0, .1, (trial, v_y))
# x = np.random.normal(0, 1, (trial, 1))
# x_md = np.repeat(x, v_x, axis=1) + noise_x
# y_md = 2*np.repeat(x, v_y, axis=1) + noise_y

# fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# # # plotting the t-values
# vmax = max(np.max(x_md),np.max(y_md))
# vmin = min(np.min(x_md),np.min(y_md))

# # plt.subplot(311)
# im = ax[0].imshow(x_md, aspect='equal', cmap='Spectral',extent=[0,v_x, 0, trial],
#                   vmin=np.min(x_md), vmax=np.max(x_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[0])
# cbar.ax.tick_params(labelsize=15)
# ax[0].set_title('ROI X', fontsize=fontsize )
# ax[0].set_xlabel('Vertex', fontsize=fontsize )
# ax[0].set_ylabel('Trial', fontsize=fontsize )
# # fig.suptitle('UV effect', fontsize=fontsize )

# im = ax[1].imshow(y_md, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y_md), vmax=np.max(y_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[1])
# cbar.ax.tick_params(labelsize=15)

# ax[1].set_title('ROI Y', fontsize=fontsize )

# ax[1].set_xlabel('Vertex', fontsize=fontsize )
# ax[1].set_ylabel('Trial', fontsize=fontsize )
# plt.savefig('/home/sr05/Method_dev/method_fig/UV_effect')

# # # plt.close('all')

# ##############################################################################
# # # ## MD effect imshow
# trial = 10
# v_x = 6
# v_y = 4
# fontsize = 22

# T = np.random.normal(0, 1, (v_x, v_y))
# noise = np.random.normal(0, .1, (trial, v_y))

# x_md = np.random.normal(0, 1, (trial, v_x))
# y_md = np.matmul(x_md, T) + noise


# fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# # # plotting the t-values
# vmax = max(np.max(x_md),np.max(y_md))
# vmin = min(np.min(x_md),np.min(y_md))

# # plt.subplot(311)
# im = ax[0].imshow(x_md, aspect='equal', cmap='Spectral',extent=[0,v_x, 0, trial],
#                   vmin=np.min(x_md), vmax=np.max(x_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[0])
# cbar.ax.tick_params(labelsize=15)

# ax[0].set_title('ROI X', fontsize=fontsize)
# ax[0].set_xlabel('Vertex', fontsize=fontsize)
# ax[0].set_ylabel('Trial', fontsize=fontsize)
# # fig.suptitle('MD effect', fontsize=fontsize)

# # im = ax[1].imshow(T, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, v_x],
# #                   vmin=np.min(T), vmax=np.max(T), origin='lower')
# # cbar = fig.colorbar(im, ax=ax[1])
# # cbar.ax.tick_params(labelsize=15)

# # ax[1].set_title('T', fontsize=fontsize)

# # ax[1].set_xlabel('Vertex', fontsize=fontsize)
# # ax[1].set_ylabel('Vertex', fontsize=fontsize)

# im = ax[1].imshow(y_md, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y_md), vmax=np.max(y_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[1])
# cbar.ax.tick_params(labelsize=15)

# ax[1].set_title('ROI Y', fontsize=fontsize)

# ax[1].set_xlabel('Vertex', fontsize=fontsize)
# ax[1].set_ylabel('Trial', fontsize=fontsize)
# plt.savefig('/home/sr05/Method_dev/method_fig/MD_effect')
# # # # # plt.close('all')

# ##############################################################################
# # # ## NN effect imshow
# trial = 10
# v_x = 4
# v_y = 4
# fontsize = 22

# x0 = np.random.normal(0, 1, (trial, v_x))
# T0 = np.random.normal(0, 1, (v_x, v_y))
# y0 = np.matmul(x0, T0)

# y1 = np.tanh(y0)

# T1 = np.random.normal(0, 1, (v_y, v_y))
# y2 = np.matmul(y1, T1)


# fig, ax = plt.subplots(1, 4, figsize=(18, 5))

# # # plotting the t-values
# vmax = max(np.max(x0),np.max(y2))
# vmin = min(np.min(x0),np.min(y2))

# # plt.subplot(311)
# im = ax[0].imshow(x0, aspect='equal', cmap='Spectral',extent=[0,v_x, 0, trial],
#                   vmin=np.min(x0), vmax=np.max(x0), origin='lower')
# # cbar = fig.colorbar(im, ax=ax[0])
# # cbar.ax.tick_params(labelsize=15)

# ax[0].set_title('ROI X', fontsize=fontsize)
# ax[0].set_xlabel('Vertex', fontsize=fontsize)
# ax[0].set_ylabel('Trial', fontsize=fontsize)


# im = ax[1].imshow(y0, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y0), vmax=np.max(y0), origin='lower')
# # cbar = fig.colorbar(im, ax=ax[1])
# # cbar.ax.tick_params(labelsize=15)
# # ax[1].set_title('ROI Y', fontsize=fontsize)

# # ax[1].set_xlabel('Vertex', fontsize=fontsize)
# # ax[1].set_ylabel('Trial', fontsize=fontsize)


# im = ax[2].imshow(y1, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y1), vmax=np.max(y1), origin='lower')
# # cbar = fig.colorbar(im, ax=ax[2])
# # cbar.ax.tick_params(labelsize=15)

# # ax[2].set_title('ROI Y', fontsize=fontsize)

# # ax[2].set_xlabel('Vertex', fontsize=fontsize)
# # ax[2].set_ylabel('Trial', fontsize=fontsize)


# im = ax[3].imshow(y2, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y2), vmax=np.max(y2), origin='lower')
# # cbar = fig.colorbar(im, ax=ax[3])
# # cbar.ax.tick_params(labelsize=15)

# ax[3].set_title('ROI Y', fontsize=fontsize)

# ax[3].set_xlabel('Vertex', fontsize=fontsize)
# ax[3].set_ylabel('Trial', fontsize=fontsize)
# plt.savefig('/home/sr05/Method_dev/method_fig/NN_NL_effect')
# # # # # plt.close('all')

# ##############################################################################
# # ## Noise effect imshow
# trial = 10
# v_x = 6
# v_y = 4
# fontsize = 22

# x_md = np.random.normal(0, 1, (trial, v_x))
# y_md = np.random.normal(0, 1, (trial, v_y))


# fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# # # plotting the t-values
# vmax = max(np.max(x_md),np.max(y_md))
# vmin = min(np.min(x_md),np.min(y_md))

# # plt.subplot(311)
# im = ax[0].imshow(x_md, aspect='equal', cmap='Spectral',extent=[0,v_x, 0, trial],
#                   vmin=np.min(x_md), vmax=np.max(x_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[0])
# cbar.ax.tick_params(labelsize=15)

# ax[0].set_title('ROI X', fontsize=fontsize)
# ax[0].set_xlabel('Vertex', fontsize=fontsize)
# ax[0].set_ylabel('Trial', fontsize=fontsize)
# # fig.suptitle('Noise', fontsize=fontsize)

# im = ax[1].imshow(y_md, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y_md), vmax=np.max(y_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[1])
# cbar.ax.tick_params(labelsize=15)

# ax[1].set_title('ROI Y', fontsize=fontsize)

# ax[1].set_xlabel('Vertex', fontsize=fontsize)
# ax[1].set_ylabel('Trial', fontsize=fontsize)
# plt.savefig('/home/sr05/Method_dev/method_fig/noise_effect')

# # # # plt.close('all')


# # ##########################################################################
# # ## X Y patterns example
# trial = 10
# v_x = 5
# v_y = 5
# fontsize = 22

# x_md = np.random.normal(0, 1, (trial, v_x))
# y_md = np.random.normal(0, 1, (trial, v_y))


# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# # # plotting the t-values
# vmax = max(np.max(x_md),np.max(y_md))
# vmin = min(np.min(x_md),np.min(y_md))

# # plt.subplot(311)
# im = ax[0].imshow(x_md, aspect='equal', cmap='Spectral',extent=[0,v_x, 0, trial],
#                   vmin=np.min(x_md), vmax=np.max(x_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[0])
# cbar.ax.tick_params(labelsize=15)

# ax[0].set_title('ROI X', fontsize=fontsize)
# ax[0].set_xlabel('Vertex (X)', fontsize=fontsize)
# ax[0].set_ylabel('Trials', fontsize=fontsize)
# # fig.suptitle('Noise', fontsize=fontsize)

# im = ax[1].imshow(y_md, aspect='equal', cmap='Spectral',extent=[0,v_y, 0, trial],
#                   vmin=np.min(y_md), vmax=np.max(y_md), origin='lower')
# cbar = fig.colorbar(im, ax=ax[1])
# cbar.ax.tick_params(labelsize=15)

# ax[1].set_title('ROI Y', fontsize=fontsize)

# ax[1].set_xlabel('Vertex (Y)', fontsize=fontsize)
# ax[1].set_ylabel('Trials', fontsize=fontsize)
# plt.savefig('/home/sr05/Method_dev/method_fig/X-Y_patterns_53')
