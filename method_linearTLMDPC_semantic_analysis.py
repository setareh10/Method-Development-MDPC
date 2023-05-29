#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 18:12:03 2023

@author: sr05
"""


import os
import time
import pickle
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.stats import permutation_cluster_1samp_test


t_down_sampling = 25.0


def mask_function(X, cut_off=None):
    if cut_off is not None:
        r, c = X.shape
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                if X[i, j] < cut_off:
                    X[i, j] = cut_off
    return X


data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
snr = C.snr
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()


s = time.time()

labels = ["lATL", "rATL", "PTC", "IFG", "AG", "PVA"]


###########################################################

d = 24
x_sd_md = np.zeros([len(C.subjects), 36, d, d])
x_sd_ud = np.zeros([len(C.subjects), 36, d, d])

x_ld_md = np.zeros([len(C.subjects), 36, d, d])
x_ld_ud = np.zeros([len(C.subjects), 36, d, d])

cut_off = 0.0
p = 0

for roi_y in np.arange(0, 6):
    for roi_x in np.arange(0, 6):
        if roi_y == roi_x:
            print(p)
            p += 1
        else:

            print(p)
            for i in np.arange(0, len(C.subjects)):
                gof_sd_md = np.zeros([d, d])
                gof_sd_ud = np.zeros([d, d])

                for cond in ["fruit", "odour", "milk"]:
                    file_name = (
                        os.path.expanduser("~")
                        + "/semnet-project/json_files/test/l/trans_"
                        + cond
                        + "_x"
                        + str(roi_x)
                        + "-y"
                        + str(roi_y)
                        + "_sub_"
                        + str(i)
                        + "_"
                        + str(int(t_down_sampling))
                        + "_clusters.json"
                        # + "_pca.json"
                    )

                    with open(file_name, "rb") as fp:  # Unpickling
                        a = pickle.load(fp)

                    gof_sd_md = gof_sd_md + \
                        mask_function(a["ev_md"], cut_off=cut_off)
                    gof_sd_ud = gof_sd_ud + \
                        mask_function(a["ev_uv"], cut_off=cut_off)

                sd_md = gof_sd_md / 3
                sd_ud = gof_sd_ud / 3

                x_sd_md[i, p, :, :] = sd_md
                x_sd_ud[i, p, :, :] = sd_ud

                for cond in ["LD"]:
                    file_name = (
                        os.path.expanduser("~")
                        + "/semnet-project/json_files/test/l/trans_"
                        + cond
                        + "_x"
                        + str(roi_x)
                        + "-y"
                        + str(roi_y)
                        + "_sub_"
                        + str(i)
                        + "_"
                        + str(int(t_down_sampling))
                        + "_clusters.json"
                        # + "_pca.json"                   
                        )

                    with open(file_name, "rb") as fp:  # Unpickling
                        a = pickle.load(fp)
                        
                    x_ld_md[i, p, :, :] = mask_function(
                        a["ev_md"], cut_off=cut_off)
                    
                    x_ld_ud[i, p, :, :] = mask_function(
                        a["ev_uv"], cut_off=cut_off)

            p += 1
#############################################################################

d = 24
step = 1
x_sd_md_avg = np.zeros([18, 15, d, d])
x_sd_ud_avg = np.zeros([18, 15, d, d])

x_ld_md_avg = np.zeros([18, 15, d, d])
x_ld_ud_avg = np.zeros([18, 15, d, d])

p = 0

## Run this part for BV
# for roi_y in np.arange(0, 6):
#     for roi_x in np.arange(roi_y + 1, 6):

#         k1 = roi_y * 6 + roi_x
#         k2 = roi_x * 6 + roi_y
#         print(k1, k2)
#         for i in np.arange(len(subjects)):
#             x_sd_md_avg[i, p, :, :] = (
#                 x_sd_md[i, k1, :, :] + x_sd_md[i, k2, :, :].transpose()
#             ) / 2
#             x_sd_ud_avg[i, p, :, :] = (
#                 x_sd_ud[i, k1, :, :] + x_sd_ud[i, k2, :, :].transpose()
#             ) / 2

#             x_ld_md_avg[i, p, :, :] = (
#                 x_ld_md[i, k1, :, :] + x_ld_md[i, k2, :, :].transpose()
#             ) / 2
#             x_ld_ud_avg[i, p, :, :] = (
#                 x_ld_ud[i, k1, :, :] + x_ld_ud[i, k2, :, :].transpose()
#             ) / 2

#         p += 1


## Run this part for MV
for roi_y in np.arange(0, 6):
    for roi_x in np.arange(roi_y + 1, 6):

        k1 = roi_y * 6 + roi_x
        k2 = roi_x * 6 + roi_y
        print(k1, k2)
        for i in np.arange(len(subjects)):
            x_sd_md_avg[i, p, :, :] = (
                np.tril(x_sd_md[i, k1, :, :], k=-1) +
                np.tril(x_sd_md[i, k2, :, :], k=-1).transpose()
            )
            np.fill_diagonal(x_sd_md_avg[i, p, :, :], (np.diagonal(
                x_sd_md[i, k1, :, :])+np.diagonal(x_sd_md[i, k2, :, :]))/2)


            x_sd_ud_avg[i, p, :, :] = (
                np.tril(x_sd_ud[i, k1, :, :], k=-1) +
                np.tril(x_sd_ud[i, k2, :, :], k=-1).transpose()
            )
            np.fill_diagonal(x_sd_ud_avg[i, p, :, :], (np.diagonal(
                x_sd_ud[i, k1, :, :])+np.diagonal(x_sd_ud[i, k2, :, :]))/2)
            
            
            

            x_ld_md_avg[i, p, :, :] = (
                np.tril(x_ld_md[i, k1, :, :], k=-1) +
                np.tril(x_ld_md[i, k2, :, :], k=-1).transpose()
            )
            np.fill_diagonal(x_ld_md_avg[i, p, :, :], (np.diagonal(
                x_ld_md[i, k1, :, :])+np.diagonal(x_ld_md[i, k2, :, :]))/2)


            x_ld_ud_avg[i, p, :, :] = (
                np.tril(x_ld_ud[i, k1, :, :], k=-1) +
                np.tril(x_ld_ud[i, k2, :, :], k=-1).transpose()
            )
            np.fill_diagonal(x_ld_ud_avg[i, p, :, :], (np.diagonal(
                x_ld_ud[i, k1, :, :])+np.diagonal(x_ld_ud[i, k2, :, :]))/2)
        

        p += 1

lb = C.rois_labels
t1, t2 = [-100, 475]

p = 0
for roi_y in np.arange(0, 6):
    for roi_x in np.arange(roi_y + 1, 6):

        k1 = roi_y * 6 + roi_x
        k2 = roi_x * 6 + roi_y
        print(k1, k2)

        ## Run for linear bvMD/mvMD
        z = x_sd_md_avg[:, p, :, :] - x_ld_md_avg[:, p, :, :]
        z1 = np.mean(x_sd_md_avg[:, p, :, :].copy(), 0)
        z2 = np.mean(x_ld_md_avg[:, p, :, :].copy(), 0)

        ## Run for linear bvUD
        # z = x_sd_ud_avg[:, p, :, :]-x_ld_ud_avg[:, p, :, :]
        # z1 = np.mean(x_sd_ud_avg[:, p, :, :].copy(), 0)
        # z2 = np.mean(x_ld_ud_avg[:, p, :, :].copy(), 0)

        ## Run to compare bvMD vs bvUD
        # z = (x_sd_md_avg[:, p, :, :] - x_ld_md_avg[:, p, :, :]) - (x_sd_ud_avg[:, p, :, :]-x_ld_ud_avg[:, p, :, :])
        # z1 = np.mean((x_sd_md_avg[:, p, :, :] - x_ld_md_avg[:, p, :, :]), 0)
        # z2 = np.mean((x_sd_ud_avg[:, p, :, :] - x_ld_ud_avg[:, p, :, :]), 0)

        T_obs1, clusters1, cluster_p_values1, H01 = permutation_cluster_1samp_test(
            z,
            n_permutations=C.n_permutations,
            threshold=C.t_threshold,
            tail=C.tail,
            out_type="mask",
            verbose=True,
        )

        T_obs_plot1 = np.nan * np.ones_like(T_obs1)
        for c, p_val in zip(clusters1, cluster_p_values1):
            if (p_val <= C.pvalue and len(np.where(c == True)[0]) >= d/2):
                T_obs_plot1[c] = T_obs1[c]

        fig, ax = plt.subplots(1, 3, figsize=(21, 5))



        vmax = max(z1.copy().max(),
                   z2.copy().max())
        vmin = min(z1.copy().min(),
                   z2.copy().min())

        # vmax = .4
        # vmin = -.1

        im = ax[0].imshow(
            z1, cmap=C.cm1, extent=[t1, t2, t1, t2], aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[0])
        ax[0].set_title("SD", fontsize=18)

        ax[0].set_xlabel(labels[roi_x] + " - time(ms)", fontsize=18)
        ax[0].set_ylabel(
            labels[roi_y] + " - time(ms)", fontsize=18
        )  # fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])
        fig.patch.set_facecolor(C.background_color)
        ax[0].xaxis.label.set_color(C.font_color)
        ax[0].yaxis.label.set_color(C.font_color)
        ax[0].tick_params(
            axis="both", colors=C.font_color, width=2, length=4, labelsize=16
        )
        cbar.ax.tick_params(
            color=C.font_color, width=2, length=2, labelcolor=C.font_color, labelsize=14
        )

        im = ax[1].imshow(
            z2, cmap=C.cm1, extent=[t1, t2, t1, t2], aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[1])
        # cbar.set_label('Expained Var', color=font_color)

        ax[1].set_title("LD", fontsize=18)
        # ax[1].set_xlabel(labels[ROI_x]+ '- time(ms)')
        # ax[1].set_ylabel(labels[ROI_y]+ '- time(ms)')
        ax[1].xaxis.label.set_color(C.font_color)
        ax[1].yaxis.label.set_color(C.font_color)
        ax[1].tick_params(
            axis="both", colors=C.font_color, width=2, length=4, labelsize=16
        )
        # cbar.ax.tick_params(color=C.font_color, width=2, length= 2, labelcolor=font_color,labelsize=14)
        cbar.ax.tick_params(
            color=C.font_color, width=2, length=4, labelcolor=C.font_color, labelsize=14
        )

        vmax = np.max(np.nan_to_num(T_obs1))
        vmin = np.min(np.nan_to_num(T_obs1))
        v = max(vmax, np.abs(vmin))
        v = 11.5
        vmax = v
        vmin = -v
        # v = max(abs(vmax), abs(vmin))
        # plt.subplot(313)
        im1 = ax[2].imshow(
            T_obs1,
            cmap=C.cm3,
            extent=[t1, t2, t1, t2],
            aspect="equal",
            origin="lower",
            vmin=-11.5,
            vmax=11.5
        )
        # cbar = fig.colorbar(im1, ax=ax[2])
        cbar.ax.tick_params(color=C.font_color, width=2,
                            length=4, labelcolor=C.font_color, labelsize=14)


        im = ax[2].imshow(
            T_obs_plot1,
            cmap=C.cm2,
            extent=[t1, t2, t1, t2],
            aspect="equal",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        cbar = fig.colorbar(im, ax=ax[2])
        # cbar.set_label('t-values', color=font_color)
        ax[2].set_title("t-test")
        # ax[2].set_xlabel(labels[ROI_x]+ '- time(ms)',fontsize=14)
        # ax[2].set_ylabel(labels[ROI_y]+ '- time(ms)',fontsize=14)
        ax[2].xaxis.label.set_color(C.font_color)
        ax[2].yaxis.label.set_color(C.font_color)
        ax[2].tick_params(axis="x", colors=C.font_color,
                          width=2, length=4, labelsize=16)
        ax[2].tick_params(axis="y", colors=C.font_color,
                          width=2, length=4, labelsize=16)

        cbar.ax.tick_params(
            color=C.font_color, width=2, length=4, labelcolor=C.font_color, labelsize=14
        )
        fig.suptitle(labels[roi_y] + "-" + labels[roi_x], fontsize=12)

        plt.savefig(
            '/home/sr05/Method_dev/method_fig/Linear_Transformation_clusters_25ms_grey'+lb[roi_y] + '_'+lb[roi_x]+'1way_equalT')



        p += 1
# 

###########################
# connectivity_strength = np.zeros([15, 2])
# p = 0
# for roi_y in np.arange(0, 6):
#     for roi_x in np.arange(roi_y + 1, 6):
#         # for ROI_x in np.arange(2, 3):

#         k1 = roi_y * 6 + roi_x
#         k2 = roi_x * 6 + roi_y
#         print(k1, k2)

#         z = x_sd_md_avg[:, p, :, :] - x_ld_md_avg[:, p, :, :]
#         z1 = np.mean(x_sd_md_avg[:, p, :, :].copy(), 0)
#         z2 = np.mean(x_ld_md_avg[:, p, :, :].copy(), 0)

#         # z = x_sd_ud_avg[:, p, :, :]-x_ld_ud_avg[:, p, :, :]
#         # z1 = np.mean(x_sd_ud_avg[:, p, :, :].copy(), 0)
#         # z2 = np.mean(x_ld_ud_avg[:, p, :, :].copy(), 0)

#         T_obs1, clusters1, cluster_p_values1, H01 = permutation_cluster_1samp_test(
#             z,
#             n_permutations=C.n_permutations,
#             threshold=t_threshold,
#             tail=tail,
#             out_type="mask",
#             verbose=True,
#         )

#         T_obs_plot1 = np.nan * np.ones_like(T_obs1)
#         for c, p_val in zip(clusters1, cluster_p_values1):
#             if (p_val <= C.pvalue and len(np.where(c == True)[0]) >= d/2):
#                 T_obs_plot1[c] = T_obs1[c]

#         # time_win_1 = T_obs_plot1[3:13, 3:13]
#         # time_win_1 = np.nan_to_num(time_win_1, nan=0)

#         # time_win_2 = T_obs_plot1[13:24, 13:24]
#         # time_win_2 = np.nan_to_num(time_win_2, nan=0)
        
                
#         x_2_y =  np.tril(T_obs_plot1, k=-1) 
#         y_2_x =  np.triu(T_obs_plot1, k=1)
        
#         x_2_y = np.nan_to_num(x_2_y, nan=0)
#         y_2_x = np.nan_to_num(y_2_x, nan=0)



#         conn_strength_x_2_y = np.sum(x_2_y)
#         conn_strength_y_2_x = np.sum(y_2_x)

#         connectivity_strength[p, 0] = conn_strength_x_2_y
#         connectivity_strength[p, 1] = conn_strength_y_2_x

#         # conn_strength_win1 = np.sum(time_win_1)
#         # conn_strength_win2 = np.sum(time_win_2)

#         # connectivity_strength[p, 0] = conn_strength_win1
#         # connectivity_strength[p, 1] = conn_strength_win2

#         p += 1


# # connectivity_strength_uv = connectivity_strength
# # connectivity_strength_md = connectivity_strength

# # # plt.close('all')
