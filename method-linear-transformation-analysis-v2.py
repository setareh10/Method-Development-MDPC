#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:19:42 2021

@author: sr05
"""
import os
import mne
import sys
import time
import pickle
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model
from joblib import Parallel, delayed
from mne.epochs import equalize_epoch_counts
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
import multiprocessing
from functools import partial
from mne.stats import (
    permutation_cluster_1samp_test,
    f_threshold_mway_rm,
    summarize_clusters_stc,
    permutation_cluster_test,
    f_mway_rm,
)
from scipy import stats as stats
from matplotlib.colors import LinearSegmentedColormap

t_down_sampling = 25.0


def mask_function(X, cut_off=None):
    if cut_off is not None:
        r, c = X.shape
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                if X[i, j] < cut_off:
                    X[i, j] = cut_off
    return X


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()
# ROI_x=1
# ROI_y=0
s = time.time()

# Cond= 'LD'
labels = ["lATL", "rATL", "PTC", "IFG", "AG", "PVA"]


###########################################################

d = 24
x_sd_md = np.zeros([len(C.subjects), 36, d, d])
x_sd_uv = np.zeros([len(C.subjects), 36, d, d])

x_ld_md = np.zeros([len(C.subjects), 36, d, d])
x_ld_uv = np.zeros([len(C.subjects), 36, d, d])

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
                gof_sd_uv = np.zeros([d, d])

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
                    )

                    with open(file_name, "rb") as fp:  # Unpickling
                        a = pickle.load(fp)
                    gof_sd_md = gof_sd_md + a["ev_md"]
                    gof_sd_uv = gof_sd_uv + a["ev_uv"]

                sd_md = gof_sd_md / 3
                sd_uv = gof_sd_uv / 3

                x_sd_md[i, p, :, :] = mask_function(sd_md, cut_off=cut_off)
                x_sd_uv[i, p, :, :] = mask_function(sd_uv, cut_off=cut_off)

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
                    )

                    with open(file_name, "rb") as fp:  # Unpickling
                        a = pickle.load(fp)
                    x_ld_md[i, p, :, :] = mask_function(
                        a["ev_md"], cut_off=cut_off)
                    x_ld_uv[i, p, :, :] = mask_function(
                        a["ev_uv"], cut_off=cut_off)

            p += 1
#############################################################################

d = 24
step = 1
x_sd_md_avg = np.zeros([18, 15, d, d])
x_sd_uv_avg = np.zeros([18, 15, d, d])

x_ld_md_avg = np.zeros([18, 15, d, d])
x_ld_uv_avg = np.zeros([18, 15, d, d])

p = 0
for roi_y in np.arange(0, 6):
    for roi_x in np.arange(roi_y + 1, 6):

        k1 = roi_y * 6 + roi_x
        k2 = roi_x * 6 + roi_y
        print(k1, k2)
        for i in np.arange(len(subjects)):
            x_sd_md_avg[i, p, :, :] = (
                x_sd_md[i, k1, :, :] + x_sd_md[i, k2, :, :].transpose()
            ) / 2
            x_sd_uv_avg[i, p, :, :] = (
                x_sd_uv[i, k1, :, :] + x_sd_uv[i, k2, :, :].transpose()
            ) / 2

            x_ld_md_avg[i, p, :, :] = (
                x_ld_md[i, k1, :, :] + x_ld_md[i, k2, :, :].transpose()
            ) / 2
            x_ld_uv_avg[i, p, :, :] = (
                x_ld_uv[i, k1, :, :] + x_ld_uv[i, k2, :, :].transpose()
            ) / 2

        p += 1


lb = C.rois_labels
t1, t2 = [-100, 475]
# # difference of SD (0:6) and LD(6:12) for aech ROI and individual

# colors = [  'darkslategrey','teal','cadetblue','lightseagreen', 'white','yellow', 'gold','orange', 'red', 'darkred']
colors = [
    "darkslategrey",
    "teal",
    "cadetblue",
    "lightseagreen",
    "darkred",
    "red",
    "orange",
    "gold",
    "yellow",
    "white",
]
colors2 = ["blue", "green", "white", "yellow", "red"]
colors3 = ["black", "gray"]

background_color = "white"
font_color = "black"

# colors = ['black','teal','red','orange','yellow','white']  # R -> G -> B

cmap_name = "my_list"
n_bin = 100
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors2, N=n_bin)
cm3 = LinearSegmentedColormap.from_list(cmap_name, colors3, N=n_bin)

tail = 0
t_threshold = -stats.distributions.t.ppf(C.pvalue / 2.0, len(C.subjects) - 1)
p = 0
for roi_y in np.arange(0, 6):
    for roi_x in np.arange(roi_y + 1, 6):
        # for ROI_x in np.arange(2, 3):

        k1 = roi_y * 6 + roi_x
        k2 = roi_x * 6 + roi_y
        print(k1, k2)

        z = x_sd_md_avg[:, p, :, :] - x_ld_md_avg[:, p, :, :]
        z1 = np.mean(x_sd_md_avg[:, p, :, :].copy(), 0)
        z2 = np.mean(x_ld_md_avg[:, p, :, :].copy(), 0)

        # z = x_sd_uv_avg[:, p, :, :]-x_ld_uv_avg[:, p, :, :]
        # z1 = np.mean(x_sd_uv_avg[:, p, :, :].copy(), 0)
        # z2 = np.mean(x_ld_uv_avg[:, p, :, :].copy(), 0)

        T_obs1, clusters1, cluster_p_values1, H01 = permutation_cluster_1samp_test(
            z,
            n_permutations=C.n_permutations,
            threshold=t_threshold,
            tail=tail,
            out_type="mask",
            verbose=True,
        )

        T_obs_plot1 = np.nan * np.ones_like(T_obs1)
        for c, p_val in zip(clusters1, cluster_p_values1):
            if (p_val <= C.pvalue and len(np.where(c==True)[0])>=d/2):
                T_obs_plot1[c] = T_obs1[c]

        # plt.figure(figsize=(3,10))
        fig, ax = plt.subplots(1, 3, figsize=(21, 5))

        # # plotting the t-values
        # vmax = max(np.mean(x_sd_avg[:, p, :, :].copy(), 0).max(),
        #             np.mean(x_ld_avg[:, p, :, :].copy(), 0).max())
        # vmin = min(np.mean(x_sd_avg[:, p, :, :].copy(), 0).min(),
        #             np.mean(x_ld_avg[:, p, :, :].copy(), 0).min())

        # vmax = .4
        # vmin = -.1
        # vmax = 0.35
        # vmin = -0.05
        # plt.subplot(311)
        im = ax[0].imshow(
            z1, cmap=cm, extent=[t1, t2, t1, t2], aspect="equal", origin="lower"
        )  # , vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[0])
        ax[0].set_title("SD", fontsize=18)

        ax[0].set_xlabel(labels[roi_x] + " - time(ms)", fontsize=18)
        ax[0].set_ylabel(
            labels[roi_y] + " - time(ms)", fontsize=18
        )  # fig.suptitle('Y: '+lb[ROI_y] + ' | X: '+lb[ROI_x])
        fig.patch.set_facecolor(background_color)
        ax[0].xaxis.label.set_color(font_color)
        ax[0].yaxis.label.set_color(font_color)
        ax[0].tick_params(
            axis="both", colors=font_color, width=2, length=4, labelsize=16
        )
        cbar.ax.tick_params(
            color=font_color, width=2, length=2, labelcolor=font_color, labelsize=14
        )

        im = ax[1].imshow(
            z2, cmap=cm, extent=[t1, t2, t1, t2], aspect="equal", origin="lower"
        )  # , vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[1])
        # cbar.set_label('Expained Var', color=font_color)

        ax[1].set_title("LD", fontsize=18)
        # ax[1].set_xlabel(labels[ROI_x]+ '- time(ms)')
        # ax[1].set_ylabel(labels[ROI_y]+ '- time(ms)')
        ax[1].xaxis.label.set_color(font_color)
        ax[1].yaxis.label.set_color(font_color)
        ax[1].tick_params(
            axis="both", colors=font_color, width=2, length=4, labelsize=16
        )
        # cbar.ax.tick_params(color=font_color, width=2, length= 2, labelcolor=font_color,labelsize=14)
        cbar.ax.tick_params(
            color=font_color, width=2, length=4, labelcolor=font_color, labelsize=14
        )

        vmax = np.max(np.nan_to_num(T_obs1))
        vmin = np.min(np.nan_to_num(T_obs1))
        # v = max(vmax , np.abs(vmin))
        v = 11.1
        vmax = v
        vmin = -v
        # v = max(abs(vmax), abs(vmin))
        # plt.subplot(313)
        im1 = ax[2].imshow(
            T_obs1.transpose(),
            cmap=cm3,
            extent=[t1, t2, t1, t2],
            aspect="equal",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        # cbar = fig.colorbar(im1, ax=ax[2])
        cbar.ax.tick_params(color=font_color, width=2,
                            length=2, labelcolor=font_color)

        im = ax[2].imshow(
            T_obs_plot1.transpose(),
            cmap=cm2,
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
        ax[2].xaxis.label.set_color(font_color)
        ax[2].yaxis.label.set_color(font_color)
        ax[2].tick_params(axis="x", colors=font_color,
                          width=2, length=4, labelsize=16)
        ax[2].tick_params(axis="y", colors=font_color,
                          width=2, length=4, labelsize=16)

        cbar.ax.tick_params(
            color=font_color, width=2, length=4, labelcolor=font_color, labelsize=14
        )
        fig.suptitle(labels[roi_y] + "-" + labels[roi_x], fontsize=12)

        plt.savefig(
            '/home/sr05/Method_dev/method_fig/Linear_Transformation_clusters_25ms_'+lb[roi_y] + '_'+lb[roi_x]+'new_equalT')
        # plt.savefig(
        #     '/home/sr05/Method_dev/method_fig/Linear_Transformation_UV_25ms_'+lb[roi_y] + '_'+lb[roi_x]+'new')

        p += 1
# plt.close('all')
#_equalT_trasposed