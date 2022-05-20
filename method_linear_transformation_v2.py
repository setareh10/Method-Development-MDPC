#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:45:16 2022

@author: sr05
"""


import os
import sys

import mne
import time
import pickle
import numpy as np
import sn_config as c
from sklearn.cluster import KMeans
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import RidgeCV
from SN_semantic_ROIs import SN_semantic_ROIs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import cross_validate, KFold
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
import warnings
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
# path to raw data
data_path = c.data_path
main_path = c.main_path
subjects = c.subjects
mri_sub = c.subjects_mri

# Parameters
lambda2 = c.lambda2_epoch
label_path = c.label_path
roi = SN_semantic_ROIs()
fs = 1000
f_down_sampling = 40  # 100Hz, 20Hz
t_down_sampling = fs / f_down_sampling  # 10ms, 50ms


def method_linear_transformation_main(cond, roi_y, roi_x, i, normalize):
    print("***Running sn_transformation_main...")
    print("***Running sn_transformation_main: ", cond, roi_y, roi_x, i)
    warnings.filterwarnings("ignore")

    meg = subjects[i]
    sub_to = mri_sub[i][1:15]
    s = time.time()

    # file_name to save the output
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

    # morph labels from fsaverage to each subject
    morphed_labels = mne.morph_labels(
        roi, subject_to=sub_to, subject_from="fsaverage", subjects_dir=data_path
    )

    # read,crop and resample epochs
    epoch_name = data_path + meg + "block_" + cond + "_words_epochs-epo.fif"
    epoch_condition = mne.read_epochs(epoch_name, preload=True)
    epochs = (
        epoch_condition["words"].copy().crop(-0.100, 0.510).resample(f_down_sampling)
    )

    # inverse operator
    inverse_fname_epoch = data_path + meg + "InvOp_" + cond + "_EMEG-inv.fif"

    print("***Running SN_transformation_io...")
    # prepares the patterns of roi_x and roi_y over time
    output = method_linear_transformation_io(
        epochs, inverse_fname_epoch, morphed_labels, sub_to, roi_x, roi_y
    )
    print("***Running SN_transformation...")
    # computes the connectivity (explained_variance) of the patterns of roi_x
    # and roi_y over time: roi_x patterns= output[0], roi_y patterns= output[1]
    gof = method_linear_transformation(output[0], output[1], normalize)
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(gof, fp)
    e = time.time()
    print("time: ", e - s, " /for: ", cond, roi_y, roi_x)
    return gof


def method_linear_transformation_io(
    epoch, inv_fname_epoch, labels, sub_to, roi_x, roi_y
):
    # extracts patterns of roi_x and roi_y over time
    output = [0] * 2
    # read and apply inverse operator
    inverse_operator = read_inverse_operator(inv_fname_epoch)
    stc = apply_inverse_epochs(
        epoch,
        inverse_operator,
        lambda2,
        method="MNE",
        pick_ori="normal",
        return_generator=False,
    )

    for i, roi_idx in enumerate([roi_x, roi_y]):
        labels[roi_idx].subject = sub_to
        # defines matrix dimensions (vertices x timepoints), & initializing
        n_vertices, n_timepoints = stc[0].in_label(labels[roi_idx]).data.shape
        x = np.zeros([len(stc), n_vertices, n_timepoints])
        # creates output array of size (trials x vertices x timepoints)
        for n_trial, stc_trial in enumerate(stc):
            pattern = stc_trial.in_label(labels[roi_idx]).data
            x[n_trial, :, :] = pattern

        output[i] = x
    return output


def method_linear_transformation(x, y, normalize):
    # computes the explained_variance of different latencies

    gof = {}
    # initialize the explained variance array of size n_timepoints X n_timepoints
    gof_ev_md = np.zeros([y.shape[-1], x.shape[-1]])
    gof_ev_uv = np.zeros([y.shape[-1], x.shape[-1]])

    for t1 in range(y.shape[-1]):
        for t2 in range(x.shape[-1]):
            # gof_explained_variance = np.zeros([idx_total.shape[0], idx_total.shape[0]])

            print("timepoint: ", t1, t2)
            n_trials = x.shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10
            print("clustering started!")

            x_scaled = my_scaler(x[:, :, t2])
            y_scaled = my_scaler(y[:, :, t1])

            x_cluster = vertices_clustering(x_scaled.transpose())
            y_cluster = vertices_clustering(y_scaled.transpose())

            x_uv = x_cluster.copy().mean(1).reshape(n_trials, 1)
            y_uv = y_cluster.copy().mean(1).reshape(n_trials, 1)

            print("clustering finished!")
            kf = KFold(n_splits=n_splits)

            gof_ev_md[t1, t2] = l_estimation(x_cluster, y_cluster, normalize, kf)
            gof_ev_uv[t1, t2] = l_estimation(x_uv, y_uv, normalize, kf)

    gof["ev_md"] = gof_ev_md
    gof["ev_uv"] = gof_ev_uv

    return gof


def vertices_clustering(x):
    print("clustering...")
    # x should be of size vertex*stimulus
    x0 = x.copy()
    x1 = x.copy()
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 20))
    visualizer.fit(x0)
    n_clusters = visualizer.elbow_value_
    if visualizer.elbow_value_ is None:
        n_clusters = 10
    model = KMeans(n_clusters=n_clusters)
    model.fit(x1)
    yhat = model.predict(x1)
    clusters = np.unique(yhat)
    x_clusters = np.zeros([x.shape[1], clusters.shape[0]])  # stimulus*vertex
    for n, cluster in enumerate(clusters):
        row_ix = np.where(yhat == cluster)
        data = x[row_ix, :].transpose()[:, :, 0]
        # data=x[row_ix, :].reshape([x[row_ix, :].shape[2],x[row_ix, :].shape[1]])

        cluster_var = data.var(0)
        idx = np.where(cluster_var == cluster_var.max())
        x_clusters[:, n] = x[idx, :]

    return x_clusters


def my_scaler(x):

    trial, v_x = x.shape
    x_vector = x.reshape(trial * v_x, 1)
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


# conditions = ['fruit', 'milk', 'odour', 'LD']
# # conditions = ["LD"]

# combinations = []
# for cond in conditions:
#     for roi_y in range(0, 6):
#         for roi_x in range(0, 6):
#             if roi_y != roi_x:
#                 combinations.append([cond, roi_y, roi_x])

combinations = [
    ["odour", 5, 2]
  
]
normalization = False
if len(sys.argv) == 1:

    sbj_ids = np.array([10])

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


n_jobs = 15
n_start = 0
start_time1 = time.monotonic()
start_time2 = time.perf_counter()
for s in sbj_ids:
    # with parallel_backend("loky", inner_max_num_threads=2):
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(method_linear_transformation_main)(cond, roi_y, roi_x, s, normalization)
        for cond, roi_y, roi_x in combinations[n_start : n_jobs + n_start]
    )
# , prefer="threads" multiprocessing

print(time.perf_counter() - start_time2)

print(time.monotonic() - start_time1)
print("FINISHED!")
