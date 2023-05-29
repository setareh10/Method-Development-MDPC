#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:28:07 2023

@author: sr05
"""



import mne
import numpy as np
import sn_config as c
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator




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



def create_io(epoch, inv_fname_epoch, labels, sub_to, roi_x, roi_y):
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


def read_epochs(cond, i):
    
    meg = subjects[i]
    sub_to = mri_sub[i][1:15]

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

    return epochs, inverse_fname_epoch, morphed_labels, sub_to











