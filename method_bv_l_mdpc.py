#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:44:30 2023

@author: sr05
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
from joblib import Parallel, delayed
from prepare_io import read_epochs, create_bv_io
from compute_timelagged_mdpc import compute_bv_linear_timelagged_mdpc

fs = 1000
f_down_sampling = int(input(f"The sampling frequency is {fs}Hz. Please enter your "\
                            "desired down sampling frequency"\
                            "(e.g. 20Hz, 40Hz, 100Hz):"))

t_down_sampling = fs / f_down_sampling


def main_linear_timelagged_mdpc(cond, roi_y, roi_x, i):


    epochs, inverse_fname_epoch, morphed_labels, sub_to = read_epochs(cond, i)

    output = create_bv_io(epochs, inverse_fname_epoch, morphed_labels, sub_to, roi_x, roi_y)

    
    gof = compute_bv_linear_timelagged_mdpc(output[0], output[1])
  
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
        
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(gof, fp)
    e = time.time()
    print("time: ", e - s, " /for: ", cond, roi_y, roi_x)
    return gof






if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")

    conditions = ['fruit', 'milk', 'odour', 'LD']

    combinations = []
    for cond in conditions:
        for roi_y in range(0, 6):
            for roi_x in range(0, 6):
                if roi_y != roi_x:
                    combinations.append([cond, roi_y, roi_x])



    if len(sys.argv) == 1:

        sbj_ids = np.array([7, 11, 17])

    else:

        sbj_ids = [int(aa) for aa in sys.argv[1:]]

    n_jobs = 20
    n_start = 0
    start_time1 = time.monotonic()
    
    for s in sbj_ids:
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(main_linear_timelagged_mdpc)(cond, roi_y, roi_x, s)
            for cond, roi_y, roi_x in combinations[n_start: n_jobs + n_start]
        )


    print(time.monotonic() - start_time1)
    print("FINISHED!")
