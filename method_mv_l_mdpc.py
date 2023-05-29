#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:23:30 2023

@author: sr05
"""


import os
import sys
import time
import pickle
import warnings
import numpy as np
from joblib import Parallel, delayed
from prepare_io import read_epochs, create_mv_io
from compute_timelagged_mdpc import compute_mv_linear_timelagged_mdpc


fs = 1000
f_down_sampling = int(input(f"The sampling frequency is {fs}Hz. Please enter your "\
                            "desired down sampling frequency"\
                            "(e.g. 20Hz, 40Hz, 100Hz):"))

t_down_sampling = fs / f_down_sampling


def main_multivariate_timelagged_mdpc(cond, i):



    epochs, inverse_fname_epoch, morphed_labels, sub_to = read_epochs(cond, i)

    output = create_mv_io(epochs, inverse_fname_epoch, morphed_labels, sub_to)
    
    gof = compute_mv_linear_timelagged_mdpc(output)
    
    file_name = (
        os.path.expanduser("~")
        + "/semnet-project/json_files/mv/trans_mv_"
        + cond
        + "_sub_"
        + str(i)
        + "_"
        + str(int(t_down_sampling))
        + "_clusters.json"
    )
    
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(gof, fp)
        
    return gof







    
if __name__ == "__main__":

    
    warnings.filterwarnings("ignore")

    conditions = ["fruit", "milk", "odour", "LD"]
    
    if len(sys.argv) == 1:
    
        sbj_ids = np.arange(0,18)
    
    else:
    
        sbj_ids = [int(aa) for aa in sys.argv[1:]]
    
    
    n_jobs = 25
    
    start_time1 = time.monotonic()
    
    for s in sbj_ids:
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(main_multivariate_timelagged_mdpc)(cond, s)
            for cond in conditions
        )
    
    
    print(time.monotonic() - start_time1)
    print("FINISHED!")
    
