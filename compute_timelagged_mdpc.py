#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:32:45 2023

@author: sr05
"""

import numpy as np
from mdpc.preprocessing import cluster_vertices, scale_patterns, create_mv_pattern
from mpdc import MDPC, BiVariateMDPC, MultiVariateMDPC


def compute_bv_linear_timelagged_mdpc(x, y):

    goodness_of_fit = {}
    gof_ev_md = np.zeros([y.shape[-1], x.shape[-1]])
    gof_ev_uv = np.zeros([y.shape[-1], x.shape[-1]])

    for t1 in range(y.shape[-1]):
        for t2 in range(x.shape[-1]):

            n_trials = x.shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10

            x_scaled = scale_patterns(x[:, :, t2])
            y_scaled = scale_patterns(y[:, :, t1])

            x_cluster = cluster_vertices(x_scaled.transpose())
            y_cluster = cluster_vertices(y_scaled.transpose())
            
            x_abs = np.abs(x[:, :, t2])
            y_abs = np.abs(y[:, :, t1])

            
            x_ud = scale_patterns(np.mean(x_abs,1).reshape(n_trials, 1))
            y_ud = scale_patterns(np.mean(y_abs,1).reshape(n_trials, 1))


            mdpc = MDPC(n_splits,  normalize=False,  alphas=np.logspace(-3, 3, 5), 
                 scoring="explained_variance", n_jobs=1)
            
            bv_mdpc = BiVariateMDPC(mdpc, x_cluster, y_cluster)
            bv_ud = BiVariateMDPC(mdpc, x_ud, y_ud)
            

            gof_ev_md[t1, t2] = max(bv_mdpc.bv_linear(),0)
            gof_ev_uv[t1, t2] = max(bv_ud.bv_linear(),0)


    goodness_of_fit["ev_md"] = gof_ev_md
    goodness_of_fit["ev_uv"] = gof_ev_uv

    return goodness_of_fit



def compute_bv_nonlinear_timelagged_mdpc(x, y):

    goodness_of_fit = {}
    gof_ev_L = np.zeros([y.shape[-1], x.shape[-1]])
    gof_ev_NL = np.zeros([y.shape[-1], x.shape[-1]])

    for t1 in np.arange(y.shape[-1]):
        for t2 in np.arange(x.shape[-1]):

            n_trials = x.shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10
                
            x_scaled = scale_patterns(x[:, :, t2])
            y_scaled = scale_patterns(y[:, :, t1])
            
            x_cluster = cluster_vertices(x_scaled.transpose())
            y_cluster = cluster_vertices(y_scaled.transpose())
            
            ni = x_cluster.shape[1]
            no = y_cluster.shape[1]
            h = int(np.floor((ni + no) / 2))

            
            mdpc = MDPC(n_splits,  normalize=False,  alphas=np.logspace(-3, 3, 5), 
                 scoring="explained_variance", n_jobs=1)
            
            bv_mdpc = BiVariateMDPC(mdpc, x_cluster, y_cluster)

            gof_ev_L[t1, t2] = bv_mdpc.bv_nonlinear(hidden_layer_sizes=h, activation_function='identity')
            gof_ev_NL[t1, t2] = bv_mdpc.bv_nonlinear(hidden_layer_sizes=h, activation_function='tanh')


    goodness_of_fit["EV_L"] = gof_ev_L
    goodness_of_fit["EV_NL"] = gof_ev_NL

    return goodness_of_fit


def compute_bv_deep_nonlinear_timelagged_mdpc(x, y):

    goodness_of_fit = {}
    gof_ev_L = np.zeros([y.shape[-1], x.shape[-1]])
    gof_ev_NL = np.zeros([y.shape[-1], x.shape[-1]])

    for t1 in np.arange(y.shape[-1]):
        for t2 in np.arange(x.shape[-1]):

            n_trials = x.shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10
                
            x_scaled = scale_patterns(x[:, :, t2])
            y_scaled = scale_patterns(y[:, :, t1])
            
            x_cluster = cluster_vertices(x_scaled.transpose())
            y_cluster = cluster_vertices(y_scaled.transpose())
            
            ni = x_cluster.shape[1]
            no = y_cluster.shape[1]
            h = int(np.floor((ni + no) / 2))
            
            units_l1 = np.random.randint(min(ni,no), max(ni,no))
            nn = [np.random.randint(min(ni,no), max(ni,no)) for _ in range(3)]
            nl = [0,1,2,3]
            activation = ['relu', 'tanh']
            optimizer= ['sgd', 'Adam']
            
            
            mdpc = MDPC(n_splits,  normalize=False,  alphas=np.logspace(-3, 3, 5), 
                 scoring="explained_variance", n_jobs=1)
            
            bv_mdpc = BiVariateMDPC(mdpc, x_cluster, y_cluster)

            gof_ev_L[t1, t2] = bv_mdpc.bv_nonlinear(hidden_layer_sizes=h, activation_function='identity')
            gof_ev_NL[t1, t2] = bv_mdpc.bv_deep_nonlinear(units_l1, nl, nn, activation, optimizer)


    goodness_of_fit["EV_L"] = gof_ev_L
    goodness_of_fit["EV_NL"] = gof_ev_NL

    return goodness_of_fit


def compute_mv_linear_timelagged_mdpc(output):


    goodness_of_fit = {}
   
    gof_mvmd = [[np.zeros([output[0].shape[-1], output[0].shape[-1]])]for k in range(15)]
    
    for t2 in np.arange(0, output[0].shape[-1]):
        for t1 in np.arange(t2, output[0].shape[-1]):
            
    
            if t1 == t2:
                h1_total, v1_total = create_mv_pattern(output, t1)
            else:

                h1_total, v1_total = create_mv_pattern(output, t1)

                h2_total, v2_total = create_mv_pattern(output, t2)


            n_trials = output[0][:, :, t1].shape[0]
            n_splits = 5
            if (n_trials / 5) > 10:
                n_splits = 10
            
            mdpc = MDPC(n_splits,  normalize=False,  alphas=np.logspace(-3, 3, 5), 
                scoring="explained_variance", n_jobs=1)
            
            p = 0
            for idx_y in range(len(v1_total)):
                for idx_x in range(idx_y+1, len(v1_total)):


                    if t1 == t2:

                        z_xy = MultiVariateMDPC(mdpc, h1_total, h1_total, v1_total, v1_total, idx_x, idx_y).mv_linear()

                        z_yx = MultiVariateMDPC(mdpc, h1_total, h1_total, v1_total, v1_total, idx_y, idx_x).mv_linear()

                        gof_mvmd[p][0][t1, t2] = (max(z_xy,0)+max(z_yx,0))/2
                    else:


                        z_yx = MultiVariateMDPC(mdpc, h2_total, h1_total, v2_total, v1_total, idx_y, idx_x).mv_linear()	
                        z_xy = MultiVariateMDPC(mdpc, h2_total, h1_total, v2_total, v1_total, idx_x, idx_y).mv_linear()

        
                        gof_mvmd[p][0][t2, t1] = max(z_yx,0)
                        gof_mvmd[p][0][t1, t2] = max(z_xy,0)


                    p = p+1

    goodness_of_fit["ev"] = gof_mvmd

    return goodness_of_fit




