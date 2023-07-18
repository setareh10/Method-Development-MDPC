#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:15:11 2023

@author: sr05
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

 
class MDPC:
    
    def __init__(self, n_splits,  normalize=False,  alphas=np.logspace(-3, 3, 5), 
                 scoring="explained_variance", n_jobs=1):
        

        self.normalize = normalize
        self.n_splits = n_splits
        self.alphas = alphas  
        self.scoring = scoring  
        self.n_jobs = n_jobs




class BiVariateMDPC:
    
    def __init__(self, mdpc, x, y):

        self.mdpc = mdpc
        self.x = x
        self.y = y
   
    
    def bv_linear(self,):
        
        kf = KFold(n_splits=self.mdpc.n_splits)
        regr_cv = RidgeCV(
        alphas=self.mdpc.alphas, scoring=self.mdpc.scoring, normalize=self.mdpc.normalize
        )
        scores = cross_validate(
        regr_cv,
        self.x,
        self.y,
        scoring=self.mdpc.scoring,
        cv=kf,
        n_jobs=self.mdpc.n_jobs,
        )

        gof_ev = np.mean(scores["test_score"])
       
        return gof_ev

        
        
        
    def bv_nonlinear(self, hidden_layer_sizes, activation_function, max_iter=500, solver='lbfgs'):
        
        kf = KFold(n_splits=self.mdpc.n_splits)

        parameters_l = {'hidden_layer_sizes': [hidden_layer_sizes], 'activation': [activation_function], 'solver': [solver],
                            'max_iter': [max_iter], 'alpha': self.mdpc.alphas}
    
        mlp = MLPRegressor()
    
        mlp_reg = GridSearchCV(
            mlp, parameters_l,  scoring=self.mdpc.scoring, cv=kf)
    
        mlp_reg.fit(self.x, self.y)
        
        explained_var = np.max(mlp_reg.cv_results_["mean_test_score"])
        
        return explained_var


def bv_deep_nonlinear(self, units_l1, nl, nn, activation, optimizer):
        
        kf = KFold(n_splits=self.mdpc.n_splits)
        
        params = {'nl':nl, 'nn':nn, 'activation':activation, 'omptimizer':optimizer}
        
        model = BiVariateMDPC.get_deep_model(self.v_x, self.v_y, units_l1, nl, nn, activation, optimizer)
        
        model = KerasRegressor(model)
        
        deep_reg = RandomizedSearchCV(model, params, scoring=self.mdpc.scoring, cv=kf)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3) 

        deep_reg.fit(self.x, self.y, callbacks=[early_stopping])
                
        explained_var = np.max(deep_reg.cv_results_["mean_test_score"])
        
        return explained_var
    
    
    
    def get_deep_model(input_shape, outut_shape, units, nl, nn, activation, optimizer):
        
        model = Sequential()
        
        model.add(Dense(units=units, input_shape=(input_shape,), activation=activation, ))
      
        for i in range(nl):
            
            model.add(Dense(units=nn, activation=activation))

       
        model.add(Dense(units=outut_shape ))
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
   

class MultiVariateMDPC:
    
    def __init__(self, mdpc, h_x, h_y, v_x, v_y,  train, test, idx_x, idx_y):
        
        self.mdpc = mdpc
        self.v_x = v_x
        self.v_y = v_y
        self.h_x = h_x
        self.train = train
        self.test = test
        self.idx_x = idx_x
        self.idx_y = idx_y
        

    
    def mv_linear(self, ):
                   
        ev_s = np.zeros(self.mdpc.n_splits)
        kf = KFold(n_splits=self.mdpc.n_splits)
        for s, (self.train, self.test) in enumerate(kf.split(self.h_x, self.h_y)):
    
            ev_s[s] = MultiVariateMDPC.mv_linear_regressor()
            
        ev_total = np.mean(ev_s)
    
        return ev_total

    
        
    def mv_linear_regressor(self,):
        
        s_x = int(sum(self.v_x[:self.idx_x]))
        s_y = int(sum(self.v_y[:self.idx_y]))
        s_y_prime = int(sum(self.v_x[:self.idx_y]))
        
        x = self.h_x[:, s_x:s_x+int(self.v_x[self.idx_x])]
        y = self.h_y[:, s_y:s_y+int(self.v_y[self.idx_y])]
        h_x_prime = np.delete(self.h_x, range(s_y_prime,s_y_prime+int(self.v_x[self.idx_y])), axis=1)
        
        regr = RidgeCV(alphas=self.mdpc.alphas)
        regr.fit(h_x_prime[self.train, :], y[self.train, :])
        trans_coef_all = (regr.coef_).transpose()
        
        intercepts = np.repeat(regr.intercept_.reshape(
            1, regr.intercept_.shape[0]), self.test.shape[0], axis=0)
        
       
        if self.idx_y<self.idx_x:
            v_x_prime = np.delete(self.v_x,self.idx_y)
            s_x = int(sum(v_x_prime[:self.idx_x-1]))
            
        trans_coef = trans_coef_all[s_x:s_x+int(self.v_x[self.idx_x]), :]
        y_predicted = np.matmul(x[self.test,:],trans_coef) + intercepts
        y_true = y[self.test,:]
        ev = explained_variance_score(y_true, y_predicted)
        
        return ev
    
    

class Preprocessing:
    
    
    def cluster_vertices(self, x):

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
    
    
    def scale_patterns(self, x):
    
        trial, v_x = x.shape
        x_vector = x.reshape(trial*v_x, 1)
        scaler = StandardScaler()
        x_vec_scaled = scaler.fit_transform(x_vector)
        x_scaled = x_vec_scaled.reshape(trial, v_x)
        
        return x_scaled
    
    
    def create_mv_pattern(self, T, t):
        
        H = []
        v = np.zeros(len(T))
        for k in range(len(T)):
            scaled = Preprocessing.scale_patterns(T[k][:, :, t])
            a = Preprocessing.cluster_vertices(scaled.transpose())
            v[k] = int(a.shape[1])
            if k == 0:
                H = a
            else:
                H = np.append(H, a, axis=1)
                
        return H, v
    
    
        
