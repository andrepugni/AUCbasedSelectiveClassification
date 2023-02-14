# -*- coding: utf-8 -*-
"""
Code for Oracle experiment
"""

import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import numpy as np
from cmeta import AUCross
from exp_tabdata import *
from time import time
from sklearn.metrics import roc_auc_score
import matplotlib.colors as mpc
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

def plot_surface(coverage, res, r, ls_z = 4):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    tmp2 = r[r['desired_coverage']==coverage].copy()
    actual_coverage = tmp2['coverage'].values[0]
    tmp = res[res['coverage']>=actual_coverage].copy()
    dataset = res['dataset'].values[0]
    x = tmp.sort_values(['auc','theta_l'], ascending = False).theta_l
    y = tmp.sort_values(['auc','theta_l'], ascending = False).theta_u
    z = tmp.sort_values(['auc','theta_l'], ascending = False).auc
    max_auc = np.max(z)
    max_point = tmp.sort_values('auc', ascending=False).iloc[0,:]
    max_auc = max_point['auc']
    theta_l_star = max_point['theta_l']
    theta_u_star = max_point['theta_u']
    theta_l_star_est = tmp2['theta_l'].values[0]
    theta_u_star_est = tmp2['theta_u'].values[0]
    auc_star = tmp2['auc'].values[0]
    ax.plot_trisurf(x, y, z, label='parametric curve', color='orange', alpha=0.5)
    c2 = ax.scatter(theta_l_star, theta_u_star, max_auc, color='blue', marker ='*',
               label = 'AUC: {} - coord: ({},{})'.format(max_auc, theta_l_star, theta_u_star))
    c3 = ax.scatter(theta_l_star_est, theta_u_star_est, auc_star, color='black', marker ='o',
               label = 'AUC: {} - coord: ({},{})'.format(auc_star, round(theta_l_star_est,3), round(theta_u_star_est,3)))
    
    c2._facecolors2d = mpc.to_rgba('blue', alpha=None)
    c2._edgecolors2d = mpc.to_rgba('blue', alpha=None)
    c3._facecolors2d = mpc.to_rgba('black', alpha=None)
    c3._edgecolors2d = mpc.to_rgba('black', alpha=None)
    
    
    ax.set_xlabel(r'$\theta_l$', fontsize=15)
    ax.set_ylabel(r'$\theta_u$', fontsize=15)
    ax.set_zlabel('AUC', fontsize= 15, rotation=90,  labelpad=ls_z)
#     ax.set_xticks(fontsize=20)
#     ax.set_yticks(fontsize=20)
#     ax.set_zticks(fontsize=20)
    label1 = 'AUC Oracle: {} - coord: ({},{})'.format(round(max_auc,4), theta_l_star, theta_u_star)
    label2 = 'AUC AUCross: {} - coord: ({},{})'.format(round(auc_star,4), round(theta_l_star_est,3), round(theta_u_star_est,3))
    ax.legend([c2,c3], [label1, label2])
#     plt.legend(loc="upper right")
    plt.title("Optimal surface {} - coverage {}".format(dataset,coverage))
    plt.savefig("surfaces/opt_surface_{}_{}.png".format(dataset,coverage), dpi=300, bbox_inches='tight', facecolor='w')

    
atts_dict = {'lending': atts_lending, 'lendingNOH': atts_lendingNOH, 'CSDS1': atts_CSDS1, 'CSDS2':atts_CSDS2, 'CSDS3':atts_CSDS3, 
                 'GiveMe':atts_giveme, 'adultNM':atts_adult, 'adultNMNOH':atts_adultNOH, 'UCI_credit' : atts_credit}

#for file in files:                 
def main(file, atts_dict):
    """
    

    Parameters
    ----------
    file : str
        the name of the dataset.
    atts_dict : dict
        A dictionary containing all the features names for a specific dataset.

    Returns
    -------
    results_combo : pd.DataFrame
        A dataframe containing all the results for ORACLE.
    results_class : pd.DataFrame
        A dataframe containing all the results for AUCross.

    """
    results_class = pd.DataFrame()
    results_combo = pd.DataFrame()
    print(file)
    X_train, X_test, y_train, y_test = read_datasets(file, atts_dict)
    clf_base = lgb.LGBMClassifier(random_state=42, n_jobs=8)
    clf = AUCross(clf_base)
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:,1]
    bands = clf.qband(X_test)
    actual_res = [[1-q,len(y_test[bands>i])/len(y_test),
                         roc_auc_score(y_test[bands>i], scores[bands>i])
                          ]for i,q in enumerate(sorted(clf.quantiles,reverse=False))]
    r = pd.DataFrame(actual_res, columns =['desired_coverage','coverage', 'auc',])
    r['theta_l'] = [theta[0] for theta in clf.thetas]
    r['theta_u'] = [theta[1] for theta in clf.thetas]
    r['dataset'] = file
    results_class = pd.concat([results_class, r], axis=0)
    res = pd.DataFrame()
    d = sorted([el for el in np.unique(np.round(scores,3))])
    start_time = time()
    v = { a:len(scores[scores>=a]) for a in d}
    end_time = time()
    print((end_time-start_time))
    start_time = time()
    len_test = len(scores)
    combinations = [(a,b) for a in d for b in d if (a < b)&(((v[a]-v[b])/len_test)<=(1-0.7))]
    end_time = time()
    print((end_time-start_time))
    for combo in tqdm(combinations):
        score_sel = scores[(scores<=combo[0])|(scores>combo[1])]
        true_sel = y_test[(scores<=combo[0])|(scores>combo[1])]
        if (np.sum(true_sel)==0) or (np.sum(true_sel)==len(true_sel)):
            auc = 0
            coverage = 0
        else:
            auc = roc_auc_score(true_sel, score_sel)
            coverage = len(true_sel)/len(y_test)
        tmp = pd.DataFrame([[combo[0], combo[1], auc, coverage]], columns = ['theta_l','theta_u', 'auc', 'coverage'])
        tmp = tmp[tmp['coverage']>=.7]
        res = pd.concat([res,tmp], axis=0)
    res['dataset'] = file
    results_combo = pd.concat([results_combo, res], axis=0)
    results_class.to_csv('AUCross_results_{}_lgbm.csv'.format(file))
    results_combo.to_csv('combinations_results_{}_lgbm.csv'.format(file))
    #for c in clf.quantiles:
    #    print(1-c)
    #    plot_surface(1-c,res, r, ls_z=10)
    return results_combo, results_class
    

if __name__=='__main__':
    atts_dict = {'lending': atts_lending, 'lendingNOH': atts_lendingNOH, 'CSDS1': atts_CSDS1, 'CSDS2':atts_CSDS2, 'CSDS3':atts_CSDS3, 
                 'GiveMe':atts_giveme, 'adultNM':atts_adult, 'adultNMNOH':atts_adultNOH, 'UCI_credit' : atts_credit}
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--n_jobs', type=int, required=False, default=1)
    args = parser.parse_args()
    n_jobs = args.n_jobs
    filelist = ['CSDS1', 'CSDS2', 'CSDS3', 'GiveMe', 'adultNM', 'UCI_credit']

    if n_jobs==1:
        results_combo = pd.DataFrame()
        results_class = pd.DataFrame()
        for file in tqdm(filelist):
            print(file)
            tmp1, tmp2 = main(file, atts_dict)
            results_combo = pd.concat([results_combo, tmp1], axis=0)
            results_class = pd.concat([results_class, tmp2], axis=0)
        results_class.to_csv('exp_oracle_final_results_lgbm_class.csv')
        results_combo.to_csv('exp_oracle_final_results_lgbm_combo.csv')
    else:
        tmp1, tmp2 = Parallel(n_jobs=n_jobs, verbose = 0)([delayed(main)(file, atts_dict) for file in tqdm(filelist)])
        results_combo = pd.concat(tmp1, axis=0)
        results_class = pd.concat(tmp2, axis=0)
        results_class.to_csv('exp_oracle_final_results_lgbm_class.csv')
        results_combo.to_csv('exp_oracle_final_results_lgbm_combo.csv')
