# -*- coding: utf-8 -*-
"""
Metrics for binary classifiers validation
"""

# standard modules
import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import sklearn.metrics as skm
import SelectiveClassifier

# get unique values in numpy array, pandas series or list
def unique(y):
    if isinstance(y, np.ndarray):
        return np.unique(y)
    if isinstance(y, pd.Series):
        return np.unique(y.values)
    if isinstance(y, list):
        return np.array(set(y))
    raise RuntimeError('unknown data type', type(y))    

def get_pipe(clf, enc=None, cat_atts=[], meta=None, cal_cv=5, cross_cv=5,\
             sampling_strategy=1/5, sl_X_train=None, sl_target=None, pca=False,\
             sc_quantiles=[.01, .05, .10, .15, .20, .25], ptest=0.018, seed = 42):
    ''' build pipe with encoding, classifiers, metaclassifiers, calibrators '''
    pipe = [] if not pca else [ ('scaler', StandardScaler()), ('reduce_dim', PCA())]
    if meta is not None:
        for act in meta.split('.'):
            if act=='cross':
                clf = AUCross(clf, cv=cross_cv, quantiles=sc_quantiles, seed=seed)
            elif act=='scross':
                clf = SCross(clf, cv=cross_cv, quantiles=sc_quantiles, seed=seed)
            elif act=='plugin':
                clf = PlugInRule(clf, quantiles=sc_quantiles, seed=seed)

            else:
                raise Exception('unknown calibration: '+act)
    pipe.append(('clf', clf))
    return Pipeline(pipe)
    


# abstract class for composition of classifiers
class CompClassifier(BaseEstimator, ClassifierMixin):
    
    def predict_proba(self, X, pred_contrib=False):
        if isinstance(self.clf_base, list):
            if pred_contrib:
                scaled = np.mean([clf.predict_proba(X, pred_contrib=True) for clf in self.clf_base], axis=0)
            else:
                scaled = np.mean([clf.predict_proba(X) for clf in self.clf_base], axis=0)
            return scaled
        if isinstance(self.clf_base, CompClassifier):
            return self.clf_base.predict_proba(X, pred_contrib=pred_contrib)
        if pred_contrib:
            return self.clf_base.predict_proba(X, pred_contrib=pred_contrib)
        return self.clf_base.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # test if classifier output error predictions    
    def has_err(self):
        """ true if it implements predict_err() """
        if self.clf_err is not None:
            return True
        if isinstance(self.clf_base, CompClassifier):
            return self.clf_base.has_err()
        if isinstance(self.clf_base, list):
            for clf in self.clf_base:
                if isinstance(clf, CompClassifier):
                    if clf.has_err():
                        continue
                return False
            return True
        return False

    def bound(self, y_pred):
        return np.where(y_pred>1,1,np.where(y_pred<0,0,y_pred))

    # get an estimate of absolute error of positive score
    def predict_err(self, X):
        if self.clf_err is not None:
            return np.mean([self.bound(clf.predict(X)) for clf in self.clf_err], axis=0)
        if isinstance(self.clf_base, CompClassifier):
            return self.clf_base.predict_err(X)
        return np.mean([clf.predict_err(X) for clf in self.clf_base], axis=0)
    
    # call as predict_rating( predict_proba(X_test)[:,1] ) to get rating with log binning
    def predict_rating(self, y_scores, n_bins=8):
        log_bins = np.logspace(-n_bins, 0, n_bins, base=2)  
        log_bins[-1] += 10**(-6) # to cover score=1
        return np.digitize(y_scores, log_bins)


        
#pluginrule
class PlugInRule(CompClassifier):
    """
    Class for PlugIn algorithm
    References
    """
    def __init__(self, clf_base, quantiles=[.01, .05, .10, .15, .20, .25], seed=42):
        self.quantiles = quantiles
        self.seed = seed
        self.thetas = np.zeros(len(quantiles))
        self.clf_base = [copy.deepcopy(clf_base)]
        self.clf_err = None 

    def fit(self, X, y, sample_weight=None):
        self.classes_ = unique(y)
        X_train, X_hold, y_train, y_hold = train_test_split(X,y, stratify=y, random_state=self.seed, test_size=0.1)
        self.clf_base[0].fit(X_train, y_train)
        # quantiles
        probas = self.clf_base[0].predict_proba(X_hold)
        confs = np.max(probas, axis=1)
        self.thetas = [np.quantile(confs, q) for q in self.quantiles]
            
    def qband(self, X):
        probas = self.predict_proba(X)
        confs = np.max(probas, axis=1)
        return np.digitize(confs, self.thetas)
        


#pluginruleAUC based
class PlugInRuleAUC(CompClassifier):
    """
    Class for PlugInAUC
    """
    def __init__(self, clf_base, quantiles=[.01, .05, .10, .15, .20, .25], seed=42):
        self.quantiles = quantiles
        self.seed = seed
        self.thetas = np.zeros(len(quantiles))
        self.clf_base = [copy.deepcopy(clf_base)]
        self.clf_err = None 
        
    def fit(self, X, y, sample_weight=None):
        self.classes_ = unique(y)
        localthetas = []
        X_train, X_hold, y_train, y_hold = train_test_split(X,y, stratify=y, random_state=self.seed, test_size=0.1)
        self.clf_base[0].fit(X_train, y_train)
        # quantiles
        y_scores = self.clf_base[0].predict_proba(X_hold)[:,1]
        auc_roc = skm.roc_auc_score(y_hold, y_scores)
        n, npos = len(y_hold), np.sum(y_hold)
        pneg = 1-np.mean(y_hold)
        u_pos = int(auc_roc*pneg*n)
        pos_sorted = np.argsort(y_scores)
        if isinstance(y_hold, pd.Series):
            tp = np.cumsum(y_hold.iloc[pos_sorted[::-1]])
        else:
            tp = np.cumsum(y_hold[pos_sorted[::-1]])
        l_pos = n-np.searchsorted(tp, auc_roc*npos+1, side='right')
        #print('Local bounds:', l_pos, '<= rank <=', u_pos, ' pct', (u_pos-l_pos+1)/n)
        #print('Local bounds:', y_scores[pos_sorted[l_pos]], '<= score <=', y_scores[pos_sorted[u_pos]])
        pos = (u_pos+l_pos)/2
        locallist = []
        for q in self.quantiles:
            delta = int(n*q/2)
            t1 = y_scores[pos_sorted[max(0,round(pos-delta))]]
            t2 = y_scores[pos_sorted[min(round(pos+delta), n-1)]]
            locallist.append( [t1, t2] )
            #print('Local thetas:', [t1, t2])
        print(locallist)
        self.thetas = locallist
            

    def qband(self, X):
        confs = self.predict_proba(X)[:,1]
        m = len(self.quantiles)
        res = np.zeros(len(confs))+m
        for i, t in enumerate(reversed(self.thetas)):
            t1, t2 = t[0], t[1]
            #print(i, t1, t2)
            res[ ((t1 <= confs) & (confs <= t2)) ] = m-i-1
        return res                              

        
class AUCross(CompClassifier):
    """
    Class for AUCross
    """
    def __init__(self, clf_base, cv=5, quantiles=[.01, .05, .10, .15, .20, .25], seed=42):
        self.cv = cv
        self.seed = seed
        self.quantiles = quantiles
        self.thetas = [None for _ in range(len(quantiles))]
        self.clf_base = [copy.deepcopy(clf_base) for _ in range(cv)]
        self.clf_err = None 
        self.clf_score = copy.deepcopy(clf_base)
                
    def fit(self, X, y, sample_weight=None):
        self.classes_ = unique(y)
        self.pneg = 1-np.mean(y)
        self.classes_ = unique(y)
        self.clf_base[0].fit(X, y)
        z = []
        localthetas = []
        idx = []
        skf = StratifiedKFold(n_splits=len(self.clf_base), shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
            else:
                X_train = X[train_index]
                X_test = X[test_index]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_index]
                # y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                # y_test = y[test_index]
            self.clf_base[i].fit(X_train, y_train)

            probas = self.clf_base[i].predict_proba(X_test)[:,1]
            z.append(probas)
            idx.append(test_index)
        self.clf_score.fit(X,y)
        z = np.concatenate(z).ravel()
        self.z = z
        idx = np.concatenate(idx).ravel()
        self.idx = idx
        if isinstance(y, pd.Series):
          sc = pd.DataFrame(np.c_[y.iloc[idx], z], columns=['y_true','y_scores'])
        else:
          sc = pd.DataFrame(np.c_[y[idx], z], columns=['y_true','y_scores'])
        sc.sort_index(inplace=True)
        sc1, sc2 = train_test_split(sc, stratify=sc['y_true'], test_size=.5)
        list_u = []
        list_l = []
        dict_q = {q:[] for q in self.quantiles}
        for db in [sc1, sc2, sc]:
            db = db.reset_index()
            auc_roc = skm.roc_auc_score(db['y_true'], db['y_scores'])
            n, npos = len(db['y_true']), np.sum(db['y_true'])
            pneg = 1-np.mean(db['y_true'])
            u_pos = int(auc_roc*pneg*n)
            pos_sorted = np.argsort(db['y_scores'])
            if isinstance(db['y_true'], pd.Series):
                tp = np.cumsum(db['y_true'].iloc[pos_sorted[::-1]])
            else:
                tp = np.cumsum(db['y_true'][pos_sorted[::-1]])
            l_pos = n-np.searchsorted(tp, auc_roc*npos+1, side='right')
            u = db['y_scores'][pos_sorted[u_pos]]
            l = db['y_scores'][pos_sorted[l_pos]]
            list_u.append(u)
            list_l.append(l)
            #print('Local bounds:', l_pos, '<= rank <=', u_pos, ' pct', (u_pos-l_pos+1)/n)
            #print('Local bounds:', y_scores[pos_sorted[l_pos]], '<= score <=', y_scores[pos_sorted[u_pos]])
        tau = 1/np.sqrt(2)
        u_star = list_u[2]*tau +(1-tau)*(.5*list_u[1]+.5*list_u[0])
        l_star = list_l[2]*tau +(1-tau)*(.5*list_l[1]+.5*list_l[0])
        pos = (u_star+l_star)*.5
        print(pos)
        sorted_scores = np.sort(self.z)
        base = np.searchsorted(sorted_scores, pos)
        for i,q in enumerate(self.quantiles):
                delta = int(n*q/2)
                l_b = max(0,round(base-delta))
                u_b = min(n-1, round(base+delta))
                t1 = sorted_scores[l_b]
                t2 = sorted_scores[u_b]
                #locallist.append( [t1, t2] )
                self.thetas[i] = [t1, t2]
                dict_q[q].append([t1,t2])
                print(t1,t2)
                #print('Local thetas:', [t1, t2])
        self.dict_q = dict_q
        """
        for i,q in enumerate((self.quantiles)):
            t1 = tau*(dict_q[q][0][0])+(1-tau)*(.5*dict_q[q][1][0]+.5*dict_q[q][2][0])
            t2 = tau*(dict_q[q][0][1])+(1-tau)*(.5*dict_q[q][1][1]+.5*dict_q[q][2][1])
            self.thetas[i] = [t1, t2]
            #print('Global thetas:', self.thetas[i])
         """  


    def predict_proba(self,X):
        scores = self.clf_score.predict_proba(X)
        return scores 
         
    def predict(self,X):
        scores = self.clf_score.predict(X)
        return scores  
        
    def qband(self, X):
        confs = self.predict_proba(X)[:,1]
        m = len(self.quantiles)
        res = np.zeros(len(confs))+m
        for i, t in enumerate(reversed(self.thetas)):
            t1, t2 = t[0], t[1]
            #print(i, t1, t2)
            res[ ((t1 <= confs) & (confs <= t2)) ] = m-i-1
        return res


class SCross(CompClassifier):
    """
    Class for SCrpss
    """
    def __init__(self, clf_base, cv=5, quantiles=[.01, .05, .10, .15, .20, .25], seed=42):
        self.cv = cv
        self.seed=seed
        self.quantiles = quantiles
        self.thetas = []
        self.clf_base = [copy.deepcopy(clf_base) for _ in range(cv)]
        self.clf_err = None 
        self.clf_score = copy.deepcopy(clf_base)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = unique(y)
        z = []
        localthetas = []
        skf = StratifiedKFold(n_splits=len(self.clf_base), shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
            else:
                X_train = X[train_index]
                X_test = X[test_index]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_index]
                # y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                # y_test = y[test_index]
            self.clf_base[i].fit(X_train, y_train)
            # quantiles
            probas = self.clf_base[i].predict_proba(X_test)
            confs = np.max(probas, axis=1)
            z.append(confs)
        self.z = z
        confs = np.concatenate(z).ravel()
        sub_confs_1, sub_confs_2 = train_test_split(confs, test_size=.5, random_state=42)
        tau = (1/np.sqrt(2))
        self.thetas = [(tau*np.quantile(confs, q) + (1-tau)*(.5*np.quantile(sub_confs_1, q)+.5*np.quantile(sub_confs_2, q))) for q in self.quantiles]
        self.clf_score.fit(X,y)
        
        
    def predict_proba(self,X):
        scores = self.clf_score.predict_proba(X)
        return scores 
         
    def predict(self,X):
        scores = self.clf_score.predict(X)
        return scores  
            
    def qband(self, X):
        probas = self.predict_proba(X)
        confs = np.max(probas, axis=1)
        return np.digitize(confs, self.thetas)
