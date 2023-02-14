import time
import random
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
import numpy as np
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import random


def cross_entropy_vectorized(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labaels.
    y_pred : Pytorch Tensor
        The predicted values from auxiliary head.

    Returns
    -------
    loss : Pytorch tensor
        The cross entropy loss for auxiliary head.

    """
    n_batch, n_class = y_pred.shape
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if len(y_true.shape)==1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    y_true = y_true.to(torch.int64)
    log_ypred = torch.log(torch.gather(y_pred,1,y_true)+0.00000001)
    loss = -(torch.sum(log_ypred))/n_batch
    return loss
    
def cross_entropy_selection_vectorized(y_true, hg, theta=.5, lamda=32, c=0.9):
    """
    

    Parameters
    ----------
    y_true : Pytorch Tensor
        The tensor with actual labels.
    hg : Pytorch Tensor
        The outputs of predictive head and selecticve head.
    theta : float, optional
        The threshold to make g(x)=1. The default is .5.
    lamda : float, optional
        Parameter to weigh the importance of constraint for coverage. The default is 32.
    c : float, optional
        The desired coverage. The default is 0.9.

    Returns
    -------
    loss : Pytorch Tensor
        The selective loss from Geifman et al. (2019).

    """
    n_batch, n_class = hg[:,:-1].shape
    if len(y_true.shape)==1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    if c==1:
        selected = n_batch
    else:
        selected = torch.sum(hg[:,-1])+0.00000001
    selection = torch.unsqueeze(hg[:,-1],dim=-1)
    y_true = y_true.to(torch.int64)
    log_ypred = torch.log(torch.gather(hg[:,:-1],1,y_true)+0.00000001)*selection
    loss = -((torch.sum(log_ypred))/(selected))+lamda*(max(0,c-(selected/n_batch)))**2
    
    return loss


def deep_gambler_loss(outputs, targets, reward):
    outputs = F.softmax(outputs, dim=1)
    outputs, reservation = outputs[:,:-1], outputs[:,-1]
    # gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
    gain = outputs[torch.arange(targets.shape[0]), targets]
    doubling_rate = (gain.add(reservation.div(reward))).log()
    return -doubling_rate.mean()


class SelfAdativeTraining():
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom
        self.num_classes = num_classes

    def _update_prob(self, prob, index, y):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y.shape[0]), y] = 1
        prob_history = self.prob_history[index].clone().to(prob.device)

        # if not inited, use onehot label to initialize runnning vector
        cond = (self.updated[index] == 1).to(prob.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot)

        # momentum update
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob

        self.updated[index] = 1
        self.prob_history[index] = prob_mom.to(self.prob_history.device)

        return prob_mom

    def __call__(self, logits, y, index):
        prob = F.softmax(logits.detach()[:, :self.num_classes], dim=1)
        prob = self._update_prob(prob, index, y)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
        soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]
        soft_label = F.normalize(soft_label, dim=1, p=1)
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        return torch.mean(loss)
        
class ImageLoaderExp(Dataset):
    def __init__(self, dataset, transform=None, resize=None, start=None, end=None, idx=[]):
        self.data = []  # some images are CMYK, Grayscale, check only RGB
        self.transform = transform
        self.idx = idx
        random.seed(42)
        if start == None:
            start = 0
        if end == None:
            end = dataset.__len__()
        if resize == None:
            if self.idx == []:
                for i in range(start, end):
                    self.data.append((dataset[i]))
            else:
                for i in self.idx:
                    self.data.append((dataset[i]))
        else:
            if self.idx == []:
                for i in range(start, end):
                    item = dataset[i]
                    random.seed(42)
                    torch.manual_seed(42)
                    self.data.append((transforms.functional.center_crop(
                        transforms.functional.resize(item[0], resize, InterpolationMode.BILINEAR), resize), item[1]))
            else:
                for i in self.idx:
                    item = dataset[i]
                    random.seed(42)
                    torch.manual_seed(42)
                    self.data.append((transforms.functional.center_crop(
                        transforms.functional.resize(item[0], resize, InterpolationMode.BILINEAR), resize), item[1]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.transform is not None:
            random.seed(42)
            torch.manual_seed(42)
            return self.transform(self.data[index][0]), self.data[index][1], index
        else:
            return self.data[index][0], self.data[index][1], index

        
def coverage_and_res(level, y, y_scores, y_pred, bands, perc_train = 0.2):
    """
    

    Parameters
    ----------
    classifier : sklearn.BaseEstimator
        A sklearn style classifier.
    level : int 
        The level for which we want to obtain the coverage.
    X : pd.DataFrame or np.array
        Test set features.
    y : pd.DataFrame or np.array
        Test set target.
    y_scores : np.array
        The scores predicted by classifier.
    y_pred : np.array
        The predicted class by classifier.
    perc_train : float, optional
        The percentage of positives in the train set. The default is 0.2.

    Returns
    -------
    coverage : float
        The empirical coverage.
    auc : float
        The empirical AUC.
    accuracy : float
        The selective accuracy.
    brier : float
        Brier score.
    bss : float
        BSS.
    perc_pos : float
        The positive rate.


    """
    covered = bands>=level
    coverage = len(y[covered])/len(y)
    y = y.astype(np.int64)
    if (np.sum(y[covered])>0) & (np.sum(y[covered])<len(y[covered])):
      try:
        auc = roc_auc_score(y[covered],y_scores[covered])
      except:
        import pdb; pdb.set_trace()
      accuracy = accuracy_score(y[covered], y_pred[covered])
      brier = brier_score_loss(y[covered], y_scores[covered])
      bss_denom = brier_score_loss(y[covered], np.repeat(perc_train, len(y))[covered])
      bss = 1-brier/bss_denom
      perc_pos = np.sum(y[covered])/len(y[covered])
    else:
      auc = 0
      accuracy = 0
      brier = -1
      bss = -1
      perc_pos = 0
    return coverage, auc, accuracy, brier, bss, perc_pos
        
def train_sat(model, device, trainloader, opt, max_epochs, pretrain,  num_examp, gamma=.5,  td=True,
              epochs_lr=[24,49,74,99,124,149,174,199,224,249,274], crit = 'sat', reward=2):
    """


    device : torch.device
        The device over which training will be performed.
    model : torch.module
        The network architecture to train.
    trainloader : torch.dataset
        The training dataset.
    opt : torch.optimizer
        The optimizer to perform network training.
    max_epochs : int
        The number of epochs.
    pretrain : int
        The number of epochs for pretraining in SAT.
    num_examples: int
        The number of examples to use for training.

    gamma: float
        A value to determine the drop over epochs. The default is .5
    td : bool
        A boolean value to determine whether the learning rate drops over epochs. The default is True.
    epochs_lr : list, optional
        The epochs when we perform the change in learning rate. The default is every 25 epochs until 300 epochs.
    crit : str
        The criterion for loss. The options are:
            - 'sat' for SelfAdaptiveTraining, num_examples=50000,
            - 'ce' for Cross Entropy
        The default is 'sat'.

    Returns
    -------

    """
    model.to(device)
    running_loss = 0
    for epoch in range(1, max_epochs+1):
        model.train()
        if td:
            if epoch in epochs_lr:
                for param_group in opt.param_groups:
                    param_group['lr'] *= gamma
                    print("\n: lr now is: {}\n".format(param_group['lr']))
            with tqdm(trainloader, unit="batch") as tepoch:
                for i,batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    X_cont, y, indices = batch           
                    X_cont, y, indices = X_cont.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        opt.zero_grad()
                        outputs = model.forward(X_cont)
                        if crit=='sat':
                            if (epoch==1)& (i==0):
                                print("\n criterion is {} \n".format(crit))
                            if epoch>pretrain:
                                if (epoch==pretrain+1)&(i==0):
                                    print("switching to Adaptive")
                                criterion = SelfAdativeTraining(num_examples=num_examp, num_classes=model.output_dim, mom=.99)
                                loss = criterion(outputs, y, indices)
                            else:
                                loss = torch.nn.functional.cross_entropy(outputs[:, :-1], y)
                        elif crit=='ce':
                            if (epoch==1) & (i==0):
                                print("\n criterion is {} \n".format(crit)) 
                            loss = torch.nn.functional.cross_entropy(outputs, y)
                        elif crit=='dg':
                            if (epoch==1)& (i==0):
                                print("\n criterion is {} \n".format(crit))
                            loss = deep_gambler_loss(outputs, y, reward)
                        loss.backward()
                        opt.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(trainloader) + i + 1
                            running_loss = 0.
                        tepoch.set_postfix(loss=loss.item())
def train_sel(model, device, trainloader, opt, max_epochs, coverage, alpha=.5, lamda=32,
              td=True, gamma =.5, epochs_lr=[24,49,74,99,124,149,174,199,224,249,274]):
    model.to(device)
    running_loss = 0
    for epoch in range(1, max_epochs+1):
        model.train()
        if td:
            if epoch in epochs_lr:
                for param_group in opt.param_groups:
                    param_group['lr'] *= gamma
                    print("\n: lr now is: {}\n".format(param_group['lr']))
            with tqdm(trainloader, unit="batch") as tepoch:
                for i,batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    X_cont, y, indices = batch           
                    X_cont, y, indices = X_cont.to(device), y.to(device), indices.to(device)
                    if len(y)==1:
                        pass
                    else:
                        opt.zero_grad()
                        hg, aux = model.forward(X_cont)
                        loss1 = cross_entropy_selection_vectorized(y,hg, lamda=lamda, c=coverage)
                        loss2 = cross_entropy_vectorized(y,aux)
                        if coverage ==1:
                            loss = loss2
                        else:
                            loss = (alpha*loss1)+((1-alpha)*loss2)
                        loss.backward()
                        opt.step()
                        running_loss += loss.item()
                        if i % 1000 == 10:
                            last_loss = running_loss / 1000 # loss per batch
            #                 print('  batch {} loss: {}'.format(i + 1, last_loss))
                            tb_x = epoch * len(trainloader) + i + 1
                            running_loss = 0.
                        tepoch.set_postfix(loss=loss.item())
                        
def get_scores(model, device, test_dl, coverage=.9, crit='sat'):
    """

    Parameters
    ----------
    model : torch.module
        The network architecture to be used for prediction
    device : torch.device
        The device used.
    test_dl : torch.datasets.DataLoader
        The set over which scores are computed.
    coverage : float, optional
        The target coverage. Used if model.selective is True. The default is .9
    crit : str
        The criterion used to train the model.
        The default is 'sat'.

    Returns
    -------
    y_hat : np.array
        An array containing scores
    """
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            if crit in ['sat','dg']:
                outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
            elif crit=='ce':
                outputs = torch.nn.functional.softmax(outputs, dim=1)
            score = outputs.detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            if coverage==1:
                score = aux.detach().cpu().numpy()
            else:
                score = hg[:,:-1].detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    return y_hat

def get_preds(model, device, test_dl, coverage=.9, crit='sat'):
    """
        Parameters
    ----------
    model : torch.module
        The network architecture to be used for prediction
    device : torch.device
        The device used.
    test_dl : torch.datasets.DataLoader
        The set over which predictions are computed.
    coverage : float, optional
        The target coverage. Used if model.selective is True. The default is .9
    crit : str
        The criterion used to train the model.
        The default is 'sat'.

    Returns
    -------
    preds : np.array
        An array containing predictions
    """
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            if crit in ['sat','dg']:
                outputs = torch.nn.functional.softmax(outputs[:,:-1], dim=1)
            elif crit=='ce':
                outputs = torch.nn.functional.softmax(outputs, dim=1)
            score = outputs.detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            if coverage==1:
                score = aux.detach().cpu().numpy()
            else:
                score = hg[:,:-1].detach().cpu().numpy()
            scores.append(score)
        y_hat = np.vstack(scores)
    preds = np.argmax(y_hat, axis=1)
    return preds

def get_confs(model, device, test_dl):
    """

    Parameters
    ----------
    model : torch.module
        The network architecture to be used for prediction
    device : torch.device
        The device used.
    test_dl : torch.datasets.DataLoader
        The set over which confidences are computed.
    Returns
    -------
    y_hat : np.array
        An array containing confidence values.
    """
    model.eval()
    scores = []
    model.to(device)
    if model.selective==False:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            outputs = model(x_cont)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            conf = outputs[:,-1].detach().cpu().numpy().reshape(-1,1)
            scores.append(conf)
        y_hat = np.vstack(scores).flatten()
    else:
        for batch in test_dl:
            x_cont, y, indices = batch[0], batch[1], batch[2]
            x_cont  = x_cont.to(device)
            hg, aux = model(x_cont)
            score = hg[:,-1].detach().cpu().numpy().reshape(-1,1)
            scores.append(score)
        y_hat = np.vstack(scores).flatten()
    return y_hat
                        
def get_theta(model, device, validloader, meta, coverage=.9, quantiles = [.01,.05,.1,.15,.2,.25]):
    """
    Function used to get theta values for selection function.

    Parameters
    ----------
    model : torch.module
        The network architecture to be used for prediction
    device : torch.device
        The device used
    validloader : torch.datasets.DataLoader
        A dataloader for the validation set used to calibrate selection function
    meta : str
        The strategy for which we need to compute theta. Possible choices are:
            - 'selnet' for SelectiveNet
            - 'plugin' for PlugIn
            - 'pluginAUC for PlugInAUC
            - 'sat' for SAT

    coverage : float
        The coverage for which selnet has to be tuned
    quantiles
        The quantiles to be computed.
    Returns
    -------
    theta : float
        The single theta value for specific coverage if meta is selnet
    OR
    thetas : list
        The list of values for list of coverages if meta is one of plugin pluginAUC sat
    """
    if meta=='selnet':
        tmp = get_confs(model,device,validloader)
        theta = np.quantile(tmp, 1-coverage, method='nearest')
        return theta
    elif meta in ['sat','dg']:
        tmp = get_confs(model, device, validloader)
        thetas = [np.quantile(tmp, 1-cov,method='nearest') for cov in sorted(quantiles, reverse=True)]
        return thetas
    elif meta=='plugin':
        scores = get_scores(model,device,validloader, crit='ce')
        tmp = np.max(scores, axis=1)
        thetas = [np.quantile(tmp, 1-cov, method='nearest') for cov in quantiles]
        return thetas
    elif meta=='pluginAUC':
        y_hold = get_true(validloader)
        y_scores = get_scores(model,device,validloader, crit='ce')[:,1]
        auc_roc = roc_auc_score(y_hold, y_scores)
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
        thetas = []
        for q in quantiles:
            delta = int(n*q/2)
            t1 = y_scores[pos_sorted[max(0,round(pos-delta))]]
            t2 = y_scores[pos_sorted[min(round(pos+delta), n-1)]]
            thetas.append( [t1, t2] )
            #print('Local thetas:', [t1, t2])
        return thetas
    
def get_true(testloader):
    """

    Parameters
    ----------
    testloader : torch.datasets.DataLoader
        The dataloader for a set

    Returns
    -------
    y_true : np.array
        The true label values
    """
    y_true = []
    for batch in testloader:
        y = batch[1].detach().cpu().numpy().reshape(-1,1)
        y_true.append(y)
    y_true = np.vstack(y_true).flatten()
    return y_true

def qband(model, device, testloader, meta, thetas):
    """

    Parameters
    ----------
    model : torch.module
        The network architecture to be used for prediction
    device : torch.device
        The device used
    testloader : torch.datasets.DataLoader
        A dataloader for the validation set used to calibrate selection function
    meta : str
        The strategy for which we need to compute theta. Possible choices are:
            - 'selnet' for SelectiveNet
            - 'plugin' for PlugIn
            - 'pluginAUC for PlugInAUC
            - 'sat' for SAT
    thetas : the value/values of theta/thetas to compute levels

    Returns
    -------
    band : np.array
        The values for levels over set
    """
    if len(thetas)==1:
        band = get_confs(model, device, testloader)
        return np.where(band > thetas, 1, 0)
    else:
        if meta in ['sat', 'dg']:
            band = np.digitize(get_confs(model, device, testloader), sorted(thetas, reverse=True), right=True)
        
            return band

