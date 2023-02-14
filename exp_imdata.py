import random

import torch.utils.data

from utils_catsdogs_ import *
from model2 import *
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse

def main(metas, cv=5, max_epochs=300, lr=.1, boot_iter=1000, td=True,
         filename='cats_dogs', quantiles=[.01, .05, .1, .15, .2, .25], cuda_dev=0, num_workers=8, add_epochs=0):
    """

    Parameters
    ----------
    metas:  list
            the list of metas available. Possible choices are:
                -'aucross' for AUCross algorithm
                -'scross' for SCross algorithm
                -'plugin' for PlugIn algorithm
                -'pluginAUC' for PlugInAUC algorithm
                -'sat' for Self Adaptive Training
                -'selnet' for SelectiveNet
    cv: int
        the number of K folds for SCross and AUCross. Default is 5.
    max_epochs: int
        the number of epochs used to train neural networks. Default is 300
    lr: float
        the learning rate to be used in training. Default is .1
    boot_iter: int
        the number of bootstrap iterations. Default is 1000.
    td: bool
        value to specify whether to use or not Time Decay in training. Default is True
    filename: str
        the name of the dataset. Default is 'cats_dogs'.
    quantiles: list
        the list of quantiles to compute. Default is [.01, .05, .1, .15, .2, .25]
    cuda_dev: int
        the cuda device to use. Default is 0.
    num_workers: int
        the number of workers to use in dataloaders. Default is 8.
    add_epochs: int
        the number of additional epochs to use in training nets on the folds. Default is 0.
    Returns
    -------

    """
    torch.manual_seed(42)
    random.seed(42)
    print(max_epochs)
    print(boot_iter)
    use_cuda = torch.cuda.is_available()
    # Random seed
    random.seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)
    if filename == 'cats_dogs':
        # set number of classes
        num_classes = 2
        # set input size
        input_size = 64
        # set transformations over images
        transform_train = transforms.Compose([
            transforms.RandomCrop(input_size, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # use the "ImageFolder" datasets
        assert os.path.exists("./data/{}/train".format(filename)) and os.path.exists("./data/{}/test".format(
            filename)), "Please download and put the 'cats vs dogs' dataset to paths 'data/cats_dogs/train' and 'data/cats_dogs/test'"
        # build training set
        trainset = datasets.ImageFolder("./data/{}/train".format(filename))
        # build test set
        testset = datasets.ImageFolder("./data/{}/test".format(filename))
        perc_train = .5
    elif filename == 'cifar10':
        # set number of classes
        num_classes = 2
        # set input size
        input_size = 32
        # set transformations over images
        transform_train = transforms.Compose([
            transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # use the "ImageFolder" datasets
        assert os.path.exists("./data/{}/train".format(filename)) and os.path.exists("./data/{}/test".format(
            filename)), "Please download and put the 'cifar-10' dataset to paths 'data/cifar10/train' and 'data/cifar10/test'"
        # use only cat as target variable
        cats_dict = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        # build training set
        trainset = datasets.ImageFolder('./data/{}/train'.format(filename), target_transform=lambda x: cats_dict[x])
        # build test set
        testset = datasets.ImageFolder('./data/{}/test'.format(filename), target_transform=lambda x: cats_dict[x])
        perc_train = .1
    # define indexes
    indexes = [el for el in range(len(trainset))]
    # define split of validation and training set
    train_idx_, valid_idx_ = train_test_split(indexes, random_state=42, test_size=.1)
    # define training image loader
    trs_ = ImageLoaderExp(trainset, transform=transform_train, idx=train_idx_, resize=input_size)
    vld_ = ImageLoaderExp(trainset, transform=transform_test, idx=valid_idx_, resize=input_size)
    tes_ = ImageLoaderExp(testset, transform=transform_test, resize=input_size)
    trainloader_ = torch.utils.data.DataLoader(trs_, batch_size=128, shuffle=True, num_workers=num_workers)
    validloader_ = torch.utils.data.DataLoader(vld_, batch_size=128, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(tes_, batch_size=128, shuffle=False, num_workers=num_workers)
    num_examp = len(trainset)
    trs = ImageLoaderExp(trainset, transform=transform_train, resize=input_size)
    # tes = ImageLoaderExp(testset, transform=transform_test, resize=input_size)
    trainloader = torch.utils.data.DataLoader(trs, batch_size=128, shuffle=True, num_workers=num_workers)
    # testloader = torch.utils.data.DataLoader(tes, batch_size=128, shuffle=False, num_workers=num_workers)
    device = torch.device('cuda:{}'.format(cuda_dev) if torch.cuda.is_available() else 'cpu')
    print(device)
    coverages = [1 - q for q in quantiles]
    for meta in tqdm(metas):
        # define results dataframe
        results = pd.DataFrame()
        if meta == 'selnet':
            #training selnet
            model_dict = {k: sel_vgg16_bn(selective=True, output_dim=num_classes, input_size=input_size)
                          for k in coverages}
            thetas_dict = {k: .5 for k in coverages}
            for k in coverages:
                optimizer = torch.optim.SGD
                if td == True:
                    opt = optimizer(model_dict[k].parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                else:
                    opt = optimizer(model_dict[k].parameters(), lr, momentum=0.9, nesterov=True)
                start_time = time()
                path_model = 'models/SelnetVGG16_coverage{}_{}_{}_cuda{}.pt'.format(k, max_epochs, filename, cuda_dev)
                if os.path.exists(path_model):
                    model_dict[k].load_state_dict(torch.load(path_model))
                    model_dict[k].eval()
                else:
                    train_sel(model_dict[k], device=device, trainloader=trainloader_, max_epochs=max_epochs, opt=opt,
                              td=td,
                              coverage=k, alpha=.5, lamda=32)
                    torch.save(model_dict[k].state_dict(), path_model)
                end_time = time()
                time_to_fit = end_time - start_time
                theta = get_theta(model_dict[k], device, validloader_, meta, coverage=k)
                thetas_dict[k] = theta
                scores = get_scores(model_dict[k], device, testloader)[:, 1]
                preds = get_preds(model_dict[k], device, testloader)
                confs = get_confs(model_dict[k], device, testloader)
                y_test = get_true(testloader)
                covered = confs > theta
                y_test = y_test.astype(np.int64)
                yy = pd.DataFrame(np.c_[y_test, scores, preds, covered], columns=['true', 'scores', 'preds', 'select'])
                for b in range(boot_iter + 1):
                    if b == 0:
                        db = yy.copy()
                    else:
                        db = yy.sample(len(y_test), random_state=b, replace=True)
                    db = db.reset_index()
                    if (np.sum(db['select'] > theta) == 0):
                        auc = 0
                        acc = 0
                        perc_pos = 0
                        coverage = 0
                        bss = -1
                        brier = 1
                    elif (np.sum(db['true'][db['select'] > theta]) == 0) | (
                            np.sum(db['true'][db['select'] > theta]) == len(y_test)):
                        auc = 0
                        acc = 0
                        perc_pos = 0
                        coverage = 0
                        bss = -1
                        brier = 1
                    else:
                        auc = roc_auc_score(db['true'][db['select'] > theta], db['scores'][db['select'] > theta])
                        acc = accuracy_score(db['true'][db['select'] > theta], db['preds'][db['select'] > theta])
                        perc_pos = np.sum(db['true'][db['select'] > theta]) / len(db['true'][db['select'] > theta])
                        coverage = len(db['true'][db['select'] > theta]) / len(db['true'])
                        ### compute brier
                        brier = brier_score_loss(db['true'][covered], db['scores'][covered])
                        ### compute brier score adj
                        bss_denom = brier_score_loss(db['true'][covered],
                                                     np.repeat(perc_train, len(db['true']))[covered])
                        bss = 1 - brier / bss_denom
                    res_cols = ['dataset', 'desired_coverage', 'coverage', 'accuracy', 'auc', 'perc_pos', 'brier',
                                'bss']
                    tmp = pd.DataFrame([[filename, k, coverage, acc, auc, perc_pos, brier, bss ]], columns=res_cols)
                    tmp['meta'] = meta
                    tmp['boot_iter'] = b
                    tmp['model'] = "{}_{}".format(meta, 'vgg')
                    tmp['time_to_fit'] = time_to_fit
                    tmp['thetas'] = theta
                    tmp = tmp[['dataset', 'desired_coverage', 'coverage', 'accuracy', 'auc', 'perc_pos', 'brier',
                               'bss', 'time_to_fit', 'model',  'boot_iter', 'thetas']].copy()
                    results = pd.concat([results, tmp], axis=0)
                    results.to_csv("results/results_{}_{}_{}_cuda{}.csv".format(meta, filename, boot_iter, cuda_dev))
        else:
            if meta == 'sat':
                #training SAT
                model = sel_vgg16_bn(selective=False, output_dim = num_classes+1, input_size=input_size)
                optimizer = torch.optim.SGD
                if td == True:
                    opt = optimizer(model.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                else:
                    opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
                path_model = 'models/SATVGG16_{}_{}_{}_cuda{}.pt'.format(max_epochs, boot_iter, filename, cuda_dev)
                start_time = time()
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model))
                    model.eval()
                else:
                    train_sat(model=model, device=device, trainloader=trainloader_, opt=opt, max_epochs=max_epochs,
                          pretrain=0, num_examp=len(trainset))
                    torch.save(model.state_dict(), path_model)
                end_time = time()
                scores = get_scores(model, device, testloader)[:,1]
                preds = get_preds(model, device, testloader)
                confs = get_confs(model, device, testloader)
                thetas = get_theta(model,device,validloader_, meta=meta)
                y_test = get_true(testloader)
                bands = qband(model, device, testloader, meta=meta,thetas=thetas)
                time_to_fit = (end_time - start_time)
                yy = pd.DataFrame(np.c_[y_test,scores, preds, bands], columns = ['true', 'scores','preds', 'bands'])
            elif meta == 'plugin':
                #training PlugIn
                model = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                results = pd.DataFrame()
                optimizer = torch.optim.SGD
                start_time = time()
                path_model = 'models/PlugInVGG16_Score_{}_{}_cuda{}_TEST.pt'.format(max_epochs, filename, cuda_dev)
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model))
                    model.eval()
                else:
                    if td == True:
                        opt = optimizer(model.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                    else:
                        opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
                    train_sat(model=model, device=device, trainloader=trainloader_, opt=opt, max_epochs=max_epochs,
                              pretrain=max_epochs, num_examp=num_examp, crit='ce')
                    torch.save(model.state_dict(), path_model)
                end_time = time()
                time_to_fit = (end_time - start_time)
                thetas = get_theta(model, device, validloader_, meta)
                scores = get_scores(model, device, testloader, crit='ce')
                confs = np.max(scores, axis=1)
                bands = np.digitize(confs, thetas)
                preds = get_preds(model, device, testloader, crit='ce')
                y_test = get_true(testloader)
                time_to_fit = (end_time - start_time)
                yy = pd.DataFrame(np.c_[y_test, scores[:, 1], preds, bands], columns=['true', 'scores', 'preds', 'bands'])
            elif meta == 'pluginAUC':
                #training plugInAUC
                model = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                optimizer = torch.optim.SGD
                start_time = time()
                path_model = 'models/PlugInAUCVGG16_Score_{}_{}_cuda{}_TEST.pt'.format(max_epochs, filename, cuda_dev)
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model))
                    model.eval()
                else:
                    if td == True:
                        opt = optimizer(model.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                    else:
                        opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
                    train_sat(model=model, device=device, trainloader=trainloader_, opt=opt, max_epochs=max_epochs,
                              pretrain=max_epochs, num_examp=num_examp, crit='ce')
                    torch.save(model.state_dict(), path_model)
                end_time = time()
                time_to_fit = (end_time - start_time)
                thetas = get_theta(model, device, validloader_, meta)
                scores = get_scores(model, device, testloader, crit='ce')
                m = len(quantiles)
                res = np.zeros(len(scores)) + m
                for i, t in enumerate(reversed(thetas)):
                    t1, t2 = t[0], t[1]
                    # print(i, t1, t2)
                    res[((t1 <= scores[:, 1]) & (scores[:, 1] <= t2))] = m - i - 1
                bands = res
                preds = get_preds(model, device, testloader, crit='ce')
                y_test = get_true(testloader)
                time_to_fit = (end_time - start_time)
                yy = pd.DataFrame(np.c_[y_test, scores[:, 1], preds, bands], columns=['true', 'scores', 'preds', 'bands'])
            elif meta == 'aucross':
                #training AUCross
                skf = StratifiedKFold(cv, shuffle=True, random_state=42)
                # we initialize thetas
                thetas = []
                # we initialize empty lists
                z = []
                idx = []
                #get targets
                yx = np.array([el[1] for el in trs.data])
                ys = []
                start_time = time()
                for num, d in enumerate(skf.split(indexes, yx)):
                    if filename=='cifar10':
                        trainset = datasets.ImageFolder('./data/{}/train'.format(filename),
                                                        target_transform=lambda x: cats_dict[x])
                        # build test set
                        testset = datasets.ImageFolder('./data/{}/test'.format(filename),
                                                       target_transform=lambda x: cats_dict[x])
                    elif filename=='cats_dogs':
                        trainset = datasets.ImageFolder('./data/{}/train'.format(filename))
                        # build test set
                        testset = datasets.ImageFolder('./data/{}/test'.format(filename))
                    # define indexes
                    train_idx__ = d[0].copy()
                    valid_idx__ = d[1]
                    random.shuffle(train_idx__)
                    # define training image loader
                    trs_k = ImageLoaderExp(trainset, transform=transform_train, idx=train_idx__, resize=input_size)
                    vld_k = ImageLoaderExp(trainset, transform=transform_test, idx=valid_idx__, resize=input_size)
                    train_dl = torch.utils.data.DataLoader(trs_k, batch_size=128, shuffle=True, num_workers=num_workers)
                    valid_dl = torch.utils.data.DataLoader(vld_k, batch_size=128, shuffle=False, num_workers=num_workers)
                    num_examp = len(d[0])
                    # we specify path model to see if they already exist, otherwise we train them
                    path_model = 'models/AUCrossVGG16TEST_fold{}_{}_{}_{}_cuda{}.pt'.format(num, max_epochs+add_epochs, cv,
                                                                                        filename, cuda_dev)
                    model_ = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                    optimizer = torch.optim.SGD
                    if os.path.exists(path_model):
                        model_.load_state_dict(torch.load(path_model))
                        model_.eval()
                    else:
                        if td == True:
                            opt = optimizer(model_.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                        else:
                            opt = optimizer(model_.parameters(), lr, momentum=0.9, nesterov=True)
                        train_sat(model=model_, device=device, trainloader=train_dl, opt=opt, max_epochs=max_epochs,
                                  pretrain=max_epochs, num_examp=num_examp, crit='ce')
                        torch.save(model_.state_dict(), path_model)
                    # we get scores
                    scores = get_scores(model_, device, valid_dl, crit='ce')[:, 1]
                    y_true = get_true(valid_dl)
                    # we get the true values
                    # here we store the scores and ids
                    print(roc_auc_score(y_true, scores))
                    z.append(scores)
                    idx.append(d[1])
                    ys.append(y_true)
                z = np.concatenate(z).ravel()
                idxes = np.concatenate(idx).ravel()
                ys = np.concatenate(ys).ravel()
                # we build full sample and then we split it in two parts for quantile estimates as in Theorem 4
                sc = pd.DataFrame(np.c_[ys, z], columns=['y_true', 'y_scores'])
                print(sc['y_true'].unique())
                sc.sort_index(inplace=True)
                sc1, sc2 = train_test_split(sc, stratify=sc['y_true'], test_size=.5, random_state=42)
                list_u = []
                list_l = []
                dict_q = {q: [] for q in quantiles}
                for db in [sc1, sc2, sc]:
                    db = db.reset_index()
                    auc_roc = roc_auc_score(db['y_true'], db['y_scores'])
                    n, npos = len(db['y_true']), np.sum(db['y_true'])
                    pneg = 1 - np.mean(db['y_true'])
                    u_pos = int(auc_roc * pneg * n)
                    pos_sorted = np.argsort(db['y_scores'])
                    if isinstance(db['y_true'], pd.Series):
                        tp = np.cumsum(db['y_true'].iloc[pos_sorted[::-1]])
                    else:
                        tp = np.cumsum(db['y_true'][pos_sorted[::-1]])
                    l_pos = n - np.searchsorted(tp, auc_roc * npos + 1, side='right')
                    u = db['y_scores'][pos_sorted[u_pos]]
                    l = db['y_scores'][pos_sorted[l_pos]]
                    list_u.append(u)
                    list_l.append(l)
                # better estimate
                tau = 1 / np.sqrt(2)
                u_star = list_u[2] * tau + (1 - tau) * (.5 * list_u[1] + .5 * list_u[0])
                l_star = list_l[2] * tau + (1 - tau) * (.5 * list_l[1] + .5 * list_l[0])
                pos = (u_star + l_star) * .5
                print(pos)
                sorted_scores = np.sort(z)
                base = np.searchsorted(sorted_scores, pos)
                for i, q in enumerate(quantiles):
                    delta = int(n * q / 2)
                    l_b = max(0, round(base - delta))
                    u_b = min(n - 1, round(base + delta))
                    t1 = sorted_scores[l_b]
                    t2 = sorted_scores[u_b]
                    # locallist.append( [t1, t2] )
                    thetas.append([t1, t2])
                    dict_q[q].append([t1, t2])
                    print(t1, t2)
                # we build a model for final score
                num_examp = len(z)
                if filename == 'cifar10':
                    trainset = datasets.ImageFolder('./data/{}/train'.format(filename),
                                                    target_transform=lambda x: cats_dict[x])
                    # build test set
                    testset = datasets.ImageFolder('./data/{}/test'.format(filename),
                                                   target_transform=lambda x: cats_dict[x])
                elif filename == 'cats_dogs':
                    trainset = datasets.ImageFolder('./data/{}/train'.format(filename))
                    # build test set
                    testset = datasets.ImageFolder('./data/{}/test'.format(filename))
                # define indexes
                # define training image loader
                trs_k = ImageLoaderExp(trainset, transform=transform_train, resize=input_size)
                # vld_k = ImageLoaderExp(trainset, transform=transform_test, resize=input_size)
                trainloader = torch.utils.data.DataLoader(trs_k, batch_size=128, shuffle=True, num_workers=num_workers)
                model = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                optimizer = torch.optim.SGD
                path_model = 'models/AUCrossVGG16TEST_Score_{}_{}_{}_cuda{}.pt'.format(max_epochs, filename, cv, cuda_dev)
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model))
                    model.eval()
                else:
                    if td == True:
                        opt = optimizer(model.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                    else:
                        opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
                    train_sat(model=model, device=device, trainloader=trainloader, opt=opt, max_epochs=max_epochs,
                              pretrain=max_epochs, num_examp=num_examp, crit='ce')
                    torch.save(model.state_dict(), path_model)
                end_time = time()
                scores = get_scores(model, device, testloader, crit='ce')[:, 1]
                m = len(quantiles)
                # compute bands
                bands = np.zeros(len(scores)) + m
                for i, t in enumerate(reversed(thetas)):
                    t1, t2 = t[0], t[1]
                    bands[((t1 <= scores) & (scores <= t2))] = m - i - 1
                preds = get_preds(model, device, testloader, crit='ce')
                y_test = get_true(testloader)
                time_to_fit = (end_time - start_time)
                yy = pd.DataFrame(np.c_[y_test, scores, preds, bands], columns=['true', 'scores', 'preds', 'bands'])
            elif meta == 'scross':
                #training SCross
                skf = StratifiedKFold(cv, shuffle=True, random_state=42)
                # we initialize empty lists
                z = []
                idx = []
                yx = np.array([el[1] for el in trs.data])
                ys = []
                start_time = time()
                for num, d in enumerate(skf.split(indexes, yx)):
                    if filename=='cifar10':
                        trainset = datasets.ImageFolder('./data/{}/train'.format(filename),
                                                        target_transform=lambda x: cats_dict[x])
                        # build test set
                        testset = datasets.ImageFolder('./data/{}/test'.format(filename),
                                                       target_transform=lambda x: cats_dict[x])
                    elif filename=='cats_dogs':
                        trainset = datasets.ImageFolder('./data/{}/train'.format(filename))
                        # build test set
                        testset = datasets.ImageFolder('./data/{}/test'.format(filename))
                    # define indexes
                    train_idx__ = d[0].copy()
                    valid_idx__ = d[1]
                    random.shuffle(train_idx__)
                    # define training image loader
                    trs_k = ImageLoaderExp(trainset, transform=transform_train, idx=train_idx__, resize=input_size)
                    vld_k = ImageLoaderExp(trainset, transform=transform_test, idx=valid_idx__, resize=input_size)
                    train_dl = torch.utils.data.DataLoader(trs_k, batch_size=128, shuffle=True, num_workers=num_workers)
                    valid_dl = torch.utils.data.DataLoader(vld_k, batch_size=128, shuffle=False, num_workers=num_workers)
                    num_examp = len(d[0])

                    # we specify path model to see if they already exist, otherwise we train them
                    path_model = 'models/SCrossVGG16TEST_fold{}_{}_{}_{}_cuda{}.pt'.format(num, max_epochs + add_epochs,
                                                                                        cv,
                                                                                        filename, cuda_dev)
                    model_ = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                    optimizer = torch.optim.SGD
                    y_train = get_true(trainloader_)
                    if os.path.exists(path_model):
                        model_.load_state_dict(torch.load(path_model))
                        model_.eval()
                    else:
                        if td == True:
                            opt = optimizer(model_.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                        else:
                            opt = optimizer(model_.parameters(), lr, momentum=0.9, nesterov=True)
                        train_sat(model=model_, device=device, trainloader=train_dl, opt=opt, max_epochs=max_epochs,
                                  pretrain=max_epochs, num_examp=num_examp, crit='ce')
                        torch.save(model_.state_dict(), path_model)
                    # we get scores
                    scores = get_scores(model_, device, valid_dl, crit='ce')
                    confs = np.max(scores, axis=1)
                    ys.append(get_true(valid_dl))
                    z.append(confs)
                confs = np.concatenate(z).ravel()
                sub_confs_1, sub_confs_2 = train_test_split(confs, test_size=.5, random_state=42)
                tau = (1 / np.sqrt(2))
                thetas = [(tau * np.quantile(confs, q) +
                           (1 - tau) * (.5 * np.quantile(sub_confs_1, q) +
                                        .5 * np.quantile(sub_confs_2, q))) for q in quantiles]

                num_examp = len(z)
                if filename == 'cifar10':
                    trainset = datasets.ImageFolder('./data/{}/train'.format(filename),
                                                    target_transform=lambda x: cats_dict[x])
                    # build test set
                    testset = datasets.ImageFolder('./data/{}/test'.format(filename),
                                                   target_transform=lambda x: cats_dict[x])
                elif filename == 'cats_dogs':
                    trainset = datasets.ImageFolder('./data/{}/train'.format(filename))
                    # build test set
                    testset = datasets.ImageFolder('./data/{}/test'.format(filename))
                # define indexes
                # define training image loader
                trs_k = ImageLoaderExp(trainset, transform=transform_train, resize=input_size)
                # vld_k = ImageLoaderExp(trainset, transform=transform_test, resize=input_size)
                trainloader = torch.utils.data.DataLoader(trs_k, batch_size=128, shuffle=True, num_workers=num_workers)
                model = sel_vgg16_bn(selective=False, output_dim=num_classes, input_size=input_size)
                optimizer = torch.optim.SGD
                path_model = 'models/SCrossVGGTEST_Score_{}_{}_{}_cuda{}.pt'.format(max_epochs+add_epochs, filename, cv, cuda_dev)
                if os.path.exists(path_model):
                    model.load_state_dict(torch.load(path_model))
                    model.eval()
                else:
                    if td == True:
                        opt = optimizer(model.parameters(), lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
                    else:
                        opt = optimizer(model.parameters(), lr, momentum=0.9, nesterov=True)
                    train_sat(model=model, device=device, trainloader=trainloader, opt=opt, max_epochs=max_epochs,
                              pretrain=max_epochs, num_examp=num_examp, crit='ce')
                    torch.save(model.state_dict(), path_model)
                end_time = time()
                scores = get_scores(model, device, testloader, crit='ce')
                confs = np.max(scores, axis=1)
                bands = np.digitize(confs, thetas)
                preds = get_preds(model, device, testloader, crit='ce')
                y_test = get_true(testloader)
                time_to_fit = (end_time - start_time)
                yy = pd.DataFrame(np.c_[y_test, scores[:, 1], preds, bands],
                                  columns=['true', 'scores', 'preds', 'bands'])
            for b in range(boot_iter + 1):
                    if b == 0:
                        db = yy.copy()
                    else:
                        db = yy.sample(len(yy), random_state=b, replace=True)
                    db = db.reset_index()
                    res = [coverage_and_res(level, db['true'].values, db['scores'].values, db['preds'].values,
                                            db['bands'].values, perc_train=perc_train)
                           for level in range(len(quantiles) + 1)]
                    tmp = pd.DataFrame(res, columns=['coverage', 'auc',
                                                     'accuracy', 'brier', 'bss', 'perc_pos'])
                    tmp['desired_coverage'] = [1, 0.99, 0.95, 0.9, 0.85, 0.80, 0.75]
                    tmp['dataset'] = filename
                    tmp['model'] = '{}_VGG'.format(meta)
                    tmp['time_to_fit'] = [time_to_fit for i in range(7)]
                    tmp['boot_iter'] = b
                    list_thetas = [(0.5, 0.5)]
                    list_thetas_app = thetas
                    new_list = list_thetas + (list_thetas_app)
                    tmp['thetas'] = new_list
                    tmp = tmp[['dataset', 'desired_coverage', 'coverage', 'accuracy', 'auc', 'perc_pos', 'brier',
                                'bss','time_to_fit','model','boot_iter','thetas']].copy()
                    results = pd.concat([results, tmp], axis=0)
                    results.to_csv(
                        'results/resultsTEST_{}_{}_{}_{}_{}_{}_add{}.csv'.format(meta, filename, max_epochs, boot_iter, cv,
                                                                       cuda_dev, add_epochs))



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    use_cuda = torch.cuda.is_available()
    # Random seed
    random.seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--boot_iter', type=int, required=False, default=1000)
    parser.add_argument('--max_epochs', type=int, required=False, default=300)
    parser.add_argument('--cv', type=int, required=False, default=5)
    parser.add_argument('--cuda', type=int, required=False, default=1)
    parser.add_argument('-m', '--metas', nargs='+', required=True)
    parser.add_argument('-f', '--files', nargs='+', required=False, default=['cats_dogs', 'cifar10'])
    parser.add_argument('--add_epochs', type=int, required=False, default=0)
    args = parser.parse_args()
    metas = args.metas
    print(metas)
    for file in tqdm(args.files):
        main(metas, max_epochs=args.max_epochs, boot_iter=args.boot_iter, cv=args.cv, cuda_dev=args.cuda,
             add_epochs=args.add_epochs,
             filename=file)

