import torch.nn.parallel
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch.utils.data
from utils_catsdogs_ import *
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from model2 import *
import pandas as pd
import argparse


def oracle_aucross_catdogs(meta='aucross', cv=5, max_epochs=300, lr=.1, boot_iter=1000, td=True,
                           gamma=.5, input_size=64, filename='cats_dogs', cuda_dev = 1,
                           quantiles=[.01, .05, .1, .15, .2, .25], num_workers=8):
    torch.manual_seed(42)
    print(max_epochs)
    print(boot_iter)
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
    if meta == 'aucross':
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
        results = pd.DataFrame()
        skf = StratifiedKFold(cv, shuffle=True, random_state=42)
        # we initialize thetas
        thetas = []
        # we initialize empty lists
        z = []
        idx = []
        # get targets
        yx = np.array([el[1] for el in trs.data])
        ys = []
        start_time = time()
        for num, d in enumerate(skf.split(indexes, yx)):
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
            path_model = 'models/AUCrossVGG16TEST_fold{}_{}_{}_{}_cuda0.pt'.format(num, max_epochs, cv,
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
                # torch.save(model_.state_dict(), path_model)
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
        path_model = 'models/AUCrossVGG16TEST_Score_{}_{}_{}_cuda0.pt'.format(max_epochs, filename, cv, cuda_dev)
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
            # torch.save(model.state_dict(), path_model)
        end_time = time()
        scores = get_scores(model, device, testloader, crit='ce')[:, 1]
        m = len(quantiles)
        bands = np.zeros(len(scores)) + m
        for i, t in enumerate(reversed(thetas)):
            t1, t2 = t[0], t[1]
            # print(i, t1, t2)
            bands[((t1 <= scores) & (scores <= t2))] = m - i - 1
        preds = get_preds(model, device, testloader, crit='ce')
        y_test = get_true(testloader)
        time_to_fit = (end_time - start_time)
        print(filename)

        results_class = pd.DataFrame()
        results_combo = pd.DataFrame()
        actual_res = [[1 - q, len(y_test[bands > i]) / len(y_test),
                       roc_auc_score(y_test[bands > i], scores[bands > i])
                       ] for i, q in enumerate(sorted(quantiles, reverse=False))]
        r = pd.DataFrame(actual_res, columns=['desired_coverage', 'coverage', 'auc', ])
        r['theta_l'] = [theta[0] for theta in thetas]
        r['theta_u'] = [theta[1] for theta in thetas]
        r['dataset'] = filename
        results_class = pd.concat([results_class, r], axis=0)
        res = pd.DataFrame()
        d = sorted([el for el in np.unique(np.round(scores, 3))])
        start_time = time()
        v = {a: len(scores[scores >= a]) for a in d}
        end_time = time()
        print((end_time - start_time))
        start_time = time()
        len_test = len(scores)
        combinations = [(a, b) for a in d for b in d if (a < b) & (((v[a] - v[b]) / len_test) <= (1 - 0.7))]
        end_time = time()
        print((end_time - start_time))
        for combo in tqdm(combinations):
            score_sel = scores[(scores <= combo[0]) | (scores > combo[1])]
            true_sel = y_test[(scores <= combo[0]) | (scores > combo[1])]
            if (np.sum(true_sel) == 0) or (np.sum(true_sel) == len(true_sel)):
                auc = 0
                coverage = 0
            else:
                auc = roc_auc_score(true_sel, score_sel)
                coverage = len(true_sel) / len(y_test)
            tmp = pd.DataFrame([[combo[0], combo[1], auc, coverage]], columns=['theta_l', 'theta_u', 'auc', 'coverage'])
            tmp = tmp[tmp['coverage'] >= .7]
            res = pd.concat([res, tmp], axis=0)
        res['dataset'] = filename
        results_combo = pd.concat([results_combo, res], axis=0)
        results_class.to_csv('AUCross_results_{}_VGG16.csv'.format(filename))
        results_combo.to_csv('combinations_results_{}_VGG16.csv'.format(filename))



if __name__ == '__main__':
    oracle_aucross_catdogs(filename='cats_dogs')
    oracle_aucross_catdogs(filename='cifar10')