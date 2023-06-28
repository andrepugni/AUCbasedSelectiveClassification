# AUCbasedSelectiveClassification
This is the repository for [AUC-based Selective Classification](https://proceedings.mlr.press/v206/pugnana23a.html).


## System specifics

All the code was run on a machine with Ubuntu 20.04.4 and using programming language Python 3.8.12.

## Usage

Download this repository from GitHub.

Data can be downloaded from [here](https://www.dropbox.com/sh/114h860wxf85q0j/AAAI7bFVthqWWC5U8iaRpzSJa?dl=0).
Place them in folder 'code/data'.

Deep Neural Network models can be downloaded from [here](https://www.dropbox.com/sh/zwtskpq5f4tuuh0/AABEWccp0In_KqRaCSiqRGBPa?dl=0).
Place them in folder 'code/models'.

We suggest to create a new environment using:

```bash
 $ conda create -n ENVNAME --file environment.yml
  ```
Activate environment and go to the code folder by using:

```bash
 $ conda activate ENVNAME
 $ cd code
  ```


To run experiments on tabular data for Table 3 AUCross, SCross, PlugIn, PlugInAUC:

```bash
$ python exp_tabdata.py --model lgbm --boot_iter 1000 --cv 5 --metas cross scross plugin pluginAUC
```

For Table 3 SAT:

```bash
$ python exp_tabdata.py --model resnet --metas sat --boot_iter 1000 --max_epochs 300
```

For Table 3 SELNET:
```bash
$ python exp_tabdata_selnet.py --model resnet --boot_iter 1000 --max_epochs 300
```

To run experiments for CatsVsDogs and CIFAR10 for Table 3 (check the paths):
```bash
$ python exp_imdata.py --metas aucross scross plugin pluginAUC selnet sat
```

To run experiments for Table 2

```bash
$ python exp_oracle_tabdata.py
$ python exp_oracle_imdata.py
```



To run experiments for Appendix results on classifiers

```bash
$ python exp_tabdata.py --model DESIRED_CLASSIFIER --boot_iter 1000 --cv 5 --metas cross scross plugin pluginAUC
```
where DESIRED_CLASSIFIER  can take as value 'rf' for Random Forest, 'logistic' for Logistic Regression, 'resnet' for ResNet and 'xgboost' for XGBoost.

If you want to retrain models from scratch (e.g. you want to check training times),
go on exp_imdata.py and change all the lines referring directly to the models' path.


