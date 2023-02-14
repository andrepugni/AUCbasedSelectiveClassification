# AUCbasedSelectiveClassification
This is the repository for AUC-based Selective Classification


## System specifics

All the code was run on a machine with Ubuntu 20.04.4 and using programming language Python 3.8.12.

## Usage

Download this repository from github and then place downladed data in 'code/data'.
We suggest to create a new environment using:

```bash
 $ conda env create -n ENVNAME --file environment.yml
  ```
Activate environment and go to the code folder by using:

```bash
 $ conda activate ENVNAME
 $ cd code
  ```


To run experiments on tabular data for Table 3:

For AUCross, SCross, PlugIn, PlugInAUC

```bash
$ python exp_tabdata.py --model lgbm --boot_iter 1000 --cv 5 --metas cross scross plugin pluginAUC
```

For SAT

```bash
$ python exp_tabdata.py --model resnet --metas sat --boot_iter 1000 --max_epochs 300
```

For SELNET
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
where DESIRED_CLASSIFIER  can take as value rf for Random Forest, logistic for Logistic Regression, resnet for ResNet and xgboost for XGBoost.


All the models used for image data are stored in this link here[](), to avoid retraining them. 
If you want to retrain models from scratch (e.g. you want to check training times),
go on exp_imdata.py and change all the lines refrerring directly to the models path.


