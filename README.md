# Bopeto: Unsupervised data cleaning for OoD detection
This repository implements Bopeto method and collects different unsupervised machine learning algorithms to detect OoD instances.
## Implemented models
For OoD detection we have used the following models. Our implementations of ALAD and DeepSVDD closely follows the original implementations already available on GitHub.
- [x] [AE]()
- [x] [ALAD]()
- [x] [DSEBM]()
- [x] [DAGMM]()
- [x] [DeepSVDD]()
- [x] [IsolationForest]()
- [x] [LocalOutlierFactor]()
- [x] [One-class SVM]()

## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn
- seaborn

## Installation
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
```
Replace `[ENV_NAME]` with the name of your environment.


Our model contains the following parameters:
- `--batch-size`: size of a training batch (**required**)
- `--dataset`: name of the selected dataset. Choices are `ciciot`, `credit`, `ecg`, `ids`, `kdd`, `kitsune` (**required**).
- `-epochs`: number of training epochs (default=20)
Please note that datasets must be stored in `.npz`. 

## Example
Cleaning KDD dataset using Bopeto:
```
$ python cleaning.py --dataset kdd 
```
Detecting OoD instances using a DAGMM on the KDD dataset:
```
$ python detection.py --dataset kdd --model dagmm --epochs 10
```
You can automate the whole process (data cleaning and OoD detection) using the following
```
$ chmod +x main.sh
$ ./main.sh
```
Make sure that your dataset is saved with a correct name as a .npz file with one key as your dataset name
