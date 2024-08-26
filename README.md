# STMAE
### Revealing the Power of Masked Autoencoders in Traffic Forecasting (CIKM'2024)
by [Jiarui Sun](https://ece.illinois.edu/about/directory/grad-students/jsun57) et. al.


[[Paper](https://arxiv.org/pdf/2309.15169)]

This repository contains the official PyTorch implementation of STMAE.

We propose Spatial-Temporal Masked AutoEncoders (STMAE), a plug-and-play framework designed to enhance existing spatial-temporal models on traffic forecasting.

## Installation


### 1. Environment

<details> 
<summary>Python/conda/mamba environment</summary>
<p>

```
Coming Soon!
```
</p>
</details> 


### 2. Datasets

#### PEMS-03/04/07/08

We follow https://github.com/liuxu77/STGCL for dataset preparation. The generated data files should be placed inside **./data/pems_0<3/4/7/8>** directories.

## Evaluation
Run the following scripts to evaluate STMAE:

```
./test_all.sh
```

## Training
Run the following scripts to train STMAE:

```
python train_pf.py --cfg <configuration_file_name>
```


## Citation
If you find our work useful in your research, please consider citing our paper:
```
Coming Soon!
```
**Note:** 
We borrow parts from [AGCRN](https://github.com/LeiBAI/AGCRN), [DCRNN](https://github.com/chnsh/DCRNN_PyTorch), [MTGNN](https://github.com/nnzhan/MTGNN) and [STGCL](https://github.com/liuxu77/STGCL).