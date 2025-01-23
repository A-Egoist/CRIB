# CRIB
This is the PyTorch implementation of our paper "Beyond the Latest: Correcting Release Interval Bias in Short-video Recommendation". This repository contains the code and instructions to reproduce the experiments described in the paper.

## Repository Structure

==TODO==

```
.
├── README.md            # Instructions for running the experiments
├── requirements.txt     # Python dependencies
├── data/                # Directory for datasets (not included; see below for download instructions)
├── src/                 # Source code for experiments
│   ├── mf/              # Code for Matrix Factorization (MF) experiments
│   ├── lightgcn/        # Code for LightGCN experiments
│   ├── mf_crib/         # Code for MF-CRIB experiments
│   └── lightgcn_crib/   # Code for LightGCN-CRIB experiments
├── scripts/             # Scripts for running the experiments
└── results/             # Directory for saving experimental results
```

## Requirements

Install the required packages using the following command:

```bash
conda env create -f ./environment.yaml -n new_env_name
```

## Datasets

==TODO==

The datasets used in the experiments (e.g., KuaiRand-Pure) are not included in this repository. Please download the datasets from their respective official websites and place them in the `data/` directory. For example:

```
data/
└── movielens/
    └── ml-100k/
```

### Data preprocessing

Run the following command to process the `KuaiRand-Pure` dataset:

```bash
python src/data_processing.py --dataset_name kuairand_pure
```

## Parameters

==TODO==

Key parameters for training and evaluation:

*   --dataset_name: 
*   --model_name: 
*   --loss_name: 
*   --pre_trained: 
*   --embedding_dim: 
*   --num_layers: 
*   --lr: learning rate
*   --epochs: 
*   --batch_size: 
*   --lamb: 
*   --test_only: 
*   --alpha: 

## Simply Reproduce the Results

#### Matrix Factorization (MF)

Run the following command to reproduce the MF results:

```bash
python main.py --dataset_name kuairand_pure --model_name MF --epochs 20 --lamb 0.1 --test_only True
```

#### LightGCN

Run the following command to reproduce the LightGCN results:

```bash
python main.py --dataset_name kuairand_pure --model_name LightGCN --num_layers 2 --epochs 20 --lamb 0.1 --test_only True
```

#### MF-CRIB

Run the following command to reproduce the MF-CRIB results:

```bash
python main.py --dataset_name kuairand_pure --model_name MF_CRIB --epochs 20 --lamb 0.1 --alpha 0.5 --test_only True
```

#### LightGCN-CRIB

Run the following command to reproduce the LightGCN-CRIB results:

```bash
python main.py --dataset_name kuairand_pure --model_name LightGCN_CRIB --num_layers 2 --epochs 20 --lamb 0.05 --alpha 0.5 --test_only True
```

## Results

The results of all experiments will be saved in the `log/Top-K.log`. 

## Acknowledgments

==TODO==

### NFM, TaFR, LDRI, and LDRI-iter Experiments

These experiments are reproduced using the code provided by the authors of the [LDRI paper](https://github.com/anonymous-ldri-repo). Please refer to their repository for instructions on how to set up and run these experiments.

## Citation

If you find this repository helpful, please cite our paper.
