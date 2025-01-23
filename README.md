# CRIB
This is the PyTorch implementation of our paper "Beyond the Latest: Correcting Release Interval Bias in Short-video Recommendation". This repository contains the code and instructions to reproduce the experiments described in the paper.



## Repository Structure

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

The datasets used in the experiments (e.g., KuaiRand-Pure) are not included in this repository. Please download the datasets from their respective official websites and place them in the `data/` directory. For example:

```
data/
└── movielens/
    └── ml-100k/
```

## Parameters

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

Run the following command to reproduce the MF experiments:

```bash
python src/mf/train.py --config configs/mf_config.yaml
```

#### LightGCN

Run the following command to reproduce the LightGCN experiments:

```bash
python src/lightgcn/train.py --config configs/lightgcn_config.yaml
```

#### MF-CRIB

Run the following command to reproduce the MF-CRIB experiments:

```bash
python src/mf_crib/train.py --config configs/mf_crib_config.yaml
```

#### LightGCN-CRIB

Run the following command to reproduce the LightGCN-CRIB experiments:

```bash
python src/lightgcn_crib/train.py --config configs/lightgcn_crib_config.yaml
```



## Results

The results of all experiments will be saved in the `results/` directory. Each experiment will generate logs, performance metrics, and model checkpoints.

## Acknowledgments

### NFM, TaFR, LDRI, and LDRI-iter Experiments

These experiments are reproduced using the code provided by the authors of the [LDRI paper](https://github.com/anonymous-ldri-repo). Please refer to their repository for instructions on how to set up and run these experiments.

## Citation

If you find this repository helpful, please cite our paper:

```
@inproceedings{anonymous2025sigir,
  title     = {Title of Your SIGIR 2025 Paper},
  author    = {Anonymous},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2025}
}
```
