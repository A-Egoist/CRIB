# CRIB
This is the PyTorch implementation of our paper "Beyond the Latest: Correcting Release Interval Bias in Short-video Recommendation". This repository contains the code and instructions to reproduce the experiments described in the paper.

## Repository Structure

```
.
│  .gitignore
│  environment.yaml      # Environment file
│  main.py               # Main script to run experiments
│  README.md
│
├─data                   # Directory for datasets
│      kuairand_pure-test.csv   # Test dataset
│      kuairand_pure-train.csv  # Training dataset
│      kuairand_pure-valid.csv  # Validation dataset
│      README.md                # Dataset description
│
├─log                    # Directory for logs
│      Top-K.log         # Log file for Top-K results
│
├─save_model             # Directory for saved models
│      kuairand_pure-LightGCN-BCE-0.1.pt        # LightGCN model checkpoint
│      kuairand_pure-LightGCN_CRIB-BCE-0.05.pt  # LightGCN-CRIB model checkpoint
│      kuairand_pure-MF-BCE-0.1.pt              # MF model checkpoint
│      kuairand_pure-MF_CRIB-BCE-0.1.pt         # MF-CRIB model checkpoint
│
└─src                    # Source code for the project
    │  data_processing.py  # Preprocessing and data loading
    │  metrics.py          # Evaluation metrics implementation
    │  test.py             # Testing script
    │  train.py            # Training script
    │  utils.py            # Utility functions
    │  __init__.py
    │
    └─model              # Model implementations
            LightGCN.py      # LightGCN model
            LightGCN_CRIB.py # LightGCN-CRIB model
            MF.py            # Matrix Factorization model
            MF_CRIB.py       # MF-CRIB model
            __init__.py
```

## Requirements

Install the required packages using the following command:

```bash
conda env create -f ./environment.yaml -n new_env_name
```

## Datasets

The datasets used in the experiments (i.e., [KuaiRand](https://kuairand.com/), [KuaiRec](https://kuairec.com/), and [KuaiSAR](https://kuaisar.github.io/)) are not included in this repository. Please download the datasets from their respective official websites and place them in the `data/` directory. For more details, see the `data/README.md` file. Due to the large size of the datasets, only the processed version of the `KuaiRand-Pure` dataset is provided in this repository.

### Data preprocessing

Run the following command to process the `KuaiRand-Pure` dataset:

```bash
python src/data_processing.py --dataset_name kuairand_pure
```

After preprocessing, three new files will be generated in `data/`:

*   `kuairand_pure-train.csv`: Processed Training dataset
*   `kuairand_pure-valid.csv`: Processed Validation dataset
*   `kuairand_pure-test.csv`: Processed Test dataset

These files are used for training, validation, and testing the models.

## Parameters

Key parameters for training and evaluation:

*   `--dataset_name`: Name of the dataset (`kuairand_pure`, `kuairand_1k`, `kuairec_small`, `kuairec_big`, `kuaisar_small`, `kuaisar`).
*   `--model_name`: Model to use (`MF`, `LightGCN`, `MF_CRIB`, or `LightGCN_CRIB`).
*   `--num_layers`: Number of layers in the LightGCN model.
*   `--lr`: Learning rate for the optimizer.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Batch size for training.
*   `--lamb`: Regularization coefficient.
*   `--test_only`: If `True`, runs only the testing phase.
*   `--alpha`: Weight for combining long-term preference and short-term temporal preference.

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

The results of all results will be saved in the `log/Top-K.log`.

## Train Our CRIB from Scratch

Follow these steps to train an MF-CRIB model from scratch:

1.  Preprocess the dataset using the following command:

    ```bash
    python src/data_processing.py --dataset_name kuairand_pure
    ```

    This will generate the processed training, validation, and test datasets in the `data/` directory.

2.  Train the MF-CRIB model with the following command:

    ```bash
    python main.py --dataset_name kuairand_pure --model_name MF_CRIB --epochs 20 --lamb 0.1 --alpha 0.5
    ```

    This will train the model for 20 epochs using the specified regularization coefficient (`lamb`) and weight for preference combination (`alpha`).

3.  For other datasets and models, adjust the dataset name, model name, and parameters according to the provided hyperparameter table:

    |               | MF        | MF-CRIB               | LightGCN   | LightGCN-CRIB         |
    | ------------- | --------- | --------------------- | ---------- | --------------------- |
    | KuaiRand-Pure | lamb=0.1  | lamb=0.1, alpha=0.5   | lamb=0.1   | lamb=0.05, alpha=0.5  |
    | KuaiRand-1K   | lamb=0.01 | lamb=0.1, alpha=0.85  | lamb=0.1   | lamb=0.05, alpha=0.15 |
    | KuaiRec-small | lamb=0.3  | lamb=0.1, alpha=0.95  | lamb=0.015 | lamb=0.01, alpha=0.85 |
    | KuaiRec-big   | lamb=0.26 | lamb=0.01, alpha=0.85 | lamb=0.1   | lamb=0.01, alpha=0.95 |
    | KuaiSAR-small | lamb=0.1  | lamb=0.1, alpha=0.8   | lamb=0.1   | lamb=0.01, alpha=0.8  |
    | KuaiSAR       | lamb=0.1  | lamb=0.01, alpha=0.7  | lamb=0.5   | lamb=0.01, alpha=0.75 |

## Acknowledgments

The implementations of NFM, TaFR, LDRI, and LDRI-iter from the [LDRI](https://github.com/ECNU-Text-Computing/LDRI) GitHub repository were used for performance comparison in this work. We acknowledge the authors of LDRI for making their code publicly available, which greatly facilitated our experimental benchmarks.

## Citation

If you find this repository helpful, please cite our paper.
