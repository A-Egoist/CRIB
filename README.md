# CRIB
This is the PyTorch implementation of our paper "Beyond the Latest: Correcting Release Interval Bias in Short-video Recommendation". This repository contains the code and instructions to reproduce the experiments described in the paper.

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

*   `--dataset_name`: Name of the dataset (`kuairand_pure`, `kuairand_1k`, `kuairec_big`, `kuaisar_small`, `kuaisar`).
*   `--model_name`: Model to use (`MF-CRIB`, or `LightGCN-CRIB`).
*   `--num_layers`: Number of layers in the LightGCN model.
*   `--lr`: Learning rate for the optimizer.
*   `--epochs`: Number of training epochs.
*   `--lamb`: Regularization coefficient.
*   `--test_only`: If `True`, runs only the testing phase.
*   `--CRIB_alpha`: Weight for combining long-term preference and short-term temporal preference.

## Simply Reproduce the Results

#### MF-CRIB

Run the following command to reproduce the CRIB with MF as the base model:

```bash
python main.py --dataset_name kuairand_pure --model_name MF-CRIB --lamb 0.1 --CRIB_alpha 0.5 --test_only True
```

#### LightGCN-CRIB

Run the following command to reproduce the CRIB with LightGCN as the base model:

```bash
python main.py --dataset_name kuairand_pure --model_name LightGCN-CRIB --lamb 0.05 --CRIB_alpha 0.5 --test_only True
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
    python main.py --dataset_name kuairand_pure --model_name MF-CRIB --epochs 20 --lamb 0.1 --CRIB_alpha 0.5
    ```

    This will train the model for 20 epochs using the specified regularization coefficient (`lamb`) and weight for preference combination (`CRIB_alpha`).

3.  For other datasets and models, adjust the dataset name, model name, and parameters according to the provided hyperparameter table:

    |               | MF-CRIB                         | LightGCN-CRIB                    |
    | ------------- | ------------------------------- | -------------------------------- |
    | KuaiRand-Pure | lr=0.0001, lamb=0.1, alpha=0.4  | lr=0.0001, lamb=0.05, alpha=0.5  |
    | KuaiRand-1K   | lr=0.0001, lamb=0.05, alpha=0.6 | lr=0.0001, lamb=0.05, alpha=0.15 |
    | KuaiRec-big   | lr=0.0001, lamb=0.01, alpha=0.9 | lr=0.0001, lamb=0.01, alpha=0.95 |
    | KuaiSAR-small | lr=0.0001, lamb=0.1, alpha=0.9  | lr=0.0001, lamb=0.01, alpha=0.8  |
    | KuaiSAR       | lr=0.0001, lamb=0.01, alpha=0.8 | lr=0.0001, lamb=0.01, alpha=0.75 |

## Citation

If you find this repository helpful, please cite our paper.

