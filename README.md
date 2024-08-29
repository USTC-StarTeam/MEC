# Readme

This is the source code for the paper "MEC: A Model-Agnostic Embedding Compression Framework For CTR Prediction."

## Data Preprocessing

Currently using [Criteo_x1](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x1) and [Avazu_x1](https://github.com/reczoo/Datasets/tree/main/Avazu/Avazu_x1) from fuxictr. Please download the target files into the data folder. For the Criteo dataset, the default input path is `/data/criteo` with the following structure:

```
|-criteo
    |-test.csv
    |-train.csv
    |-valid.csv
```

Convert the data to parquet format by running in the root directory:

```bash
cd build_csv_to_parquet
python build_criteo_to_parquet.py
```

After conversion, check if the following files are generated under `/data/criteo`:

```
|-criteo
    |-test.csv
    |-train.csv
    |-valid.csv
    |-criteo_x1_default
        |-feature_map.json
        |-feature_processor.pkl
        |-feature_vocab.json
        |-test.parquet
        |-train.parquet
        |-valid.parquet
```

For the Avazu dataset, replace `build_criteo_to_parquet.py` with `build_avazu_to_parquet.py`.

## Recommendation Model Pretraining (Stage 1)

For initial pretraining, run the following in the current folder:

```bash
cd pretrain_embedding
python pretrain.py --model deepfm --dataset criteo_x1_default
python pretrain.py --model deepfm --dataset avazu_x1_default
```

## PQ Pretraining (Stage 2)

Generate PQ code and codebook:

```bash
cd pretrain_embedding
python vqvae_with_freq_and_contras.py --model {model} --dataset_id {dataset_id} --layers {layers} --code_dim {code_dim} --cut_down 100000 --gpu {gpu_id} --batch_size {batch_size}
```

An example of training:

```bash
cd pretrain_embedding
python vqvae_with_freq_and_contras.py --model deepfm --dataset_id criteo_x1_default --layers 4 --code_dim 256 --cut_down 100000 --gpu 0 --batch_size 12000
```

## Main Program Training (Stage 3)

Example of main program training:

```bash
cd main_ctr
python main_run.py --verbose 0 --embedding vqemb --useid False --pre_emb_model deepfm --dataset_id criteo_x1_default --expid GDCN_default --layers 4 --code_dim 2048 --cutdown 100000 --use_freq True --gpu 1
```

The following hyperparameters can be modified:

```
verbose: 0 means no detailed information is printed, 1 means detailed information is printed.
pre_emb_model: Choose from fm, deepfm, dcnv2.
dataset_id: Choose from criteo_x1_default and avazu_x1_default.
expid: Choose from PNN_default, GDCN_default.
code_dim: Choose from 256, 512, 1024, 2048.
gpu: GPU index to use, starting from 0.
```