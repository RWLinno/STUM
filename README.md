This is the official repo for the paper 'Cross Spatial and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting'

The lastest source code will be released when the paper accepted.

**<font color='red'>[Highlight]</font> This code is the version as of November 2024, and the updated code will be released upon acceptance of the paper.**
**<font color='red'>Part of the information will be hidden during the review phase.</font>**

## 🔗Citing  STUM
(🌟It's very important for me~~~)

If you find this resource helpful, please consider to star this repository and cite our research:

## 💿Requirements

- python >= 3.7

- torch==1.13.1

All dependencies can be installed using the following command:

```
conda create -n stlora python==3.7
conda activate stlora
pip install -r requirements.txt
```

## 📚repo structure
```
.
|   README.md
|   requirements.txt
|   train_stum_ori.py # which is used for train our proposed STUM model(vanilla version) from scratch.
|   main.py  # which is used to train the STUM model enhanced by STGNNs from scratch.
+---experiments
|   \---[model_name]
|           [saved_checkpoints.pt]
|           [train_record.log]
+---data
|   |   generate_data_for_training.py
|   +---sensor_graph
|   |   |   [adj_mx.pkl]
|   |   |   [graph_sensors.csv]
|   |    \--- ...
|   +---pems03
|   |   +---[year|[few-shot]]
|   |   |   |   his.npz
|   |   |   |   idx_test.npy
|   |   |   |   idx_train.npy
|   |   |   \---idx_val.npy
|   |    \--- ...
|   +---pems04
|   |    \--- ...
|   +---pems07
|   |    \--- ...
|   \---pems08
|        \--- ...
+---save
|    \--- ... # convenient to record embeddings / models / experimental results
+---tutorial
|    \--- ... # some codes and raw meterials for analysis and visualization
\---src
    |   __init__.py 
    +---stum
    |   |   __init__.py
    |   |   ASTUC.py
    |   |   GCN.py # here is a try to replace MLP in STUM architecture.
    |   |   MLP.py
    |   |   MLRF.py
    |   \---STUM.py
    +---baselines
    |   |   __init__.py
    |   |   ...  # baselines used in experiments
    |   \---agcrn.py 
    +---base
    |   |   basemodel.py
    |   \---engine.py
    \---utils
        |   __init__.py
        |   args.py
        |   dataloader.py
        |   graph_algo.py
        \---metrics.py
```

## 📦Dataset

You can download datasets used in the paper via this link: [Google Drive](https://drive.google.com/drive/folders/1vtfAlMufZJxzoLsdJXFasE39pfc1Xcqn?usp=sharing)
or use `./download_datasets.sh` to download datasets.


## ⭐Quick Start

```
python main.py [-dataset] [-device] [-pre_train] [-seed] [-epochs] ...
```
