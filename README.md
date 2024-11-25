This is the official repo for the paper 'Cross Spatial and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting'

The latest source code will be released when the paper is accepted.

**<font color='red'>[Highlight]</font> This code is the version as of November 2024, and the updated code will be released upon acceptance of the paper.**
**<font color='red'>Part of the information will be hidden during the review phase.</font>**

## 💿Requirements

- python >= 3.7

- torch==1.13.1

All dependencies can be installed using the following command:

```
conda create -n stum python==3.7
conda activate stum
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
1. train and save the baselines.
```
python main.py --mode=train [--device] [--dataset] [--year] [--model_name] [-seed] [--batch_size] [--seq_length] [--horizon] [--input_dim] [--output_dim]
... # Please import the model from the code
# copy the following example
python main.py --device=cuda:2 --dataset=PEMS07 --years=2017  --batch_size=64 --seq_length=12 --horizon=12 --input_dim=3 --output_dim=1 --mode=train --model_name=stgcn --save='pre_trained_stgcn_model.pt'
```

2. train the vanilla version STUM model.
```
python train_stum_ori.py --enhance --num_mlrfs=4 --num_cells=8 --embed_dim=16 [--mlp] [--without_backbone] [--frozen] [--bias] [--pre_train] [--save]
# optimization setting: [--max_epochs] [--save_interval] [--patience] [--lrate] [--wdecay] [--step_size] [--gamma] [--dropout] [--clip_grad_value] [--adj_type]
```

3. train a STUM model enhanced by STGNNs. 
```
# A. We train together
python main.py --enhance --device=cuda:2 --dataset=PEMS07 --years=2017  --batch_size=64 --seq_length=12 --horizon=12 --input_dim=3 --output_dim=1 --mode=train

# B. Reload the pre-trained STGNNs used in (more efficient)
python main.py --enhance --device=cuda:2 --dataset=PEMS07 --years=2017  --batch_size=64 --seq_length=12 --horizon=12 --input_dim=3 --output_dim=1 --mode=train --pre_train='pre_trained_stgcn_model.pt'
```

4. ablation study and other analysis
Stay tuned for the latest repo/experiments

## 🔗Citing  STUM
(🌟It's very important for me~~~)

If you find this resource helpful, please consider star this repository and cite our research:
```
@article{ruan2024cross,
  title={Cross Space and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting},
  author={Ruan, Weilin and Wang, Wenzhuo and Zhong, Siru and Chen, Wei and Liu, Li and Liang, Yuxuan},
  journal={arXiv preprint arXiv:2411.09251},
  year={2024}
}
```
