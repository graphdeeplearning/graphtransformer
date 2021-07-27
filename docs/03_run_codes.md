# Reproducibility


<br>

## 1. Usage


<br>

### In terminal

```
# Run the main file (at the root of the project)
python main_molecules_graph_regression.py --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' # for CPU
python main_molecules_graph_regression.py --gpu_id 0 --config 'configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json' # for GPU
```
The training and network parameters for each experiment is stored in a json file in the [`configs/`](../configs) directory.




<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json`](../configs/molecules_GraphTransformer_LapPE_ZINC_500k_sparse_graph_BN.json) file).  

If `out_dir = 'out/ZINC_sparse_LapPE_BN/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/ZINC_sparse_LapPE_BN/results` to view all result text files.
2. Directory `out/ZINC_sparse_LapPE_BN/checkpoints` contains model checkpoints.

#### 2.2 To see the training logs in Tensorboard on local machine
1. Go to the logs directory, i.e. `out/ZINC_sparse_LapPE_BN/logs/`.
2. Run the commands
```
source activate graph_transformer
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.3 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/ZINC_sparse_LapPE_BN/logs/`.
2. Run the [script](../scripts/TensorBoard/script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.



<br>

## 3. Reproduce results 


```
# At the root of the project 

# reproduce main results (Table 1 in paper) 
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k.sh

# reproduce WL-PE ablation results (Table 3 in paper)
bash scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k_WL_ablation.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER_500k_WL_ablation.sh
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN_500k_WL_ablation.sh
```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 

 <br>

## 4. Generate statistics obtained over mulitple runs 
After running a script, statistics (mean and standard variation) can be generated from a notebook. For example, after running the script `scripts/ZINC/script_main_molecules_graph_regression_ZINC_500k.sh`, go to the results folder `out/ZINC_sparse_LapPE_LN/results/`, and run the [notebook](../scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC.ipynb) `scripts/StatisticalResults/generate_statistics_molecules_graph_regression_ZINC.ipynb` to generate the statistics.


















<br><br><br>
