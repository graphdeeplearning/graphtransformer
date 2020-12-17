# Download datasets

All the datasets work with DGL 0.5.x or later. Please update the environment using the yml files in the root directory if the use of these datasets throw error(s).



<br>

## 1. ZINC molecular dataset
ZINC size is 58.9MB.  

```
# At the root of the project
cd data/ 
bash script_download_molecules.sh
```
Script [script_download_molecules.sh](../data/script_download_molecules.sh) is located here. Refer to [benchmarking-gnns repo](https://github.com/graphdeeplearning/benchmarking-gnns) for details on preparation.


<br>

## 2. PATTERN/CLUSTER SBM datasets
PATTERN size is 1.98GB and CLUSTER size is 1.26GB.

```
# At the root of the project
cd data/ 
bash script_download_SBMs.sh
```
Script [script_download_SBMs.sh](../data/script_download_SBMs.sh) is located here. Refer to [benchmarking-gnns repo](https://github.com/graphdeeplearning/benchmarking-gnns) for details on preparation.

<br>

## 3. All datasets

```
# At the root of the project
cd data/ 
bash script_download_all_datasets.sh
```

Script [script_download_all_datasets.sh](../data/script_download_all_datasets.sh) is located here. 


<br><br><br>
