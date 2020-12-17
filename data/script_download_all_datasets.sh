

# Command to download dataset:
#   bash script_download_all_datasets.sh



############
# ZINC
############

DIR=molecules/
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
fi

cd ..


############
# PATTERN and CLUSTER 
############

DIR=SBMs/
cd $DIR

FILE=SBM_CLUSTER.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl -o SBM_CLUSTER.pkl -J -L -k
fi

FILE=SBM_PATTERN.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/SBM_PATTERN.pkl -o SBM_PATTERN.pkl -J -L -k
fi


cd ..











