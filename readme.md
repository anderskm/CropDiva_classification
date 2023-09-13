# Installation
1. conda create --name CropDiva python=3.8
2. conda activate CropDiva
2. conda install -c conda-forge cudatoolkit=11.8.0
3. pip install nvidia-cudnn-cu11==8.6.0.163
3. conda install -c nvidia cuda-nvcc=11.8
3. Locate libdevice: locate nvvm\libdevice
3. Change to match your libdevice path. export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/anders/anaconda3/pkgs/cuda-nvcc-12.2.140-0/
3. pip install -r requirements.txt


# Usage
## Create datasets
1. python main_0_make_dataframe.py
1. python main_1_filter_dataframe.py
1. python main_2_split_dataset.py

## Train network
1. python train_network.py --dataset_id 1bd80a6348083b4c4b2c7602 --dataset_folder Datasets/ --image_folder ~/CropDiva/TrainingData/05SEP23_scale224px_50px/ --image_size 224 224 --stratify --weights imagenet --finetune --batch_size 32 --optimizer_params "{'learning_rate': 0.0001}"