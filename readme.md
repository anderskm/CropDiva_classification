# Installation
1. Install NVidia drivers: `sudo apt install nvidia-driver-470`
2. Reboot: `sudo shutdown --reboot now`
3. Create a conda environment: `conda create --name CropDiva python=3.8`
4. Activate environment: `conda activate CropDiva`
5. Install CUDA 11.8: `conda install -c conda-forge cudatoolkit=11.8.0`
6. Install CUDNN: `pip install nvidia-cudnn-cu11==8.6.0.163`
7. Set LD Libray path:
      1. `CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))`
      2. `export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH`
      3. **NOTE**: Must be done everytime the environment is activated.
10. Install TensorFlow 2.13:
      1. `pip install tensorflow==2.13.*`
      2. Test the installation: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
11. Install CUDA Compiler Driver: `conda install -c nvidia cuda-nvcc=11.8`
12. Set nvvm libdevice path:
      1. Locate libdevice: `locate nvvm\libdevice`
      2. Change to match your libdevice path. `export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/anders/anaconda3/pkgs/cuda-nvcc-12.2.140-0/`
      3. **NOTE**: Must be done everytime the environment is activated.
16. Install Python packages: `pip install -r requirements.txt`


# Usage
## Activate environment
1. Activate environment: `conda activate CropDiva`
2. Set LD Libray path:
      1. `CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))`
      2. `export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH`
      3. **NOTE**: Must be done everytime the environment is activated.
4. Set nvvm libdevice path:
      1. Locate libdevice: `locate nvvm\libdevice`
      2. Change to match your libdevice path. `export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/anders/anaconda3/pkgs/cuda-nvcc-12.2.140-0/`
      3. **NOTE**: Must be done everytime the environment is activated.

## Create datasets
1. Create dataframe of all the images: `python main_0_make_dataframe.py --output_df RNJ_test.pkl --data_folder ..\TrainingData\05SEP23_scale224px_50px\ --img_ext png`
1. Filter images: `python main_1_filter_dataframe.py --input_df RNJ_test.pkl --min_sample_count 500`
1. Split dataset into train, validation and test: `python main_2_split_dataset.py --input_df .\RNJ_test__filtered.pkl`
      
      2. **NOTE**: Take note of the hex (dataset id) at the end, as it is needed when starting a training.

## Train network
1. `python train_network.py --dataset_id 1bd80a6348083b4c4b2c7602 --dataset_folder Datasets/ --image_folder ~/CropDiva/TrainingData/05SEP23_scale224px_50px/ --image_size 224 224 --stratify --weights imagenet --finetune --batch_size 32 --optimizer_params "{'learning_rate': 0.0001}"`
2. `python train_network.py --dataset_folder Datasets --dataset_id e231d20a83d645e767d4384c --image_folder /media/data/CropDiva/data/16OCT23_Approved_scale224px_min50px_padding/ --image_size 224 224 --basenet EfficientNetV2S --pooling avg --weights imagenet --loss_func FocalCrossEntropy --optimizer Adam --optimizer_params "{'learning_rate': 0.0001}" --batch_size 32 --epochs 400 --name 16OCT_Approved`


## Useful links:
CUDA version and Compute Compability:
<https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions>
Tensorflow and cuda version
<https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible>
<https://www.tensorflow.org/install/source#gpu>