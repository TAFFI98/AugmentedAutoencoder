# CREATE ENVIRONMENT
cd /home/user/Orientation_learning/
conda env create -f env.yml 
conda activate aae_py37_tf26 
conda install -c anaconda cython 
conda install -c conda-forge glfw 
conda install -c "conda-forge/label/cf202003" glfw 
pip install cyglfw3


# TRAIN AUGMENTED AUTOENCODER
conda activate aae_py37_tf26
cd /home/user/Orientation_learning/AugmentedAutoencoder
pip install . ; 
export AE_WORKSPACE_PATH="/home/user/Orientation_learning/AugmentedAutoencoder/AAE_w3" ; 
echo $AE_WORKSPACE_PATH ; 
export PYOPENGL_PLATFORM='egl' 

ae_train exp_group/my_autoencoder -d


# TRAIN RETINANET
cd /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet
conda activate aae_py37_tf26
pip install . 
retinanet-train  --epochs 50 --batch-size  2 --steps 64 csv /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/examples/dataset/annotations.csv /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/examples/dataset/classes.csv --val-annotations /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/examples/dataset/annotations_val.csv 