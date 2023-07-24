conda activate aae_py37_tf26
cd /home/user/Orientation_learning/AugmentedAutoencoder
pip install . ; 
export AE_WORKSPACE_PATH="/home/user/Orientation_learning/AugmentedAutoencoder/AAE_w3" ; 
echo $AE_WORKSPACE_PATH ; 
export PYOPENGL_PLATFORM='egl' 

ae_train exp_group/my_autoencoder -d
