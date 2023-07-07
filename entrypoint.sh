#!/bin/bash --login
set -e
conda env list
cd /home/user/Orientation_learning/AugmentedAutoencoder 
pip install . ; 
export AE_WORKSPACE_PATH="/home/user/Orientation_learning/AugmentedAutoencoder/AAE_workspace" ; 
echo $AE_WORKSPACE_PATH ; 
export PYOPENGL_PLATFORM='egl' 
exec "$@"