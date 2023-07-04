#!/bin/bash --login
set -e
conda activate aae_py37_tf26
cd /home/user/Orientation_learning/AugmentedAutoencoder
pip install .
export AE_WORKSPACE_PATH="$(pwd)/AAE_workspace"
echo $AE_WORKSPACE_PATH


export PYOPENGL_PLATFORM='egl'


exec "$@"


