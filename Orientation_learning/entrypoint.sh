
v4l2-ctl --list-devices

# CREATE ENVIRONMENT
cd /home/user/Orientation_learning/ # nel Dockerfile
conda env create -f env.yml # nel Dockerfile

conda activate aae_py37_tf26 
conda install -c anaconda cython --yes
conda install -c conda-forge glfw --yes
conda install -c "conda-forge/label/cf202003" glfw --yes
pip install cyglfw3


# TRAIN AUGMENTED AUTOENCODER
conda activate aae_py37_tf26
cd /home/user/Orientation_learning/AugmentedAutoencoder
pip install . ; 
export AE_WORKSPACE_PATH="/home/user/Orientation_learning/AugmentedAutoencoder/AAE_w3" ; 
echo $AE_WORKSPACE_PATH ; 
export PYOPENGL_PLATFORM='egl' 



[Paths]
PLY_MODEL_PATH: /home/user/Orientation_learning/AugmentedAutoencoder/3D_model/manico_ASCII.PLY
BACKGROUND_IMAGES_PATH: /home/user/Orientation_learning/AugmentedAutoencoder/background_images/*.jpg

# Create dataset
ae_train exp_group/my_autoencoder -d
# Train model 
ae_train exp_group/my_autoencoder    
# Create embeddings
ae_embed exp_group/my_autoencoder
# Test on test images
python /home/user/Orientation_learning/AugmentedAutoencoder/auto_pose/test/aae_image.py exp_group/my_autoencoder -f /home/user/Orientation_learning/AugmentedAutoencoder/test_images_tool/ 


# TRAIN RETINANET
cd /home/user/Orientation_learning/AugmentedAutoencoder/
git clone https://github.com/fizyr/keras-retinanet.git
cd /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet
conda activate aae_py37_tf26
pip install . 
#train
retinanet-train  --epochs 50 --batch-size  2 --steps 64 csv /home/user/Orientation_learning/Tool\ Finder.retinanet/train/annotations_train.csv /home/user/Orientation_learning/Tool\ Finder.retinanet/classes.csv --val-annotations /home/user/Orientation_learning/Tool\ Finder.retinanet/valid/annotations_val.csv
#convert model
retinanet-convert-model /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/snapshots/resnet50_csv_13.h5 /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/snapshots/resnet50_csv_13_inference.h5
#test detector on image and save bbox
python /home/user/Orientation_learning/AugmentedAutoencoder/keras-retinanet/examples/resnet50_retinanet.py
#TEST POSE ESTIMATION WITH RETINA
python /home/user/Orientation_learning/AugmentedAutoencoder/auto_pose/test/aae_image_RETINA.py  -test_config /home/user/Orientation_learning/AugmentedAutoencoder/auto_pose/ae/cfg_eval/aae_retina_webcam.cfg -f /home/user/Orientation_learning/AugmentedAutoencoder/test_images_tool/