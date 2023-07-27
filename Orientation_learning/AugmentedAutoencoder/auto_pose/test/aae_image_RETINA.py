import cv2
try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
except:
    import tensorflow as tf
import numpy as np
import glob
import os
import time
import argparse
import configparser
from aae_retina_pose_estimator import AePoseEstimator
from auto_pose.ae import factory, utils
import pathlib

from auto_pose.ae.utils import get_dataset_path
parser = argparse.ArgumentParser()
parser.add_argument("-test_config")
parser.add_argument("-f", "--file_str", required=True, help='folder or filename to image(s)')
parser.add_argument("-vis", action='store_true', default=True)
args = parser.parse_args()

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print('Please define a workspace path:\n')
    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
    exit(-1)

test_configpath = os.path.join(workspace_path,'cfg_eval',args.test_config)
test_args = configparser.ConfigParser()
test_args.read(test_configpath)

ae_pose_est = AePoseEstimator(test_configpath)
if args.vis:
    from auto_pose.meshrenderer import meshrenderer

    ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in ae_pose_est.all_train_args]
    cad_reconst = [str(train_args.get('Dataset','MODEL')) for train_args in ae_pose_est.all_train_args]
    
    renderer = meshrenderer.Renderer(ply_model_paths, 
                    samples=1, 
                    vertex_tmp_store_folder=get_dataset_path(workspace_path),
                    vertex_scale=float(1)) # float(1) for some models




full_name = eval(test_args.get('AAE','experiment'))[0].split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''

file_str = args.file_str
if os.path.isdir(file_str):
    files = sorted(glob.glob(os.path.join(str(file_str),'*.png'))+glob.glob(os.path.join(str(file_str),'*.jpg')))
    bboxes = sorted(glob.glob(os.path.join(str(file_str)+'bboxes/','*.txt')))
else:
    files = [file_str]
    bboxes = [pathlib.Path(file_str).parents[0] + pathlib.Path(file_str).stem +'.txt']
color_dict = [(0,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10
for file,bbox in zip(files,bboxes):
        im = cv2.imread(file)
        im = cv2.resize(im, (960,720))
        with open(bbox) as f:
            bbox = f.readlines()[0].split(',')
        boxes = bbox[0].strip(']').strip('[]').split()
        boxes = [float(i) for i in boxes]
        scores, labels = float(bbox[1]),float(bbox[2])

        print(boxes, scores, labels )
        boxes, scores, labels = boxes, scores, labels = ae_pose_est.process_detection(im, [boxes], [scores], [labels] )

        all_pose_estimates, all_class_idcs = ae_pose_est.process_pose(boxes, labels, im)

        # dataset = ae_pose_est.all_datasets[0]
        # print(dataset)
        # pred_view = dataset.render_rot( all_pose_estimates[0][:3,:3],downSample = 1)

        # cv2.imshow('resized img', cv2.resize(im/255.,(256,256)))
        # cv2.imshow('pred_view', cv2.resize(pred_view,(256,256)))
        # cv2.waitKey(0)

        if args.vis:
                bgr, depth,_ = renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                            W = ae_pose_est._width,
                            H = ae_pose_est._height,
                            K = ae_pose_est._camK, 
                            # R = transform.random_rotation_matrix()[:3,:3],
                            Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                            ts = [pose_est[:3,3] for pose_est in all_pose_estimates],
                            near = 10,
                            far = 10000,
                            random_light=False,
                            phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})

                bgr = cv2.resize(bgr,(ae_pose_est._width,ae_pose_est._height))
                
                g_y = np.zeros_like(bgr)
                g_y[:,:,1]= bgr[:,:,1]    
                mask = (g_y[:,:,1]==0).astype(np.uint8)
                print(mask.shape)
                print(im.shape)
                im_bg = cv2.bitwise_and(im,im,mask=mask)                 
                image_show = cv2.addWeighted(im_bg,1,g_y,1,0)

                #cv2.imshow('pred view rendered', pred_view)
                for label,box,score in zip(labels,boxes,scores):
                    box = box.astype(np.int32)
                    xmin,ymin,xmax,ymax = box[0],box[1],box[0]+box[2],box[1]+box[3]
                    print(label)
                    cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
                    cv2.rectangle(image_show,(xmin,ymin),(xmax,ymax),(255,0,0),2)

                #cv2.imshow('', bgr)
                cv2.imshow('real', image_show)
                cv2.waitKey(0)

